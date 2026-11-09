"""Score saved checkpoints on an external cohort (AUROC, sharp AUROC, severity Spearman), without retraining."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from prd.config import Config
from prd.data import AudioClassifierDataset
from prd.models import ClassifierHead, ConsensusAudioModel, get_audio_encoder
from prd.utils import save_json

ENCODERS = ("CNN", "CRNN", "TRANSFORMER")


def _encoder_of(parts: tuple[str, ...]) -> str:
    for part in parts:
        if part in ENCODERS:
            return part
    return "CNN"


def _build(method: str, encoder: str) -> torch.nn.Module:
    if method == "dtca":
        return ConsensusAudioModel(get_audio_encoder(encoder), Config.NUM_CLASSES, Config.SHARED_DIM)
    return ClassifierHead(get_audio_encoder(encoder), Config.NUM_CLASSES)


@torch.no_grad()
def _patient_probabilities(model, loader) -> pd.DataFrame:
    model.eval()
    accumulated: dict[int, list[np.ndarray]] = {}
    truth: dict[int, int] = {}
    for spec, label, pid in loader:
        probabilities = torch.softmax(model(spec.to(Config.DEVICE)), dim=1).cpu().numpy()
        for index in range(len(pid)):
            key = int(pid[index])
            accumulated.setdefault(key, []).append(probabilities[index])
            truth[key] = int(label[index])
    rows = [{"patient_id": key, "y_true": truth[key],
             **{f"prob_{name}": value for name, value in zip(Config.CLASSES, np.mean(values, axis=0))}}
            for key, values in accumulated.items()]
    return pd.DataFrame(rows)


def _scores(frame: pd.DataFrame, gold: pd.Series | None) -> dict:
    healthy, copd = f"prob_{Config.CLASSES[0]}", f"prob_{Config.CLASSES[1]}"
    score = frame[copd] / (frame[healthy] + frame[copd] + 1e-9)
    out: dict[str, float] = {"patients": int(len(frame))}
    binary = frame["y_true"].isin([0, 1])
    if binary.sum() > 5 and frame.loc[binary, "y_true"].nunique() == 2:
        out["auroc"] = float(roc_auc_score(frame.loc[binary, "y_true"], score[binary]))
    if gold is not None:
        grades = gold.reindex(frame["patient_id"]).to_numpy(dtype=float)
        valid = ~np.isnan(grades)
        if valid.sum() > 5:
            rho, p = spearmanr(grades[valid], score.to_numpy()[valid])
            out["severity_spearman"] = float(rho)
            out["severity_pvalue"] = float(p)
        sharp = valid & np.isin(grades, [0, 3, 4])
        if sharp.sum() > 5 and len(set(grades[sharp])) > 1:
            out["auroc_sharp"] = float(
                roc_auc_score((grades[sharp] >= 3).astype(int), score.to_numpy()[sharp])
            )
            out["sharp_patients"] = int(sharp.sum())
    return out


def _parse_run(run: str) -> dict:
    info = {"view": "", "loss": "", "knob": float("nan"), "age": "", "ablation": "none",
            "seed": float("nan")}
    for part in str(run).split("/"):
        if part.startswith("seed"):
            info["seed"] = int(part[4:])
        elif part.startswith("abl_"):
            info["ablation"] = part[4:]
        elif part.startswith("dl_"):
            info["loss"] = part[3:]
        elif part.startswith("age"):
            info["age"] = part[3:]
        elif part.startswith("w") and part[1:].replace(".", "", 1).isdigit():
            info["knob"] = float(part[1:])
        elif part in ("ct_clinical", "omics"):
            info["view"] = part
    return info


def paired_summary(table: pd.DataFrame) -> pd.DataFrame:
    # the external cohort is fixed, so a method and its baseline differ only by
    # training seed; pairing on seed removes that shared variance
    if table.empty:
        return table
    parsed = pd.DataFrame([_parse_run(r) for r in table["run"]])
    frame = pd.concat([table.reset_index(drop=True), parsed], axis=1)
    metrics = [c for c in ("auroc", "auroc_sharp", "severity_spearman") if c in frame.columns]
    base = frame[frame["method"] == "baseline_audio"]
    lookup = {m: {(r.encoder, r.seed): getattr(r, m) for r in base.itertuples()} for m in metrics}

    keys = ["method", "view", "encoder", "loss", "knob", "age", "ablation"]
    rows = []
    for config, group in frame[frame["method"] != "baseline_audio"].groupby(keys, dropna=False):
        row = dict(zip(keys, config))
        row["n_seeds"] = int(len(group))
        for metric in metrics:
            deltas = [getattr(r, metric) - lookup[metric][(r.encoder, r.seed)]
                      for r in group.itertuples()
                      if (r.encoder, r.seed) in lookup[metric]
                      and pd.notna(getattr(r, metric))]
            if deltas:
                values = np.asarray(deltas, dtype=float)
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"delta_{metric}"] = float(values.mean())
                row[f"n_positive_{metric}"] = int((values > 0).sum())
        rows.append(row)
    for (encoder,), group in base.groupby(["encoder"]):
        row = {"method": "baseline_audio", "encoder": encoder, "view": "", "loss": "",
               "knob": float("nan"), "age": "", "ablation": "none", "n_seeds": int(len(group))}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    sort_key = "delta_auroc" if "delta_auroc" in out.columns else out.columns[0]
    return out.sort_values(sort_key, ascending=False, na_position="last")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved checkpoints on an external cohort.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--processed", required=True)
    parser.add_argument("--tag", default="ext")
    parser.add_argument("--results_root", default=None)
    args = parser.parse_args()

    root = Path(args.results_root) if args.results_root else Config.RESULTS_DIR
    manifest = pd.read_csv(Config.PROJECT_ROOT / args.manifest)
    processed = Config.PROJECT_ROOT / args.processed
    gold = (manifest.drop_duplicates("pid").set_index("pid")["gold"]
            if "gold" in manifest.columns else None)
    loader = DataLoader(
        AudioClassifierDataset(manifest, processed_dir=processed),
        batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY,
    )

    rows = []
    checkpoints = sorted(root.glob("*/**/checkpoints/fold_0.pth"))
    print(f"found {len(checkpoints)} run(s) under {root}")
    for first in checkpoints:
        run_dir = first.parent.parent
        parts = run_dir.relative_to(root).parts
        method, encoder = parts[0], _encoder_of(parts)
        folds = sorted(run_dir.glob("checkpoints/fold_*.pth"))
        frames = []
        for fold_path in folds:
            model = _build(method, encoder).to(Config.DEVICE)
            try:
                model.load_state_dict(torch.load(fold_path, map_location=Config.DEVICE))
            except RuntimeError as error:  # architecture mismatch
                print(f"  skip {run_dir.name}: {str(error)[:80]}")
                frames = []
                break
            frames.append(_patient_probabilities(model, loader))
        if not frames:
            continue
        columns = [f"prob_{name}" for name in Config.CLASSES]
        ensemble = (pd.concat(frames, ignore_index=True)
                    .groupby("patient_id", as_index=False)
                    .agg({"y_true": "first", **{c: "mean" for c in columns}}))
        (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
        ensemble.to_csv(run_dir / "predictions" / f"{args.tag}_preds_ensemble.csv", index=False)
        rows.append({"run": str(run_dir.relative_to(root)), "method": method,
                     "encoder": encoder, "folds": len(frames), **_scores(ensemble, gold)})
        print(f"  {rows[-1]['run'][:60]:60s} {rows[-1]}")

    table = pd.DataFrame(rows)
    out_dir = root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"external_{args.tag}.csv", index=False)

    paired = paired_summary(table)
    if not paired.empty:
        paired.to_csv(out_dir / f"external_{args.tag}_paired.csv", index=False)
        show = [c for c in ("method", "view", "encoder", "loss", "ablation", "n_seeds",
                            "auroc_mean", "delta_auroc", "n_positive_auroc",
                            "severity_spearman_mean", "delta_severity_spearman")
                if c in paired.columns]
        print("\npaired vs same-encoder baseline (top rows):")
        print(paired[show].head(15).to_string(index=False))

    save_json({"manifest": args.manifest, "runs": len(table)},
              out_dir / f"external_{args.tag}_meta.json")
    print(f"\nsaved -> {out_dir / f'external_{args.tag}.csv'}")
    print(f"saved -> {out_dir / f'external_{args.tag}_paired.csv'}")


if __name__ == "__main__":
    main()
