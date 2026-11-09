"""Train the audio student with masked distillation onto the bridge targets."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from prd.config import Config, Hparams
from prd.data import AudioClassifierDataset, DistilledAudioDataset
from prd.losses import masked_distillation_loss
from prd.models import ClassifierHead, DistillationProjector, get_audio_encoder
from prd.splitting import build_nested_patient_splits, select_patient_rows
from prd.utils import (
    EarlyStopping,
    bootstrap_ci,
    evaluate_patient_level,
    evaluate_task_loss,
    fold_metrics,
    macro_f1,
    method_run_dir,
    patient_weighted_sampler,
    prediction_frame,
    save_classification_report,
    save_json,
)

# controls, ordered by how much of the target they remove (also used by audit_shuffle.py)
ABLATIONS = [
    "none",               # full bridge: label + sex + age
    "class_prototype",    # label-only groups
    "shuffled_targets",   # within-class permutation
    "random_prototype",   # one random norm-matched vector per class
    "crossed_targets",    # real teacher targets drawn from the opposite class
    "global_prototype",   # a single target for every file, no class dependence
    "random_teacher",     # same bridge, untrained teacher encoder
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Masked feature distillation student.")
    parser.add_argument("--seed", type=int, default=Config.SEED)
    parser.add_argument(
        "--encoder",
        type=str,
        default=Config.AUDIO_ENCODER,
        choices=["CNN", "CRNN", "TRANSFORMER"],
    )
    parser.add_argument("--view", type=str, default="omics", choices=["omics", "ct_clinical"])
    parser.add_argument("--distill_weight", type=float, default=Hparams.DISTILL_WEIGHT)
    parser.add_argument(
        "--distill_loss", type=str, default=Hparams.DISTILL_LOSS,
        choices=["cosine", "mse", "smooth_l1", "cosine_mse", "relational"],
        help="relational = RKD",
    )
    parser.add_argument(
        "--age_tolerance", type=float, default=None,
        help="use bridge targets built at this age window; omit for the default file",
    )
    parser.add_argument(
        "--ablation", type=str, default="none", choices=ABLATIONS,
        help="target control; see ABLATIONS",
    )
    return parser.parse_args()


def resolve_target_path(view: str, ablation: str, seed: int, age_tolerance=None) -> Path:
    """Path to this run's targets; ablated targets go to a temporary file."""
    base = Config.PROJECT_ROOT / "data" / "distillation"
    if ablation == "random_teacher":
        # built separately by bridge.py --random_teacher
        if age_tolerance is None:
            raise ValueError("--ablation random_teacher requires --age_tolerance "
                             "(targets are written per tolerance and seed).")
        path = base / f"{view}_targets_randinit_age{int(age_tolerance)}_seed{seed}.npy"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing random-init targets: {path}. Build them first with "
                f"python scripts/bridge.py --view {view} --seed {seed} "
                f"--age_tolerance {int(age_tolerance)} --random_teacher")
        return path
    if age_tolerance is None:
        candidates = [base / f"{view}_targets.npy"]
    else:
        candidates = [
            base / f"{view}_targets_age{int(age_tolerance)}_seed{seed}.npy",
            base / f"{view}_targets_age{int(age_tolerance)}.npy",
        ]
    target_path = next((p for p in candidates if p.is_file()), None)
    if target_path is None:
        raise FileNotFoundError(
            f"Missing distillation targets for view '{view}': tried {[str(p) for p in candidates]}. "
            f"Generate them first with bridge.py"
            + ("" if age_tolerance is None else f" --age_tolerance {age_tolerance}") + "."
        )
    if ablation == "none":
        return target_path

    target_map = np.load(target_path, allow_pickle=True).item()
    manifest = pd.read_csv(Config.FULL_CSV).set_index("filename")["disease"]
    keys = [k for k in target_map if k in manifest.index]
    classes = manifest.loc[keys].to_numpy()
    rng = np.random.default_rng(seed)
    new_map = dict(target_map)

    if ablation == "shuffled_targets":
        # within class: ~96% of targets are COPD, so a global shuffle would
        # leave most files with a COPD-like target anyway
        for label in np.unique(classes):
            group = [k for k, c in zip(keys, classes) if c == label]
            permuted = list(group)
            rng.shuffle(permuted)
            for destination, source in zip(group, permuted):
                new_map[destination] = target_map[source]
    elif ablation == "class_prototype":
        for label in np.unique(classes):
            group = [k for k, c in zip(keys, classes) if c == label]
            mean = np.mean(np.stack([target_map[k] for k in group]), axis=0)
            for k in group:
                new_map[k] = mean
    elif ablation == "random_prototype":
        # norm-matched to the class mean so the loss stays on the same scale
        for label in np.unique(classes):
            group = [k for k, c in zip(keys, classes) if c == label]
            mean = np.mean(np.stack([target_map[k] for k in group]), axis=0)
            vector = rng.standard_normal(mean.shape).astype(mean.dtype)
            vector *= float(np.linalg.norm(mean)) / max(float(np.linalg.norm(vector)), 1e-8)
            for k in group:
                new_map[k] = vector
    elif ablation == "crossed_targets":
        # within-class spread is small next to the between-class distance, so
        # only crossing classes actually breaks the label-to-target link
        for label in np.unique(classes):
            group = [k for k, c in zip(keys, classes) if c == label]
            donors = [k for k, c in zip(keys, classes) if c != label]
            if not donors:
                continue
            picks = rng.integers(0, len(donors), size=len(group))
            for destination, pick in zip(group, picks):
                new_map[destination] = target_map[donors[int(pick)]]
    elif ablation == "global_prototype":
        # mean of the class means; a file-weighted mean would sit on the COPD
        # centroid and behave like class_prototype
        centroid = np.mean(
            np.stack([
                np.mean(np.stack([target_map[k] for k, c in zip(keys, classes) if c == label]),
                        axis=0)
                for label in np.unique(classes)
            ]),
            axis=0,
        )
        for k in keys:
            new_map[k] = centroid
    else:
        raise ValueError(f"unknown ablation: {ablation!r}")

    handle = tempfile.NamedTemporaryFile(
        prefix=f"{view}_targets_{ablation}_seed{seed}_", suffix=".npy", delete=False
    )
    handle.close()
    temp_path = Path(handle.name)
    np.save(temp_path, new_map)
    return temp_path


def make_loader(frame, target_path, *, seed, fold, train) -> DataLoader:
    dataset = DistilledAudioDataset(frame, target_path)
    if train:
        generator = Config.get_generator(seed * 1000 + fold)
        return DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            sampler=patient_weighted_sampler(frame, generator),
            shuffle=False,
            drop_last=True,
            num_workers=Config.NUM_WORKERS,
            persistent_workers=Config.NUM_WORKERS > 0,
            pin_memory=Config.PIN_MEMORY,
            worker_init_fn=Config.seed_worker,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        persistent_workers=Config.NUM_WORKERS > 0,
        pin_memory=Config.PIN_MEMORY,
        worker_init_fn=Config.seed_worker,
    )


def train_one_fold(train_frame, target_path, args, fold, checkpoint_path, val_loader):
    """Train the student and projector on one outer fold."""
    model = ClassifierHead(get_audio_encoder(args.encoder), Config.NUM_CLASSES).to(Config.DEVICE)
    projector = DistillationProjector(512, 512).to(Config.DEVICE)
    # same learning rates as the baseline, so the auxiliary loss is the only difference
    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": Config.BASELINE_ENCODER_LR},
            {"params": model.head.parameters(), "lr": Config.HEAD_LR},
            {"params": projector.parameters(), "lr": Config.HEAD_LR},
        ]
    )
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE, path=str(checkpoint_path), maximize=False,
        min_epochs=Config.MIN_EPOCHS,
    )
    train_loader = make_loader(train_frame, target_path, seed=args.seed, fold=fold, train=True)

    best_epoch, best_score, best_loss = -1, 0.0, np.inf
    for epoch in range(Config.FINETUNE_EPOCHS):
        model.train()
        projector.train()
        current_weight = args.distill_weight
        distill_total, mask_total, n_batches = 0.0, 0.0, 0
        for spec, label, target, mask, _ in train_loader:
            spec, label = spec.to(Config.DEVICE), label.to(Config.DEVICE)
            target, mask = target.to(Config.DEVICE), mask.to(Config.DEVICE)
            optimizer.zero_grad()
            logits, feat = model(spec, return_features=True)
            ce = criterion(logits, label)
            distill = masked_distillation_loss(projector(feat), target, mask, kind=args.distill_loss)
            loss = ce + current_weight * distill
            loss.backward()
            optimizer.step()
            distill_total += float(distill.detach().cpu())
            mask_total += float(mask.float().mean().detach().cpu())
            n_batches += 1
        # an empty mask would silently turn this into the baseline
        mask_fraction = mask_total / max(n_batches, 1)
        if mask_fraction == 0.0:
            raise RuntimeError(
                f"No training sample had a teacher target (fold {fold}). The targets in "
                f"{target_path} do not match the manifest filenames -- rebuild them with bridge.py."
            )

        val_loss = evaluate_task_loss(model, val_loader)
        truth, pred, _, _ = evaluate_patient_level(model, val_loader)
        score = macro_f1(truth, pred)
        print(f"   [distilled fold {fold}] epoch {epoch + 1:02d} | w {current_weight:.2f} "
              f"| distill {distill_total / max(n_batches, 1):.4f} | matched {mask_fraction:.2f} "
              f"| val loss {val_loss:.4f} | val macro-F1 {score:.4f}", flush=True)
        if val_loss < best_loss:
            best_loss, best_score, best_epoch = val_loss, score, epoch + 1
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
    return model, best_epoch, best_score


def main() -> None:
    args = parse_args()
    Config.set_seed(args.seed)
    target_path = resolve_target_path(args.view, args.ablation, args.seed, args.age_tolerance)

    # runs with an explicit age window are also tagged by window and loss
    study_mode = args.age_tolerance is not None
    tag = f"{args.view}/{args.encoder}"
    if study_mode:
        tag += f"/age{int(args.age_tolerance)}/dl_{args.distill_loss}"
    tag += f"/w{args.distill_weight}/seed{args.seed}"
    if args.ablation != "none":
        tag += f"/abl_{args.ablation}"
    run = method_run_dir("distilled", tag)
    save_json(
        {
            "method": "distilled",
            "view": args.view,
            "encoder": args.encoder,
            "seed": args.seed,
            "distill_weight": args.distill_weight,
            "distill_loss": args.distill_loss,
            "age_tolerance": args.age_tolerance,
            "distill_weight_schedule": "constant (no ramp)",
            "model_selection": "inner-validation cross-entropy loss",
            "min_epochs": Config.MIN_EPOCHS,
            "patience": Config.PATIENCE,
            "ablation": args.ablation,
            "target_path": str(target_path),
            "num_folds": Config.NUM_FOLDS,
            "inner_validation_fraction": Config.INNER_VALIDATION_FRACTION,
            "epochs": Config.FINETUNE_EPOCHS,
            "batch_size": Config.BATCH_SIZE,
            "encoder_lr": 1e-4,
            "projector_lr": Config.HEAD_LR,
            "classes": Config.CLASSES,
        },
        run / "config.json",
    )

    development = pd.read_csv(Config.FULL_CSV)
    patient_labels, splits = build_nested_patient_splits(
        development,
        folds=Config.NUM_FOLDS,
        repeat_seed=args.seed,
        validation_fraction=Config.INNER_VALIDATION_FRACTION,
    )

    external = None
    if Config.EXTERNAL_CSV.exists():
        external = pd.read_csv(Config.EXTERNAL_CSV)

    fold_rows, pooled_frames = [], []
    for split in splits:
        fold = split.outer_fold
        train_frame = select_patient_rows(development, split.train_ids)
        val_frame = select_patient_rows(development, split.validation_ids)
        test_frame = select_patient_rows(development, split.test_ids)
        save_json(split.audit(patient_labels), run / "metrics" / f"split_audit_fold_{fold}.json")

        val_loader = make_loader(val_frame, target_path, seed=args.seed, fold=fold, train=False)
        test_loader = make_loader(test_frame, target_path, seed=args.seed, fold=fold, train=False)

        checkpoint_path = run / "checkpoints" / f"fold_{fold}.pth"
        model, best_epoch, best_score = train_one_fold(
            train_frame, target_path, args, fold, checkpoint_path, val_loader
        )

        truth, pred, probs, ids = evaluate_patient_level(model, test_loader)
        save_classification_report(truth, pred, run / "reports", f"fold_{fold}")
        preds = prediction_frame(truth, pred, probs, ids)
        preds["seed"] = args.seed
        preds["fold"] = fold
        preds.to_csv(run / "predictions" / f"patient_preds_fold_{fold}.csv", index=False)
        pooled_frames.append(preds)

        fold_rows.append(
            {
                "fold": fold,
                "seed": args.seed,
                **fold_metrics(truth, pred),
                "best_epoch": best_epoch,
                "best_val_macro_f1": float(best_score),
            }
        )

        if external is not None:
            ext_loader = DataLoader(
                AudioClassifierDataset(external, processed_dir=Config.EXTERNAL_PROCESSED_DIR),
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
            )
            ext_truth, ext_pred, ext_probs, ext_ids = evaluate_patient_level(model, ext_loader)
            ext_frame = prediction_frame(ext_truth, ext_pred, ext_probs, ext_ids)
            ext_frame["seed"] = args.seed
            ext_frame["fold"] = fold
            ext_frame.to_csv(run / "predictions" / f"cwlsd_preds_fold_{fold}.csv", index=False)

    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(run / "metrics" / "fold_metrics.csv", index=False)
    save_json(
        {
            metric: {
                "mean": float(fold_frame[metric].mean()),
                "std": float(fold_frame[metric].std(ddof=0)),
            }
            for metric in ("accuracy", "macro_f1", "weighted_f1")
        },
        run / "metrics" / "summary_metrics.json",
    )

    pooled = pd.concat(pooled_frames, ignore_index=True)
    save_classification_report(pooled["y_true"], pooled["y_pred"], run / "reports", "pooled")
    save_json(bootstrap_ci(pooled), run / "metrics" / "bootstrap_ci.json")
    print(f"Distilled audio outputs saved under: {run}")


if __name__ == "__main__":
    main()
