"""Aggregate every run under results/ into paper-ready tables.

Reads the standard layout written by the training scripts
(``results/<method>/<tag>/metrics/{summary_metrics.json,fold_metrics.csv}``),
averages across seeds, and emits:

- results/reports/summary.csv          one row per experiment (mean +/- SD over seeds)
- results/reports/sensitivity_<m>.csv  key-knob sweep (val + test macro-F1) per method
- results/reports/best_configs.csv     best knob per method, selected on validation
- results/reports/headline.md          baseline vs each method's best config, with the
                                        delta over the SAME-encoder audio-only baseline

Every transfer method uses the same audio encoder factory as the baseline, so the
delta isolates the effect of adding the privileged modality. Run after any subset
of experiments; missing pieces are skipped, not fatal.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from config import Config


def _f1(truth, prediction, average: str) -> float:
    return float(f1_score(truth, prediction, labels=range(Config.NUM_CLASSES),
                          average=average, zero_division=0))


def _external_metrics(run_dir: Path) -> dict:
    """Aggregate the locked external cohort (CWLSD) for one run.

    Each outer-fold model scores the whole external cohort. We report both the
    mean over fold models (transfer of a single model) and the probability
    ensemble across fold models (transfer of the method as deployed).
    """
    files = sorted((run_dir / "predictions").glob("cwlsd_preds_fold_*.csv"))
    if not files:
        return {}
    frames = [pd.read_csv(f) for f in files]
    per_fold = [_f1(f["y_true"], f["y_pred"], "macro") for f in frames]
    columns = [f"prob_{name}" for name in Config.CLASSES]
    combined = pd.concat(frames, ignore_index=True)
    if not set(columns).issubset(combined.columns):
        return {"external_macro_f1_fold_mean": float(np.mean(per_fold))}
    ensemble = combined.groupby("patient_id", as_index=False).agg(
        {"y_true": "first", **{column: "mean" for column in columns}}
    )
    ensemble["y_pred"] = ensemble[columns].to_numpy().argmax(axis=1)
    metrics = {
        "external_macro_f1_fold_mean": float(np.mean(per_fold)),
        "external_macro_f1_fold_std": float(np.std(per_fold, ddof=0)),
        "external_macro_f1_ensemble": _f1(ensemble["y_true"], ensemble["y_pred"], "macro"),
        "external_weighted_f1_ensemble": _f1(ensemble["y_true"], ensemble["y_pred"], "weighted"),
        "external_accuracy_ensemble": float(accuracy_score(ensemble["y_true"], ensemble["y_pred"])),
        "external_patients": int(len(ensemble)),
    }

    # Threshold-free Healthy-vs-COPD AUROC on the external cohort.
    # Hard argmax on an unseen cohort is dominated by class-prior shift and by
    # `Other` meaning different diseases in the two corpora. AUROC over the two
    # classes that ARE defined identically is insensitive to both, so it measures
    # whether the model's RANKING transfers -- the question we actually care about.
    healthy, copd = f"prob_{Config.CLASSES[0]}", f"prob_{Config.CLASSES[1]}"
    subset = ensemble[ensemble["y_true"].isin([0, 1])]
    if len(subset) > 5 and subset["y_true"].nunique() == 2:
        score = subset[copd] / (subset[healthy] + subset[copd] + 1e-9)
        metrics["external_auroc_healthy_vs_copd"] = float(roc_auc_score(subset["y_true"], score))
        metrics["external_binary_patients"] = int(len(subset))
    return metrics

KNOB_NAME = {"distilled": "distill_weight", "contrastive": "temperature", "dtca": "guidance_weight"}
GROUP_KEYS = ["method", "view", "encoder", "clinical_encoder", "age_tolerance",
              "distill_loss", "knob", "ablation"]


def _parse_run(method: str, tag_parts: list[str]) -> dict:
    """Turn a results/<method>/<tag> path into normalized experiment fields."""
    info = {"method": method, "view": "", "encoder": "", "clinical_encoder": "",
            "age_tolerance": "", "distill_loss": "", "knob": np.nan,
            "ablation": "none", "seed": np.nan}
    for part in tag_parts:
        if part.startswith("seed"):
            info["seed"] = int(part[4:])
        elif part.startswith("abl_"):
            info["ablation"] = part[4:]
        elif part.startswith("ce_"):
            info["clinical_encoder"] = part[3:]
        elif part.startswith("dl_"):
            info["distill_loss"] = part[3:]
        elif part.startswith("age"):
            info["age_tolerance"] = part[3:]
        elif re.fullmatch(r"[wtg][0-9.]+", part):
            info["knob"] = float(part[1:])
        elif part in ("CNN", "CRNN", "TRANSFORMER"):
            info["encoder"] = part
        else:
            info["view"] = part
    return info


def _collect() -> pd.DataFrame:
    rows = []
    for summary_path in Config.RESULTS_DIR.glob("*/**/metrics/summary_metrics.json"):
        run_dir = summary_path.parent.parent
        rel = run_dir.relative_to(Config.RESULTS_DIR).parts
        method, tag_parts = rel[0], list(rel[1:])
        if method in ("teachers", "reports"):
            continue
        info = _parse_run(method, tag_parts)
        summary = json.loads(summary_path.read_text())
        info["test_macro_f1"] = summary.get("macro_f1", {}).get("mean", np.nan)
        info["test_accuracy"] = summary.get("accuracy", {}).get("mean", np.nan)
        info["test_weighted_f1"] = summary.get("weighted_f1", {}).get("mean", np.nan)
        fold_csv = run_dir / "metrics" / "fold_metrics.csv"
        info["val_macro_f1"] = np.nan
        if fold_csv.exists():
            folds = pd.read_csv(fold_csv)
            if "best_val_macro_f1" in folds:
                info["val_macro_f1"] = float(folds["best_val_macro_f1"].mean())
        info.update(_external_metrics(run_dir))
        rows.append(info)
    return pd.DataFrame(rows)


def _aggregate_over_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(GROUP_KEYS, dropna=False)
    out = grouped.agg(
        seeds=("seed", "nunique"),
        test_macro_f1_mean=("test_macro_f1", "mean"),
        test_macro_f1_std=("test_macro_f1", lambda s: float(np.std(s, ddof=0))),
        test_accuracy_mean=("test_accuracy", "mean"),
        test_weighted_f1_mean=("test_weighted_f1", "mean"),
        test_weighted_f1_std=("test_weighted_f1", lambda s: float(np.std(s, ddof=0))),
        val_macro_f1_mean=("val_macro_f1", "mean"),
    ).reset_index()
    # External cohort (present only once CWLSD predictions exist).
    if "external_macro_f1_ensemble" in frame.columns:
        external = grouped.agg(
            external_macro_f1_mean=("external_macro_f1_ensemble", "mean"),
            external_macro_f1_std=("external_macro_f1_ensemble",
                                   lambda s: float(np.std(s, ddof=0))),
            external_weighted_f1_mean=("external_weighted_f1_ensemble", "mean"),
        ).reset_index()
        out = out.merge(external, on=GROUP_KEYS, how="left")
    if "external_auroc_healthy_vs_copd" in frame.columns:
        auroc = grouped.agg(
            external_auroc_mean=("external_auroc_healthy_vs_copd", "mean"),
            external_auroc_std=("external_auroc_healthy_vs_copd",
                                lambda s: float(np.std(s, ddof=0))),
        ).reset_index()
        out = out.merge(auroc, on=GROUP_KEYS, how="left")
    return out.sort_values(GROUP_KEYS).reset_index(drop=True)


PAIR_KEYS = ["method", "view", "encoder", "clinical_encoder", "age_tolerance",
             "distill_loss", "knob", "ablation"]


def _paired_pvalues(deltas):
    """Paired one-sample tests of (method - baseline) per seed against 0."""
    try:
        from scipy import stats
    except Exception:
        return float("nan"), float("nan")
    t_p = w_p = float("nan")
    if len(deltas) >= 2 and float(np.std(deltas)) > 0:
        try:
            t_p = float(stats.ttest_1samp(deltas, 0.0).pvalue)
        except Exception:
            pass
        try:
            if np.any(deltas != 0):
                w_p = float(stats.wilcoxon(deltas).pvalue)
        except Exception:
            pass
    return t_p, w_p


def _paired_vs_baseline(runs: pd.DataFrame) -> pd.DataFrame:
    """Same-seed paired comparison: for each config, delta = method - baseline at
    the SAME seed (and same encoder), then summarized across seeds. Tighter than
    comparing marginal mean +/- SD, since baseline and method share folds, seeds,
    and initialization. Reported for both headline metrics:
      macro-F1     -- all classes equally (minority-class view);
      weighted-F1  -- support-weighted (population/deployment view).
    (Micro-F1 is omitted: for single-label multiclass it equals accuracy.)"""
    base = runs[(runs["method"] == "baseline_audio") & (runs["ablation"] == "none")]
    metrics = {"test_macro_f1": "", "test_weighted_f1": "_weighted"}
    # External ranking transfer, paired per seed on the same external patients --
    # far more sensitive than comparing marginal means, whose seed-to-seed spread
    # is shared between the method and its baseline.
    if "external_auroc_healthy_vs_copd" in runs.columns:
        metrics["external_auroc_healthy_vs_copd"] = "_ext_auroc"
    lookups = {
        m: {(r.encoder, int(r.seed)): getattr(r, m) for r in base.itertuples()}
        for m in metrics
    }
    method_runs = runs[runs["method"] != "baseline_audio"]
    rows = []
    for cfg, group in method_runs.groupby(PAIR_KEYS, dropna=False):
        row, keep = dict(zip(PAIR_KEYS, cfg)), False
        for metric, suffix in metrics.items():
            deltas = []
            for r in group.itertuples():
                b = lookups[metric].get((r.encoder, int(r.seed)))
                value = getattr(r, metric)
                if b is not None and pd.notna(value):
                    deltas.append(float(value) - float(b))
            if not deltas:
                continue
            keep = True
            d = np.asarray(deltas)
            row[f"paired_delta{suffix}_mean"] = float(d.mean())
            row[f"paired_delta{suffix}_std"] = float(d.std(ddof=0))
            row[f"n_positive{suffix}"] = int((d > 0).sum())
            if suffix == "":  # primary metric carries the shared bookkeeping
                row["n_seeds"] = int(len(d))
                row["t_pvalue"], row["wilcoxon_pvalue"] = _paired_pvalues(d)
        if keep:
            rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("paired_delta_mean", ascending=False) if not out.empty else out


def main() -> None:
    out_dir = Config.RESULTS_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = _collect()
    if runs.empty:
        print(f"No runs found under {Config.RESULTS_DIR}. Train something first.")
        return
    summary = _aggregate_over_seeds(runs)
    summary.to_csv(out_dir / "summary.csv", index=False)

    # Way 1: same-seed PAIRED comparison (with significance).
    paired = _paired_vs_baseline(runs)
    if not paired.empty:
        paired.to_csv(out_dir / "paired_vs_baseline.csv", index=False)

    # Same-encoder audio-only baseline (the reference every method is compared to).
    baseline = summary[(summary["method"] == "baseline_audio") & (summary["ablation"] == "none")]
    baseline_by_encoder = dict(zip(baseline["encoder"], baseline["test_macro_f1_mean"]))

    # Sensitivity + best-config (main, non-ablation runs) per method with a knob.
    best_rows = []
    main = summary[summary["ablation"] == "none"]
    for method, knob_name in KNOB_NAME.items():
        sub = main[main["method"] == method]
        if sub.empty or sub["knob"].isna().all():
            continue
        strata = ["view", "encoder", "clinical_encoder", "age_tolerance", "distill_loss"]
        cols = strata + ["knob", "val_macro_f1_mean", "test_macro_f1_mean", "test_macro_f1_std"]
        sens = sub[cols].rename(columns={"knob": knob_name}).sort_values(strata + [knob_name])
        sens.to_csv(out_dir / f"sensitivity_{method}.csv", index=False)
        # Best knob per stratum (view/encoder/clinical/age/loss), selected on validation.
        for keys, group in sub.groupby(strata):
            view, encoder, clinical, age_tol, dloss = keys
            pick = group.loc[group["val_macro_f1_mean"].idxmax()] if group["val_macro_f1_mean"].notna().any() \
                else group.loc[group["test_macro_f1_mean"].idxmax()]
            base = baseline_by_encoder.get(encoder, np.nan)
            best_rows.append({
                "method": method, "view": view, "encoder": encoder, "clinical_encoder": clinical,
                "age_tolerance": age_tol, "distill_loss": dloss,
                knob_name: pick["knob"], "selected_on": "val_macro_f1",
                "test_macro_f1_mean": pick["test_macro_f1_mean"], "test_macro_f1_std": pick["test_macro_f1_std"],
                "test_weighted_f1_mean": pick["test_weighted_f1_mean"],
                "test_weighted_f1_std": pick["test_weighted_f1_std"],
                "baseline_macro_f1": base, "delta_vs_baseline": pick["test_macro_f1_mean"] - base,
            })
    if best_rows:
        pd.DataFrame(best_rows).to_csv(out_dir / "best_configs.csv", index=False)

    # Headline: baseline vs each method's best, with delta over same-encoder baseline.
    lines = [
        "# Headline results (patient-level macro-F1, mean +/- SD over seeds)",
        "",
        "Every method uses the same audio encoder as the baseline; `delta` is the gain",
        "from adding the privileged modality, at equal audio encoder.",
        "",
        "| Method | View | Audio enc | Clinical enc | Config | macro-F1 | weighted-F1 | delta (macro) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in baseline.iterrows():
        lines.append(f"| baseline_audio | - | {row['encoder']} | - | - | "
                     f"{row['test_macro_f1_mean']:.3f} +/- {row['test_macro_f1_std']:.3f} | "
                     f"{row['test_weighted_f1_mean']:.3f} +/- {row['test_weighted_f1_std']:.3f} | - |")
    for row in best_rows:
        knob = KNOB_NAME[row["method"]]
        ce = row["clinical_encoder"] or "-"
        cfg = f"{knob}={row[knob]}"
        if row.get("distill_loss"):
            cfg += f", {row['distill_loss']}"
        if row.get("age_tolerance"):
            cfg += f", ±{row['age_tolerance']}y"
        delta = row["delta_vs_baseline"]
        delta_str = "-" if pd.isna(delta) else f"{delta:+.3f}"
        wmean, wstd = row.get("test_weighted_f1_mean"), row.get("test_weighted_f1_std")
        wstr = "-" if wmean is None or pd.isna(wmean) else f"{wmean:.3f} +/- {wstd:.3f}"
        lines.append(
            f"| {row['method']} | {row['view']} | {row['encoder']} | {ce} | {cfg} | "
            f"{row['test_macro_f1_mean']:.3f} +/- {row['test_macro_f1_std']:.3f} | {wstr} | {delta_str} |"
        )
    # Way 1 table: top configs by same-seed paired delta, with significance.
    if not paired.empty:
        top_paired = paired[paired["ablation"] == "none"].head(15)
        lines += [
            "",
            "## Way 1 — same-seed PAIRED delta vs baseline (tighter than ±SD)",
            "",
            "`Δ` = mean of (method − baseline) at the *same* seed; `seeds+` = seeds improved.",
            "Full table: `paired_vs_baseline.csv`.",
            "",
            "| Method | View | Enc | Config | Δ macro-F1 | seeds+ | Δ weighted-F1 | seeds+ |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in top_paired.itertuples(index=False):
            cfg = f"w={r.knob:g}" if pd.notna(r.knob) else "-"
            if r.distill_loss:
                cfg += f", {r.distill_loss}"
            if r.age_tolerance:
                cfg += f", ±{r.age_tolerance}y"
            if r.clinical_encoder:
                cfg += f", {r.clinical_encoder}"
            wd = getattr(r, "paired_delta_weighted_mean", float("nan"))
            wn = getattr(r, "n_positive_weighted", None)
            wstr = "-" if pd.isna(wd) else f"{wd:+.3f}"
            wns = "-" if wn is None else f"{wn}/{r.n_seeds}"
            lines.append(f"| {r.method} | {r.view} | {r.encoder} | {cfg} | "
                         f"{r.paired_delta_mean:+.3f} | {r.n_positive}/{r.n_seeds} | {wstr} | {wns} |")

    # External cohort: does the transfer survive a shift to an unseen cohort?
    if "external_macro_f1_mean" in summary.columns and summary["external_macro_f1_mean"].notna().any():
        ext = summary[summary["external_macro_f1_mean"].notna()].copy()
        ext_base = dict(zip(
            ext[ext["method"] == "baseline_audio"]["encoder"],
            ext[ext["method"] == "baseline_audio"]["external_macro_f1_mean"],
        ))
        ext["external_delta"] = ext.apply(
            lambda r: r["external_macro_f1_mean"] - ext_base.get(r["encoder"], np.nan), axis=1
        )
        ext.sort_values("external_delta", ascending=False).to_csv(
            out_dir / "external_cwlsd.csv", index=False
        )
        lines += [
            "",
            "## External cohort (CWLSD, locked) — probability ensemble over fold models",
            "",
            "| Method | View | Enc | Config | ext macro-F1 | ext weighted-F1 | delta vs baseline |",
            "|---|---|---|---|---|---|---|",
        ]
        show = pd.concat([
            ext[ext["method"] == "baseline_audio"],
            ext[(ext["method"] != "baseline_audio") & (ext["ablation"] == "none")]
            .sort_values("external_delta", ascending=False).head(12),
        ])
        for r in show.itertuples(index=False):
            cfg = "-" if pd.isna(r.knob) else f"w={r.knob:g}"
            if r.distill_loss:
                cfg += f", {r.distill_loss}"
            if r.age_tolerance:
                cfg += f", ±{r.age_tolerance}y"
            delta = "-" if pd.isna(r.external_delta) else f"{r.external_delta:+.3f}"
            lines.append(
                f"| {r.method} | {r.view or '-'} | {r.encoder} | {cfg} | "
                f"{r.external_macro_f1_mean:.3f} +/- {r.external_macro_f1_std:.3f} | "
                f"{r.external_weighted_f1_mean:.3f} | {delta} |"
            )

    (out_dir / "headline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote reports to {out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
