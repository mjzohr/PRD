"""Teacher macro-F1 and weighted-F1 per privileged view.

Reads the artifacts written by train_teachers.py; nothing is retrained.
Macro-F1 comes from metrics/teacher_metrics.json. Weighted-F1 comes from the
"weighted avg" row of the saved classification report -- that report is written
with the three audio class names even though the teacher task is binary, so its
per-class rows and its own macro average are not usable, but the weighted row
is unaffected because the spurious third label has zero support.

    RESULTS_ROOT=results_A10 python teacher_table.py --latex
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from config import Config


def _load(run: Path) -> dict | None:
    metrics = run / "metrics" / "teacher_metrics.json"
    report = run / "reports" / "classification_report_locked_test.json"
    if not metrics.exists():
        return None
    m = json.loads(metrics.read_text(encoding="utf-8"))
    row = {
        "seed": run.name,
        "macro_f1": float(m["locked_test_macro_f1"]),
        "accuracy": float(m["locked_test_accuracy"]),
        "sharp": bool(m.get("sharp", False)),
        "train": int(m.get("train_subjects", 0)),
        "val": int(m.get("validation_subjects", 0)),
        "test": int(m.get("locked_test_subjects", 0)),
        "weighted_f1": None,
    }
    if report.exists():
        r = json.loads(report.read_text(encoding="utf-8"))
        if "weighted avg" in r:
            row["weighted_f1"] = float(r["weighted avg"]["f1-score"])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher macro-F1 / weighted-F1 table.")
    parser.add_argument("--latex", action="store_true", help="emit the booktabs table body")
    args = parser.parse_args()

    root = Config.RESULTS_DIR / "teachers"
    if not root.is_dir():
        raise SystemExit(f"no teacher runs under {root} -- set RESULTS_ROOT")

    summary = []
    for view_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        rows = [r for r in (_load(s) for s in sorted(view_dir.glob("seed*"))) if r]
        if not rows:
            continue
        missing = [r["seed"] for r in rows if r["weighted_f1"] is None]
        sharp = {r["sharp"] for r in rows}
        subjects = {r["train"] + r["val"] + r["test"] for r in rows}

        def agg(key):
            vals = [r[key] for r in rows if r[key] is not None]
            return (st.mean(vals), st.pstdev(vals)) if vals else (float("nan"), float("nan"))

        mf, mfs = agg("macro_f1")
        wf, wfs = agg("weighted_f1")
        summary.append((view_dir.name, len(rows), mf, mfs, wf, wfs, sharp, subjects, missing))

        print(f"== {view_dir.name}  (n={len(rows)} seeds)")
        print(f"   macro-F1     {mf:.4f} +/- {mfs:.4f}")
        print(f"   weighted-F1  {wf:.4f} +/- {wfs:.4f}"
              + (f"   [MISSING for {missing}]" if missing else ""))
        print(f"   accuracy     {agg('accuracy')[0]:.4f}")
        print(f"   subjects     {sorted(subjects)}  (train+val+test per seed)")
        print(f"   sharp        {sharp}")
        if len(sharp) > 1:
            print("   WARNING: seeds disagree on the sharp-phenotype flag -- do not pool these")
        if len(subjects) > 1:
            print("   WARNING: seeds disagree on subject count")
        print()

    if args.latex:
        print("% paste into Table II")
        for name, n, mf, mfs, wf, wfs, *_ in summary:
            label = {"ct_clinical": "CT", "omics": "omics"}.get(name, name)
            print(f"{label:6s} & ${mf:.3f}{{\\pm}}{mfs:.3f}$ & ${wf:.3f}{{\\pm}}{wfs:.3f}$ \\\\")


if __name__ == "__main__":
    main()
