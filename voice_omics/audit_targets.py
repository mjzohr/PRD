"""Audit how much information the bridge targets actually carry.

The bridge averages every demographically-compatible privileged subject into one
target per audio file. When many subjects match, that average collapses toward a
per-stratum (and largely per-class) constant, and the transferred signal reduces
to class/demographic identity rather than individual biology. This script
quantifies that collapse so the limitation can be reported rather than assumed.

Reports, per targets file:
  * distinct target vectors vs number of files (the collapse factor);
  * within-class spread vs between-class distance (how class-dominated targets are);
  * class balance of the target pool (which determines whether a global shuffle is
    even a valid control).

    python audit_targets.py --view ct_clinical --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from utils import save_json


def audit(view: str, seed: int, age_tolerance: int | None, targets_path=None) -> dict:
    base = Config.PROJECT_ROOT / "data" / "distillation"
    if targets_path is not None:
        # Explicit file. Needed for the random-init teacher targets, whose names
        # the view/seed scheme below does not cover. Metrics are computed in that
        # file's OWN space, which is the only way they mean anything: those
        # targets come from a different encoder, so they are not comparable
        # point-by-point with the trained-teacher targets.
        path = Path(targets_path)
        if not path.is_absolute():
            path = base / path
        if not path.is_file():
            raise FileNotFoundError(f"No targets at {path}")
    else:
        name = (f"{view}_targets.npy" if age_tolerance is None
                else f"{view}_targets_age{int(age_tolerance)}_seed{seed}.npy")
        path = base / name
        if not path.is_file():
            path = base / f"{view}_targets_age{int(age_tolerance)}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"No targets found for {view} (looked for {name})")

    targets = np.load(path, allow_pickle=True).item()
    manifest = pd.read_csv(Config.FULL_CSV).set_index("filename")["disease"]
    keys = [k for k in targets if k in manifest.index]
    values = np.stack([targets[k] for k in keys])
    classes = manifest.loc[keys].to_numpy()

    distinct = len(np.unique(values.round(5), axis=0))
    per_class, means = {}, {}
    for label in np.unique(classes):
        subset = values[classes == label]
        means[label] = subset.mean(axis=0)
        per_class[str(label)] = {
            "files": int(len(subset)),
            "distinct_targets": int(len(np.unique(subset.round(5), axis=0))),
            "within_class_spread": float(np.linalg.norm(subset - subset.mean(0), axis=1).mean()),
        }
    labels = sorted(means)
    between = (float(np.linalg.norm(means[labels[0]] - means[labels[1]]))
               if len(labels) > 1 else float("nan"))
    within = float(np.mean([v["within_class_spread"] for v in per_class.values()]))

    report = {
        "targets_file": str(path),
        "files_with_targets": int(len(keys)),
        "distinct_target_vectors": distinct,
        "collapse_factor": round(len(keys) / max(distinct, 1), 1),
        "within_class_spread_mean": within,
        "between_class_distance": between,
        "within_over_between": round(within / between, 3) if between == between else None,
        "per_class": per_class,
        "class_balance": {str(k): int(v) for k, v in pd.Series(classes).value_counts().items()},
    }
    print(f"\n=== {path.name}")
    print(f"  files with targets : {report['files_with_targets']}")
    print(f"  distinct vectors   : {distinct}  (collapse factor {report['collapse_factor']}x)")
    print(f"  within-class spread: {within:.4f}")
    print(f"  between-class dist : {between:.4f}")
    print(f"  within/between     : {report['within_over_between']}"
          "   <- small means targets are essentially class identity")
    print(f"  class balance      : {report['class_balance']}")
    if report["within_over_between"] is not None and report["within_over_between"] < 0.25:
        print("  NOTE: targets are class-dominated; a GLOBAL shuffle is not a valid\n"
              "        control (use --ablation shuffled_targets, which permutes within class).")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit bridge-target information content.")
    parser.add_argument("--view", default="ct_clinical", choices=["omics", "ct_clinical"])
    parser.add_argument("--seed", type=int, default=Config.SEED)
    parser.add_argument("--age_tolerance", type=int, default=5)
    parser.add_argument("--targets", default=None,
                        help="Audit this targets .npy directly (absolute, or relative to "
                             "data/distillation). Use for the random-init teacher files, "
                             "whose names the view/seed scheme does not cover.")
    args = parser.parse_args()
    report = audit(args.view, args.seed, args.age_tolerance, args.targets)
    name = (f"target_audit_{Path(args.targets).stem}.json" if args.targets
            else f"target_audit_{args.view}_age{args.age_tolerance}.json")
    out = Config.RESULTS_DIR / "reports" / name
    save_json(report, out)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
