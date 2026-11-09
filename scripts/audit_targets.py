"""Measure how much information the bridge targets carry beyond class identity."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

from prd.config import Config
from prd.utils import save_json


def audit(view: str, seed: int, age_tolerance: int | None, targets_path=None) -> dict:
    base = Config.PROJECT_ROOT / "data" / "distillation"
    if targets_path is not None:
        # random-init teacher targets live in a different encoder space, so
        # they are only ever audited against themselves
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
        print("  targets are class-dominated; a global shuffle is not a valid control\n"
              "  (use --ablation shuffled_targets, which permutes within class)")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", default="ct_clinical", choices=["omics", "ct_clinical"])
    parser.add_argument("--seed", type=int, default=Config.SEED)
    parser.add_argument("--age_tolerance", type=int, default=5)
    parser.add_argument("--targets", default=None,
                        help="targets .npy to audit (absolute or relative to data/distillation)")
    args = parser.parse_args()
    report = audit(args.view, args.seed, args.age_tolerance, args.targets)
    name = (f"target_audit_{Path(args.targets).stem}.json" if args.targets
            else f"target_audit_{args.view}_age{args.age_tolerance}.json")
    out = Config.RESULTS_DIR / "reports" / name
    save_json(report, out)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
