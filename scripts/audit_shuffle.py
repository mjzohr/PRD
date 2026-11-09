"""Measure how much each ablation actually perturbs the distillation targets."""
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
from train_distilled import ABLATIONS as LADDER, resolve_target_path

# random_teacher targets come from a different encoder, so displacement from the
# real targets means nothing; audit those with audit_targets.py --targets instead
ABLATIONS = [a for a in LADDER if a not in ("none", "random_teacher")]


def _load(view: str, ablation: str, seed: int, age: int | None):
    path = resolve_target_path(view, ablation, seed, age)
    return np.load(path, allow_pickle=True).item()


def audit(view: str, seed: int, age: int | None) -> dict:
    base_map = _load(view, "none", seed, age)
    manifest = pd.read_csv(Config.FULL_CSV).set_index("filename")["disease"]
    keys = [k for k in base_map if k in manifest.index]
    base = np.stack([base_map[k] for k in keys])
    classes = manifest.loc[keys].to_numpy()

    labels = sorted(np.unique(classes))
    means = {c: base[classes == c].mean(axis=0) for c in labels}
    between = (float(np.linalg.norm(means[labels[0]] - means[labels[1]]))
               if len(labels) > 1 else float("nan"))

    def class_recoverable(values):
        # nearest-class-mean accuracy against the original class means
        stacked = np.stack([means[c] for c in labels])
        d = np.linalg.norm(values[:, None, :] - stacked[None, :, :], axis=2)
        pred = np.array(labels)[d.argmin(axis=1)]
        return float((pred == classes).mean())

    # a target with no class information scores the majority share, not 0%
    majority = float(max((classes == c).mean() for c in labels))

    report = {
        "view": view, "seed": seed, "age_tolerance": age,
        "files": int(len(keys)), "between_class_distance": between,
        "baseline_class_recoverable": class_recoverable(base),
        "majority_class_share": majority,
        "ablations": {},
    }
    print(f"== {view} seed{seed}: {len(keys)} files, between-class distance {between:.3f}")
    print(f"   class recoverable from the true targets: "
          f"{report['baseline_class_recoverable'] * 100:.1f}%")
    print(f"   reference: majority-class share = {majority * 100:.1f}%  "
          f"(a target with no class information scores this, not 0%)")

    for ablation in ABLATIONS:
        new_map = _load(view, ablation, seed, age)
        new = np.stack([new_map[k] for k in keys])
        step = np.linalg.norm(new - base, axis=1)
        unchanged = float((step == 0).mean())
        per_class = {
            str(c): {
                "files": int((classes == c).sum()),
                "unchanged_share": float((step[classes == c] == 0).mean()),
                "mean_step_over_between": float(step[classes == c].mean() / between),
            }
            for c in labels
        }
        entry = {
            "unchanged_share": unchanged,
            "mean_step_over_between": float(step.mean() / between),
            "class_recoverable": class_recoverable(new),
            "per_class": per_class,
        }
        report["ablations"][ablation] = entry
        print(f"\n   -- {ablation}")
        print(f"      targets left bitwise unchanged : {unchanged * 100:5.1f}%")
        print(f"      mean move, in class-gap units  : {entry['mean_step_over_between']:5.3f}")
        print(f"      class still recoverable        : {entry['class_recoverable'] * 100:5.1f}%")
        for c, v in per_class.items():
            print(f"        {c:<8} n={v['files']:<5} unchanged={v['unchanged_share'] * 100:5.1f}%"
                  f"  move={v['mean_step_over_between']:.3f}")

    print("\n   Read:")
    print("     * high 'unchanged' share or a small move -> the control barely perturbs the")
    print("       targets, so a null result says nothing about privileged structure.")
    print(f"     * class recoverable near {majority * 100:.0f}% (the majority share) -> the target")
    print("       carries no class information at all  [expected for global_prototype].")
    print("     * class recoverable near 0% -> the target is systematically the wrong class")
    print("       [expected for crossed_targets: inverted, not merely uninformative].")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", default="ct_clinical")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--age_tolerance", type=int, default=int(Config.AGE_TOLERANCE))
    parser.add_argument("--all-seeds", action="store_true",
                        help="also check that the permutation differs across seeds")
    args = parser.parse_args()

    report = audit(args.view, args.seed, args.age_tolerance)

    if args.all_seeds:
        maps = {s: _load(args.view, "shuffled_targets", s, args.age_tolerance)
                for s in Config.SEEDS}
        keys = sorted(set(maps[Config.SEEDS[0]]))
        agree = {}
        for a, b in zip(Config.SEEDS, Config.SEEDS[1:]):
            same = np.mean([np.array_equal(maps[a][k], maps[b][k]) for k in keys])
            agree[f"{a}_vs_{b}"] = float(same)
            print(f"   seeds {a} vs {b}: {same * 100:.1f}% of files got the same shuffled target")
        report["cross_seed_agreement"] = agree

    out = Config.RESULTS_DIR / "reports" / f"shuffle_audit_{args.view}_seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
