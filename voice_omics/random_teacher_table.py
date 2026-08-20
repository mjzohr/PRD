"""Compare a trained teacher against an untrained one, on ICBHI.

Three arms, paired per seed:

    audio only            <REF_ROOT>/baseline_audio/<enc>/seed<N>
    PRD (trained)         <REF_ROOT>/distilled/.../seed<N>
    PRD (untrained)       <RESULTS_ROOT>/distilled/.../seed<N>/abl_random_teacher

The first two are read from the existing run, so only the third arm is new.
The reported contrast is trained minus untrained at the same seed, which
cancels the baseline noise both arms share.

    REF_ROOT=results_A10 RESULTS_ROOT=results_randteacher python random_teacher_table.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
from pathlib import Path

from config import Config

VIEW, LOSS, WEIGHT, AGE = "ct_clinical", "smooth_l1", "1.0", 5
ENCODERS = ["CNN", "CRNN", "TRANSFORMER"]
SEEDS = [11, 23, 42, 73, 101, 3, 5, 7, 13, 19]


def _value(run: Path, metric: str = "macro_f1"):
    """That run's patient-level score (its mean over the five outer folds)."""
    path = run / "metrics" / "summary_metrics.json"
    if not path.is_file():
        return None
    return float(json.loads(path.read_text(encoding="utf-8"))[metric]["mean"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Trained vs untrained teacher, ICBHI.")
    parser.add_argument("--metric", default="macro_f1",
                        choices=["macro_f1", "weighted_f1", "accuracy"])
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    ref = Path(os.environ.get("VOICE_OMICS_ROOT", Config.PROJECT_ROOT)) / os.environ.get(
        "REF_ROOT", "results_A10")
    new = Config.RESULTS_DIR
    stem = f"distilled/{VIEW}/%s/age{AGE}/dl_{LOSS}/w{WEIGHT}/seed%s"

    print(f"reference (already reported) : {ref}")
    print(f"random-init teacher (new)    : {new}")
    print(f"metric: patient-level {args.metric}, {len(SEEDS)} seeds\n")

    rows, missing = {}, []
    for enc in ENCODERS:
        arms = {}
        for seed in SEEDS:
            b = _value(ref / f"baseline_audio/{enc}/seed{seed}", args.metric)
            t = _value(ref / (stem % (enc, seed)), args.metric)
            r = _value(new / (stem % (enc, seed)) / "abl_random_teacher", args.metric)
            if None in (b, t, r):
                missing.append(f"{enc}/seed{seed}"
                               + ("" if r is not None else " [random_teacher not run]"))
                continue
            arms[seed] = (b, t, r)
        rows[enc] = arms

    hdr = "%-12s %-22s %-22s %-22s" % ("student", "audio only", "PRD (trained teacher)",
                                       "PRD (UNTRAINED teacher)")
    print(hdr); print("-" * len(hdr))
    pooled = []
    for enc in ENCODERS:
        arms = rows[enc]
        if not arms:
            print("%-12s  no complete seeds" % enc)
            continue
        b = [v[0] for v in arms.values()]
        t = [v[1] for v in arms.values()]
        r = [v[2] for v in arms.values()]
        print("%-12s %-22s %-22s %-22s"
              % (enc,
                 "%.4f +/- %.4f" % (st.mean(b), st.pstdev(b)),
                 "%.4f  (%+.4f)" % (st.mean(t), st.mean(t) - st.mean(b)),
                 "%.4f  (%+.4f)" % (st.mean(r), st.mean(r) - st.mean(b))))
        pooled += [(x, y) for x, y in zip(t, r)]

    print("\nPAIRED CONTRAST  trained minus untrained, same seed (baseline noise cancels)")
    print("%-12s %10s %10s %10s %9s %s" % ("student", "mean", "SD", "SE", "mean/SE", "trained wins"))
    for enc in ENCODERS:
        arms = rows[enc]
        if len(arms) < 2:
            continue
        d = [v[1] - v[2] for v in arms.values()]
        se = st.pstdev(d) / len(d) ** 0.5
        print("%-12s %+10.4f %10.4f %10.4f %9.2f %d/%d"
              % (enc, st.mean(d), st.pstdev(d), se,
                 st.mean(d) / se if se else float("nan"),
                 sum(1 for x in d if x > 0), len(d)))
    if len(pooled) > 1:
        d = [x - y for x, y in pooled]
        se = st.pstdev(d) / len(d) ** 0.5
        print("%-12s %+10.4f %10.4f %10.4f %9.2f %d/%d"
              % ("ALL", st.mean(d), st.pstdev(d), se,
                 st.mean(d) / se if se else float("nan"),
                 sum(1 for x in d if x > 0), len(d)))
        print("\n  NOTE: the pooled row treats 3 encoders x 10 seeds as 30 independent")
        print("  pairs, but encoders share a partition within a seed, so its SE is")
        print("  optimistic. The per-encoder rows above are the honest unit.")

    if missing:
        print("\nWARNING: incomplete, excluded from every column: " + ", ".join(missing[:12])
              + (" ..." if len(missing) > 12 else ""))

    if args.latex:
        print("\n% ---- table body ----")
        for enc in ENCODERS:
            arms = rows[enc]
            if not arms:
                continue
            b = st.mean([v[0] for v in arms.values()])
            t = st.mean([v[1] for v in arms.values()])
            r = st.mean([v[2] for v in arms.values()])
            print("%-12s & $%.3f$ & $%.3f$ & $%.3f$ \\\\" % (enc, b, t, r))


if __name__ == "__main__":
    main()
