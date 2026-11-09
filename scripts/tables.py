"""Result tables: teacher scores, the signal-specificity ablation, and the random-teacher control."""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from prd.config import Config

ENCODERS = ["CNN", "CRNN", "TRANSFORMER"]


# teachers

def _load_teacher(run: Path) -> dict | None:
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
    # the saved report uses the three audio class names for a binary task, so
    # only its weighted row (where the empty third class has zero support) is valid
    if report.exists():
        r = json.loads(report.read_text(encoding="utf-8"))
        if "weighted avg" in r:
            row["weighted_f1"] = float(r["weighted avg"]["f1-score"])
    return row


def teachers(args) -> None:
    root = Config.RESULTS_DIR / "teachers"
    if not root.is_dir():
        raise SystemExit(f"no teacher runs under {root} -- set RESULTS_ROOT")

    summary = []
    for view_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        rows = [r for r in (_load_teacher(s) for s in sorted(view_dir.glob("seed*"))) if r]
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
              + (f"   [missing for {missing}]" if missing else ""))
        print(f"   accuracy     {agg('accuracy')[0]:.4f}")
        print(f"   subjects     {sorted(subjects)}  (train+val+test per seed)")
        print(f"   sharp        {sharp}")
        if len(sharp) > 1:
            print("   warning: seeds disagree on the sharp-phenotype flag; do not pool these")
        if len(subjects) > 1:
            print("   warning: seeds disagree on subject count")
        print()

    if args.latex:
        print("% table body")
        for name, n, mf, mfs, wf, wfs, *_ in summary:
            label = {"ct_clinical": "CT", "omics": "omics"}.get(name, name)
            print(f"{label:6s} & ${mf:.3f}{{\\pm}}{mfs:.3f}$ & ${wf:.3f}{{\\pm}}{wfs:.3f}$ \\\\")


# ablation: internal and external paired deltas at the ablation operating point

ABL_VIEW, ABL_LOSS, ABL_WEIGHT, ABL_AGE = "ct_clinical", "smooth_l1", 1.0, "5"
ABL_LADDER = ["none", "class_prototype", "shuffled_targets", "random_prototype",
              "crossed_targets", "global_prototype", "random_teacher"]
ABL_PRETTY = {
    "none": "full bridge",
    "class_prototype": "label-only groups",
    "shuffled_targets": "within-class shuffle",
    "random_prototype": "random per-class vector",
    "crossed_targets": "targets from other class",
    "global_prototype": "one target for all",
    "random_teacher": "untrained teacher",
}


def _rows(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path, encoding="utf-8"))) if path.is_file() else []


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _collect_ablation() -> tuple[dict, list[str]]:
    reports = Config.RESULTS_DIR / "reports"
    warnings: list[str] = []

    internal = [r for r in _rows(reports / "paired_vs_baseline.csv")
                if r.get("method") == "distilled" and r.get("view") == ABL_VIEW
                and r.get("distill_loss") == ABL_LOSS and _f(r.get("knob")) == ABL_WEIGHT
                and str(r.get("age_tolerance")) == ABL_AGE]
    ext = {}
    for tag in ("tr", "cwlsd_matched"):
        ext[tag] = [r for r in _rows(reports / f"external_{tag}_paired.csv")
                    if r.get("method") == "distilled" and r.get("view") == ABL_VIEW
                    and r.get("loss") == ABL_LOSS and _f(r.get("knob")) == ABL_WEIGHT
                    and str(r.get("age")) == ABL_AGE]
        if not ext[tag]:
            warnings.append(f"no external_{tag}_paired.csv rows -- run evaluate_external.py --tag {tag}")

    cells = {}
    for enc in ENCODERS:
        for abl in ABL_LADDER:
            i = next((r for r in internal if r["encoder"] == enc and r["ablation"] == abl), None)
            entry = {"internal": None, "tr": None, "cwlsd_matched": None}
            if i:
                entry["internal"] = (_f(i["paired_delta_mean"]), i["n_positive"], i["n_seeds"])
            for tag in ("tr", "cwlsd_matched"):
                e = next((r for r in ext[tag]
                          if r["encoder"] == enc and r["ablation"] == abl), None)
                if e:
                    entry[tag] = (_f(e["delta_auroc"]), e["n_positive_auroc"], e["n_seeds"])
            cells[(enc, abl)] = entry
    return cells, warnings


def _fmt(cell) -> str:
    if cell is None:
        return "      --      "
    d, pos, n = cell
    return "%+7.4f %3s/%-2s" % (d, str(pos).rstrip(".0") or "0", n)


def ablation(args) -> None:
    cells, warnings = _collect_ablation()
    print(f"Signal specificity -- {ABL_VIEW}, {ABL_LOSS}, w={ABL_WEIGHT:g}, tau=+/-{ABL_AGE}, "
          f"{Config.RESULTS_DIR.name}")
    print("paired delta vs the architecture-matched audio-only baseline, "
          "with seeds improved\n")
    head = "%-12s %-26s %-16s %-16s %-16s" % ("encoder", "control", "ICBHI macro-F1",
                                              "TR AUROC", "CWLSD AUROC")
    print(head)
    print("-" * len(head))
    for enc in ENCODERS:
        for abl in ABL_LADDER:
            c = cells[(enc, abl)]
            print("%-12s %-26s %-16s %-16s %-16s"
                  % (enc if abl == ABL_LADDER[0] else "", ABL_PRETTY[abl],
                     _fmt(c["internal"]), _fmt(c["tr"]), _fmt(c["cwlsd_matched"])))
        print()

    print("mean over encoders")
    print("%-26s %-16s %-16s %-16s" % ("control", "ICBHI", "TR", "CWLSD"))
    means = {}
    for abl in ABL_LADDER:
        row = []
        for key in ("internal", "tr", "cwlsd_matched"):
            v = [cells[(e, abl)][key][0] for e in ENCODERS if cells[(e, abl)][key]]
            row.append(st.mean(v) if len(v) == len(ENCODERS) else None)
        means[abl] = row
        print("%-26s %-16s %-16s %-16s"
              % (ABL_PRETTY[abl],
                 *["%+7.4f" % x if x is not None else "   --   " for x in row]))

    out = Config.RESULTS_DIR / "reports" / "ablation_ladder.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["encoder", "ablation", "internal_delta", "internal_pos", "internal_n",
                    "tr_delta_auroc", "tr_pos", "tr_n",
                    "cwlsd_delta_auroc", "cwlsd_pos", "cwlsd_n"])
        for enc in ENCODERS:
            for abl in ABL_LADDER:
                c = cells[(enc, abl)]
                w.writerow([enc, abl,
                            *(c["internal"] or ("", "", "")),
                            *(c["tr"] or ("", "", "")),
                            *(c["cwlsd_matched"] or ("", "", ""))])
    print(f"\nwrote {out}")

    missing = [f"{e}/{a}" for e in ENCODERS for a in ABL_LADDER
               if cells[(e, a)]["internal"] is None]
    for w_ in warnings:
        print("warning: " + w_)
    if missing:
        print("warning: no internal result yet for: " + ", ".join(missing))

    if args.latex:
        print("\n% table body")
        for abl in ABL_LADDER:
            r = means[abl]
            cellstr = " & ".join("$%+.3f$" % x if x is not None else "---" for x in r)
            print("%-26s & %s \\\\" % (ABL_PRETTY[abl], cellstr))


# random teacher: trained vs untrained teacher on ICBHI, paired per seed.
# baseline and trained arms are read from REF_ROOT; the untrained arm from RESULTS_ROOT.

RT_VIEW, RT_LOSS, RT_WEIGHT, RT_AGE = "ct_clinical", "smooth_l1", "1.0", 5
RT_SEEDS = [11, 23, 42, 73, 101, 3, 5, 7, 13, 19]


def _value(run: Path, metric: str = "macro_f1"):
    path = run / "metrics" / "summary_metrics.json"
    if not path.is_file():
        return None
    return float(json.loads(path.read_text(encoding="utf-8"))[metric]["mean"])


def random_teacher(args) -> None:
    ref = Path(os.environ.get("PRD_ROOT", Config.PROJECT_ROOT)) / os.environ.get(
        "REF_ROOT", "results_A10")
    new = Config.RESULTS_DIR
    stem = f"distilled/{RT_VIEW}/%s/age{RT_AGE}/dl_{RT_LOSS}/w{RT_WEIGHT}/seed%s"

    print(f"reference (trained teacher) : {ref}")
    print(f"random-init teacher         : {new}")
    print(f"metric: patient-level {args.metric}, {len(RT_SEEDS)} seeds\n")

    rows, missing = {}, []
    for enc in ENCODERS:
        arms = {}
        for seed in RT_SEEDS:
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
                                       "PRD (untrained teacher)")
    print(hdr)
    print("-" * len(hdr))
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

    print("\npaired contrast: trained minus untrained, same seed")
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
        print("\n  note: encoders share a partition within a seed, so the pooled SE is")
        print("  optimistic; the per-encoder rows are the unit to read.")

    if missing:
        print("\nwarning: incomplete, excluded from every column: " + ", ".join(missing[:12])
              + (" ..." if len(missing) > 12 else ""))

    if args.latex:
        print("\n% table body")
        for enc in ENCODERS:
            arms = rows[enc]
            if not arms:
                continue
            b = st.mean([v[0] for v in arms.values()])
            t = st.mean([v[1] for v in arms.values()])
            r = st.mean([v[2] for v in arms.values()])
            print("%-12s & $%.3f$ & $%.3f$ & $%.3f$ \\\\" % (enc, b, t, r))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="table", required=True)
    for name in ("teachers", "ablation", "random_teacher"):
        p = sub.add_parser(name)
        p.add_argument("--latex", action="store_true", help="also print the LaTeX table body")
        if name == "random_teacher":
            p.add_argument("--metric", default="macro_f1",
                           choices=["macro_f1", "weighted_f1", "accuracy"])
    args = parser.parse_args()
    {"teachers": teachers, "ablation": ablation, "random_teacher": random_teacher}[args.table](args)


if __name__ == "__main__":
    main()
