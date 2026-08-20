"""Join the internal and external paired deltas into one ablation table.

Reads the reports written by report.py and evaluate_external.py for the
ablation operating point (CT teacher, smooth-L1, w=1, tau=+/-5) and prints the
controls in order of how much of the target they remove.

    RESULTS_ROOT=results_A10 python ablation_table.py --latex
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

from config import Config

VIEW, LOSS, WEIGHT, AGE = "ct_clinical", "smooth_l1", 1.0, "5"
ENCODERS = ["CNN", "CRNN", "TRANSFORMER"]
LADDER = ["none", "class_prototype", "shuffled_targets", "random_prototype",
          "crossed_targets", "global_prototype", "random_teacher"]
PRETTY = {
    "none": "full bridge",
    "class_prototype": "label-only groups",
    "shuffled_targets": "within-class shuffle",
    "random_prototype": "random per-class vector",
    "crossed_targets": "targets from other class",
    "global_prototype": "one target for all",
    "random_teacher": "UNTRAINED teacher",
}


def _rows(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path, encoding="utf-8"))) if path.is_file() else []


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def collect() -> tuple[dict, list[str]]:
    reports = Config.RESULTS_DIR / "reports"
    warnings: list[str] = []

    internal = [r for r in _rows(reports / "paired_vs_baseline.csv")
                if r.get("method") == "distilled" and r.get("view") == VIEW
                and r.get("distill_loss") == LOSS and _f(r.get("knob")) == WEIGHT
                and str(r.get("age_tolerance")) == AGE]
    ext = {}
    for tag in ("tr", "cwlsd_matched"):
        ext[tag] = [r for r in _rows(reports / f"external_{tag}_paired.csv")
                    if r.get("method") == "distilled" and r.get("view") == VIEW
                    and r.get("loss") == LOSS and _f(r.get("knob")) == WEIGHT
                    and str(r.get("age")) == AGE]
        if not ext[tag]:
            warnings.append(f"no external_{tag}_paired.csv rows -- run evaluate_external.py --tag {tag}")

    cells = {}
    for enc in ENCODERS:
        for abl in LADDER:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Signal-specificity ladder.")
    parser.add_argument("--latex", action="store_true", help="also emit the table body")
    args = parser.parse_args()

    cells, warnings = collect()
    print(f"Signal specificity -- {VIEW}, {LOSS}, w={WEIGHT:g}, tau=+/-{AGE}, "
          f"{Config.RESULTS_DIR.name}")
    print("paired delta vs the architecture-matched audio-only baseline, "
          "with seeds improved\n")
    head = "%-12s %-26s %-16s %-16s %-16s" % ("encoder", "control", "ICBHI macro-F1",
                                              "TR AUROC", "CWLSD AUROC")
    print(head); print("-" * len(head))
    for enc in ENCODERS:
        for abl in LADDER:
            c = cells[(enc, abl)]
            print("%-12s %-26s %-16s %-16s %-16s"
                  % (enc if abl == LADDER[0] else "", PRETTY[abl],
                     _fmt(c["internal"]), _fmt(c["tr"]), _fmt(c["cwlsd_matched"])))
        print()

    print("MEAN OVER ENCODERS")
    print("%-26s %-16s %-16s %-16s" % ("control", "ICBHI", "TR", "CWLSD"))
    means = {}
    for abl in LADDER:
        row = []
        for key in ("internal", "tr", "cwlsd_matched"):
            v = [cells[(e, abl)][key][0] for e in ENCODERS if cells[(e, abl)][key]]
            row.append(st.mean(v) if len(v) == len(ENCODERS) else None)
        means[abl] = row
        print("%-26s %-16s %-16s %-16s"
              % (PRETTY[abl],
                 *["%+7.4f" % x if x is not None else "   --   " for x in row]))

    out = Config.RESULTS_DIR / "reports" / "ablation_ladder.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["encoder", "ablation", "internal_delta", "internal_pos", "internal_n",
                    "tr_delta_auroc", "tr_pos", "tr_n",
                    "cwlsd_delta_auroc", "cwlsd_pos", "cwlsd_n"])
        for enc in ENCODERS:
            for abl in LADDER:
                c = cells[(enc, abl)]
                w.writerow([enc, abl,
                            *(c["internal"] or ("", "", "")),
                            *(c["tr"] or ("", "", "")),
                            *(c["cwlsd_matched"] or ("", "", ""))])
    print(f"\nwrote {out}")

    missing = [f"{e}/{a}" for e in ENCODERS for a in LADDER
               if cells[(e, a)]["internal"] is None]
    for w_ in warnings:
        print("WARNING: " + w_)
    if missing:
        print("WARNING: no internal result yet for: " + ", ".join(missing))

    if args.latex:
        print("\n% ---- table body ----")
        for abl in LADDER:
            r = means[abl]
            cellstr = " & ".join("$%+.3f$" % x if x is not None else "---" for x in r)
            print("%-26s & %s \\\\" % (PRETTY[abl], cellstr))


if __name__ == "__main__":
    main()
