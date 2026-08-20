#!/usr/bin/env bash
# Verify the overnight run produced everything the paper needs.
# Prints a checklist and flags anything missing or degenerate.
#   bash scripts/morning_check.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PYTHON_BIN:-python}"

for root in results_A10 results_B10; do
  echo ""
  echo "=================== $root ==================="
  RESULTS_ROOT="$root" $PY - <<'PYEOF'
import json, os, glob
from pathlib import Path
import pandas as pd

root = Path(os.environ["RESULTS_ROOT"])
if not root.exists():
    print("  MISSING result root"); raise SystemExit

runs = glob.glob(str(root / "*/**/metrics/summary_metrics.json"), recursive=True)
print(f"  runs completed            : {len(runs)}")

rep = root / "reports"
for name in ("summary.csv", "paired_vs_baseline.csv", "headline.md",
             "external_tr.csv", "external_cwlsd.csv", "external_cwlsd_matched.csv"):
    path = rep / name
    print(f"  {'OK ' if path.is_file() else 'MISSING'} {name}")

# Seed coverage for the reported configurations.
summary = rep / "summary.csv"
if summary.is_file():
    s = pd.read_csv(summary)
    base = s[s.method == "baseline_audio"][["encoder", "seeds"]]
    print("\n  seeds per baseline encoder:")
    for r in base.itertuples(index=False):
        flag = "OK " if r.seeds >= 10 else "LOW"
        print(f"    {flag} {r.encoder:12s} n={r.seeds}")
    main = s[(s.method == "distilled") & (s.ablation == "none")]
    low = main[main.seeds < 10]
    print(f"\n  distilled configs at n>=10 : {(main.seeds >= 10).sum()} / {len(main)}")
    if len(low):
        print("    (exploratory grid stays at n=5 by design)")

# External sanity: a collapsed model predicts one class and shows AUROC ~0.5 or below.
for tag in ("tr", "cwlsd_matched"):
    path = rep / f"external_{tag}.csv"
    if not path.is_file():
        continue
    e = pd.read_csv(path)
    if "auroc" not in e.columns:
        continue
    b = e[e.method == "baseline_audio"]
    print(f"\n  external [{tag}] baseline AUROC by encoder:")
    for enc, grp in b.groupby("encoder"):
        m = grp["auroc"].mean()
        flag = "OK " if m > 0.6 else ("INVERTED" if m < 0.45 else "WEAK")
        print(f"    {flag} {enc:12s} {m:.3f}  (n_seeds={len(grp)})")
PYEOF
done

echo ""
echo "=================== shared artifacts ==================="
for f in ../data/manifests/cwlsd_matched_slices.csv ../data/manifests/tr_external_slices.csv \
         ../results_A10/reports/device_confound.json; do
  [ -f "$f" ] && echo "  OK      $(basename "$f")" || echo "  MISSING $(basename "$f")"
done
echo ""
echo "failed jobs (non-empty = investigate):"
grep -l "Traceback" ../log_*_train_*.log 2>/dev/null | head || echo "  none"
