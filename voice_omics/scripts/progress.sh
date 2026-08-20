#!/usr/bin/env bash
# How much of the overnight run is left?
# Recomputes the outstanding job list per results root (the generator skips
# anything already finished), so the count is exact rather than inferred.
#   bash scripts/progress.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PYTHON_BIN:-python}"
WORK=".overnight"

for pair in "A:results_A10" "B:results_B10"; do
  label="${pair%%:*}"; root="${pair##*:}"
  planned_file="$WORK/${label}_train.txt"
  [ -d "../$root" ] || { echo "== $label ($root): not started"; continue; }

  planned=$([ -f "$planned_file" ] && wc -l < "$planned_file" || echo 0)
  remaining=$(RESULTS_ROOT="$root" $PY make_paper_jobs.py --phase confirm 2>/dev/null | wc -l)
  if [ "$planned" -gt 0 ]; then
    done=$(( planned - remaining )); pct=$(( 100 * done / planned ))
  else
    done=0; pct=0
  fi
  echo ""
  echo "== $label ($root)"
  printf "   %d/%d done (%d%%), %d remaining\n" "$done" "$planned" "$pct" "$remaining"

  # Rate + ETA from the most recently written run artifacts.
  RESULTS_ROOT="$root" REMAIN="$remaining" $PY - <<'PYEOF'
import glob, os, time
from pathlib import Path
root = Path(os.environ["RESULTS_ROOT"])
remain = int(os.environ["REMAIN"])
files = glob.glob(str(root / "*/**/metrics/summary_metrics.json"), recursive=True)
if len(files) < 5 or remain == 0:
    raise SystemExit
ts = sorted(os.path.getmtime(f) for f in files)[-40:]      # recent window
span, n = ts[-1] - ts[0], len(ts) - 1
if span <= 0 or n <= 0:
    raise SystemExit
per = span / n                                              # wall-clock sec per completed run
idle = time.time() - ts[-1]
print(f"   ~{per:.1f}s between completions | last finished {idle/60:.1f} min ago")
print(f"   ETA ~{remain * per / 3600:.1f}h")
if idle > 900:
    print("   WARNING: nothing completed in 15+ min -- check the logs")
PYEOF
done

echo ""
echo "== activity"
running=$(pgrep -fc "train_baseline_audio.py|train_distilled.py" 2>/dev/null || echo 0)
echo "   training processes alive: $running"
echo "   recent errors:"
grep -l "Traceback" ../log_*_train_*.log 2>/dev/null | head -5 || echo "     none"
