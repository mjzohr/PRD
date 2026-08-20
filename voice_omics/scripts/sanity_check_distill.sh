#!/usr/bin/env bash
# Fast guard against the "distillation had no effect" failure mode.
#
# Runs three single-seed configs that MUST differ from each other:
#   (1) mse  w=2   (2) smooth_l1 w=2   (3) mse w=2 with shuffled targets
# If any two report the same macro-F1, the auxiliary loss is not reaching the
# model and a full sweep would be wasted compute.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export RESULTS_ROOT="${RESULTS_ROOT:-results_sanity}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="${PYTHON_BIN:-python}"
SEED="${SEED:-42}"
VIEW="${VIEW:-ct_clinical}"

echo "==> sanity check (results -> $RESULTS_ROOT, seed $SEED, view $VIEW)"
$PY train_baseline_audio.py --encoder CNN --seed "$SEED"
$PY train_distilled.py --view "$VIEW" --encoder CNN --age_tolerance 5 \
    --distill_loss mse --distill_weight 2.0 --seed "$SEED"
$PY train_distilled.py --view "$VIEW" --encoder CNN --age_tolerance 5 \
    --distill_loss smooth_l1 --distill_weight 2.0 --seed "$SEED"
$PY train_distilled.py --view "$VIEW" --encoder CNN --age_tolerance 5 \
    --distill_loss mse --distill_weight 2.0 --seed "$SEED" --ablation shuffled_targets

echo ""
echo "==> macro-F1 by config (all four values must be DISTINCT):"
$PY - <<'PYEOF'
import json, glob
from pathlib import Path
from config import Config          # resolve exactly like the training scripts do

root = Config.RESULTS_DIR
files = sorted(glob.glob(str(root / "**" / "summary_metrics.json"), recursive=True))
print(f"  (found {len(files)} run(s) under {root})")
seen = {}
for f in files:
    tag = str(Path(f).parent.parent.relative_to(root))
    value = json.load(open(f))["macro_f1"]["mean"]
    print(f"  {value:.12f}   {tag}")
    seen.setdefault(round(value, 10), []).append(tag)
dupes = {k: v for k, v in seen.items() if len(v) > 1}
print()
if len(files) < 4:
    print(f"INCONCLUSIVE: expected 4 runs, found {len(files)} -- nothing was compared.")
elif dupes:
    print("FAIL: identical macro-F1 across different configs -> auxiliary loss is inert:")
    for value, tags in dupes.items():
        for tag in tags:
            print(f"   {value}  {tag}")
else:
    print("PASS: every config produced a distinct score; the auxiliary loss is active.")
PYEOF
