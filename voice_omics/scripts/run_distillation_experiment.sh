#!/usr/bin/env bash
# ============================================================================
# Focused study: MASKED FEATURE DISTILLATION only.
#
# Sweeps three axes, holding the distillation weight fixed by default:
#   1. audio architecture      : CNN, CRNN, TRANSFORMER
#   2. bridge age-tolerance    : +/- 3, 5, 7 years  (matching sensitivity)
#   3. distillation objective  : cosine (current method) + mse, smooth_l1,
#                                relational (RKD, a more defensible variant that
#                                transfers pairwise geometry, not absolute vectors)
#
# Results are ISOLATED under results_distillation/ (RESULTS_ROOT), so this study
# never touches the main results/. A same-encoder audio-only baseline is included
# so report.py can compute delta-vs-baseline.
#
# Scope (defaults): 2 views x 3 encoders x 3 ages x 4 losses x 1 weight x 5 seeds
#   = 360 distillation runs (+ teachers, bridges, baselines). Override any axis
#   via env. To ALSO sweep the weight, set DISTILL_WEIGHTS="0.5 1.0 2.0 4.0".
#
# Multi-GPU: shard by SEED. Every artifact is seed-tagged, so seed shards share no
# files and need no coordination -- each runs the full prep+distill for its seeds.
#   # 2-GPU box: split the 5 seeds 3/2 across the two GPUs, report once at the end.
#   CUDA_VISIBLE_DEVICES=0 SEEDS="11 23 42" RUN_REPORT=0 bash scripts/run_distillation_experiment.sh &
#   CUDA_VISIBLE_DEVICES=1 SEEDS="73 101"   RUN_REPORT=0 bash scripts/run_distillation_experiment.sh &
#   wait; RESULTS_ROOT=results_distillation python report.py
# Single GPU: just `bash scripts/run_distillation_experiment.sh`.
# ============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export RESULTS_ROOT="${RESULTS_ROOT:-results_distillation}"
PY="${PYTHON_BIN:-python}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
# Avoid BLAS thread explosions when several runs share the box.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# Use the precomputed feature cache if present (precompute_features.py); warn otherwise.
if [ ! -d "../data/feature_cache/ichbi" ] && [ ! -d "data/feature_cache/ichbi" ]; then
  echo "NOTE: no feature cache found -- run 'python precompute_features.py' once for a ~10x speedup."
fi

SEEDS="${SEEDS:-11 23 42 73 101}"
VIEWS="${VIEWS:-ct_clinical omics}"
ENCODERS="${ENCODERS:-CNN CRNN TRANSFORMER}"
AGE_TOLERANCES="${AGE_TOLERANCES:-3 5 7}"
DISTILL_LOSSES="${DISTILL_LOSSES:-cosine mse smooth_l1 relational}"
DISTILL_WEIGHTS="${DISTILL_WEIGHTS:-2.0}"
SKIP_PREP="${SKIP_PREP:-0}"
RUN_REPORT="${RUN_REPORT:-1}"

echo "results -> $RESULTS_ROOT | SEEDS=[$SEEDS] VIEWS=[$VIEWS] ENCODERS=[$ENCODERS]"
echo "ages=[$AGE_TOLERANCES] losses=[$DISTILL_LOSSES] weights=[$DISTILL_WEIGHTS] skip_prep=$SKIP_PREP"

if [ "$SKIP_PREP" != "1" ]; then
  # 1. Privileged teachers (per view, seed) -- into $RESULTS_ROOT.
  for view in $VIEWS; do for seed in $SEEDS; do
    echo "== teacher $view seed$seed"
    $PY train_teachers.py --view "$view" --seed "$seed"
  done; done

  # 2. Bridge targets per (view, seed, age tolerance).
  for view in $VIEWS; do for seed in $SEEDS; do for age in $AGE_TOLERANCES; do
    echo "== bridge $view seed$seed age$age"
    $PY run_bridge.py --view "$view" --seed "$seed" --age_tolerance "$age"
  done; done; done

  # 3. Same-encoder audio-only baseline (the comparison bar).
  for enc in $ENCODERS; do for seed in $SEEDS; do
    echo "== baseline $enc seed$seed"
    $PY train_baseline_audio.py --encoder "$enc" --seed "$seed"
  done; done
fi

# Stop after prep (teachers + bridges + baselines) so distillation can then be
# sharded across GPUs by any axis, all reading the prebuilt seed-tagged targets.
if [ "${PREP_ONLY:-0}" = "1" ]; then
  echo "== PREP_ONLY done (teachers + bridges + baselines). Skipping distillation."
  exit 0
fi

# 4. Distillation sweep: architecture x age-tolerance x loss (x weight).
for view in $VIEWS; do for enc in $ENCODERS; do for age in $AGE_TOLERANCES; do
  for loss in $DISTILL_LOSSES; do for w in $DISTILL_WEIGHTS; do for seed in $SEEDS; do
    echo "== distilled $view $enc age$age $loss w$w seed$seed"
    $PY train_distilled.py --view "$view" --encoder "$enc" --age_tolerance "$age" \
      --distill_loss "$loss" --distill_weight "$w" --seed "$seed"
  done; done; done; done; done; done

# 5. Report (isolated under $RESULTS_ROOT/reports/).
if [ "$RUN_REPORT" = "1" ]; then
  $PY report.py
  echo "== done -> $RESULTS_ROOT/reports/headline.md and summary.csv"
fi
