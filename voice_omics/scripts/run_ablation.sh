#!/usr/bin/env bash
# Signal-specificity ablation: six controls x 3 encoders x 10 seeds, at the
# confirmatory operating point (CT teacher, smooth-L1, w=1, tau=+/-5, Config A).
# Anything already computed is skipped. Each control is then scored on both
# external cohorts from its saved checkpoints.
#
#   bash scripts/run_ablation.sh
#   DRY_RUN=1 bash scripts/run_ablation.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT_DIR="$(cd .. && pwd)"

PY="${PYTHON_BIN:-python}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUM_WORKERS="${NUM_WORKERS:-2}"
GPUS="${GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-8}"
SHARDS=$(( GPUS * JOBS_PER_GPU ))
WORK=".ablation"; mkdir -p "$WORK"

banner () { echo ""; echo "============================================================"; echo "$1"; echo "============================================================"; }

# Config A is the DEFAULT front-end (full band, no normalisation). Clear anything
# inherited from the calling shell first: an exported AUDIO_F_MAX from an earlier
# session would silently mix Config B features into the Config A results root.
unset AUDIO_F_MAX NORMALIZE_FEATURES MFCC_N_MELS
export RESULTS_ROOT="results_A10"

banner "0. front-end check (must read F_MAX=None NORM=False)"
$PY -c "from config import Config; from dataset import feature_cache_dir; \
print('   F_MAX=%s NORM=%s cache=%s' % (Config.F_MAX, Config.NORMALIZE_FEATURES, \
feature_cache_dir(Config.PROCESSED_DIR).name))"

banner "1. what the targets actually look like under each control"
$PY audit_shuffle.py --view ct_clinical --seed 42 || true

banner "2. plan"
$PY make_paper_jobs.py --phase ablation > "$WORK/jobs.txt"
TOTAL=$(wc -l < "$WORK/jobs.txt")
echo "   $TOTAL job(s) outstanding"
if [ "$TOTAL" -eq 0 ]; then
  echo "   nothing to train -- skipping to scoring"
else
  head -5 "$WORK/jobs.txt"; echo "   ..."
fi
if [ "${DRY_RUN:-0}" = "1" ]; then echo "DRY_RUN: stopping here"; exit 0; fi

banner "3. training"
if [ "$TOTAL" -gt 0 ]; then
  for (( i=0; i<SHARDS; i++ )); do
    gpu=$(( i % GPUS ))
    awk -v n="$SHARDS" -v r="$i" 'NR % n == r' "$WORK/jobs.txt" > "$WORK/shard_${i}.sh"
    CUDA_VISIBLE_DEVICES=$gpu nohup bash "$WORK/shard_${i}.sh" \
      > "$ROOT_DIR/log_abl_${i}.log" 2>&1 &
  done
  wait
fi

banner "4. external scoring (both cohorts, from saved checkpoints)"
$PY evaluate_external.py --manifest data/manifests/tr_external_slices.csv \
    --processed data/processed_audio/tr --tag tr
$PY evaluate_external.py --manifest data/manifests/cwlsd_matched_slices.csv \
    --processed data/processed_audio/cwlsd --tag cwlsd_matched

banner "5. reports"
$PY report.py
$PY ablation_table.py

banner "DONE"
echo "   internal + external ladder: $ROOT_DIR/$RESULTS_ROOT/reports/ablation_ladder.csv"
echo "   check for stragglers:  grep -l Traceback $ROOT_DIR/log_abl_*.log"
