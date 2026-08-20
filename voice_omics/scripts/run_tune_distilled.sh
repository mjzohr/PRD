#!/usr/bin/env bash
# Optuna tuning of the distilled student for the encoders that still need it.
# One (encoder, view) study per GPU: 4 studies x TRIALS trials.
# Prereqs: teachers + bridges for the search seed exist (PREP_ONLY run), and the
# feature cache is built (precompute_features.py).
#
#   bash scripts/run_tune_distilled.sh            # launches all 4 in background
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export RESULTS_ROOT="${RESULTS_ROOT:-results_distillation}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="${PYTHON_BIN:-python}"
TRIALS="${TRIALS:-20}"
SEARCH_SEEDS="${SEARCH_SEEDS:-42}"

CUDA_VISIBLE_DEVICES=0 nohup $PY tune_distilled.py --encoder CRNN        --view ct_clinical --trials "$TRIALS" --seeds $SEARCH_SEEDS > ../log_tune_crnn_ct.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup $PY tune_distilled.py --encoder CRNN        --view omics       --trials "$TRIALS" --seeds $SEARCH_SEEDS > ../log_tune_crnn_om.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup $PY tune_distilled.py --encoder TRANSFORMER --view ct_clinical --trials "$TRIALS" --seeds $SEARCH_SEEDS > ../log_tune_trf_ct.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup $PY tune_distilled.py --encoder TRANSFORMER --view omics       --trials "$TRIALS" --seeds $SEARCH_SEEDS > ../log_tune_trf_om.log 2>&1 &
echo "4 tuning studies launched; watch ../log_tune_*.log"
