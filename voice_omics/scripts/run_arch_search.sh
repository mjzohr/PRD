#!/usr/bin/env bash
# Architecture search for the CRNN and Transformer audio encoders.
# Ranks configs by inner-validation macro-F1 and logs the best architecture for
# each. Results isolated under results_arch_search/. Small by design:
#   CRNN: 8 configs (hidden x depth x dropout)
#   TRANSFORMER: 8 configs (width x depth x positional-encoding)
#
# Single GPU:  bash scripts/run_arch_search.sh
# 2 GPUs (one encoder each):
#   CUDA_VISIBLE_DEVICES=0 ENCODERS="CRNN"        bash scripts/run_arch_search.sh &
#   CUDA_VISIBLE_DEVICES=1 ENCODERS="TRANSFORMER" bash scripts/run_arch_search.sh &
#   wait
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export RESULTS_ROOT="${RESULTS_ROOT:-results_arch_search}"
export NUM_WORKERS="${NUM_WORKERS:-16}"
PY="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-42}"                 # add more for a sturdier ranking, e.g. "11 42"
EPOCHS="${EPOCHS:-25}"
ENCODERS="${ENCODERS:-CRNN TRANSFORMER}"

echo "arch search -> $RESULTS_ROOT | encoders=[$ENCODERS] seeds=[$SEEDS] epochs=$EPOCHS"
$PY arch_search.py --encoders $ENCODERS --seeds $SEEDS --epochs "$EPOCHS"
