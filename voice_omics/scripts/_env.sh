#!/usr/bin/env bash
# Shared settings for every experiment script. Source this first.
# Override any of these from the environment, e.g. `SEEDS="42" ./run_all.sh`.
set -euo pipefail

# Run from the package directory (flat imports expect this cwd).
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY="${PYTHON_BIN:-python}"

# Publication uses all five seeds; default to one for a quick pass.
SEEDS="${SEEDS:-42}"
# Full publication run: SEEDS="11 23 42 73 101"

VIEWS="${VIEWS:-omics ct_clinical}"
ENCODERS="${ENCODERS:-CNN}"                 # add "CRNN TRANSFORMER" for the encoder ablation
# Clinical-text encoder variants (contrastive method): bioclinicalbert gatortron modernbert
CLINICAL_ENCODERS="${CLINICAL_ENCODERS:-bioclinicalbert}"

# Key-knob sweep grids (kept small; see config.Hparams).
DISTILL_WEIGHTS="${DISTILL_WEIGHTS:-0.5 1.0 2.0 4.0}"
TEMPERATURES="${TEMPERATURES:-0.05 0.07 0.1 0.2}"
GUIDANCE_WEIGHTS="${GUIDANCE_WEIGHTS:-0.25 0.5 1.0 2.0}"

echo "PY=$PY | SEEDS=[$SEEDS] | VIEWS=[$VIEWS] | ENCODERS=[$ENCODERS]"
