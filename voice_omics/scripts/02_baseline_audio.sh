#!/usr/bin/env bash
# Audio-only baselines (one per encoder in ENCODERS).
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
for enc in $ENCODERS; do
  for seed in $SEEDS; do
    echo "==> baseline: encoder=$enc seed=$seed"
    $PY train_baseline_audio.py --encoder "$enc" --seed "$seed"
  done
done
