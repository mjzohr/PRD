#!/usr/bin/env bash
# Masked feature distillation, sweeping the distill_weight knob.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
for view in $VIEWS; do
  for enc in $ENCODERS; do
    for w in $DISTILL_WEIGHTS; do
      for seed in $SEEDS; do
        echo "==> distilled: view=$view enc=$enc distill_weight=$w seed=$seed"
        $PY train_distilled.py --view "$view" --encoder "$enc" --distill_weight "$w" --seed "$seed"
      done
    done
  done
done
