#!/usr/bin/env bash
# Disease-Topic Consensus Alignment, sweeping the guidance_weight knob.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
for view in $VIEWS; do
  for enc in $ENCODERS; do
    for g in $GUIDANCE_WEIGHTS; do
      for seed in $SEEDS; do
        echo "==> dtca: view=$view enc=$enc guidance_weight=$g seed=$seed"
        $PY train_dtca.py --view "$view" --encoder "$enc" --guidance_weight "$g" --seed "$seed"
      done
    done
  done
done
