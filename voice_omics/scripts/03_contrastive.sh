#!/usr/bin/env bash
# Contrastive clinical-pivot: sweep temperature, over each clinical-text encoder.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
for view in $VIEWS; do
  for enc in $ENCODERS; do
    for ce in $CLINICAL_ENCODERS; do
      for t in $TEMPERATURES; do
        for seed in $SEEDS; do
          echo "==> contrastive: view=$view enc=$enc clinical=$ce temperature=$t seed=$seed"
          $PY train_contrastive.py --view "$view" --encoder "$enc" \
            --clinical_encoder "$ce" --temperature "$t" --seed "$seed"
        done
      done
    done
  done
done
