#!/usr/bin/env bash
# Stage 1: privileged teachers (sharp GOLD 0 vs 3&4) + audio->teacher bridge targets.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
for view in $VIEWS; do
  for seed in $SEEDS; do
    echo "==> teacher: view=$view seed=$seed"
    $PY train_teachers.py --view "$view" --seed "$seed"
    echo "==> bridge:  view=$view seed=$seed"
    $PY run_bridge.py --view "$view" --seed "$seed"
  done
done
