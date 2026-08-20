#!/usr/bin/env bash
# Ablations / matched audio-only controls at the default knob. Each turns off one
# part of the privileged signal so its contribution is isolated (same audio encoder):
#   distilled  -> shuffled_targets (breaks audio<->teacher correspondence)
#   contrastive-> audio_text_only (drops the privileged modality; matched control),
#                 vanilla / direct (loss-form ablations)
#   dtca       -> audio_task_only (matched control), no_disease, no_topic,
#                 shuffled_labels, random_features
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
for view in $VIEWS; do
  for enc in $ENCODERS; do
    for seed in $SEEDS; do
      echo "==> ablations: view=$view enc=$enc seed=$seed"
      $PY train_distilled.py --view "$view" --encoder "$enc" --seed "$seed" --ablation shuffled_targets
      for ce in $CLINICAL_ENCODERS; do
        for lt in audio_text_only vanilla direct; do
          $PY train_contrastive.py --view "$view" --encoder "$enc" \
            --clinical_encoder "$ce" --seed "$seed" --loss_type "$lt"
        done
      done
      for abl in audio_task_only no_disease no_topic shuffled_labels random_features; do
        $PY train_dtca.py --view "$view" --encoder "$enc" --seed "$seed" --ablation "$abl"
      done
    done
  done
done
