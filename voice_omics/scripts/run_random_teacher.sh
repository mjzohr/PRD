#!/usr/bin/env bash
# Random-init teacher control: the full pipeline with the teacher's trained
# weights withheld. Results go to their own root; the audio-only and trained-
# teacher arms are read from REF_ROOT for the paired comparison, never recomputed.
#
# The run is split because the two stages need different roots. Bridges read the
# fitted teacher from REF_ROOT and write targets to data/distillation, which is
# independent of RESULTS_ROOT; students read those targets and write to NEW_ROOT.
#
#   bash scripts/run_random_teacher.sh
#   DRY_RUN=1 bash scripts/run_random_teacher.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT_DIR="$(cd .. && pwd)"

PY="${PYTHON_BIN:-python}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUM_WORKERS="${NUM_WORKERS:-2}"
GPUS="${GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-8}"
SHARDS=$(( GPUS * JOBS_PER_GPU ))
WORK=".randteacher"; mkdir -p "$WORK"

REF_ROOT="${REF_ROOT:-results_A10}"          # where teachers + reported runs live
NEW_ROOT="${NEW_ROOT:-results_randteacher}"  # where this control writes
export REF_ROOT

banner () { echo ""; echo "============================================================"; echo "$1"; echo "============================================================"; }

# Config A is the default front-end. Clear inherited settings so a stale
# AUDIO_F_MAX cannot mix front-ends into this root.
unset AUDIO_F_MAX NORMALIZE_FEATURES MFCC_N_MELS

banner "0. front-end check (must read F_MAX=None NORM=False)"
RESULTS_ROOT="$NEW_ROOT" $PY -c "from config import Config; from dataset import feature_cache_dir; \
print('   F_MAX=%s NORM=%s cache=%s' % (Config.F_MAX, Config.NORMALIZE_FEATURES, \
feature_cache_dir(Config.PROCESSED_DIR).name))"
echo "   reference root : $REF_ROOT   (teachers + already-reported arms)"
echo "   new root       : $NEW_ROOT   (this control only)"

banner "1. plan"
RESULTS_ROOT="$REF_ROOT" $PY make_paper_jobs.py --phase random_teacher_prep > "$WORK/bridges.txt"
RESULTS_ROOT="$NEW_ROOT" $PY make_paper_jobs.py --phase random_teacher      > "$WORK/students.txt"
echo "   $(wc -l < "$WORK/bridges.txt") bridge(s), $(wc -l < "$WORK/students.txt") student run(s)"
head -2 "$WORK/students.txt"; echo "   ..."
if [ "${DRY_RUN:-0}" = "1" ]; then echo "DRY_RUN: stopping here"; exit 0; fi

# Bridges must finish before any student starts: students read what they write.
banner "2. random-init bridges  (RESULTS_ROOT=$REF_ROOT, reads the fitted teachers)"
if [ -s "$WORK/bridges.txt" ]; then
  RESULTS_ROOT="$REF_ROOT" bash "$WORK/bridges.txt" || { echo "BRIDGE FAILED -- stopping"; exit 1; }
else
  echo "   already built"
fi

banner "3. what the untrained teacher's targets look like"
RESULTS_ROOT="$NEW_ROOT" $PY audit_targets.py \
  --targets "ct_clinical_targets_randinit_age5_seed42.npy" || true

banner "4. students  (RESULTS_ROOT=$NEW_ROOT)"
if [ -s "$WORK/students.txt" ]; then
  for (( i=0; i<SHARDS; i++ )); do
    gpu=$(( i % GPUS ))
    awk -v n="$SHARDS" -v r="$i" 'NR % n == r' "$WORK/students.txt" > "$WORK/shard_${i}.sh"
    CUDA_VISIBLE_DEVICES=$gpu RESULTS_ROOT="$NEW_ROOT" nohup bash "$WORK/shard_${i}.sh" \
      > "$ROOT_DIR/log_rt_${i}.log" 2>&1 &
  done
  wait
else
  echo "   nothing to do"
fi

banner "5. result"
RESULTS_ROOT="$NEW_ROOT" $PY random_teacher_table.py

banner "DONE"
echo "   errors, if any:  grep -l Traceback $ROOT_DIR/log_rt_*.log"
