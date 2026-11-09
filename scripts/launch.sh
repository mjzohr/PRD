#!/usr/bin/env bash
#   preprocess      slice audio, build manifests and the feature cache
#   quick           one seed, CNN, CT teacher: baseline vs distilled
#   main            teachers, bridges, baselines and the full student grid
#   ablation        signal-specificity controls, scored on both external cohorts
#   random_teacher  same pipeline with an untrained teacher encoder
#   report          tables and figures from whatever has finished
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS="${GPUS:-1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-4}"
export RESULTS_ROOT="${RESULTS_ROOT:-results}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
LOGS="${RESULTS_ROOT}/logs"

# deal a job list round-robin over GPUS x JOBS_PER_GPU background workers
run_jobs() {
  local file="$1" name="$2" shards=$(( GPUS * JOBS_PER_GPU )) i
  [ -s "$file" ] || { echo "$name: nothing to run"; return; }
  echo "$name: $(wc -l < "$file") jobs on $GPUS gpu(s)"
  for (( i=0; i<shards; i++ )); do
    awk -v n="$shards" -v r="$i" 'NR % n == r' "$file" > "$LOGS/${name}_${i}.sh"
    CUDA_VISIBLE_DEVICES=$(( i % GPUS )) bash "$LOGS/${name}_${i}.sh" > "$LOGS/${name}_${i}.log" 2>&1 &
  done
  wait
  grep -l Traceback "$LOGS/${name}"_*.log || true
}

plan() { mkdir -p "$LOGS"; python scripts/jobs.py "$1" > "$LOGS/$1.txt"; run_jobs "$LOGS/$1.txt" "$1"; }

external() {
  python scripts/evaluate_external.py --manifest data/manifests/tr_external_slices.csv \
      --processed data/processed_audio/tr --tag tr
  python scripts/evaluate_external.py --manifest data/manifests/cwlsd_external_slices.csv \
      --processed data/processed_audio/cwlsd --tag cwlsd
  python scripts/evaluate_external.py --manifest data/manifests/cwlsd_matched_slices.csv \
      --processed data/processed_audio/cwlsd --tag cwlsd_matched
}

case "${1:-}" in
  preprocess)
    python scripts/preprocess_icbhi.py
    python scripts/preprocess_tr.py
    python scripts/preprocess_cwlsd.py
    python scripts/match_cwlsd.py
    python scripts/cache_features.py
    ;;
  quick)
    python scripts/train_teachers.py --view ct_clinical --seed 42
    python scripts/bridge.py --view ct_clinical --seed 42 --age_tolerance 5
    python scripts/train_baseline.py --encoder CNN --seed 42
    python scripts/train_distilled.py --view ct_clinical --encoder CNN --seed 42 \
        --age_tolerance 5 --distill_loss smooth_l1 --distill_weight 1.0
    python scripts/report.py
    ;;
  main)
    plan prep
    plan main
    plan sensitivity
    external
    python scripts/report.py
    ;;
  ablation)
    python scripts/audit_shuffle.py --view ct_clinical --seed 42 || true
    plan ablation
    external
    python scripts/report.py
    python scripts/tables.py ablation
    ;;
  random_teacher)
    # bridges read the trained teachers from RESULTS_ROOT; students go to a separate root
    plan random_teacher_prep
    export REF_ROOT="$RESULTS_ROOT" RESULTS_ROOT="${RESULTS_ROOT}_randteacher"
    LOGS="${RESULTS_ROOT}/logs"
    plan random_teacher
    python scripts/tables.py random_teacher
    ;;
  report)
    python scripts/report.py
    python scripts/tables.py teachers
    python scripts/figures.py --reports "${RESULTS_ROOT}/reports"
    ;;
  *)
    echo "usage: bash scripts/launch.sh {preprocess|quick|main|ablation|random_teacher|report}"; exit 2 ;;
esac
