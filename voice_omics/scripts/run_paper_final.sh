#!/usr/bin/env bash
# ============================================================================
# FINAL PAPER RUN -- completes only what the write-up still needs.
#
# Resumable: already-finished runs are skipped, so re-running is safe and cheap.
# Jobs are emitted in priority order and dealt round-robin across GPUs, so if
# you stop early the most important blocks are the ones that finished.
#
#   bash scripts/run_paper_final.sh              # 4 GPUs (default)
#   GPUS=2 bash scripts/run_paper_final.sh       # fewer GPUs
#   DRY_RUN=1 bash scripts/run_paper_final.sh    # show the plan, run nothing
# ============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export RESULTS_ROOT="${RESULTS_ROOT:-results_distillation}"
# Each training process is small; with several sharing a GPU, a few loader
# workers each is plenty and avoids oversubscribing the CPU.
export NUM_WORKERS="${NUM_WORKERS:-4}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="${PYTHON_BIN:-python}"
GPUS="${GPUS:-4}"
# These models use a small fraction of a GPU, so run several per device.
# Raise if `nvidia-smi` still shows low memory/utilization.
JOBS_PER_GPU="${JOBS_PER_GPU:-4}"
SHARDS=$(( GPUS * JOBS_PER_GPU ))
WORK="${WORK:-.paper_jobs}"
mkdir -p "$WORK"

# ---- 0. feature cache (the ~10x speedup; safe to re-run, it skips existing) --
echo "==> ensuring feature cache"
$PY precompute_features.py

# ---- 1. plan ---------------------------------------------------------------
$PY make_paper_jobs.py --phase prep  > "$WORK/jobs_prep.txt"
$PY make_paper_jobs.py --phase train > "$WORK/jobs_train.txt"
n_prep=$(wc -l < "$WORK/jobs_prep.txt")
n_train=$(wc -l < "$WORK/jobs_train.txt")
echo "==> plan: $n_prep prep job(s), $n_train training run(s) on $GPUS GPU(s)"
echo "    (at ~6 min/run that is ~$(( n_train * 6 / GPUS / 60 ))h; at ~12 min/run ~$(( n_train * 12 / GPUS / 60 ))h)"
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "--- prep ---"; cat "$WORK/jobs_prep.txt"
  echo "--- train (first 20) ---"; head -20 "$WORK/jobs_train.txt"
  exit 0
fi

run_sharded () {  # $1 = job file, $2 = log prefix
  local file="$1" prefix="$2" total i gpu
  total=$(wc -l < "$file")
  [ "$total" -eq 0 ] && { echo "    nothing to do"; return; }
  echo "    $total job(s) over $SHARDS shard(s) ($JOBS_PER_GPU per GPU x $GPUS GPUs)"
  for (( i=0; i<SHARDS; i++ )); do
    gpu=$(( i % GPUS ))
    awk -v n="$SHARDS" -v r="$i" 'NR % n == r' "$file" > "$WORK/${prefix}_shard${i}.sh"
    CUDA_VISIBLE_DEVICES=$gpu nohup bash "$WORK/${prefix}_shard${i}.sh" \
      > "../log_${prefix}_s${i}_gpu${gpu}.log" 2>&1 &
  done
  wait
}

# ---- 2. prep (teachers + bridges for the new seeds) ------------------------
echo "==> phase 1/2: prep"
run_sharded "$WORK/jobs_prep.txt" "prep"

# ---- 3. training -----------------------------------------------------------
echo "==> phase 2/2: training ($n_train runs)"
run_sharded "$WORK/jobs_train.txt" "train"

# ---- 4. paper artifacts ----------------------------------------------------
echo "==> building report"
$PY report.py
$PY make_figures.py || echo "    (figures skipped -- install matplotlib for Fig. 4)"
echo "==> done. See $RESULTS_ROOT/reports/{headline.md,summary.csv,paired_vs_baseline.csv,sensitivity_distilled.csv}"
