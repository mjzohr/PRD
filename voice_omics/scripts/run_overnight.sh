#!/usr/bin/env bash
# ============================================================================
# OVERNIGHT RUN -- takes both headline configurations from n=5 to n=10 seeds and
# produces every artifact the paper needs by morning.
#
# Results land in NEW roots so the existing runs are preserved untouched:
#   results_A10/  Config A (full band, no normalization)  <- seeded from results_distillation
#   results_B10/  Config B (2 kHz + per-channel CMVN)     <- seeded from results_transfer
# Existing 5-seed runs are copied in first, so only the 5 NEW seeds are computed.
#
#   bash scripts/run_overnight.sh                # full plan
#   DRY_RUN=1 bash scripts/run_overnight.sh      # show the plan, run nothing
# ============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT_DIR="$(cd .. && pwd)"

PY="${PYTHON_BIN:-python}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUM_WORKERS="${NUM_WORKERS:-2}"
GPUS="${GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-8}"
SHARDS=$(( GPUS * JOBS_PER_GPU ))
WORK=".overnight"; mkdir -p "$WORK"

banner () { echo ""; echo "============================================================"; echo "$1"; echo "============================================================"; }

run_sharded () {   # $1 = job file, $2 = log prefix
  local file="$1" prefix="$2" total i gpu
  total=$(wc -l < "$file")
  if [ "$total" -eq 0 ]; then echo "   nothing to do"; return; fi
  echo "   $total job(s) across $SHARDS shards"
  for (( i=0; i<SHARDS; i++ )); do
    gpu=$(( i % GPUS ))
    awk -v n="$SHARDS" -v r="$i" 'NR % n == r' "$file" > "$WORK/${prefix}_${i}.sh"
    CUDA_VISIBLE_DEVICES=$gpu nohup bash "$WORK/${prefix}_${i}.sh" \
      > "$ROOT_DIR/log_${prefix}_${i}.log" 2>&1 &
  done
  wait
}

# ---- seed the new roots from the existing 5-seed results -------------------
banner "0. preparing result roots (existing runs are preserved)"
for pair in "results_distillation:results_A10" "results_transfer:results_B10"; do
  src="$ROOT_DIR/${pair%%:*}"; dst="$ROOT_DIR/${pair##*:}"
  if [ ! -d "$dst" ] && [ -d "$src" ]; then
    echo "   copying $(basename "$src") -> $(basename "$dst")"
    [ "${DRY_RUN:-0}" = "1" ] || cp -r "$src" "$dst"
  else
    echo "   $(basename "$dst") already present (or no source) -- leaving as is"
  fi
done

# ---- one config: prep, confirm runs, external evaluations ------------------
run_config () {          # $1 = results root, $2 = label, $3.. = env assignments
  local root="$1" label="$2"; shift 2
  # Clear any front-end settings inherited from the caller's shell BEFORE applying
  # this configuration's own. Without this, an exported AUDIO_F_MAX from a previous
  # session silently leaks into Config A and mixes two front-ends in one results root.
  unset AUDIO_F_MAX NORMALIZE_FEATURES MFCC_N_MELS
  export RESULTS_ROOT="$root"
  for assignment in "$@"; do export "$assignment"; done
  banner "$label  ($root)  [front-end: AUDIO_F_MAX=${AUDIO_F_MAX:-none} NORMALIZE_FEATURES=${NORMALIZE_FEATURES:-0}]"
  $PY -c "from config import Config; from dataset import feature_cache_dir; \
print('   verify -> F_MAX=%s NORM=%s cache=%s' % (Config.F_MAX, Config.NORMALIZE_FEATURES, \
feature_cache_dir(Config.PROCESSED_DIR).name))"

  echo "-- feature cache"
  [ "${DRY_RUN:-0}" = "1" ] || $PY precompute_features.py

  $PY make_paper_jobs.py --phase confirm_prep > "$WORK/${label}_prep.txt"
  $PY make_paper_jobs.py --phase confirm      > "$WORK/${label}_train.txt"
  echo "-- planned: $(wc -l < "$WORK/${label}_prep.txt") prep, $(wc -l < "$WORK/${label}_train.txt") training runs"
  if [ "${DRY_RUN:-0}" = "1" ]; then head -5 "$WORK/${label}_train.txt"; return; fi

  echo "-- teachers + bridges (new seeds)"; run_sharded "$WORK/${label}_prep.txt" "${label}_prep"
  echo "-- training (new seeds)";           run_sharded "$WORK/${label}_train.txt" "${label}_train"

  echo "-- external cohorts"
  $PY evaluate_external.py --manifest data/manifests/tr_external_slices.csv \
      --processed data/processed_audio/tr --tag tr
  $PY evaluate_external.py --manifest data/manifests/cwlsd_external_slices.csv \
      --processed data/processed_audio/cwlsd --tag cwlsd
  $PY evaluate_external.py --manifest data/manifests/cwlsd_matched_slices.csv \
      --processed data/processed_audio/cwlsd --tag cwlsd_matched

  echo "-- reports"
  $PY report.py
  $PY audit_targets.py --view ct_clinical --seed 42 || true
  $PY make_figures.py 2>/dev/null || echo "   (figures need matplotlib -- generate locally)"
}

# ---- matched CWLSD manifest (cheap, needed by both configs) ----------------
banner "1. matched CWLSD manifest (drops heart failure / fibrosis / effusion)"
[ "${DRY_RUN:-0}" = "1" ] || $PY make_matched_manifest.py

# ---- benchmark confound audit (config-independent) -------------------------
banner "2. ICBHI device-confound audit"
[ "${DRY_RUN:-0}" = "1" ] || RESULTS_ROOT=results_A10 $PY audit_device_confound.py

# ---- Config A then Config B ------------------------------------------------
run_config results_A10 "A"                                    # defaults = full band, no norm
run_config results_B10 "B" AUDIO_F_MAX=2000 NORMALIZE_FEATURES=1

banner "DONE -- run scripts/morning_check.sh to verify completeness"
