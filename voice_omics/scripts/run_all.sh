#!/usr/bin/env bash
# End-to-end pipeline. For the full publication run:
#   SEEDS="11 23 42 73 101" ./scripts/run_all.sh
#
# Resolve the scripts dir to an ABSOLUTE path first: _env.sh changes the working
# directory, so a relative path would break for the steps below.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/_env.sh"

bash "$here/00_preprocess.sh"
bash "$here/01_teachers.sh"
bash "$here/02_baseline_audio.sh"
bash "$here/03_contrastive.sh"
bash "$here/04_distilled.sh"
bash "$here/05_dtca.sh"
bash "$here/06_ablations.sh"
bash "$here/07_report.sh"
echo "==> all done. See results/reports/headline.md"
