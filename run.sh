#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python campaign.py initialize "$@"
mkdir -p "${BASELINE_WORK:-work}/logs"
python download.py --bases-only > "${BASELINE_WORK:-work}/logs/download.log" 2>&1
python campaign.py launch
