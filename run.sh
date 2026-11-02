#!/usr/bin/env bash
# One command: environment -> COCO prompts -> models -> training -> evaluation.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$PROJECT"
usage() {
  cat <<'EOF'
Usage: ./run.sh [--gpus 0] [--workers-per-gpu 1] [--prompts FILE]
                [--config FILE] [--overrides FILE] [--prepare-only]
                [--skip-install] [--retry-failed]

Default: 5 methods x 3 students x 7 manipulations (105 trainings).
No --prompts: use config.prompt_source if present, otherwise download COCO
captions and create disjoint train/test/calibration splits automatically.
No FID/KID. Results: work/reports/metrics.csv and metrics.json.
Environment: BASELINE_WORK, BASELINE_CACHE, BASELINE_VENV, PYTHON_BIN,
             TORCH_INDEX_URL (default CUDA12.6 wheels).
Requires Linux, NVIDIA driver, git and Python3.12.
Example: BASELINE_WORK=/data/baseline/work BASELINE_CACHE=/data/baseline/cache ./run.sh --gpus 1
EOF
}
GPUS=0
WORKERS=1
CONFIG="$PROJECT/config/default.json"
PROMPTS=""
OVERRIDES=""
PREPARE_ONLY=0
SKIP_INSTALL=0
RETRY_FAILED=0
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --gpus|--workers-per-gpu|--config|--prompts|--overrides)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --gpus) GPUS="$2" ;;
        --workers-per-gpu) WORKERS="$2" ;;
        --config) CONFIG="$2" ;;
        --prompts) PROMPTS="$2" ;;
        --overrides) OVERRIDES="$2" ;;
      esac
      shift 2 ;;
    --prepare-only) PREPARE_ONLY=1; shift ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    --retry-failed) RETRY_FAILED=1; shift ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$WORKERS" =~ ^[1-9][0-9]*$ && "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
  echo 'Use positive workers and comma-separated GPU indices, e.g. --gpus 0,1' >&2; exit 2;
}
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }
for program in git nvidia-smi; do
  command -v "$program" >/dev/null || { echo "Required program not found: $program" >&2; exit 1; }
done
for gpu in ${GPUS//,/ }; do nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader; done
export BASELINE_WORK="${BASELINE_WORK:-$PROJECT/work}"
export BASELINE_CACHE="${BASELINE_CACHE:-$PROJECT/.cache}"
VENV="${BASELINE_VENV:-$PROJECT/.venv}"
mkdir -p "$BASELINE_WORK/logs" "$BASELINE_CACHE"
export HF_HOME="$BASELINE_CACHE/huggingface" TORCH_HOME="$BASELINE_CACHE/torch"
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
PYTHON="${PYTHON_BIN:-python3.12}"
if [[ ! -x "$VENV/bin/python" ]]; then
  command -v "$PYTHON" >/dev/null || { echo 'Python3.12 not found; set PYTHON_BIN.' >&2; exit 1; }
  "$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,12), "Python3.12 required"'
  "$PYTHON" -m venv --without-pip "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3,12), "Python3.12 required"'
if ! "$PY" -m pip --version >/dev/null 2>&1; then
  "$PY" - "$VENV/get-pip.py" <<'PY'
import sys, urllib.request
urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', sys.argv[1])
PY
  "$PY" "$VENV/get-pip.py" --disable-pip-version-check
  rm "$VENV/get-pip.py"
fi
if (( ! SKIP_INSTALL )); then
  echo 'Installing isolated Python environment...'
  "$PY" -m pip install --disable-pip-version-check --upgrade pip
  "$PY" -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 \
    --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
  "$PY" -m pip install --no-cache-dir -r "$PROJECT/requirements.txt"
  "$PY" -m pip check
fi
"$PY" -c 'import torch,torchvision; assert torch.__version__.split("+")[0]=="2.7.1"; assert torchvision.__version__.split("+")[0]=="0.22.1"; assert torch.cuda.is_available(), "CUDA unavailable; check NVIDIA driver and wheel selection"'
"$PY" -m pip freeze > "$BASELINE_WORK/environment.txt"
if [[ -z "$PROMPTS" ]]; then
  PROMPTS="$("$PY" -c 'import json,sys,pathlib; p=pathlib.Path(json.load(open(sys.argv[1]))["prompt_source"]); print(p if p.is_file() else "")' "$CONFIG")"
  if [[ -z "$PROMPTS" ]]; then
    "$PY" prepare_prompts.py --config "$CONFIG" --output "$BASELINE_WORK/input_prompts.json"
    PROMPTS="$BASELINE_WORK/input_prompts.json"
  fi
fi
INIT=(--config "$CONFIG" --prompts "$PROMPTS")
[[ -z "$OVERRIDES" ]] || INIT+=(--overrides "$OVERRIDES")
"$PY" tests.py
"$PY" test_training.py
"$PY" campaign.py initialize "${INIT[@]}"
echo "Downloading pinned sources and model artifacts; log: $BASELINE_WORK/logs/download.log"
if ! "$PY" download.py > "$BASELINE_WORK/logs/download.log" 2>&1; then
  tail -n 40 "$BASELINE_WORK/logs/download.log" >&2
  exit 1
fi
if (( PREPARE_ONLY )); then
  echo 'Environment, prompts and downloads ready. Rerun without --prepare-only to train.'
  exit 0
fi
"$PY" preflight.py --gpu "${GPUS%%,*}"
RUN=(--gpus "$GPUS" --workers-per-gpu "$WORKERS")
(( ! RETRY_FAILED )) || RUN+=(--retry-failed)
exec "$PY" pipeline.py "${RUN[@]}"
