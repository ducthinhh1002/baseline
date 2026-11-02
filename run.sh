#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# CPU preparation only. No model downloads or GPU jobs are started.
python -m benchmark.cli verify-sources
python -m benchmark.cli plan "$@"
