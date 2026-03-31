#!/bin/bash

set -euo pipefail

REPO_PATH="/workspace/RLinf"
MODEL_PATH="/workspace/RLinf/weight/RLinf-Pi05-SFT"
OUTPUT_DIR="/workspace/RLinf/outputs"

cd "${REPO_PATH}"

python scripts/simple_eval_libero10_pi05.py \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
