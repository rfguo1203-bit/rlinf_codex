#!/bin/bash

set -euo pipefail

REPO_PATH="/workspace/RLinf"
MODEL_PATH="/workspace/RLinf/weight/RLinf-Pi05-SFT"
OUTPUT_DIR="/workspace/RLinf/outputs"

cd "${REPO_PATH}"

if [[ "${1:-}" == "--list-tasks" ]]; then
  python scripts/simple_eval_libero10_pi05.py --list-tasks
  exit 0
fi

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  python scripts/simple_eval_libero10_pi05.py \
    --task-id "$1" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-episodes "${2:-1}"
  exit 0
fi

if [[ -n "${1:-}" ]]; then
  python scripts/simple_eval_libero10_pi05.py \
    --task-name "$1" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-episodes "${2:-1}"
  exit 0
fi

python scripts/simple_eval_libero10_pi05.py \
  --task-id 0 \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-episodes 1
