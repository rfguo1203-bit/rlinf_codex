#!/bin/bash

set -euo pipefail

REPO_PATH="/workspace/RLinf"
MODEL_PATH="/workspace/RLinf/weight/RLinf-Pi05-SFT"
OUTPUT_DIR="/workspace/RLinf/outputs"
CONFIG_NAME="libero_10_ppo_openpi_pi05"
NUM_EPISODES="1"
SAVE_FRACTION="1.0"
SEED=""
SHUFFLE="false"

# Task selection: use exactly one of the following modes.
LIST_TASKS="false"
TASK_ID="0"
TASK_NAME=""

cd "${REPO_PATH}"

ARGS=(
  --config-name "${CONFIG_NAME}"
  --model-path "${MODEL_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --num-episodes "${NUM_EPISODES}"
  --save-fraction "${SAVE_FRACTION}"
)

if [[ "${LIST_TASKS}" == "true" ]]; then
  ARGS+=(--list-tasks)
elif [[ -n "${TASK_NAME}" ]]; then
  ARGS+=(--task-name "${TASK_NAME}")
else
  ARGS+=(--task-id "${TASK_ID}")
fi

if [[ -n "${SEED}" ]]; then
  ARGS+=(--seed "${SEED}")
fi

if [[ "${SHUFFLE}" == "true" ]]; then
  ARGS+=(--shuffle)
fi

python scripts/simple_eval_libero10_pi05.py "${ARGS[@]}"
