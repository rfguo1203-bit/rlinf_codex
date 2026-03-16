#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/eval_libero_single_task.py"

export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

export HYDRA_FULL_ERROR=1

# NOTE: Set the active robot platform for OpenPI normalization/action conventions.
export ROBOT_PLATFORM=${ROBOT_PLATFORM:-"LIBERO"}
echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"

CONFIG_NAME=${1:-"libero_10_ppo_openpi_pi05"}
MODEL_PATH=${2:-"/path/to/RLinf-Pi05-LIBERO-SFT"}
TASK_SELECTOR=${3:-"0"}
OUTPUT_DIR=${4:-"${REPO_PATH}/logs/libero_single_task_eval/$(date +'%Y%m%d-%H:%M:%S')"}
CKPT_PATH=${CKPT_PATH:-""}
MAX_TRAJS=${MAX_TRAJS:-""}

mkdir -p "${OUTPUT_DIR}"

CMD=(
  python "${SRC_FILE}"
  --config-name "${CONFIG_NAME}"
  --model-path "${MODEL_PATH}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ "${TASK_SELECTOR}" =~ ^[0-9]+$ ]]; then
  CMD+=(--task-id "${TASK_SELECTOR}")
else
  CMD+=(--task-name "${TASK_SELECTOR}")
fi

if [ -n "${CKPT_PATH}" ]; then
  CMD+=(--ckpt-path "${CKPT_PATH}")
fi

if [ -n "${MAX_TRAJS}" ]; then
  CMD+=(--max-trajs "${MAX_TRAJS}")
fi

printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
