#! /bin/bash

export SCRIPT_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname "$SCRIPT_PATH")
export SRC_FILE="${SCRIPT_PATH}/infer_libero10_pi05_single_task.py"

export MUJOCO_GL=${MUJOCO_GL:-"osmesa"}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-"osmesa"}

export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

MODEL_PATH=${1:-${PI05_MODEL_PATH:-"/path/to/local/pi05_hf_model"}}
TASK_ID=${2:-0}
OUTPUT_DIR=${3:-"${REPO_PATH}/logs/infer_libero10_pi05_single_task/task_${TASK_ID}-$(date +'%Y%m%d-%H%M%S')"}
EXTRA_ARGS=("${@:4}")

CMD=(
    python "${SRC_FILE}"
    --model-path "${MODEL_PATH}"
    --task-id "${TASK_ID}"
    --num-episodes 1
    --save-video-ratio 1.0
    --output-dir "${OUTPUT_DIR}"
    "${EXTRA_ARGS[@]}"
)

printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
