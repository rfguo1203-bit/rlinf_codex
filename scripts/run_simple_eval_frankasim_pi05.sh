#! /bin/bash

export SCRIPT_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname "$SCRIPT_PATH")
export SRC_FILE="${SCRIPT_PATH}/simple_eval_frankasim_pi05.py"

export MUJOCO_GL=${MUJOCO_GL:-"egl"}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-"egl"}
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

CONFIG_NAME=${1:-frankasim_ppo_openpi_pi05}
MODEL_PATH=${2:-${PI05_MODEL_PATH:-"/path/to/local/pi05_hf_model"}}
NUM_EPISODES=${3:-200}
VIDEO_RATIO=${4:-0.2}
OUTPUT_DIR=${5:-"${REPO_PATH}/logs/simple_eval_frankasim_pi05/${CONFIG_NAME}-$(date +'%Y%m%d-%H%M%S')"}
EXTRA_ARGS=("${@:6}")

echo "Using CONFIG_NAME=${CONFIG_NAME}"
echo "Using MODEL_PATH=${MODEL_PATH}"
echo "Using NUM_EPISODES=${NUM_EPISODES}"
echo "Using VIDEO_RATIO=${VIDEO_RATIO}"
echo "Using OUTPUT_DIR=${OUTPUT_DIR}"
echo "Using Python at $(which python)"

CMD=(
    python "${SRC_FILE}"
    --config-name "${CONFIG_NAME}"
    --model-path "${MODEL_PATH}"
    --num-episodes "${NUM_EPISODES}"
    --save-video-ratio "${VIDEO_RATIO}"
    --output-dir "${OUTPUT_DIR}"
    "${EXTRA_ARGS[@]}"
)
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
