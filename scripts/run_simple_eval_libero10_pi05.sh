#! /bin/bash

export SCRIPT_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname "$SCRIPT_PATH")
export SRC_FILE="${SCRIPT_PATH}/simple_eval_libero10_pi05.py"

export MUJOCO_GL=${MUJOCO_GL:-"osmesa"}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-"osmesa"}

export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

MODEL_PATH=${1:-${PI05_MODEL_PATH:-"/path/to/local/pi05_hf_model"}}
TASK_ID=${2:-0}
NUM_EPISODES=${3:-200}
VIDEO_RATIO=${4:-0.2}
OUTPUT_DIR=${5:-"${REPO_PATH}/logs/simple_eval_libero10_pi05/task_${TASK_ID}-$(date +'%Y%m%d-%H%M%S')"}
EXTRA_ARGS=("${@:6}")

echo "Using MODEL_PATH=${MODEL_PATH}"
echo "Using TASK_ID=${TASK_ID}"
echo "Using NUM_EPISODES=${NUM_EPISODES}"
echo "Using VIDEO_RATIO=${VIDEO_RATIO}"
echo "Using OUTPUT_DIR=${OUTPUT_DIR}"
echo "Using Python at $(which python)"

CMD=(
    python "${SRC_FILE}"
    --model-path "${MODEL_PATH}"
    --task-id "${TASK_ID}"
    --num-episodes "${NUM_EPISODES}"
    --save-video-ratio "${VIDEO_RATIO}"
    --output-dir "${OUTPUT_DIR}"
    "${EXTRA_ARGS[@]}"
)
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
