#!/bin/bash
set -euo pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH="$(dirname "$(dirname "$EMBODIED_PATH")")"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"

TOTAL_TRIALS="${1:-200}"
SAVE_VIDEO_RATIO="${2:-0.2}"
GPU_ID="${3:-0}"

CONFIG_PATH="${CONFIG_PATH:-${EMBODIED_PATH}/config/libero_10_ppo_openpi_pi05.yaml}"
LOG_DIR="${LOG_DIR:-${REPO_PATH}/logs/libero10_pi05_minimal}"
EXP_NAME="${EXP_NAME:-libero10_pi05_minimal_$(date +'%Y%m%d-%H%M%S')}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"

CMD=(
  python "${EMBODIED_PATH}/eval_libero10_pi05_minimal.py"
  --config "${CONFIG_PATH}"
  --total-trials "${TOTAL_TRIALS}"
  --save-video-ratio "${SAVE_VIDEO_RATIO}"
  --log-dir "${LOG_DIR}"
  --exp-name "${EXP_NAME}"
)

printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
