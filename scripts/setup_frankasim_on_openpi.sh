#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERL_DIR="${ROOT_DIR}/env_additional/serl"
SERL_REPO="RLinf/serl"
SERL_REPO_URL="https://github.com/${SERL_REPO}.git"
SERL_BRANCH="RLinf/franka-sim"

if ! command -v git >/dev/null 2>&1; then
  echo "[ERROR] git is required but not found." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv is required but not found in current environment." >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "[ERROR] python is required but not found in current environment." >&2
  exit 1
fi

mkdir -p "${ROOT_DIR}/env_additional"

echo "[INFO] Syncing ${SERL_REPO} (${SERL_BRANCH}) into ${SERL_DIR}"
if [ ! -d "${SERL_DIR}/.git" ]; then
  git clone --branch "${SERL_BRANCH}" "${SERL_REPO_URL}" "${SERL_DIR}"
else
  git -C "${SERL_DIR}" fetch origin
  git -C "${SERL_DIR}" checkout "${SERL_BRANCH}"
  git -C "${SERL_DIR}" pull --ff-only origin "${SERL_BRANCH}"
fi

echo "[INFO] Installing minimal FrankaSim dependencies into current Python env"
uv pip install -e "${SERL_DIR}/franka_sim"
uv pip install -r "${SERL_DIR}/franka_sim/requirements.txt"

echo "[INFO] Running import sanity check"
python - <<'PY'
import franka_sim
import rlinf.envs.frankasim
print("frankasim ok")
PY

echo "[DONE] FrankaSim minimal dependencies are ready in current environment."
echo "[TIP] If running headless: export MUJOCO_GL=egl && export PYOPENGL_PLATFORM=egl"
