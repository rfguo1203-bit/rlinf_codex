#!/usr/bin/env python3

# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate a single LIBERO-10 task and save one MP4 per trajectory."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch
from hydra import compose
from hydra.initialize import initialize_config_dir
from omegaconf import open_dict

from rlinf.config import validate_cfg
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.libero.libero_env import LiberoEnv
from rlinf.envs.utils import put_info_on_image
from rlinf.models import get_model


DEFAULT_CONFIG_NAME = "libero_10_ppo_openpi_pi05"
DEFAULT_MODEL_PATH = "/path/to/RLinf-Pi05-LIBERO-SFT"
DEFAULT_OUTPUT_DIR = "logs/libero_single_task_eval"
DEFAULT_FPS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one LIBERO-10 task with RLinf OpenPI and save one MP4 per "
            "trajectory."
        )
    )
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Hydra config name under examples/embodiment/config.",
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Optional config directory override. Defaults to examples/embodiment/config.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to the pretrained RLinf-Pi05-LIBERO-SFT model.",
    )
    parser.add_argument(
        "--ckpt-path",
        default=None,
        help="Optional .pt checkpoint to load on top of model-path.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used for videos and metrics output.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Task index inside LIBERO-10.",
    )
    parser.add_argument(
        "--task-name",
        default=None,
        help="Task identifier to match against task name/language/problem folder.",
    )
    parser.add_argument(
        "--max-trajs",
        type=int,
        default=None,
        help="Optional cap on number of trajectories to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Evaluation seed override.",
    )
    return parser.parse_args()


def resolve_config_dir(args: argparse.Namespace) -> Path:
    if args.config_dir is not None:
        return Path(args.config_dir).resolve()
    return (Path(__file__).resolve().parent / "config").resolve()


def load_cfg(args: argparse.Namespace) -> Any:
    config_dir = resolve_config_dir(args)
    overrides = [
        f"runner.logger.log_path={Path(args.output_dir).resolve()}",
        f"rollout.model.model_path={Path(args.model_path).resolve()}",
        f"actor.model.model_path={Path(args.model_path).resolve()}",
        f"env.eval.seed={args.seed}",
    ]
    if args.ckpt_path:
        overrides.append(f"runner.ckpt_path={Path(args.ckpt_path).resolve()}")

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = compose(config_name=args.config_name, overrides=overrides)

    with open_dict(cfg):
        cfg.runner.only_eval = True
        cfg.runner.val_check_interval = -1
        cfg.env.eval.total_num_envs = 1
        cfg.env.eval.group_size = 1
        cfg.env.eval.auto_reset = False
        cfg.env.eval.ignore_terminations = False
        cfg.env.eval.use_fixed_reset_state_ids = False
        cfg.env.eval.use_ordered_reset_state_ids = False
        cfg.env.eval.video_cfg.save_video = False

    return validate_cfg(cfg)


def build_model(cfg: Any):
    rollout_model_cfg = copy.deepcopy(cfg.actor.model)
    with open_dict(rollout_model_cfg):
        rollout_model_cfg.precision = cfg.rollout.model.precision
        rollout_model_cfg.model_path = cfg.rollout.model.model_path

    model = get_model(rollout_model_cfg)
    if cfg.runner.get("ckpt_path", None):
        model.load_state_dict(torch.load(cfg.runner.ckpt_path, map_location="cpu"))
    model.eval()

    if cfg.rollout.get("enable_torch_compile", False):
        mode = cfg.rollout.get("torch_compile_mode", "max-autotune-no-cudagraphs")
        model.enable_torch_compile(mode=mode)

    return model


def _task_candidates(task: Any) -> dict[str, str]:
    candidates = {}
    for key in ("name", "language", "problem_folder", "bddl_file"):
        value = getattr(task, key, None)
        if value:
            candidates[key] = str(value)
    return candidates


def _normalize_task_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def resolve_task_id(env: LiberoEnv, task_id: int | None, task_name: str | None) -> int:
    num_tasks = env.task_suite.get_num_tasks()
    if task_id is not None:
        if task_id < 0 or task_id >= num_tasks:
            raise ValueError(f"task_id {task_id} out of range [0, {num_tasks - 1}]")
        return task_id

    if task_name is None:
        raise ValueError("One of --task-id or --task-name must be provided.")

    target = _normalize_task_name(task_name)
    available = []
    for index in range(num_tasks):
        task = env.task_suite.get_task(index)
        candidates = _task_candidates(task)
        available.append({"task_id": index, **candidates})
        for value in candidates.values():
            if _normalize_task_name(value) == target:
                return index

    raise ValueError(
        "Task not found. Available tasks:\n"
        + json.dumps(available, ensure_ascii=False, indent=2)
    )


def get_task_reset_state_ids(env: LiberoEnv, task_id: int) -> list[int]:
    start = 0 if task_id == 0 else int(env.cumsum_trial_id_bins[task_id - 1])
    end = int(env.cumsum_trial_id_bins[task_id])
    return list(range(start, end))


def task_metadata(env: LiberoEnv, task_id: int) -> dict[str, Any]:
    task = env.task_suite.get_task(task_id)
    metadata = _task_candidates(task)
    metadata["task_id"] = task_id
    metadata["num_trials"] = len(env.task_suite.get_task_init_states(task_id))
    primary_name = metadata.get("name") or metadata.get("problem_folder")
    if primary_name is None:
        primary_name = metadata.get("language", f"task_{task_id}")
    metadata["slug"] = _normalize_task_name(primary_name)
    return metadata


def tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def extract_frame(obs: dict[str, Any], info: dict[str, Any], extras: list[str]) -> np.ndarray:
    image = tensor_to_numpy(obs["main_images"][0])
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return put_info_on_image(image, info, extras=extras)


def save_video(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def evaluate_trajectory(
    cfg: Any,
    env: LiberoEnv,
    model: Any,
    reset_state_id: int,
    task_info: dict[str, Any],
    videos_dir: Path,
    fps: int,
) -> dict[str, Any]:
    env.is_start = False
    obs, _ = env.reset(reset_state_ids=np.array([reset_state_id]))

    success_once = False
    success_at_end = False
    episode_return = 0.0
    episode_length = 0
    frames = [
        extract_frame(
            obs,
            info={"step": 0, "reward": 0.0, "success_once": 0},
            extras=[
                f"task_id={task_info['task_id']}",
                f"reset_state_id={reset_state_id}",
                f"task={task_info.get('name', task_info.get('language', 'unknown'))}",
            ],
        )
    ]

    while episode_length < cfg.env.eval.max_episode_steps:
        with torch.no_grad():
            raw_actions, _ = model.predict_action_batch(obs, mode="eval")
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=cfg.env.eval.env_type,
            model_type=cfg.actor.model.model_type,
            num_action_chunks=cfg.actor.model.num_action_chunks,
            action_dim=cfg.actor.model.action_dim,
            policy=cfg.actor.model.get("policy_setup", None),
        )

        for step_actions in chunk_actions[0]:
            next_obs, reward, termination, truncation, _ = env.step(
                step_actions[None, ...], auto_reset=False
            )
            reward_value = float(tensor_to_numpy(reward).reshape(-1)[0])
            terminated = bool(tensor_to_numpy(termination).reshape(-1)[0])
            truncated = bool(tensor_to_numpy(truncation).reshape(-1)[0])

            episode_length += 1
            episode_return += reward_value
            success_once = success_once or terminated
            success_at_end = terminated

            frames.append(
                extract_frame(
                    next_obs,
                    info={
                        "step": episode_length,
                        "reward": reward_value,
                        "success_once": int(success_once),
                        "terminated": int(terminated),
                    },
                    extras=[
                        f"task_id={task_info['task_id']}",
                        f"reset_state_id={reset_state_id}",
                    ],
                )
            )
            obs = next_obs

            if terminated or truncated:
                outcome = "success" if success_once else "fail"
                video_name = (
                    f"{task_info['slug']}"
                    f"__task_{task_info['task_id']:02d}"
                    f"__reset_{reset_state_id:04d}"
                    f"__{outcome}.mp4"
                )
                save_video(frames, videos_dir / video_name, fps)
                return {
                    "reset_state_id": reset_state_id,
                    "episode_len": episode_length,
                    "return": episode_return,
                    "success_once": success_once,
                    "success_at_end": success_at_end,
                    "video_path": str((videos_dir / video_name).resolve()),
                }

    outcome = "success" if success_once else "fail"
    video_name = (
        f"{task_info['slug']}"
        f"__task_{task_info['task_id']:02d}"
        f"__reset_{reset_state_id:04d}"
        f"__{outcome}.mp4"
    )
    save_video(frames, videos_dir / video_name, fps)
    return {
        "reset_state_id": reset_state_id,
        "episode_len": episode_length,
        "return": episode_return,
        "success_once": success_once,
        "success_at_end": success_at_end,
        "video_path": str((videos_dir / video_name).resolve()),
    }


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args)

    output_dir = Path(args.output_dir).resolve()
    videos_dir = output_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = LiberoEnv(
        cfg=cfg.env.eval,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    model = build_model(cfg)

    selected_task_id = resolve_task_id(env, args.task_id, args.task_name)
    selected_task = task_metadata(env, selected_task_id)
    reset_state_ids = get_task_reset_state_ids(env, selected_task_id)
    if args.max_trajs is not None:
        reset_state_ids = reset_state_ids[: args.max_trajs]

    fps = int(cfg.env.eval.video_cfg.get("fps", DEFAULT_FPS))
    trajectory_results = []
    try:
        for reset_state_id in reset_state_ids:
            trajectory_results.append(
                evaluate_trajectory(
                    cfg=cfg,
                    env=env,
                    model=model,
                    reset_state_id=reset_state_id,
                    task_info=selected_task,
                    videos_dir=videos_dir,
                    fps=fps,
                )
            )
    finally:
        if hasattr(env, "env") and hasattr(env.env, "close"):
            env.env.close()

    total = len(trajectory_results)
    if total == 0:
        raise ValueError("No trajectories selected. Check task selection or --max-trajs.")
    success_once_rate = sum(item["success_once"] for item in trajectory_results) / total
    success_at_end_rate = (
        sum(item["success_at_end"] for item in trajectory_results) / total
    )

    summary = {
        "config_name": args.config_name,
        "model_path": str(Path(args.model_path).resolve()),
        "ckpt_path": str(Path(args.ckpt_path).resolve()) if args.ckpt_path else None,
        "task": selected_task,
        "num_trajectories": total,
        "success_once_rate": success_once_rate,
        "success_at_end_rate": success_at_end_rate,
        "trajectories": trajectory_results,
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
