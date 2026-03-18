#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal single-GPU LIBERO-10 evaluation for OpenPI pi0.5.

This script intentionally reuses the model settings from
`examples/embodiment/config/libero_10_ppo_openpi_pi05.yaml` and runs a fixed
number of evaluation episodes on LIBERO-10. It reports success_once as the
episode success rate and saves rollout videos for a configurable ratio of trials.

TODO(agent): This helper is specialized for the current pi0.5 + LIBERO-10 config
instead of being a generic Hydra config launcher.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import random
from datetime import datetime
from typing import Iterable


def compute_num_save_videos(total_trials: int, save_video_ratio: float) -> int:
    """Convert a video ratio into an episode count."""
    if total_trials <= 0:
        raise ValueError("total_trials must be positive")
    if not 0.0 <= save_video_ratio <= 1.0:
        raise ValueError("save_video_ratio must be between 0.0 and 1.0")
    return min(total_trials, int(round(total_trials * save_video_ratio)))


def select_video_indices(total_trials: int, num_save_videos: int) -> list[int]:
    """Pick evenly spaced episode indices whose videos should be kept."""
    if total_trials <= 0:
        raise ValueError("total_trials must be positive")
    if num_save_videos < 0:
        raise ValueError("num_save_videos must be non-negative")
    if num_save_videos == 0:
        return []
    if num_save_videos == 1:
        return [0]
    if num_save_videos >= total_trials:
        return list(range(total_trials))

    indices = []
    last_idx = -1
    for offset in range(num_save_videos):
        idx = round(offset * (total_trials - 1) / (num_save_videos - 1))
        if idx != last_idx:
            indices.append(idx)
            last_idx = idx
    return indices


def build_trial_specs(
    trial_counts_per_task: Iterable[int],
    total_trials: int,
    shuffle_seed: int = 0,
) -> list[tuple[int, int]]:
    """Build a shuffled `(task_id, init_state_id)` list and keep the first N."""
    flat_specs: list[tuple[int, int]] = []
    for task_id, task_trial_count in enumerate(trial_counts_per_task):
        flat_specs.extend((task_id, trial_id) for trial_id in range(task_trial_count))

    if total_trials > len(flat_specs):
        raise ValueError(
            f"Requested {total_trials} trials, but only {len(flat_specs)} fixed "
            "initial states are available in LIBERO-10."
        )

    ordered_indices = list(range(len(flat_specs)))
    rng = random.Random(shuffle_seed)
    rng.shuffle(ordered_indices)
    return [flat_specs[idx] for idx in ordered_indices[:total_trials]]


def load_pi05_libero10_settings(config_path: pathlib.Path) -> dict[str, object]:
    """Load the minimal settings needed for pi0.5 LIBERO-10 evaluation."""
    from omegaconf import OmegaConf

    config_root = config_path.parent
    main_cfg = OmegaConf.load(config_path)
    base_model_cfg = OmegaConf.load(config_root / "model" / "pi0_5.yaml")
    base_env_cfg = OmegaConf.load(config_root / "env" / "libero_10.yaml")

    actor_model_cfg = OmegaConf.merge(base_model_cfg, main_cfg.actor.model)
    eval_env_cfg = OmegaConf.merge(base_env_cfg, main_cfg.env.eval)

    return {
        "model_path": str(actor_model_cfg.model_path),
        "config_name": str(actor_model_cfg.openpi.config_name),
        "action_chunk": int(actor_model_cfg.num_action_chunks),
        "num_steps": int(actor_model_cfg.num_steps),
        "task_suite_name": str(eval_env_cfg.task_suite_name),
        "max_episode_steps": int(eval_env_cfg.max_episode_steps),
        "seed": int(eval_env_cfg.get("seed", 0)),
        "experiment_name": str(main_cfg.runner.logger.experiment_name),
    }


def parse_args() -> argparse.Namespace:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    default_config = (
        repo_root
        / "examples"
        / "embodiment"
        / "config"
        / "libero_10_ppo_openpi_pi05.yaml"
    )
    default_log_dir = repo_root / "logs" / "libero10_pi05_minimal"

    parser = argparse.ArgumentParser(
        description="Minimal single-GPU LIBERO-10 evaluation for OpenPI pi0.5."
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=default_config,
        help="Path to examples/embodiment/config/libero_10_ppo_openpi_pi05.yaml",
    )
    parser.add_argument(
        "--total-trials",
        type=int,
        default=200,
        help="Total number of LIBERO-10 episodes to evaluate.",
    )
    parser.add_argument(
        "--save-video-ratio",
        type=float,
        default=0.2,
        help="Fraction of evaluated episodes whose videos will be saved.",
    )
    parser.add_argument(
        "--log-dir",
        type=pathlib.Path,
        default=default_log_dir,
        help="Directory that stores logs, videos, and summary.json.",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Optional experiment name. Defaults to a timestamped name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Environment seed override. Defaults to env.eval.seed from the config.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help="Seed used to choose which fixed LIBERO-10 initial states to evaluate.",
    )
    parser.add_argument(
        "--num-steps-wait",
        type=int,
        default=10,
        help="Number of warmup simulator steps before policy inference.",
    )
    parser.add_argument(
        "--video-temp-subsample",
        type=int,
        default=10,
        help="Save every Nth frame in rollout videos.",
    )
    return parser.parse_args()


def run_eval(args: argparse.Namespace) -> dict[str, object]:
    import imageio
    import numpy as np
    import tqdm
    from libero.libero import benchmark
    from toolkits.eval_scripts_openpi import setup_logger, setup_policy
    from toolkits.eval_scripts_openpi.libero_eval import (
        LIBERO_DUMMY_ACTION,
        LIBERO_ENV_RESOLUTION,
        _get_libero_env,
        _quat2axisangle,
    )

    settings = load_pi05_libero10_settings(args.config.resolve())
    exp_name = args.exp_name or (
        f"{settings['experiment_name']}_minimal_"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    exp_dir = args.log_dir.resolve() / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(exp_name=exp_name, log_dir=str(exp_dir))
    logger.info("Using config: %s", args.config.resolve())
    logger.info("Output directory: %s", exp_dir)

    policy_args = argparse.Namespace(
        config_name=settings["config_name"],
        pretrained_path=settings["model_path"],
        num_steps=settings["num_steps"],
    )
    policy = setup_policy(policy_args)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[settings["task_suite_name"]]()
    trial_counts_per_task = [
        len(task_suite.get_task_init_states(task_id))
        for task_id in range(task_suite.n_tasks)
    ]
    trial_specs = build_trial_specs(
        trial_counts_per_task=trial_counts_per_task,
        total_trials=args.total_trials,
        shuffle_seed=args.shuffle_seed,
    )

    num_save_videos = compute_num_save_videos(
        total_trials=args.total_trials,
        save_video_ratio=args.save_video_ratio,
    )
    save_video_indices = set(
        select_video_indices(args.total_trials, num_save_videos)
    )

    logger.info(
        "Evaluating %d episodes on %s, saving %d videos.",
        args.total_trials,
        settings["task_suite_name"],
        num_save_videos,
    )

    per_task_counts = collections.Counter()
    per_task_successes = collections.Counter()
    total_successes = 0

    env_seed = settings["seed"] if args.seed is None else args.seed
    max_steps = int(settings["max_episode_steps"])
    video_dir = exp_dir / "videos"

    for global_trial_id, (task_id, init_state_id) in enumerate(
        tqdm.tqdm(trial_specs, desc="Evaluating")
    ):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(
            task, LIBERO_ENV_RESOLUTION, env_seed
        )

        policy.reset()
        env.reset()
        obs = env.set_init_state(initial_states[init_state_id])

        action_plan = collections.deque()
        replay_images = []
        done = False

        for step_id in range(max_steps + args.num_steps_wait):
            if step_id < args.num_steps_wait:
                obs, _reward, done, _info = env.step(LIBERO_DUMMY_ACTION)
                if done:
                    break
                continue

            image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_image = np.ascontiguousarray(
                obs["robot0_eye_in_hand_image"][::-1, ::-1]
            )
            replay_images.append(image)

            state = np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            )

            if not action_plan:
                observation = {
                    "observation/image": image,
                    "observation/wrist_image": wrist_image,
                    "observation/state": state,
                    "prompt": str(task_description),
                }
                action_chunk = policy.infer(observation)["actions"]
                if len(action_chunk) < int(settings["action_chunk"]):
                    raise RuntimeError(
                        "Policy returned fewer actions than action_chunk requires: "
                        f"{len(action_chunk)} < {settings['action_chunk']}"
                    )
                action_plan.extend(action_chunk[: int(settings["action_chunk"])])

            action = action_plan.popleft()
            obs, _reward, done, _info = env.step(action.tolist())
            if done:
                break

        env.close()

        per_task_counts[task_description] += 1
        if done:
            total_successes += 1
            per_task_successes[task_description] += 1

        if global_trial_id in save_video_indices and replay_images:
            suffix = "success" if done else "failure"
            sanitized_task = task_description.replace(" ", "_")
            video_path = (
                video_dir
                / f"trial_{global_trial_id:04d}_{sanitized_task}_{suffix}.mp4"
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(
                video_path,
                [
                    np.asarray(frame)
                    for frame in replay_images[:: args.video_temp_subsample]
                ],
                fps=max(1, 30 // args.video_temp_subsample),
            )

        logger.info(
            "trial=%d/%d task=%s init_state=%d success=%s running_success_once=%.4f",
            global_trial_id + 1,
            args.total_trials,
            task_description,
            init_state_id,
            done,
            total_successes / (global_trial_id + 1),
        )

    per_task_success_rate = {
        task_name: (
            per_task_successes[task_name] / count if count > 0 else 0.0
        )
        for task_name, count in sorted(per_task_counts.items())
    }

    summary = {
        "config_path": str(args.config.resolve()),
        "model_path": str(settings["model_path"]),
        "config_name": str(settings["config_name"]),
        "task_suite_name": str(settings["task_suite_name"]),
        "total_trials": args.total_trials,
        "total_successes": total_successes,
        "success_once": total_successes / args.total_trials,
        "save_video_ratio": args.save_video_ratio,
        "saved_video_count": num_save_videos,
        "seed": env_seed,
        "shuffle_seed": args.shuffle_seed,
        "action_chunk": int(settings["action_chunk"]),
        "num_steps": int(settings["num_steps"]),
        "max_episode_steps": max_steps,
        "per_task_success_once": per_task_success_rate,
        "output_dir": str(exp_dir),
    }

    summary_path = exp_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    logger.info("success_once=%.4f", summary["success_once"])
    logger.info("summary saved to %s", summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    args = parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
