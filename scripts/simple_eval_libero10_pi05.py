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

"""Run Pi0.5 inference on a single LIBERO-10 task and save local videos."""

from __future__ import annotations

import argparse
import copy
import os
import random
import re
import secrets
from itertools import accumulate
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EMBODIED_PATH = REPO_ROOT / "examples" / "embodiment"
DEFAULT_CONFIG_NAME = "libero_10_ppo_openpi_pi05"


def compute_num_save_videos(total_episodes: int, save_fraction: float) -> int:
    """Compute how many episodes should be exported as videos."""
    if total_episodes < 0:
        raise ValueError("total_episodes must be non-negative")
    clamped_fraction = min(max(save_fraction, 0.0), 1.0)
    return int(total_episodes * clamped_fraction)


def select_video_indices(total_episodes: int, num_save_videos: int) -> list[int]:
    """Choose evenly spaced episode indices for video export."""
    if total_episodes <= 0 or num_save_videos <= 0:
        return []
    if num_save_videos >= total_episodes:
        return list(range(total_episodes))
    if num_save_videos == 1:
        return [0]

    indices = [
        round(i * (total_episodes - 1) / (num_save_videos - 1))
        for i in range(num_save_videos)
    ]
    deduped: list[int] = []
    for idx in indices:
        if not deduped or deduped[-1] != idx:
            deduped.append(idx)
    return deduped


def build_task_reset_state_ids(
    cumsum_trial_id_bins: list[int], task_id: int
) -> list[int]:
    """Map a LIBERO task id to its contiguous global reset-state id range."""
    if task_id < 0 or task_id >= len(cumsum_trial_id_bins):
        raise ValueError(
            f"task_id must be in [0, {len(cumsum_trial_id_bins) - 1}], got {task_id}"
        )
    start = 0 if task_id == 0 else cumsum_trial_id_bins[task_id - 1]
    end = cumsum_trial_id_bins[task_id]
    return list(range(start, end))


def choose_reset_state_ids(
    task_reset_state_ids: list[int],
    num_episodes: int | None,
    shuffle: bool,
    seed: int,
) -> list[int]:
    """Pick which reset states to evaluate for the selected task."""
    selected_ids = list(task_reset_state_ids)
    if shuffle:
        random.Random(seed).shuffle(selected_ids)
    if num_episodes is not None:
        if num_episodes < 0:
            raise ValueError("num_episodes must be non-negative")
        selected_ids = selected_ids[:num_episodes]
    return selected_ids


def _set_runtime_env() -> None:
    os.environ.setdefault("EMBODIED_PATH", str(EMBODIED_PATH))
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")


def load_eval_cfg(config_name: str, overrides: list[str] | None = None):
    """Compose the embodied eval config through Hydra."""
    _set_runtime_env()

    import hydra
    from omegaconf import OmegaConf, open_dict

    config_dir = str(EMBODIED_PATH / "config")
    with hydra.initialize_config_dir(
        config_dir=config_dir,
        version_base="1.1",
    ):
        cfg = hydra.compose(config_name=config_name, overrides=overrides or [])

    with open_dict(cfg):
        cfg.runner.only_eval = True
    OmegaConf.resolve(cfg)
    return cfg


def load_libero10_metadata(task_suite_name: str = "libero_10") -> dict[str, Any]:
    """Load task descriptions and reset-state ranges from the LIBERO benchmark."""
    _set_runtime_env()

    from rlinf.envs.libero.utils import get_benchmark_overridden

    task_suite = get_benchmark_overridden(task_suite_name)()
    descriptions = [
        str(task_suite.get_task(task_id).language)
        for task_id in range(task_suite.get_num_tasks())
    ]
    trial_counts = [
        len(task_suite.get_task_init_states(task_id))
        for task_id in range(task_suite.get_num_tasks())
    ]
    cumsum_trial_id_bins = list(accumulate(trial_counts))
    return {
        "task_suite": task_suite,
        "task_descriptions": descriptions,
        "trial_counts": trial_counts,
        "cumsum_trial_id_bins": cumsum_trial_id_bins,
    }


def normalize_task_name(task_name: str) -> str:
    """Normalize common separators and whitespace for task matching."""
    normalized = re.sub(r"[_\-]+", " ", task_name.strip().lower())
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def resolve_task_id(
    task_descriptions: list[str],
    task_id: int | None = None,
    task_name: str | None = None,
) -> int:
    """Resolve a LIBERO-10 task from either id or language description."""
    if task_id is not None:
        if task_id < 0 or task_id >= len(task_descriptions):
            raise ValueError(
                f"task_id must be in [0, {len(task_descriptions) - 1}], got {task_id}"
            )
        return task_id

    if not task_name:
        raise ValueError("Either task_id or task_name must be provided.")

    normalized_query = normalize_task_name(task_name)
    normalized_descriptions = [
        normalize_task_name(description) for description in task_descriptions
    ]

    for idx, normalized_description in enumerate(normalized_descriptions):
        if normalized_description == normalized_query:
            return idx

    substring_matches = [
        idx
        for idx, normalized_description in enumerate(normalized_descriptions)
        if normalized_query in normalized_description
    ]
    if len(substring_matches) == 1:
        return substring_matches[0]
    if len(substring_matches) > 1:
        raise ValueError(
            "Matched multiple tasks by substring. Please use a more specific task_name."
        )

    raise ValueError(f"Could not find a LIBERO-10 task matching: {task_name}")


def _to_bool(value: Any) -> bool:
    try:
        return bool(value.item())
    except AttributeError:
        return bool(value)


def _slugify_task_name(task_name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in task_name).strip("_")


def _predict_video_path(video_base_dir: Path, seed: int, task_dir: str, video_idx: int) -> Path:
    return video_base_dir / f"seed_{seed}" / task_dir / f"{video_idx}.mp4"


def _resolve_seed(seed: int | None) -> int:
    """Use the provided seed or generate a random one when absent."""
    return seed if seed is not None else secrets.randbelow(2**31)


def _get_next_video_index(video_base_dir: Path, seed: int, task_dir: str) -> int:
    """Return the next non-overlapping MP4 index for the target task directory."""
    task_video_dir = video_base_dir / f"seed_{seed}" / task_dir
    if not task_video_dir.exists():
        return 0

    existing_indices: list[int] = []
    for mp4_path in task_video_dir.glob("*.mp4"):
        try:
            existing_indices.append(int(mp4_path.stem))
        except ValueError:
            continue
    return max(existing_indices, default=-1) + 1


def _standardize_env_obs(obs: dict[str, Any]) -> dict[str, Any]:
    """Match the observation schema produced by the standard eval pipeline.

    The official `eval_embodiment.sh` path goes through `EnvOutput.to_dict()`,
    which normalizes observations with `EnvOutput.prepare_observations()`.
    This helper mirrors that behavior so the standalone script can feed the
    model the same key set, including optional keys with `None` defaults.
    """
    return {
        "main_images": obs["main_images"] if "main_images" in obs else None,
        "wrist_images": obs["wrist_images"] if "wrist_images" in obs else None,
        "extra_view_images": (
            obs["extra_view_images"] if "extra_view_images" in obs else None
        ),
        "states": obs["states"] if "states" in obs else None,
        "task_descriptions": (
            list(obs["task_descriptions"])
            if "task_descriptions" in obs and obs["task_descriptions"] is not None
            else None
        ),
    }


def run_single_task_eval(
    task_id: int,
    config_name: str = DEFAULT_CONFIG_NAME,
    model_path: str | None = None,
    output_dir: str | None = None,
    num_episodes: int | None = 1,
    shuffle: bool = False,
    seed: int | None = None,
    save_fraction: float = 1.0,
) -> dict[str, Any]:
    """Run a single-task LIBERO-10 evaluation loop without Ray workers."""
    from omegaconf import open_dict

    from rlinf.envs import get_env_cls
    from rlinf.envs.action_utils import prepare_actions
    from rlinf.envs.wrappers import RecordVideo
    from rlinf.models import get_model

    cfg = load_eval_cfg(config_name=config_name)
    metadata = load_libero10_metadata(task_suite_name=cfg.env.eval.task_suite_name)
    task_descriptions = metadata["task_descriptions"]
    if task_id < 0 or task_id >= len(task_descriptions):
        raise ValueError(
            f"task_id must be in [0, {len(task_descriptions) - 1}], got {task_id}"
        )

    resolved_seed = _resolve_seed(seed)
    task_name = task_descriptions[task_id]
    task_slug = _slugify_task_name(task_name)
    task_reset_state_ids = build_task_reset_state_ids(
        metadata["cumsum_trial_id_bins"],
        task_id=task_id,
    )
    chosen_reset_state_ids = choose_reset_state_ids(
        task_reset_state_ids=task_reset_state_ids,
        num_episodes=num_episodes,
        shuffle=shuffle,
        seed=resolved_seed,
    )
    if not chosen_reset_state_ids:
        raise ValueError("No reset states selected for evaluation.")

    num_save_videos = compute_num_save_videos(
        total_episodes=len(chosen_reset_state_ids),
        save_fraction=save_fraction,
    )
    save_video_indices = set(
        select_video_indices(
            total_episodes=len(chosen_reset_state_ids),
            num_save_videos=num_save_videos,
        )
    )

    video_base_dir = Path(output_dir or (REPO_ROOT / "results" / "libero10_pi05_single_task"))
    with open_dict(cfg):
        if model_path is not None:
            cfg.actor.model.model_path = model_path
            cfg.rollout.model.model_path = model_path
        cfg.env.eval.total_num_envs = 1
        cfg.env.eval.auto_reset = False
        cfg.env.eval.ignore_terminations = False
        cfg.env.eval.use_fixed_reset_state_ids = False
        cfg.env.eval.seed = resolved_seed
        cfg.env.eval.video_cfg.save_video = bool(save_video_indices)
        cfg.env.eval.video_cfg.video_base_dir = str(video_base_dir)

    env_cfg = copy.deepcopy(cfg.env.eval)
    env_cls = get_env_cls(env_cfg.env_type, env_cfg)
    env = env_cls(
        cfg=env_cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    if env_cfg.video_cfg.save_video:
        env = RecordVideo(env, env_cfg.video_cfg)

    model_cfg = copy.deepcopy(cfg.actor.model)
    model = get_model(model_cfg)
    model.eval()

    episode_results: list[dict[str, Any]] = []
    saved_video_paths: list[str] = []
    next_video_index = _get_next_video_index(
        video_base_dir=video_base_dir,
        seed=resolved_seed,
        task_dir=task_slug,
    )

    try:
        for episode_idx, reset_state_id in enumerate(chosen_reset_state_ids):
            env.is_start = False
            obs, _ = env.reset(reset_state_ids=[reset_state_id])
            obs = _standardize_env_obs(obs)

            done = False
            success = False
            episode_steps = 0

            while not done and episode_steps < env_cfg.max_episode_steps:
                raw_chunk_actions, _ = model.predict_action_batch(
                    env_obs=obs,
                    mode="eval",
                    compute_values=False,
                )
                chunk_actions = prepare_actions(
                    raw_chunk_actions=raw_chunk_actions,
                    env_type=env_cfg.env_type,
                    model_type=model_cfg.model_type,
                    num_action_chunks=model_cfg.num_action_chunks,
                    action_dim=model_cfg.action_dim,
                    policy=model_cfg.get("policy_setup", None),
                    wm_env_type=env_cfg.get("wm_env_type", None),
                )

                for chunk_step in range(chunk_actions.shape[1]):
                    action = chunk_actions[:, chunk_step]
                    obs, _reward, terminations, truncations, _infos = env.step(action)
                    obs = _standardize_env_obs(obs)
                    episode_steps += 1

                    terminated = _to_bool(terminations[0])
                    truncated = _to_bool(truncations[0])
                    done = terminated or truncated
                    success = success or terminated
                    if done:
                        break

            video_path = None
            if episode_idx in save_video_indices and isinstance(env, RecordVideo):
                env.video_cnt = next_video_index
                video_path = _predict_video_path(
                    video_base_dir=video_base_dir,
                    seed=resolved_seed,
                    task_dir=task_slug,
                    video_idx=env.video_cnt,
                )
                env.flush_video(video_sub_dir=task_slug)
                next_video_index = env.video_cnt
                saved_video_paths.append(str(video_path))

            episode_results.append(
                {
                    "episode_idx": episode_idx,
                    "task_id": task_id,
                    "task_name": task_name,
                    "reset_state_id": reset_state_id,
                    "success": success,
                    "steps": episode_steps,
                    "video_path": str(video_path) if video_path is not None else None,
                }
            )
    finally:
        env.close()

    return {
        "task_id": task_id,
        "task_name": task_name,
        "seed": resolved_seed,
        "episodes": episode_results,
        "saved_video_paths": saved_video_paths,
        "video_base_dir": str(video_base_dir),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Pi0.5 on a single LIBERO-10 task and save videos."
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="LIBERO-10 task id.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="LIBERO-10 task language description or unique substring.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print all LIBERO-10 task ids and descriptions, then exit.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=DEFAULT_CONFIG_NAME,
        help="Hydra config name under examples/embodiment/config.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional override for the local pi0.5/OpenPI checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory used as video_base_dir.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="How many reset states to evaluate for the selected task.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle reset states before truncating to num_episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed used for env selection and shuffle. Defaults to a random seed.",
    )
    parser.add_argument(
        "--save-fraction",
        type=float,
        default=1.0,
        help="Fraction of evaluated episodes to export as videos.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    metadata = load_libero10_metadata()

    if args.list_tasks:
        for idx, description in enumerate(metadata["task_descriptions"]):
            print(f"{idx}: {description}")
        return

    selected_task_id = resolve_task_id(
        task_descriptions=metadata["task_descriptions"],
        task_id=args.task_id,
        task_name=args.task_name,
    )
    results = run_single_task_eval(
        task_id=selected_task_id,
        config_name=args.config_name,
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        shuffle=args.shuffle,
        seed=args.seed,
        save_fraction=args.save_fraction,
    )

    print(f"Task {results['task_id']}: {results['task_name']} (seed={results['seed']})")
    for episode in results["episodes"]:
        print(
            "Episode "
            f"{episode['episode_idx']}: reset_state_id={episode['reset_state_id']}, "
            f"success={episode['success']}, steps={episode['steps']}, "
            f"video={episode['video_path']}"
        )


if __name__ == "__main__":
    main()
