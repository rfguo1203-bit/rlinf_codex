#!/usr/bin/env python3

"""Single-process single-GPU LIBERO pi0.5 evaluation.

This script extracts the core evaluation path from
`examples/embodiment/eval_embodiment.sh libero_10_ppo_openpi_pi05`
into a single Python entrypoint without Ray/Hydra worker orchestration.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_CFG_PATH = (
    REPO_ROOT / "examples" / "embodiment" / "config" / "env" / "libero_10.yaml"
)
MODEL_CFG_PATH = (
    REPO_ROOT / "examples" / "embodiment" / "config" / "model" / "pi0_5.yaml"
)
MAIN_CFG_PATH = (
    REPO_ROOT
    / "examples"
    / "embodiment"
    / "config"
    / "libero_10_ppo_openpi_pi05.yaml"
)


def compute_num_save_videos(num_episodes: int, save_video_ratio: float) -> int:
    """Return how many episode videos should be saved."""
    if num_episodes <= 0:
        return 0
    clipped_ratio = min(max(save_video_ratio, 0.0), 1.0)
    return min(num_episodes, int(round(num_episodes * clipped_ratio)))


def select_video_indices(num_episodes: int, num_save_videos: int) -> list[int]:
    """Select evenly spaced episode indices for video saving."""
    if num_episodes <= 0 or num_save_videos <= 0:
        return []
    if num_save_videos >= num_episodes:
        return list(range(num_episodes))
    if num_save_videos == 1:
        return [0]
    step = (num_episodes - 1) / (num_save_videos - 1)
    indices = [int(step * i) for i in range(num_save_videos)]
    indices[-1] = num_episodes - 1

    deduped: list[int] = []
    used = set()
    for index in indices:
        while index in used and index + 1 < num_episodes:
            index += 1
        while index in used and index - 1 >= 0:
            index -= 1
        if index not in used:
            used.add(index)
            deduped.append(index)
    return sorted(deduped)


def build_task_reset_state_ids(
    cumsum_trial_id_bins: list[int], task_id: int
) -> list[int]:
    """Map a task id to its fixed LIBERO reset-state ids."""
    if task_id < 0 or task_id >= len(cumsum_trial_id_bins):
        raise ValueError(
            f"task_id must be in [0, {len(cumsum_trial_id_bins) - 1}], got {task_id}"
        )
    start = 0 if task_id == 0 else int(cumsum_trial_id_bins[task_id - 1])
    end = int(cumsum_trial_id_bins[task_id])
    return list(range(start, end))


def choose_reset_state_ids(
    task_reset_state_ids: list[int],
    num_episodes: int | None,
    shuffle: bool,
    seed: int,
) -> list[int]:
    """Choose which fixed reset states to evaluate for a task."""
    if not task_reset_state_ids:
        return []

    reset_state_ids = list(task_reset_state_ids)
    if shuffle:
        random.Random(seed).shuffle(reset_state_ids)

    if num_episodes is None:
        return reset_state_ids
    if num_episodes <= 0:
        raise ValueError("--num-episodes must be > 0")
    if num_episodes > len(reset_state_ids):
        raise ValueError(
            f"Requested {num_episodes} episodes, but task only has "
            f"{len(reset_state_ids)} fixed reset states. "
            "Use a smaller value or omit --num-episodes to evaluate the full task."
        )
    return reset_state_ids[:num_episodes]


def to_python(value: Any) -> Any:
    """Convert common array/tensor values into JSON-serializable Python objects."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None

    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None

    if isinstance(value, Path):
        return str(value)
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: to_python(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_python(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal single-process single-GPU LIBERO-10 pi0.5 evaluation "
            "with task-id filtering and proportional video saving."
        )
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local pi0.5 model directory used as the base rollout model.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        required=True,
        help="LIBERO-10 task id to evaluate.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Number of fixed reset states to evaluate for the task. Default: all.",
    )
    parser.add_argument(
        "--save-video-ratio",
        type=float,
        default=0.2,
        help="Fraction of evaluated episodes whose rollout videos will be saved.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: logs/simple_eval_libero10_pi05/<timestamp>/",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for task-state ordering when --shuffle-reset-states is set.",
    )
    parser.add_argument(
        "--shuffle-reset-states",
        action="store_true",
        help="Shuffle the fixed reset states for the selected task before truncation.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=480,
        help="Maximum number of interaction steps per rollout.",
    )
    parser.add_argument(
        "--num-action-chunks",
        type=int,
        default=None,
        help="Override the action chunk count. Default: value from pi0.5 config.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=256,
        help="Rendered camera height.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=256,
        help="Rendered camera width.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Saved MP4 fps.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device used for inference.",
    )
    parser.add_argument(
        "--ckpt-path",
        default=None,
        help="Optional .pt checkpoint loaded on top of --model-path.",
    )
    return parser.parse_args()


def load_main_cfg():
    from omegaconf import OmegaConf

    return OmegaConf.load(MAIN_CFG_PATH)


def build_env_cfg(args: argparse.Namespace):
    from omegaconf import OmegaConf, open_dict

    env_cfg = OmegaConf.load(ENV_CFG_PATH)
    main_cfg = load_main_cfg()
    default_max_episode_steps = int(main_cfg.env.eval.max_episode_steps)
    with open_dict(env_cfg):
        env_cfg.total_num_envs = 1
        env_cfg.auto_reset = False
        env_cfg.ignore_terminations = bool(main_cfg.env.eval.ignore_terminations)
        env_cfg.max_episode_steps = args.max_episode_steps or default_max_episode_steps
        env_cfg.max_steps_per_rollout_epoch = env_cfg.max_episode_steps
        env_cfg.use_fixed_reset_state_ids = bool(main_cfg.env.eval.use_fixed_reset_state_ids)
        env_cfg.use_ordered_reset_state_ids = False
        env_cfg.is_eval = bool(main_cfg.env.eval.is_eval)
        env_cfg.group_size = int(main_cfg.env.eval.group_size)
        env_cfg.seed = args.seed
        env_cfg.video_cfg.save_video = True
        env_cfg.video_cfg.info_on_video = True
        env_cfg.video_cfg.fps = args.fps
        env_cfg.video_cfg.video_base_dir = str(args.output_dir / "videos")
        env_cfg.init_params.camera_heights = args.camera_height
        env_cfg.init_params.camera_widths = args.camera_width
    return env_cfg


def build_model_cfg(args: argparse.Namespace):
    from omegaconf import OmegaConf, open_dict

    model_cfg = OmegaConf.load(MODEL_CFG_PATH)
    main_cfg = load_main_cfg()
    num_action_chunks = (
        args.num_action_chunks
        if args.num_action_chunks is not None
        else int(model_cfg.num_action_chunks)
    )
    with open_dict(model_cfg):
        model_cfg.model_type = "openpi"
        model_cfg.model_path = args.model_path
        model_cfg.is_lora = False
        model_cfg.add_value_head = bool(main_cfg.actor.model.add_value_head)
        model_cfg.num_action_chunks = num_action_chunks
        model_cfg.openpi.config_name = "pi05_libero"
        model_cfg.openpi.action_chunk = num_action_chunks
        model_cfg.openpi.num_steps = int(model_cfg.num_steps)
        model_cfg.openpi.action_env_dim = int(model_cfg.action_dim)
        model_cfg.openpi.add_value_head = bool(model_cfg.add_value_head)
        model_cfg.openpi.value_after_vlm = bool(main_cfg.actor.model.openpi.value_after_vlm)
    return model_cfg


def load_model(model_cfg, ckpt_path: str | None, device):
    from rlinf.models import get_model
    import torch

    model = get_model(model_cfg)
    if ckpt_path:
        model_state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
    return model


def build_env(env_cfg):
    from rlinf.envs.libero.libero_env import LiberoEnv

    return LiberoEnv(
        cfg=env_cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info={},
    )


def maybe_wrap_video(env, env_cfg, fps: int, enable_video: bool):
    if not enable_video:
        return env
    from rlinf.envs.wrappers.record_video import RecordVideo

    return RecordVideo(env, env_cfg.video_cfg, fps=fps)


def run_episode(
    env_runner,
    model,
    model_cfg,
    env_cfg,
    reset_state_id: int,
    device,
) -> dict[str, Any]:
    from rlinf.envs.action_utils import prepare_actions
    import numpy as np
    import torch

    obs, _ = env_runner.reset(reset_state_ids=np.array([reset_state_id], dtype=int))
    chunk_steps = math.ceil(env_cfg.max_episode_steps / model_cfg.num_action_chunks)

    for _ in range(chunk_steps):
        obs = {
            key: value.to(device).contiguous() if torch.is_tensor(value) else value
            for key, value in obs.items()
        }
        with torch.no_grad():
            raw_chunk_actions, _ = model.predict_action_batch(env_obs=obs, mode="eval")

        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_chunk_actions,
            env_type=env_cfg.env_type,
            model_type=model_cfg.model_type,
            num_action_chunks=model_cfg.num_action_chunks,
            action_dim=model_cfg.action_dim,
        )
        obs_list, _rewards, chunk_terminations, chunk_truncations, _infos_list = (
            env_runner.chunk_step(chunk_actions)
        )
        obs = obs_list[-1]
        if bool(torch.logical_or(chunk_terminations, chunk_truncations).any().item()):
            break

    unwrapped_env = (
        env_runner.unwrapped if hasattr(env_runner, "unwrapped") else env_runner
    )
    success_once = bool(unwrapped_env.success_once[0])
    success_at_end = bool(unwrapped_env.prev_step_reward[0] > 0)
    episode_len = int(unwrapped_env.elapsed_steps[0])
    episode_return = float(unwrapped_env.returns[0])
    return {
        "success_once": success_once,
        "success_at_end": success_at_end,
        "episode_len": episode_len,
        "return": episode_return,
        "reset_state_id": reset_state_id,
        "task_description": unwrapped_env.task_descriptions[0],
    }


def clear_video_buffer(env_runner) -> None:
    if hasattr(env_runner, "render_images"):
        env_runner.render_images = []


def main() -> None:
    args = parse_args()
    import torch
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (
            REPO_ROOT
            / "logs"
            / "simple_eval_libero10_pi05"
            / f"task_{args.task_id:02d}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)

    env_cfg = build_env_cfg(args)
    model_cfg = build_model_cfg(args)

    print(f"Loading model from {args.model_path}")
    model = load_model(model_cfg, args.ckpt_path, device)
    env = build_env(env_cfg)

    task_reset_state_ids = build_task_reset_state_ids(
        [int(x) for x in env.cumsum_trial_id_bins.tolist()],
        args.task_id,
    )
    eval_reset_state_ids = choose_reset_state_ids(
        task_reset_state_ids=task_reset_state_ids,
        num_episodes=args.num_episodes,
        shuffle=args.shuffle_reset_states,
        seed=args.seed,
    )

    num_save_videos = compute_num_save_videos(
        num_episodes=len(eval_reset_state_ids),
        save_video_ratio=args.save_video_ratio,
    )
    video_episode_indices = set(
        select_video_indices(len(eval_reset_state_ids), num_save_videos)
    )
    env_runner = maybe_wrap_video(
        env=env,
        env_cfg=env_cfg,
        fps=args.fps,
        enable_video=num_save_videos > 0,
    )

    episode_results: list[dict[str, Any]] = []
    saved_videos: list[dict[str, Any]] = []

    print(
        f"Evaluating task_id={args.task_id} on {len(eval_reset_state_ids)} fixed reset "
        f"states; saving {num_save_videos} rollout videos."
    )

    try:
        for episode_idx, reset_state_id in enumerate(eval_reset_state_ids):
            result = run_episode(
                env_runner=env_runner,
                model=model,
                model_cfg=model_cfg,
                env_cfg=env_cfg,
                reset_state_id=reset_state_id,
                device=device,
            )
            result["episode_idx"] = episode_idx
            episode_results.append(result)

            if episode_idx in video_episode_indices and hasattr(env_runner, "flush_video"):
                video_sub_dir = f"task_{args.task_id:02d}/episode_{episode_idx:04d}"
                env_runner.flush_video(video_sub_dir=video_sub_dir)
                video_path = (
                    output_dir
                    / "videos"
                    / f"seed_{env.seed}"
                    / video_sub_dir
                    / "0.mp4"
                )
                saved_videos.append(
                    {
                        "episode_idx": episode_idx,
                        "reset_state_id": reset_state_id,
                        "video_path": str(video_path),
                    }
                )
            else:
                clear_video_buffer(env_runner)

            print(
                f"[{episode_idx + 1:03d}/{len(eval_reset_state_ids):03d}] "
                f"reset_state_id={reset_state_id} "
                f"success_once={int(result['success_once'])} "
                f"success_at_end={int(result['success_at_end'])} "
                f"return={result['return']:.3f} "
                f"len={result['episode_len']}"
            )
    finally:
        if hasattr(env_runner, "close"):
            env_runner.close()
        elif hasattr(env, "close"):
            env.close()

    num_success_once = sum(int(item["success_once"]) for item in episode_results)
    num_success_at_end = sum(int(item["success_at_end"]) for item in episode_results)
    success_once_rate = num_success_once / len(episode_results)
    success_at_end_rate = num_success_at_end / len(episode_results)
    avg_episode_len = sum(item["episode_len"] for item in episode_results) / len(
        episode_results
    )
    avg_return = sum(item["return"] for item in episode_results) / len(episode_results)

    summary = {
        "task_id": args.task_id,
        "task_description": episode_results[0]["task_description"]
        if episode_results
        else None,
        "model_path": args.model_path,
        "ckpt_path": args.ckpt_path,
        "num_episodes": len(eval_reset_state_ids),
        "num_success_once": num_success_once,
        "num_success_at_end": num_success_at_end,
        "success_once_rate": success_once_rate,
        "success_at_end_rate": success_at_end_rate,
        "avg_episode_len": avg_episode_len,
        "avg_return": avg_return,
        "num_videos_saved": len(saved_videos),
        "video_save_ratio": args.save_video_ratio,
        "seed": args.seed,
        "shuffle_reset_states": args.shuffle_reset_states,
        "max_episode_steps": args.max_episode_steps,
        "num_action_chunks": int(model_cfg.num_action_chunks),
    }

    payload = {
        "summary": summary,
        "episodes": episode_results,
        "saved_videos": saved_videos,
        "evaluated_reset_state_ids": eval_reset_state_ids,
        "task_reset_state_ids": task_reset_state_ids,
    }

    metrics_path = output_dir / "metrics.json"
    episodes_path = output_dir / "episodes.json"
    videos_path = output_dir / "saved_videos.json"
    summary_path = output_dir / "summary.txt"

    metrics_path.write_text(
        json.dumps(to_python(payload), indent=2, ensure_ascii=False) + "\n"
    )
    episodes_path.write_text(
        json.dumps(to_python(episode_results), indent=2, ensure_ascii=False) + "\n"
    )
    videos_path.write_text(
        json.dumps(to_python(saved_videos), indent=2, ensure_ascii=False) + "\n"
    )
    summary_path.write_text(
        "\n".join(
            [
                f"task_id: {summary['task_id']}",
                f"task_description: {summary['task_description']}",
                f"model_path: {summary['model_path']}",
                f"ckpt_path: {summary['ckpt_path']}",
                f"num_episodes: {summary['num_episodes']}",
                f"num_success_once: {summary['num_success_once']}",
                f"num_success_at_end: {summary['num_success_at_end']}",
                f"success_once_rate: {summary['success_once_rate']:.4f}",
                f"success_at_end_rate: {summary['success_at_end_rate']:.4f}",
                f"avg_episode_len: {summary['avg_episode_len']:.2f}",
                f"avg_return: {summary['avg_return']:.4f}",
                f"num_videos_saved: {summary['num_videos_saved']}",
                f"video_save_ratio: {summary['video_save_ratio']:.4f}",
                f"seed: {summary['seed']}",
                f"shuffle_reset_states: {summary['shuffle_reset_states']}",
                f"max_episode_steps: {summary['max_episode_steps']}",
                f"num_action_chunks: {summary['num_action_chunks']}",
            ]
        )
        + "\n"
    )

    print("")
    print(
        f"Success-once rate: {success_once_rate:.4f} "
        f"({num_success_once}/{len(episode_results)})"
    )
    print(
        f"Success-at-end rate: {success_at_end_rate:.4f} "
        f"({num_success_at_end}/{len(episode_results)})"
    )
    print(f"Average episode length: {avg_episode_len:.2f}")
    print(f"Average return: {avg_return:.4f}")
    print(f"Metrics written to {metrics_path}")
    print(f"Episodes written to {episodes_path}")
    print(f"Saved video manifest written to {videos_path}")


if __name__ == "__main__":
    main()
