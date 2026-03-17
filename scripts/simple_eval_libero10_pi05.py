#!/usr/bin/env python3

"""Single-process LIBERO-10 evaluation for a local pi0.5 OpenPI model."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.libero.libero_env import LiberoEnv
from rlinf.envs.wrappers.record_video import RecordVideo
from rlinf.models import get_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a local pi0.5 OpenPI model on a single LIBERO-10 task."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local Hugging Face/OpenPI checkpoint directory.",
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
        default=200,
        help="Number of evaluation episodes to run.",
    )
    parser.add_argument(
        "--save-video-ratio",
        type=float,
        default=0.2,
        help="Fraction of episodes to save as videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/simple_eval_libero10_pi05"),
        help="Directory for metrics and videos.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for episode sampling and video sampling.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=480,
        help="Maximum environment steps per episode.",
    )
    parser.add_argument(
        "--num-action-chunks",
        type=int,
        default=10,
        help="Number of actions predicted per forward pass.",
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
        help="Output video fps.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for model inference.",
    )
    return parser.parse_args()


def build_env_cfg(args: argparse.Namespace):
    repo_root = Path(__file__).resolve().parents[1]
    env_cfg = OmegaConf.load(
        repo_root / "examples" / "embodiment" / "config" / "env" / "libero_10.yaml"
    )
    with open_dict(env_cfg):
        env_cfg.total_num_envs = 1
        env_cfg.auto_reset = False
        env_cfg.ignore_terminations = False
        env_cfg.max_episode_steps = args.max_episode_steps
        env_cfg.max_steps_per_rollout_epoch = args.max_episode_steps
        env_cfg.use_fixed_reset_state_ids = True
        env_cfg.use_ordered_reset_state_ids = False
        env_cfg.is_eval = True
        env_cfg.group_size = 1
        env_cfg.seed = args.seed
        env_cfg.video_cfg.save_video = True
        env_cfg.video_cfg.info_on_video = True
        env_cfg.video_cfg.fps = args.fps
        env_cfg.video_cfg.video_base_dir = str(args.output_dir / "videos")
        env_cfg.init_params.camera_heights = args.camera_height
        env_cfg.init_params.camera_widths = args.camera_width
    return env_cfg


def build_model_cfg(args: argparse.Namespace):
    repo_root = Path(__file__).resolve().parents[1]
    model_cfg = OmegaConf.load(
        repo_root / "examples" / "embodiment" / "config" / "model" / "pi0_5.yaml"
    )
    with open_dict(model_cfg):
        model_cfg.model_type = "openpi"
        model_cfg.model_path = args.model_path
        model_cfg.is_lora = False
        model_cfg.add_value_head = False
        model_cfg.num_action_chunks = args.num_action_chunks
        model_cfg.openpi.config_name = "pi05_libero"
        model_cfg.openpi.action_chunk = args.num_action_chunks
    return model_cfg


def build_reset_state_ids_for_task(env: LiberoEnv, task_id: int) -> list[int]:
    if task_id < 0 or task_id >= env.task_suite.get_num_tasks():
        raise ValueError(
            f"task_id must be in [0, {env.task_suite.get_num_tasks() - 1}], got {task_id}"
        )

    start = 0 if task_id == 0 else int(env.cumsum_trial_id_bins[task_id - 1])
    end = int(env.cumsum_trial_id_bins[task_id])
    return list(range(start, end))


def choose_video_episodes(
    num_episodes: int, save_video_ratio: float, rng: random.Random
) -> set[int]:
    ratio = min(max(save_video_ratio, 0.0), 1.0)
    num_videos = min(num_episodes, int(round(num_episodes * ratio)))
    if num_videos == 0:
        return set()
    return set(rng.sample(range(num_episodes), num_videos))


def to_python(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: to_python(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_python(v) for v in value]
    return value


def run_episode(
    env_like: LiberoEnv | RecordVideo,
    model,
    model_cfg,
    env_cfg,
    reset_state_id: int,
    device: torch.device,
) -> dict[str, Any]:
    obs, _ = env_like.reset(reset_state_ids=np.array([reset_state_id], dtype=int))
    done = False
    chunk_steps = math.ceil(
        env_cfg.max_episode_steps / model_cfg.num_action_chunks
    )

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
            env_like.chunk_step(chunk_actions)
        )
        obs = obs_list[-1]
        done = bool(torch.logical_or(chunk_terminations, chunk_truncations).any().item())
        if done:
            break

    success_once = bool(env_like.unwrapped.success_once[0])
    episode_len = int(env_like.unwrapped.elapsed_steps[0])
    episode_return = float(env_like.unwrapped.returns[0])
    return {
        "success_once": success_once,
        "episode_len": episode_len,
        "return": episode_return,
        "reset_state_id": reset_state_id,
        "task_description": env_like.unwrapped.task_descriptions[0],
    }


def main() -> None:
    args = parse_args()
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be > 0")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = build_env_cfg(args)
    model_cfg = build_model_cfg(args)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)

    random_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    print(f"Loading model from {args.model_path}")
    model = get_model(model_cfg)
    model = model.to(device)
    model.eval()

    env = LiberoEnv(
        cfg=env_cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info={},
    )

    task_reset_state_ids = build_reset_state_ids_for_task(env, args.task_id)
    sampled_reset_state_ids = np_rng.choice(
        task_reset_state_ids, size=args.num_episodes, replace=True
    ).tolist()
    video_episode_ids = choose_video_episodes(
        num_episodes=args.num_episodes,
        save_video_ratio=args.save_video_ratio,
        rng=random_rng,
    )

    episode_results: list[dict[str, Any]] = []
    saved_videos: list[dict[str, Any]] = []

    print(
        f"Evaluating task_id={args.task_id} for {args.num_episodes} episodes; "
        f"saving {len(video_episode_ids)} videos."
    )

    try:
        for episode_idx, reset_state_id in enumerate(sampled_reset_state_ids):
            save_video = episode_idx in video_episode_ids
            env_runner: LiberoEnv | RecordVideo = env
            if save_video:
                env_runner = RecordVideo(env, env_cfg.video_cfg, fps=args.fps)

            result = run_episode(
                env_like=env_runner,
                model=model,
                model_cfg=model_cfg,
                env_cfg=env_cfg,
                reset_state_id=reset_state_id,
                device=device,
            )
            result["episode_idx"] = episode_idx
            episode_results.append(result)

            if save_video:
                video_sub_dir = f"task_{args.task_id:02d}/episode_{episode_idx:04d}"
                env_runner.flush_video(video_sub_dir=video_sub_dir)
                env_runner.close()
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

            success = int(result["success_once"])
            print(
                f"[{episode_idx + 1:03d}/{args.num_episodes:03d}] "
                f"reset_state_id={reset_state_id} success={success} "
                f"return={result['return']:.3f} len={result['episode_len']}"
            )
    finally:
        try:
            env.close()
        except AttributeError:
            pass

    num_success = sum(int(item["success_once"]) for item in episode_results)
    success_rate = num_success / len(episode_results)
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
        "num_episodes": args.num_episodes,
        "num_success": num_success,
        "success_rate": success_rate,
        "avg_episode_len": avg_episode_len,
        "avg_return": avg_return,
        "num_videos_saved": len(saved_videos),
        "video_save_ratio": args.save_video_ratio,
        "seed": args.seed,
    }

    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.txt"
    episodes_path = output_dir / "episodes.json"
    videos_path = output_dir / "saved_videos.json"

    metrics_payload = {
        "summary": summary,
        "episodes": episode_results,
        "saved_videos": saved_videos,
        "sampled_reset_state_ids": sampled_reset_state_ids,
        "task_reset_state_ids": task_reset_state_ids,
    }
    metrics_path.write_text(
        json.dumps(to_python(metrics_payload), indent=2, ensure_ascii=False) + "\n"
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
                f"num_episodes: {summary['num_episodes']}",
                f"num_success: {summary['num_success']}",
                f"success_rate: {summary['success_rate']:.4f}",
                f"avg_episode_len: {summary['avg_episode_len']:.2f}",
                f"avg_return: {summary['avg_return']:.4f}",
                f"num_videos_saved: {summary['num_videos_saved']}",
                f"video_save_ratio: {summary['video_save_ratio']:.4f}",
                f"seed: {summary['seed']}",
            ]
        )
        + "\n"
    )

    print("")
    print(f"Success rate: {success_rate:.4f} ({num_success}/{len(episode_results)})")
    print(f"Average episode length: {avg_episode_len:.2f}")
    print(f"Average return: {avg_return:.4f}")
    print(f"Metrics written to {metrics_path}")
    print(f"Summary written to {summary_path}")
    print(f"Saved video manifest written to {videos_path}")


if __name__ == "__main__":
    main()
