#!/usr/bin/env python3

"""Single-process FrankaSim evaluation for a local pi0.5 OpenPI model."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.wrappers.record_video import RecordVideo
from rlinf.models import get_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a local pi0.5 OpenPI model on FrankaSim."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local Hugging Face/OpenPI checkpoint directory.",
    )
    parser.add_argument(
        "--config-name",
        default="frankasim_ppo_openpi_pi05",
        help="Config name under examples/embodiment/config without .yaml.",
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
        default=Path("logs/simple_eval_frankasim_pi05"),
        help="Directory for metrics and videos.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for episode seeds and video sampling.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Override episode horizon from config.",
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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def config_root() -> Path:
    return repo_root() / "examples" / "embodiment" / "config"


def _extract_default_name(main_cfg: DictConfig, prefix: str, target: str) -> str:
    defaults = OmegaConf.to_container(main_cfg.get("defaults", []), resolve=False)
    for entry in defaults:
        if isinstance(entry, str):
            if "@" not in entry:
                continue
            key, value = entry.split("@", 1)
            if key.startswith(prefix) and value == target:
                return key.split("/", 1)[1]
        if isinstance(entry, dict):
            for key, value in entry.items():
                if (
                    key.endswith(f"@{target}")
                    and key.startswith(prefix)
                    and isinstance(value, str)
                ):
                    return value
    raise ValueError(f"Could not find default for {prefix}@{target} in config.")


def load_main_cfg(config_name: str) -> DictConfig:
    cfg_path = config_root() / f"{config_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return OmegaConf.load(cfg_path)


def compose_eval_env_cfg(main_cfg: DictConfig, args: argparse.Namespace) -> DictConfig:
    env_default = _extract_default_name(main_cfg, "env/", "env.eval")
    base_env_cfg = OmegaConf.load(config_root() / "env" / f"{env_default}.yaml")
    env_cfg = OmegaConf.merge(base_env_cfg, main_cfg.env.eval)
    with open_dict(env_cfg):
        env_cfg.total_num_envs = 1
        env_cfg.group_size = 1
        env_cfg.auto_reset = False
        env_cfg.ignore_terminations = False
        env_cfg.is_eval = True
        env_cfg.seed = args.seed
        if args.max_episode_steps is not None:
            env_cfg.max_episode_steps = args.max_episode_steps
            env_cfg.max_steps_per_rollout_epoch = args.max_episode_steps
        env_cfg.video_cfg.save_video = True
        env_cfg.video_cfg.info_on_video = True
        env_cfg.video_cfg.fps = args.fps
        env_cfg.video_cfg.video_base_dir = str(args.output_dir / "videos")
    return env_cfg


def compose_model_cfg(main_cfg: DictConfig, args: argparse.Namespace) -> DictConfig:
    model_default = _extract_default_name(main_cfg, "model/", "actor.model")
    base_model_cfg = OmegaConf.load(config_root() / "model" / f"{model_default}.yaml")
    model_cfg = OmegaConf.merge(base_model_cfg, main_cfg.actor.model)
    with open_dict(model_cfg):
        model_cfg.model_path = args.model_path
        model_cfg.is_lora = False
        model_cfg.add_value_head = False
        model_cfg.openpi.add_value_head = False
        model_cfg.openpi.action_chunk = model_cfg.num_action_chunks
        model_cfg.openpi.num_steps = model_cfg.num_steps
        model_cfg.openpi.action_env_dim = model_cfg.action_dim
    return model_cfg


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


def move_obs_to_device(obs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device).contiguous() if torch.is_tensor(value) else value
        for key, value in obs.items()
    }


def run_episode(
    env_like,
    model,
    model_cfg: DictConfig,
    env_cfg: DictConfig,
    device: torch.device,
    episode_seed: int,
) -> dict[str, Any]:
    obs, _ = env_like.reset(seed=episode_seed)
    chunk_steps = math.ceil(
        int(env_cfg.max_episode_steps) / int(model_cfg.num_action_chunks)
    )

    for _ in range(chunk_steps):
        obs = move_obs_to_device(obs, device)
        with torch.no_grad():
            raw_chunk_actions, _ = model.predict_action_batch(env_obs=obs, mode="eval")

        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_chunk_actions,
            env_type=env_cfg.env_type,
            model_type=model_cfg.model_type,
            num_action_chunks=model_cfg.num_action_chunks,
            action_dim=model_cfg.action_dim,
            policy=model_cfg.get("policy_setup", "widowx_bridge"),
        )

        obs_list, _rewards, chunk_terminations, chunk_truncations, _infos_list = (
            env_like.chunk_step(chunk_actions)
        )
        obs = obs_list[-1]
        done = bool(torch.logical_or(chunk_terminations, chunk_truncations).any().item())
        if done:
            break

    base_env = env_like.unwrapped
    success_once = bool(base_env.success_once[0].item())
    fail_once = bool(base_env.fail_once[0].item())
    episode_len = int(base_env.elapsed_steps[0].item())
    episode_return = float(base_env.returns[0].item())

    task_description = base_env.task_prompt
    if isinstance(task_description, list):
        task_description = task_description[0]

    return {
        "success_once": success_once,
        "fail_once": fail_once,
        "episode_len": episode_len,
        "return": episode_return,
        "episode_seed": episode_seed,
        "task_description": str(task_description),
    }


def main() -> None:
    args = parse_args()
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be > 0")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)

    main_cfg = load_main_cfg(args.config_name)
    env_cfg = compose_eval_env_cfg(main_cfg, args)
    model_cfg = compose_model_cfg(main_cfg, args)

    random_rng = random.Random(args.seed)
    video_episode_ids = choose_video_episodes(
        num_episodes=args.num_episodes,
        save_video_ratio=args.save_video_ratio,
        rng=random_rng,
    )
    episode_seeds = [args.seed + episode_idx for episode_idx in range(args.num_episodes)]

    print(f"Loading model from {args.model_path}")
    model = get_model(model_cfg)
    model = model.to(device)
    model.eval()

    env_cls = get_env_cls(env_cfg.env_type, env_cfg)
    env = env_cls(
        cfg=env_cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info={},
    )

    episode_results: list[dict[str, Any]] = []
    saved_videos: list[dict[str, Any]] = []

    print(
        f"Evaluating FrankaSim config={args.config_name} for {args.num_episodes} episodes; "
        f"saving {len(video_episode_ids)} videos."
    )

    try:
        for episode_idx, episode_seed in enumerate(episode_seeds):
            save_video = episode_idx in video_episode_ids
            env_runner = env
            if save_video:
                env_runner = RecordVideo(env, env_cfg.video_cfg, fps=args.fps)

            result = run_episode(
                env_like=env_runner,
                model=model,
                model_cfg=model_cfg,
                env_cfg=env_cfg,
                device=device,
                episode_seed=episode_seed,
            )
            result["episode_idx"] = episode_idx
            episode_results.append(result)

            if save_video:
                video_sub_dir = f"episode_{episode_idx:04d}"
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
                        "episode_seed": episode_seed,
                        "video_path": str(video_path),
                    }
                )

            success = int(result["success_once"])
            print(
                f"[{episode_idx + 1:03d}/{args.num_episodes:03d}] "
                f"seed={episode_seed} success={success} "
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
        "config_name": args.config_name,
        "gym_id": env_cfg.get("gym_id"),
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
                f"config_name: {summary['config_name']}",
                f"gym_id: {summary['gym_id']}",
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
