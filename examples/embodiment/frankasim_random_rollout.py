#!/usr/bin/env python3
"""Run FrankaSim with random continuous actions and save a rendered video.

This script is standalone and does not launch RLinf runner/worker groups.
It extracts FrankaSim env settings from an RLinf config (default:
examples/embodiment/config/frankasim_ppo_cnn.yaml), runs one rollout with
smooth random actions, and writes an MP4 video.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
from omegaconf import DictConfig, OmegaConf

from rlinf.envs import get_env_cls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FrankaSim with random actions and save video."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="examples/embodiment/config/frankasim_ppo_cnn.yaml",
        help="Path to RLinf config yaml.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="eval",
        choices=["train", "eval"],
        help="Use env.train or env.eval settings from config.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=-1,
        help="Rollout steps. If <=0, use env.<split>.max_episode_steps.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/frankasim_random_rollout.mp4",
        help="Output video path.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Output video fps.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of env instances to run in parallel.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for action generation and env seed offset.",
    )
    parser.add_argument(
        "--action-alpha",
        type=float,
        default=0.9,
        help="Smoothing factor for random continuous actions (0~1).",
    )
    parser.add_argument(
        "--action-noise",
        type=float,
        default=0.35,
        help="Noise std for smooth random action update.",
    )
    return parser.parse_args()


def _resolve_config_path(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path


def _iter_defaults(defaults: Iterable) -> Iterable[str]:
    for item in defaults:
        if isinstance(item, str):
            yield item
        elif isinstance(item, DictConfig):
            for key in item.keys():
                yield str(key)


def _load_env_cfg(config_path: Path, split: str) -> DictConfig:
    cfg = OmegaConf.load(config_path)
    defaults = cfg.get("defaults", [])
    env_default_key = None
    target = f"@env.{split}"

    for item in _iter_defaults(defaults):
        if item.startswith("env/") and target in item:
            env_default_key = item.split("@", maxsplit=1)[0]
            break

    if env_default_key is None:
        raise ValueError(
            f"Cannot find env default for split='{split}' in {config_path}."
        )

    base_env_cfg_path = config_path.parent / f"{env_default_key}.yaml"
    if not base_env_cfg_path.exists():
        raise FileNotFoundError(f"Env config not found: {base_env_cfg_path}")

    base_env_cfg = OmegaConf.load(base_env_cfg_path)
    # Resolve interpolations (e.g. ${algorithm.group_size}) against root config first.
    split_cfg_resolved = OmegaConf.create(
        OmegaConf.to_container(cfg.env[split], resolve=True)
    )
    env_cfg = OmegaConf.merge(base_env_cfg, split_cfg_resolved)
    return env_cfg


def _extract_frame(obs: dict, env_index: int = 0) -> np.ndarray:
    if "main_images" not in obs:
        raise KeyError("Observation does not contain 'main_images'; need rgb obs_mode.")

    frame = obs["main_images"]
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    else:
        frame = np.asarray(frame)

    if frame.ndim != 4:
        raise ValueError(f"Unexpected main_images shape: {frame.shape}, expected [N,H,W,C].")

    frame = frame[env_index]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _make_smooth_random_actions(
    prev_actions: np.ndarray,
    rng: np.random.Generator,
    alpha: float,
    noise_std: float,
) -> np.ndarray:
    noise = rng.normal(loc=0.0, scale=noise_std, size=prev_actions.shape).astype(
        np.float32
    )
    updated = alpha * prev_actions + (1.0 - alpha) * noise
    return np.clip(updated, -1.0, 1.0)


def main() -> None:
    args = parse_args()
    config_path = _resolve_config_path(args.config)
    env_cfg = _load_env_cfg(config_path, args.split)

    env_cls = get_env_cls(env_cfg.env_type, env_cfg=env_cfg)
    env = env_cls(
        cfg=env_cfg,
        num_envs=args.num_envs,
        seed_offset=args.seed,
        total_num_processes=1,
        worker_info={"worker_rank": 0},
        record_metrics=False,
    )

    obs, _ = env.reset(seed=args.seed)
    frames: list[np.ndarray] = [_extract_frame(obs, env_index=0)]

    action_dim = int(np.prod(env.action_space.shape))
    steps = args.steps if args.steps > 0 else int(env_cfg.max_episode_steps)
    rng = np.random.default_rng(args.seed)
    actions = np.zeros((args.num_envs, action_dim), dtype=np.float32)

    for _ in range(steps):
        actions = _make_smooth_random_actions(
            prev_actions=actions,
            rng=rng,
            alpha=float(args.action_alpha),
            noise_std=float(args.action_noise),
        )
        obs, _, terminations, truncations, _ = env.step(actions)
        frames.append(_extract_frame(obs, env_index=0))

        done = bool((terminations[0] | truncations[0]).item())
        if done:
            break

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=args.fps)

    print(
        f"Saved video to {output_path} "
        f"(frames={len(frames)}, steps_executed={len(frames) - 1}, fps={args.fps})"
    )


if __name__ == "__main__":
    main()
