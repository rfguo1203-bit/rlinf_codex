#!/usr/bin/env python3

"""Run pi0.5 inference for one LIBERO-10 task and save rollout videos locally.

This script is a focused wrapper around `scripts/simple_eval_libero10_pi05.py`.
It is designed for the common workflow of:

1. choosing one task from the LIBERO-10 benchmark,
2. running pi0.5 inference on a small number of fixed reset states,
3. saving the rollout video(s) to a local directory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_CFG_PATH = (
    REPO_ROOT / "examples" / "embodiment" / "config" / "env" / "libero_10.yaml"
)
SIMPLE_EVAL_SCRIPT_PATH = REPO_ROOT / "scripts" / "simple_eval_libero10_pi05.py"


def normalize_task_name(task_name: str) -> str:
    """Normalize a task description for fuzzy matching."""
    normalized = task_name.strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(normalized.split())


def load_simple_eval_module():
    """Import the shared minimal LIBERO-10 pi0.5 evaluation helper script."""
    spec = importlib.util.spec_from_file_location(
        "simple_eval_libero10_pi05", SIMPLE_EVAL_SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {SIMPLE_EVAL_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_libero10_task_descriptions() -> list[str]:
    """Load all LIBERO-10 task descriptions in benchmark order."""
    from omegaconf import OmegaConf

    from rlinf.envs.libero.utils import get_benchmark_overridden

    env_cfg = OmegaConf.load(ENV_CFG_PATH)
    task_suite = get_benchmark_overridden(env_cfg.task_suite_name)()
    return [str(task_suite.get_task(task_id).language) for task_id in range(task_suite.get_num_tasks())]


def resolve_task_id(
    task_descriptions: list[str],
    task_id: int | None = None,
    task_name: str | None = None,
) -> int:
    """Resolve a LIBERO-10 task id from either `task_id` or `task_name`."""
    if task_id is not None:
        if task_id < 0 or task_id >= len(task_descriptions):
            raise ValueError(
                f"task_id must be in [0, {len(task_descriptions) - 1}], got {task_id}"
            )
        return task_id

    if not task_name:
        raise ValueError("Please provide either --task-id or --task-name.")

    normalized_query = normalize_task_name(task_name)
    exact_matches = [
        idx
        for idx, description in enumerate(task_descriptions)
        if normalize_task_name(description) == normalized_query
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise ValueError(
            "Matched multiple tasks by exact normalized name. "
            "Please use --task-id instead."
        )

    substring_matches = [
        idx
        for idx, description in enumerate(task_descriptions)
        if normalized_query in normalize_task_name(description)
    ]
    if len(substring_matches) == 1:
        return substring_matches[0]
    if len(substring_matches) > 1:
        match_lines = "\n".join(
            f"  [{idx}] {task_descriptions[idx]}" for idx in substring_matches
        )
        raise ValueError(
            "Matched multiple tasks by substring. Please use a more specific "
            f"--task-name or pass --task-id.\n{match_lines}"
        )

    available_tasks = "\n".join(
        f"  [{idx}] {description}" for idx, description in enumerate(task_descriptions)
    )
    raise ValueError(
        f"Could not find a LIBERO-10 task matching {task_name!r}.\n"
        f"Available tasks:\n{available_tasks}"
    )


def format_task_list(task_descriptions: list[str]) -> str:
    """Return a stable, readable task list."""
    return "\n".join(
        f"[{task_id}] {description}"
        for task_id, description in enumerate(task_descriptions)
    )


def to_python(value: Any) -> Any:
    """Convert Path values recursively for JSON output."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_python(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_python(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pi0.5 inference for one LIBERO-10 task and save local rollout video(s)."
        )
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print all LIBERO-10 task ids and descriptions, then exit.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local pi0.5 model directory. Required unless --list-tasks is used.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="LIBERO-10 task id to run.",
    )
    parser.add_argument(
        "--task-name",
        default=None,
        help="LIBERO-10 task description or unique substring to run.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Number of fixed reset states to evaluate for the selected task.",
    )
    parser.add_argument(
        "--save-video-ratio",
        type=float,
        default=1.0,
        help="Fraction of evaluated episodes whose rollout videos will be saved.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: logs/infer_libero10_pi05_single_task/<timestamp>/",
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
        help="Shuffle the selected task's fixed reset states before truncation.",
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


def validate_args(args: argparse.Namespace) -> None:
    """Validate user-facing CLI arguments."""
    if args.list_tasks:
        return
    if not args.model_path:
        raise ValueError("--model-path is required unless --list-tasks is used.")
    if args.task_id is None and not args.task_name:
        raise ValueError("Please provide either --task-id or --task-name.")
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be > 0.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    task_descriptions = load_libero10_task_descriptions()
    if args.list_tasks:
        print(format_task_list(task_descriptions))
        return

    task_id = resolve_task_id(
        task_descriptions=task_descriptions,
        task_id=args.task_id,
        task_name=args.task_name,
    )
    task_description = task_descriptions[task_id]

    simple_eval = load_simple_eval_module()

    import torch

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (
            REPO_ROOT
            / "logs"
            / "infer_libero10_pi05_single_task"
            / f"task_{task_id:02d}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)

    env_cfg = simple_eval.build_env_cfg(args)
    model_cfg = simple_eval.build_model_cfg(args)

    print(f"Selected task_id={task_id}")
    print(f"Task description: {task_description}")
    print(f"Loading model from {args.model_path}")

    model = simple_eval.load_model(model_cfg, args.ckpt_path, device)
    env = simple_eval.build_env(env_cfg)

    task_reset_state_ids = simple_eval.build_task_reset_state_ids(
        [int(x) for x in env.cumsum_trial_id_bins.tolist()],
        task_id,
    )
    eval_reset_state_ids = simple_eval.choose_reset_state_ids(
        task_reset_state_ids=task_reset_state_ids,
        num_episodes=args.num_episodes,
        shuffle=args.shuffle_reset_states,
        seed=args.seed,
    )

    num_save_videos = simple_eval.compute_num_save_videos(
        num_episodes=len(eval_reset_state_ids),
        save_video_ratio=args.save_video_ratio,
    )
    video_episode_indices = set(
        simple_eval.select_video_indices(len(eval_reset_state_ids), num_save_videos)
    )
    env_runner = simple_eval.maybe_wrap_video(
        env=env,
        env_cfg=env_cfg,
        fps=args.fps,
        enable_video=num_save_videos > 0,
    )

    episode_results: list[dict[str, Any]] = []
    saved_videos: list[dict[str, Any]] = []

    try:
        for episode_idx, reset_state_id in enumerate(eval_reset_state_ids):
            result = simple_eval.run_episode(
                env_runner=env_runner,
                model=model,
                model_cfg=model_cfg,
                env_cfg=env_cfg,
                reset_state_id=reset_state_id,
                device=device,
            )
            result["episode_idx"] = episode_idx
            result["task_id"] = task_id
            episode_results.append(result)

            if episode_idx in video_episode_indices and hasattr(env_runner, "flush_video"):
                video_sub_dir = f"task_{task_id:02d}/episode_{episode_idx:04d}"
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
                        "video_path": video_path,
                    }
                )
            else:
                simple_eval.clear_video_buffer(env_runner)

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

    summary = {
        "task_id": task_id,
        "task_description": task_description,
        "model_path": args.model_path,
        "ckpt_path": args.ckpt_path,
        "num_episodes": len(eval_reset_state_ids),
        "num_success_once": num_success_once,
        "num_success_at_end": num_success_at_end,
        "success_once_rate": success_once_rate,
        "success_at_end_rate": success_at_end_rate,
        "num_videos_saved": len(saved_videos),
        "output_dir": output_dir,
        "saved_video_paths": [item["video_path"] for item in saved_videos],
    }

    payload = {
        "summary": summary,
        "episodes": episode_results,
        "saved_videos": saved_videos,
        "evaluated_reset_state_ids": eval_reset_state_ids,
        "task_reset_state_ids": task_reset_state_ids,
    }

    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.txt"
    metrics_path.write_text(
        json.dumps(simple_eval.to_python(to_python(payload)), indent=2, ensure_ascii=False)
        + "\n"
    )
    summary_path.write_text(
        "\n".join(
            [
                f"task_id: {task_id}",
                f"task_description: {task_description}",
                f"model_path: {args.model_path}",
                f"ckpt_path: {args.ckpt_path}",
                f"num_episodes: {len(eval_reset_state_ids)}",
                f"success_once_rate: {success_once_rate:.4f}",
                f"success_at_end_rate: {success_at_end_rate:.4f}",
                f"num_videos_saved: {len(saved_videos)}",
                f"output_dir: {output_dir}",
            ]
        )
        + "\n"
    )

    print("")
    print(f"Task: [{task_id}] {task_description}")
    print(f"Success-once rate: {success_once_rate:.4f}")
    print(f"Success-at-end rate: {success_at_end_rate:.4f}")
    print(f"Metrics written to {metrics_path}")
    if saved_videos:
        print(f"Saved video: {saved_videos[0]['video_path']}")
    else:
        print("No video was saved. Try increasing --save-video-ratio.")


if __name__ == "__main__":
    main()
