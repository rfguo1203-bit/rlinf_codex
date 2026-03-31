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

"""Resolve one LIBERO-10 task and run Pi0.5 inference with local video export."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from simple_eval_libero10_pi05 import load_libero10_metadata, run_single_task_eval


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer Pi0.5 on one LIBERO-10 task and save local videos."
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
        default="libero_10_ppo_openpi_pi05",
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
        default=0,
        help="Random seed used for env selection and shuffle.",
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

    print(f"Task {results['task_id']}: {results['task_name']}")
    for episode in results["episodes"]:
        print(
            "Episode "
            f"{episode['episode_idx']}: reset_state_id={episode['reset_state_id']}, "
            f"success={episode['success']}, steps={episode['steps']}, "
            f"video={episode['video_path']}"
        )


if __name__ == "__main__":
    main()
