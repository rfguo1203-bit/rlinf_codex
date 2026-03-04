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

"""Single-process embodied training entry for pdb debugging.

This entry keeps the same embodied computation path as train_embodied_agent.py
for single-machine/single-GPU runs, while removing Ray and using local channels.
"""

import asyncio
import gc
import json
import os
import queue
import socket
import threading
import time
from contextlib import contextmanager
from datetime import timedelta
from typing import Any

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf, open_dict

from rlinf.config import validate_fsdp_cfg
from rlinf.data.embodied_io_struct import Trajectory, convert_trajectories_to_batch
from rlinf.envs import get_env_cls
from rlinf.envs.wrappers import RecordVideo
from rlinf.scheduler import Channel
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import (
    compute_evaluate_metrics,
    compute_split_num,
    print_metrics_table,
)
from rlinf.utils.runner_utils import check_progress
from rlinf.utils.utils import get_model_weights_id
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _init_single_process_dist():
    if dist.is_initialized():
        yield
        return

    has_cuda = torch.cuda.is_available()
    backend = "nccl" if has_cuda else "gloo"

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ.setdefault("NODE_RANK", "0")

    if has_cuda:
        torch.cuda.set_device(0)

    dist.init_process_group(
        backend=backend,
        rank=0,
        world_size=1,
        timeout=timedelta(minutes=30),
    )

    try:
        yield
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def _setup_local_worker_attrs(worker: Any, group_name: str) -> None:
    worker._rank = 0
    worker._world_size = 1
    worker._group_name = group_name
    worker._cluster_node_rank = 0
    worker._local_accelerator_rank = 0
    worker._node_local_rank = 0
    worker._node_local_world_size = 1
    worker._local_rank = 0
    worker._local_world_size = 1
    worker._is_ray_actor = False
    worker._timer_metrics = {}
    worker._logger = get_logger()
    worker._stacklevel = 3
    worker._lock = threading.Lock()
    worker._has_initialized = True


class LocalEnvWorker(EnvWorker):
    """Local EnvWorker without Ray/Cluster initialization."""

    def __init__(self, cfg: DictConfig):
        _setup_local_worker_attrs(self, cfg.env.group_name)

        self.cfg = cfg
        self.train_video_cnt = 0
        self.eval_video_cnt = 0
        self.should_stop = False

        self.env_list = []
        self.eval_env_list = []

        self.last_obs_list = []
        self.last_intervened_info_list = []

        self.gather_num = 1
        self.stage_num = self.cfg.rollout.pipeline_stage_num

        self.only_eval = getattr(self.cfg.runner, "only_eval", False)
        self.enable_eval = self.cfg.runner.val_check_interval > 0 or self.only_eval
        if not self.only_eval:
            self.train_num_envs_per_stage = (
                self.cfg.env.train.total_num_envs // self._world_size // self.stage_num
            )
        if self.enable_eval:
            self.eval_num_envs_per_stage = (
                self.cfg.env.eval.total_num_envs // self._world_size // self.stage_num
            )

    def init_worker(self):
        train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
        eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)

        if not self.only_eval:
            for stage_id in range(self.stage_num):
                env = train_env_cls(
                    cfg=self.cfg.env.train,
                    num_envs=self.train_num_envs_per_stage,
                    seed_offset=self._rank * self.stage_num + stage_id,
                    total_num_processes=self._world_size * self.stage_num,
                    worker_info=None,
                )
                if self.cfg.env.train.video_cfg.save_video:
                    env = RecordVideo(env, self.cfg.env.train.video_cfg)
                self.env_list.append(env)
        if self.enable_eval:
            for stage_id in range(self.stage_num):
                env = eval_env_cls(
                    cfg=self.cfg.env.eval,
                    num_envs=self.eval_num_envs_per_stage,
                    seed_offset=self._rank * self.stage_num + stage_id,
                    total_num_processes=self._world_size * self.stage_num,
                    worker_info=None,
                )
                if self.cfg.env.eval.video_cfg.save_video:
                    env = RecordVideo(env, self.cfg.env.eval.video_cfg)
                self.eval_env_list.append(env)

        if not self.only_eval:
            self._init_env()


class LocalRolloutWorker(MultiStepRolloutWorker):
    """Local rollout worker using in-process channels."""

    def __init__(self, cfg: DictConfig):
        _setup_local_worker_attrs(self, cfg.rollout.group_name)

        self.cfg = cfg
        self.should_stop = False

        self.actor_group_name = cfg.actor.group_name
        self.device = torch.cuda.current_device()

        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)
        self.only_eval = getattr(self.cfg.runner, "only_eval", False)
        self.enable_train = not self.only_eval
        self.enable_eval = self.cfg.runner.val_check_interval > 0 or self.only_eval

        self.actor_weight_src_rank = 0

        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.model_weights_id = ""
        self.count_update = 0

        self.total_num_train_envs = cfg.env.train.total_num_envs
        self.total_num_eval_envs = cfg.env.eval.total_num_envs
        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.train_batch_size = (
            self.total_num_train_envs // self._world_size // self.num_pipeline_stages
        )
        self.eval_batch_size = (
            self.total_num_eval_envs // self._world_size // self.num_pipeline_stages
        )
        self.enable_cuda_graph = cfg.rollout.get("enable_cuda_graph", False)
        # Compatibility shim:
        # Upstream versions may access self.placement in init_worker/_setup_dst_ranks.
        # For local single-process debug, all component world sizes are 1.
        class _LocalPlacement:
            @staticmethod
            def get_world_size(_component: str) -> int:
                return 1

        self.placement = _LocalPlacement()
        # Compatibility for newer rollout worker implementations.
        self.train_dst_ranks = [0]
        self.eval_dst_ranks = [0]

    def sync_model_from_actor_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.hf_model.load_state_dict(state_dict)
        self.model_weights_id = (
            str(get_model_weights_id(self.hf_model)) + f"_{self.count_update}"
        )
        self.count_update += 1
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

    async def recv_env_output(
        self, input_channel: Channel, mode="train"
    ) -> dict[str, torch.Tensor]:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        key = f"{self._rank}_{mode}"
        return await asyncio.to_thread(input_channel.get, key)

    def send_chunk_actions(self, output_channel: Channel, chunk_actions, mode="train"):
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        output_channel.put(item=chunk_actions, key=f"{self._rank}_{mode}")

    async def send_rollout_trajectories(
        self, rollout_result, channel: Channel
    ) -> None:
        split_num = self.get_actor_split_num()
        trajectories: Trajectory = rollout_result.to_splited_trajectories(split_num)
        for trajectory in trajectories:
            channel.put(item=trajectory)

    def get_actor_split_num(self):
        send_num = self._world_size * self.num_pipeline_stages
        recv_num = 1
        return compute_split_num(recv_num, send_num)


class LocalEmbodiedFSDPActor(EmbodiedFSDPActor):
    """Local embodied actor that keeps original FSDP training logic."""

    def __init__(self, cfg: DictConfig):
        _setup_local_worker_attrs(self, cfg.actor.group_name)

        from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager

        FSDPModelManager.__init__(self, cfg.actor, self._world_size, self._rank)

        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self.stage_num = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.actor.get("enable_offload", False)
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "torch")
        self._weight_dst_rank_in_rollout = [0]

    def _setup_rollout_weight_dst_ranks(self) -> None:
        self._weight_dst_rank_in_rollout = [0]

    def sync_model_to_rollout_state(self) -> dict[str, torch.Tensor]:
        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()
        if self.enable_offload and self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        state_dict = self.get_model_state_dict(cpu_offload=False, full_state_dict=True)
        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()
        return state_dict

    def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        send_num = 1 * self.stage_num
        recv_num = 1
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory: Trajectory = input_channel.get()
            recv_list.append(trajectory)

        self.rollout_batch = convert_trajectories_to_batch(recv_list)
        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)


class LocalEmbodiedRunner:
    """Embodied runner for local single-process debugging."""

    def __init__(
        self,
        cfg: DictConfig,
        actor: LocalEmbodiedFSDPActor,
        rollout: LocalRolloutWorker,
        env: LocalEnvWorker,
    ):
        self.cfg = cfg
        self.actor = actor
        self.rollout = rollout
        self.env = env

        self.weight_sync_interval = self.cfg.runner.weight_sync_interval
        self.env_channel = Channel.create("LocalEnv", local=True)
        self.rollout_channel = Channel.create("LocalRollout", local=True)
        self.actor_channel = Channel.create("LocalActor", local=True)

        self.global_step = 0
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs
        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.logger = get_logger()
        self.metric_logger = MetricLogger(cfg)

    def init_workers(self):
        self.actor.init_worker()
        self.rollout.init_worker()
        self.env.init_worker()

    def update_rollout_weights(self):
        state_dict = self.actor.sync_model_to_rollout_state()
        self.rollout.sync_model_from_actor_state(state_dict)

    def _run_interact_and_generate(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        error_q: queue.Queue[Exception] = queue.Queue()
        env_result: dict[str, Any] = {"metrics": None}

        def env_job():
            try:
                env_result["metrics"] = self.env.interact(
                    input_channel=self.rollout_channel,
                    output_channel=self.env_channel,
                )
            except Exception as exc:  # pragma: no cover - passthrough
                error_q.put(exc)

        thread = threading.Thread(target=env_job, daemon=True)
        thread.start()

        asyncio.run(
            self.rollout.generate(
                input_channel=self.env_channel,
                output_channel=self.rollout_channel,
                actor_channel=self.actor_channel,
            )
        )
        thread.join()

        if not error_q.empty():
            raise error_q.get()

        self.actor.recv_rollout_trajectories(input_channel=self.actor_channel)
        rollout_metrics = self.actor.compute_advantages_and_returns()
        return env_result["metrics"], rollout_metrics

    def evaluate(self):
        error_q: queue.Queue[Exception] = queue.Queue()
        env_result: dict[str, Any] = {"metrics": None}

        def env_job():
            try:
                env_result["metrics"] = self.env.evaluate(
                    input_channel=self.rollout_channel,
                    output_channel=self.env_channel,
                )
            except Exception as exc:  # pragma: no cover - passthrough
                error_q.put(exc)

        thread = threading.Thread(target=env_job, daemon=True)
        thread.start()
        asyncio.run(
            self.rollout.evaluate(
                input_channel=self.env_channel,
                output_channel=self.rollout_channel,
            )
        )
        thread.join()

        if not error_q.empty():
            raise error_q.get()

        eval_metrics = compute_evaluate_metrics([env_result["metrics"]])
        return eval_metrics

    def _save_checkpoint(self):
        self.logger.info(f"Saving checkpoint at step {self.global_step}.")
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step)

    def run(self):
        start_time = time.time()
        for step in range(self.max_steps):
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)

            with self.timer("step"):
                with self.timer("sync_weights"):
                    if step % self.weight_sync_interval == 0:
                        self.update_rollout_weights()

                with self.timer("generate_rollouts"):
                    env_metrics_raw, rollout_metrics_raw = (
                        self._run_interact_and_generate()
                    )

                training_metrics_raw = self.actor.run_training()
                self.global_step += 1

                run_val, save_model, _ = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )

                eval_metrics = {}
                if run_val:
                    with self.timer("eval"):
                        self.update_rollout_weights()
                        eval_metrics = self.evaluate()
                        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                        self.metric_logger.log(data=eval_metrics, step=step)

                if save_model:
                    self._save_checkpoint()

            time_metrics = self.timer.consume_durations()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            time_metrics.update(
                {f"time/env/{k}": v for k, v in self.env.pop_execution_times().items()}
            )
            time_metrics.update(
                {
                    f"time/rollout/{k}": v
                    for k, v in self.rollout.pop_execution_times().items()
                }
            )
            time_metrics.update(
                {
                    f"time/actor/{k}": v
                    for k, v in self.actor.pop_execution_times().items()
                }
            )

            env_metrics = compute_evaluate_metrics([env_metrics_raw])
            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
            rollout_metrics = {f"rollout/{k}": v for k, v in rollout_metrics_raw.items()}
            training_metrics = {
                f"train/{k}": v for k, v in training_metrics_raw.items()
            }

            self.metric_logger.log(env_metrics, step)
            self.metric_logger.log(rollout_metrics, step)
            self.metric_logger.log(time_metrics, step)
            self.metric_logger.log(training_metrics, step)

            logging_metrics = {}
            logging_metrics.update(time_metrics)
            logging_metrics.update(eval_metrics)
            logging_metrics.update(env_metrics)
            logging_metrics.update(rollout_metrics)
            logging_metrics.update(training_metrics)
            print_metrics_table(step, self.max_steps, start_time, logging_metrics, 0)

        self.metric_logger.finish()


def _validate_local_cfg(cfg: DictConfig) -> DictConfig:
    if cfg.runner.task_type != "embodied":
        raise ValueError("Local debug entry only supports runner.task_type=embodied.")
    if cfg.actor.training_backend != "fsdp":
        raise ValueError("Local debug entry only supports actor.training_backend=fsdp.")
    if cfg.cluster.num_nodes != 1:
        raise ValueError("Local debug entry requires cluster.num_nodes == 1.")

    with open_dict(cfg):
        cfg.runner.weight_sync_interval = cfg.runner.get("weight_sync_interval", 1)
        cfg.actor = validate_fsdp_cfg(cfg.actor)

    assert cfg.runner.weight_sync_interval > 0, (
        "runner.weight_sync_interval must be greater than 0."
    )
    assert cfg.env.train.total_num_envs > 0, "env.train.total_num_envs must be > 0."
    assert cfg.env.eval.total_num_envs > 0, "env.eval.total_num_envs must be > 0."
    assert cfg.env.train.total_num_envs % cfg.rollout.pipeline_stage_num == 0, (
        "env.train.total_num_envs must be divisible by rollout.pipeline_stage_num."
    )
    assert cfg.env.eval.total_num_envs % cfg.rollout.pipeline_stage_num == 0, (
        "env.eval.total_num_envs must be divisible by rollout.pipeline_stage_num."
    )
    return cfg


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    cfg = _validate_local_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    with _init_single_process_dist():
        actor = LocalEmbodiedFSDPActor(cfg)
        rollout = LocalRolloutWorker(cfg)
        env = LocalEnvWorker(cfg)
        runner = LocalEmbodiedRunner(cfg=cfg, actor=actor, rollout=rollout, env=env)
        runner.init_workers()
        runner.run()


if __name__ == "__main__":
    main()
