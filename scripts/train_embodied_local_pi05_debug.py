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

"""Single-process embodied training entry for pdb debugging with pi0.5 on FrankaSim."""

import json

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from train_embodied_local_debug import (
    LocalEmbodiedFSDPActor,
    LocalEmbodiedRunner,
    LocalEnvWorker,
    LocalRolloutWorker,
    _init_single_process_dist,
    _validate_local_cfg,
)


def _apply_franka_pi05_overrides(cfg: DictConfig) -> DictConfig:
    """Apply pi0.5+PPO defaults for FrankaSim local debug."""
    with open_dict(cfg):
        # Keep embodied + FSDP path unchanged.
        cfg.runner.task_type = "embodied"
        cfg.actor.training_backend = "fsdp"
        cfg.cluster.num_nodes = 1

        # Use OpenPI (pi0.5) as actor/rollout model.
        cfg.actor.model.model_type = "openpi"
        cfg.actor.model.precision = None
        cfg.rollout.model.precision = None

        # FrankaSim receives 4D action; OpenPI internally keeps higher action dim.
        cfg.actor.model.action_dim = 4
        cfg.actor.model.num_action_chunks = cfg.actor.model.get("num_action_chunks", 1)
        cfg.actor.model.num_steps = cfg.actor.model.get("num_steps", 3)
        cfg.actor.model.add_value_head = True

        if "openpi" not in cfg.actor.model or cfg.actor.model.openpi is None:
            cfg.actor.model.openpi = {}

        cfg.actor.model.openpi.config_name = cfg.actor.model.openpi.get(
            "config_name", "pi0_custom"
        )
        cfg.actor.model.openpi.pi05 = True
        cfg.actor.model.openpi.num_images_in_input = cfg.actor.model.openpi.get(
            "num_images_in_input", 2
        )
        cfg.actor.model.openpi.noise_method = cfg.actor.model.openpi.get(
            "noise_method", "flow_sde"
        )
        cfg.actor.model.openpi.noise_level = cfg.actor.model.openpi.get(
            "noise_level", 0.5
        )
        cfg.actor.model.openpi.action_chunk = cfg.actor.model.num_action_chunks
        cfg.actor.model.openpi.num_steps = cfg.actor.model.num_steps
        cfg.actor.model.openpi.action_env_dim = cfg.actor.model.action_dim
        cfg.actor.model.openpi.train_expert_only = cfg.actor.model.openpi.get(
            "train_expert_only", True
        )
        cfg.actor.model.openpi.add_value_head = cfg.actor.model.add_value_head
        cfg.actor.model.openpi.value_after_vlm = cfg.actor.model.openpi.get(
            "value_after_vlm", True
        )

        # OpenPI expects wrist image key in obs for embodied inference.
        cfg.env.train.wrap_obs_mode = "openpi"
        cfg.env.eval.wrap_obs_mode = "openpi"

        # Use PPO-style OpenPI defaults.
        cfg.algorithm.adv_type = "gae"
        cfg.algorithm.loss_type = "actor_critic"
        cfg.algorithm.reward_type = "chunk_level"
        cfg.algorithm.logprob_type = "chunk_level"
        cfg.algorithm.entropy_type = "token_level"

        # Keep gradient checkpointing disabled for OpenPI.
        cfg.actor.fsdp_config.gradient_checkpointing = False

        # Make run identity explicit in logs.
        cfg.runner.logger.experiment_name = "franka_sim_ppo_openpi_pi05_local_debug"

    return cfg


@hydra.main(version_base="1.1", config_path="config", config_name="frankasim_ppo_cnn")
def main(cfg) -> None:
    cfg = _apply_franka_pi05_overrides(cfg)
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
