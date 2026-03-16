# RLinf-USER 快速排障与调参速查（面向新会话）

适用场景：
- RLinf-USER / Franka-Sim / π0.5（OpenPI）训练
- 4xH100（单卡 82G）
- 目标：尽量吃满显存但不 OOM，并快速定位训练卡死/超时

参考：
- 论文：<https://arxiv.org/abs/2602.07837>
- 官方 RLinf-USER 文档：<https://rlinf.readthedocs.io/en/latest/rst_source/publications/rlinf_user.html>
- π0/π0.5 配置指南：<https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html>
- Placement 教程：<https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/user/placement.html>
- FAQ：<https://rlinf.readthedocs.io/en/latest/rst_source/faq.html>

---

## 1. RLinf-USER 核心特性（5 句话记住）

1. 统一抽象：奖励、算法、策略可插拔；同一框架覆盖 CNN/MLP、Flow policy、VLA。
2. 异步流水线：将数据生成、训练、权重同步解耦，论文里显示对大模型训练提速明显。
3. 资源一等公民：不仅是 GPU，机器人硬件也可被 placement 统一调度。
4. 多机与异构：支持多机器人并行采样与异构本体联合训练。
5. 同一配置语义：通过 `cluster.component_placement` + `actor/rollout/env` 参数组合完成部署与调优。

---

## 2. 参数配置心智模型（先看这个再改 YAML）

把参数分成三层：

1. 拓扑层（谁用哪张卡）
- `cluster.component_placement`
- `rollout.pipeline_stage_num`

2. 采样层（每轮收集多少数据）
- `env.train.total_num_envs`
- `env.train.max_steps_per_rollout_epoch`
- `algorithm.rollout_epoch`

3. 学习层（每次更新吃多少样本）
- `actor.micro_batch_size`
- `actor.global_batch_size`
- `algorithm.update_epoch`

关键约束（FSDP）：
- `global_batch_size % (micro_batch_size * actor_world_size) == 0`

---

## 3. 你当前配置的建议基线（4xH100 / 82G）

目标配置文件：
- `examples/embodiment/config/frankasim_ppo_openpi_pi05_local_debug.yaml`

建议基线（优先稳定）：
- placement：`env: 0`, `rollout: 0`, `actor: 1-3`
- FSDP：`sharding_strategy: no_shard`
- AMP：`bf16`（`amp.enabled: true`, `amp.precision: bf16`）
- batch：先从 `micro_batch_size=384`, `global_batch_size=1152` 起步
- env 并发：`total_num_envs=16`（train/eval 同步）

为什么不是先上 full_shard：
- 你历史日志出现过 NCCL collective 超时，full_shard 会放大 all-gather 压力。
- 先用 no_shard + bf16 找到稳定区间，再考虑更激进策略更稳。

---

## 4. 吃满 82G 的渐进调参法（每次只改 1 个旋钮）

优先级顺序：

1. `actor.micro_batch_size`（主旋钮，直接影响 actor 显存）
2. `env.total_num_envs`（主影响 rollout/env 负载）
3. `algorithm.update_epoch`（主影响每轮训练时长，不直接决定峰值显存）

建议阶梯（保持 actor_world_size=3）：

1. 档位 A：`micro=384`, `global=1152`
2. 档位 B：`micro=448`, `global=1344`
3. 档位 C：`micro=512`, `global=1536`

停止条件：
- 任一 actor GPU 峰值显存达到 79~81G（靠近 82G 但留安全边界）
- 出现 CUDA OOM / NCCL 超时即回退一档

---

## 5. 训练卡死/报错快速定位（症状 -> 首查项）

1. 报 `global_batch_size must be divisible ...`
- 先查 `actor_world_size` 与 `micro/global_batch_size` 整除关系

2. 报 NCCL watchdog timeout（ALLREDUCE/ALLGATHER）
- 先回退到 `no_shard`
- 降低 `micro_batch_size`
- 检查是否某 rank 提前报错导致其余 rank 卡通信

3. rollout 已完成但 actor 挂
- 优先看 actor worker 日志而非主进程尾部异常
- 常见根因：actor 显存/通信问题，不是环境采样逻辑

4. Ray 状态 API 异常（如 504 / 本地 dashboard 请求异常）
- 多为故障后的次生错误
- 不要把它当第一根因

---

## 6. 每次实验必记的最小实验记录模板

每次改参数只记录这 7 项：

1. commit id
2. config 名称
3. placement
4. `micro/global_batch_size`
5. `total_num_envs`
6. 峰值显存（每张卡）
7. 是否成功跑过 N steps（如 500/1000）

这样下个会话能 1 分钟恢复上下文，避免重复试错。

---

## 7. 推荐执行节奏（实战）

1. 先用基线参数跑到 300~500 steps，确认无超时/无 OOM。
2. 只提高 `micro_batch_size` 一个档位，复跑。
3. 若稳定再增 `total_num_envs`，观察 rollout 与 actor 是否失衡。
4. 每次变更后固定观察：
- `train/actor/grad_norm`
- `train/actor/approx_kl`
- `rollout/returns_mean`
- 每卡显存峰值和 NCCL 错误日志

---

## 8. 一句话原则

先用稳定拓扑（no_shard + bf16）找到“最大可用 micro_batch”，再调并发；不要同时改 placement、batch、env 并发。

