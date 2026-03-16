# FrankaSim PPO PI05 配置参数详解

## 概述

本文档详细解释了 `frankasim_ppo_openpi_pi05.yaml` 配置文件中每个参数的含义，以及为什么针对 FrankaSim 仿真环境进行这些参数调整。

---

## 1. PPO 算法核心参数

### 1.1 优势函数参数

| 参数 | 默认值 | 含义 | FrankaSim 调整理由 |
|------|--------|------|-------------------|
| `gamma` | 0.99 | 折扣因子，决定未来奖励的权重 | 标准值，平衡即时和长期奖励 |
| `gae_lambda` | 0.95 | GAE 的平滑参数 | 标准值，在偏差和方差间取得平衡 |
| `normalize_advantages` | true | 是否归一化优势函数 | **必须开启**，稳定 PPO 训练，防止梯度爆炸 |

**GAE 公式**:
```
δ_t = r_t + γV(s_{t+1}) - V(s_t)
A_t^GAE = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}
```

### 1.2 PPO 裁剪参数

| 参数 | 默认值 | 含义 | FrankaSim 调整理由 |
|------|--------|------|-------------------|
| `clip_ratio_low` | 0.2 | PPO 裁剪下界 | 标准值，防止策略更新过大 |
| `clip_ratio_high` | 0.2 | PPO 裁剪上界 | 标准值，对称裁剪 |
| `clip_ratio_c` | 3.0 | 双重裁剪系数 | 防止极端优势值导致的不稳定 |
| `value_clip` | 0.2 | 价值函数裁剪阈值 | 防止价值函数更新过快 |
| `huber_delta` | 10.0 | Huber 损失 delta | 较大的值允许更大的误差 |

**PPO 裁剪公式**:
```
L^CLIP(θ) = E[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]
```

### 1.3 探索参数

| 参数 | 默认值 | 含义 | FrankaSim 调整理由 |
|------|--------|------|-------------------|
| `entropy_bonus` | 0.005 | 熵正则化系数 | **较小值**，FrankaSim 任务相对简单，不需要太多探索 |
| `kl_beta` | 0.0 | KL 惩罚系数 | 关闭，PPO 裁剪已提供足够的约束 |

---

## 2. PI05 模型特定参数

### 2.1 核心架构参数

| 参数 | 默认值 | 含义 | FrankaSim 调整理由 |
|------|--------|------|-------------------|
| `pi05` | true | 是否使用 PI05 架构 | **关键区别**: PI05 将 state 作为离散 token 输入，而非连续输入 |
| `num_action_chunks` | 1 | 动作分块数量 | FrankaSim 任务简单，不需要长序列预测 |
| `num_steps` | 4 | 去噪步数 | **减少** (原为 10)，加快训练速度 |
| `action_dim` | 4 | 动作维度 | 根据 Franka 机械臂配置 (实际为 7+1) |

**PI0 vs PI05 区别**:
- **PI0**: state 作为连续输入，与 action 一起通过 flow matching
- **PI05**: state 作为离散 token 输入，与 language token 类似

### 2.2 Flow Matching 参数

| 参数 | 默认值 | 含义 | FrankaSim 调整理由 |
|------|--------|------|-------------------|
| `noise_method` | "flow_noise" | 噪声方法 | **推荐 flow_noise**，支持联合概率计算 |
| `joint_logprob` | true | 联合概率计算 | **必须开启**，与 flow_noise 配合使用 |
| `noise_params` | [0.16, 0.12, 200] | 噪声参数 | [起始噪声，结束噪声，退火步数] |
| `value_after_vlm` | true | value head 位置 | PI05 模式：value head 在 VLM 后 |

**Flow Matching 原理**:
```
dx_t = v(x_t, t) dt + σ(t) dW_t
```
- `flow_noise`: 在去噪过程中添加噪声，需要联合概率计算
- `flow_sde`: 随机微分方程方式
- `flow_cps`: 条件路径采样

### 2.3 输入配置

| 参数 | 默认值 | 含义 | FrankaSim 调整理由 |
|------|--------|------|-------------------|
| `num_images_in_input` | 1 | 输入图像数量 | 根据相机配置：front=1, wrist=+1, both=2 |
| `config_name` | "pi05_franka" | 配置名称 | 决定数据预处理和归一化方式 |
| `policy_setup` | "widowx_bridge" | 策略设置 | 决定动作空间归一化 |

---

## 3. 环境配置参数

### 3.1 并行环境

| 参数 | 默认值 | 含义 | FrankaSim 调整理由 |
|------|--------|------|-------------------|
| `total_num_envs` | 24 | 并行环境数量 | **大幅减少** (ManiSkill 用 320)，FrankaSim 计算开销大 |
| `max_episode_steps` | 100 | 最大 episode 步数 | FrankaPickCube 任务通常 50-80 步完成 |
| `auto_reset` | true | 自动 reset | 持续训练，无需手动干预 |

**环境数量选择原则**:
- 保证每个 step 有足够的样本数
- 考虑 GPU 显存限制
- FrankaSim: 24 个环境 × vision 观测 ≈ 适度显存占用

### 3.2 奖励配置

| 参数 | 默认值 | 含义 | FrankaSim 调整理由 |
|------|--------|------|-------------------|
| `use_rel_reward` | true | 使用相对奖励 | 更稳定的训练信号 |
| `reward_type` | "chunk_level" | 奖励粒度 | 与动作分块对应 |

---

## 4. 训练配置参数

### 4.1 批次大小

| 参数 | 默认值 | 含义 | FrankaSim 调整理由 |
|------|--------|------|-------------------|
| `micro_batch_size` | 32 | 每个 GPU 的微批次 | 根据显存大小调整 |
| `global_batch_size` | 1024 | 全局批次大小 | **减小** (ManiSkill 用 5120)，环境数少 |

**批次大小计算**:
```
global_batch = micro_batch × num_gpus × gradient_accumulation
```

### 4.2 学习率

| 参数 | 默认值 | 含义 | FrankaSim 调整理由 |
|------|--------|------|-------------------|
| `lr` | 7.91e-6 | Actor 学习率 | 较小的学习率，稳定微调 |
| `value_lr` | 1.55e-4 | Critic 学习率 | **更高**，critic 需要快速收敛 |
| `clip_grad` | 1.0 | 梯度裁剪 | 防止梯度爆炸 |

**学习率设置原则**:
- Actor: 较小，保持预训练知识
- Critic: 较大，快速适应新任务

### 4.3 训练轮次

| 参数 | 默认值 | 含义 | FrankaSim 调整理由 |
|------|--------|------|-------------------|
| `update_epoch` | 5 | 每个 batch 更新轮数 | 标准值，充分利用每个 batch |
| `val_check_interval` | 10 | 评估间隔 | 频繁评估，监控训练 |
| `save_interval` | 50 | 保存间隔 | 定期保存，防止丢失进度 |

---

## 5. 采样参数

| 参数 | 默认值 | 含义 | FrankaSim 调整理由 |
|------|--------|------|-------------------|
| `temperature_train` | 1.0 | 训练温度 | 标准采样 |
| `temperature_eval` | 0.6 | 评估温度 | **较低**，评估时更确定 |
| `top_k` | 50 | Top-K 采样 | 限制采样范围 |
| `top_p` | 1.0 | Nucleus 采样 | 不限制 (1.0) |

---

## 6. 关键修改总结

### 从 ManiSkill 到 FrankaSim 的主要修改

| 参数 | ManiSkill 值 | FrankaSim 值 | 修改理由 |
|------|-------------|-------------|----------|
| `total_num_envs` | 320 | 24 | FrankaSim 仿真计算开销大 |
| `global_batch_size` | 5120 | 1024 | 配合环境数调整 |
| `num_action_chunks` | 5 | 1 | Franka 任务简单，短序列 |
| `num_steps` | 10 | 4 | 减少去噪步，加快训练 |
| `max_episode_steps` | 80 | 100 | FrankaPickCube 需要更多步 |
| `config_name` | pi05_maniskill | pi05_franka | 适配 Franka 数据格式 |
| `model_path` | ManiSkill-SFT | RLinf-Pi05-SFT | 使用对应预训练模型 |

### 为什么这样修改

1. **环境数量减少 (320→24)**
   - FrankaSim 是高精度物理仿真
   - Vision 观测需要更多显存
   - 保证训练稳定性

2. **动作分块减少 (5→1)**
   - PickCube 任务相对简单
   - 不需要长序列动作预测
   - 减少计算量

3. **去噪步数减少 (10→4)**
   - Flow Noise 方法效率更高
   - 较少步数即可收敛
   - 显著加快训练速度

4. **学习率保持较小**
   - 基于预训练模型微调
   - 防止灾难性遗忘
   - Critic 学习率更高以快速收敛

---

## 7. 使用建议

### 训练启动命令
```bash
# 单节点 4 卡训练
python -m rlinf.train \
    --config-name frankasim_ppo_openpi_pi05_custom \
    cluster.num_nodes=1 \
    cluster.component_placement.actor,env,rollout=0-3

# 调试模式 (单卡)
python -m rlinf.train \
    --config-name frankasim_ppo_openpi_pi05_local_debug
```

### 调参建议

1. **如果训练不稳定**:
   - 降低 `lr` (如 5e-6)
   - 增加 `clip_ratio_low/high` (如 0.3)
   - 增加 `entropy_bonus` (如 0.01)

2. **如果收敛太慢**:
   - 增加 `value_lr`
   - 增加 `update_epoch` (如 8)
   - 增加 `total_num_envs` (如显存允许)

3. **如果显存不足**:
   - 减少 `micro_batch_size`
   - 减少 `total_num_envs`
   - 开启 `gradient_checkpointing`

---

## 8. 参考配置文件

- `frankasim_ppo_mlp.yaml`: FrankaSim + MLP 策略基准
- `maniskill_ppo_openpi_pi05.yaml`: ManiSkill + PI05 参考
- `frankasim_ppo_openpi_pi05.yaml`: FrankaSim + PI05 原始配置
