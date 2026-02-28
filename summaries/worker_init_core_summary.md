# Worker init_worker 核心逻辑总结（Actor / Rollout / Env）

本文按“关注核心执行逻辑、弱化分布式/多进程封装”的原则，概括 `train_embodied_agent.py` 路径中 **Actor / Rollout / Env** 三类 worker 在 `init_worker()` 中真正做的事，并指出主要调用链与核心作用。

## Actor（EmbodiedFSDPActor / EmbodiedSACFSDPPolicy）

**入口**
- `EmbodiedRunner.init_workers()` 中调用 `actor.init_worker()`
- 具体类：
  - `EmbodiedFSDPActor`（默认）
  - `EmbodiedSACFSDPPolicy`（`loss_type == embodied_sac` 时）

**核心调用链与功能（更细化）**
1. `EmbodiedFSDPActor.init_worker()`
   - 作用：**初始化训练模型与优化器，准备权重同步拓扑**
   - 分工细化：聚焦“训练端”模型构建、训练态配置、参数/优化器状态的内存策略

2. `setup_model_and_optimizer()`（FSDPModelManager）
   - 作用：**构建模型、优化器、FSDP 分布式训练结构**
   - 分工细化：模型本体 + 优化器 + 分布式并行（FSDP）统一管理
  - 关键子调用：
    - `model_provider_func()`
      - 优先 `get_model(cfg.actor.model)`（具体模型实现）
      - 若 `cfg.runner.ckpt_path` 存在，`torch.load` + `load_state_dict`
      - 分工细化：把“模型结构来源”和“权重加载来源”统一封装

3. Offload 相关（可选）
   - `offload_param_and_grad()`
   - `offload_optimizer()`
   - 作用：**把参数/梯度/优化器状态转移到 CPU 以节省显存**
   - 分工细化：决定训练态内存布局（GPU/CPU），影响训练吞吐与显存占用

4. `self._setup_rollout_weight_dst_ranks()`
   - 作用：**计算 actor → rollout 的权重同步目的 rank**
   - 分工细化：把训练权重如何分发给 rollout 的路由规则固化（模版：按 modulo 分配）
   - 说明：这是训练-采样权重同步的核心拓扑准备

**核心执行对象（聚焦“真正任务执行”）**
- **模型实现本体**：`rlinf/models/embodiment/*`（由 `get_model(cfg.actor.model)` 决定）
- **训练逻辑**：由 `FSDPModelManager` + 模型 forward / loss 计算路径驱动
- **职责边界**：Actor 只负责“训练态”与“权重同步准备”，不负责采样和环境交互

---

## Rollout（MultiStepRolloutWorker）

**入口**
- `EmbodiedRunner.init_workers()` 中调用 `rollout.init_worker()`
- 具体类：`MultiStepRolloutWorker`（HuggingFace / BasePolicy 路径）

**核心调用链与功能（更细化）**
1. `MultiStepRolloutWorker.init_worker()`
   - 作用：**加载 rollout 模型、设置采样参数、可选加速（compile / cuda graph）**
   - 分工细化：只做“推理/采样态”准备，不负责训练态优化器

2. 构建 rollout 模型配置
   - 复制 `cfg.actor.model`，覆盖为 `cfg.rollout.model` 的 precision / path
   - 目的是让 rollout 侧模型与 actor 同架构但独立加载
   - 分工细化：确保“采样模型配置”与“训练模型配置”可以独立演化（精度/路径）

3. `get_model(rollout_model_config)`
   - 作用：**真正加载 rollout 模型实现**（通常为 BasePolicy 子类）
   - 若 `cfg.runner.ckpt_path` 存在，加载权重
   - 分工细化：rollout 侧的“可执行推理模型”载入点

4. `self.hf_model.eval()`
   - 作用：**采样模式（不训练）**
   - 分工细化：禁止训练态行为（如 dropout/bn 训练）

5. 可选加速
   - `enable_torch_compile` → `self.hf_model.enable_torch_compile(...)`
   - `enable_cuda_graph`（且未 offload）→ `self.hf_model.capture_cuda_graph(...)`
   - 分工细化：提升推理吞吐，属于采样侧性能优化

6. `setup_sample_params()`
   - 作用：**设置训练 / 评估的采样参数**（temperature / top-k / top-p / max_new_tokens）
   - 分工细化：将“采样策略”与模型解耦，便于实验配置驱动

7. `offload_model()`（可选）
   - 作用：**将模型 offload 到 CPU**
   - 分工细化：推理侧内存策略控制

**核心执行对象（聚焦“真正任务执行”）**
- **模型实现本体**：`rlinf/models/embodiment/*`（由 `get_model(rollout_model_config)` 决定）
- **采样核心路径**：`MultiStepRolloutWorker.predict()` 调用模型 `predict_action_batch` / `generate`
- **职责边界**：Rollout 只负责“采样/推理态”，不做训练优化器更新

---

## Env（LiberoEnv / OffScreenRenderEnv 栈）

**入口**  
- `EnvWorker.init_worker()` 中创建 `train_env_cls(...)` / `eval_env_cls(...)`  
- 当 env_type 为 `libero`，实际构造 `LiberoEnv`

**核心调用链与功能（更细化）**  
1. `LiberoEnv.__init__()`  
   - 作用：**RLinf-LIBERO 适配层**  
   - 分工细化：  
     - 生成 reset/task/trial 索引  
     - 构造 `env_fns`（每个 env 的 OffScreenRenderEnv 构造参数）  
     - 组织任务描述、观测包装、奖励与指标初始化  

2. `LiberoEnv._init_env()`  
   - 作用：**创建子环境进程池**  
   - 调用 `ReconfigureSubprocEnv(env_fns)`  

3. `OffScreenRenderEnv(**params)`  
   - 作用：**启用 offscreen 渲染的 ControlEnv**  
   - 由 `ControlEnv` 完成真正任务构造  

4. `ControlEnv.__init__()`  
   - 作用：**LIBERO 核心任务构造器**  
   - 分工细化：  
     - 解析 BDDL  
     - 根据 `TASK_MAPPING` 实例化具体任务类  
     - 注入控制器、相机、渲染、仿真参数  

5. `Libero_* Task` → `BDDLBaseDomain` → `SingleArmEnv` → `ManipulationEnv` → `RobotEnv` → `MujocoEnv`  
   - 作用：**具体任务逻辑 + robosuite 仿真栈**  
   - 分工细化：  
     - `Libero_*` / `BDDLBaseDomain`：物体/区域/任务成功判定  
     - `SingleArmEnv`：单臂约束与 EEF 接口  
     - `ManipulationEnv`：操作任务通用逻辑（抓取判定等）  
     - `RobotEnv`：机器人、控制器、相机 obs  
     - `MujocoEnv`：仿真初始化、reset/obs 管线  

**核心执行对象（聚焦“真正任务执行”）**  
- **任务执行核心**：`OffScreenRenderEnv` → `ControlEnv` → `Libero_*` → `BDDLBaseDomain`  
- **仿真核心**：`MujocoEnv`（含 reset / obs / sim 初始化）  

---

## 总结（只保留核心执行层）

- **Actor**：加载训练模型 + 初始化优化器 + 准备权重同步拓扑。核心执行体是模型实现与训练更新逻辑。
- **Rollout**：加载采样模型 + 设置采样参数 + 可选 compile/cuda graph。核心执行体是模型推理与采样。
- **Env**：构造 OffScreenRenderEnv 栈并进入 robosuite 任务执行链。核心执行体是 BDDL 任务逻辑与 Mujoco 仿真。

---

## 核心类调用关系图（init_worker 执行路径）

```mermaid
flowchart TD
  %% Actor
  A1[EmbodiedRunner.init_workers] --> A2[EmbodiedFSDPActor.init_worker]
  A2 --> A3[setup_model_and_optimizer]
  A3 --> A4[get_model(cfg.actor.model)]
  A2 --> A5[offload_param_and_grad / offload_optimizer]
  A2 --> A6[setup_rollout_weight_dst_ranks]

  %% Rollout
  A1 --> R1[MultiStepRolloutWorker.init_worker]
  R1 --> R2[get_model(rollout_model_config)]
  R1 --> R3[setup_sample_params]
  R1 --> R4[enable_torch_compile / capture_cuda_graph]
  R1 --> R5[offload_model]

  %% Env
  A1 --> E1[EnvWorker.init_worker]
  E1 --> E2[LiberoEnv.__init__]
  E2 --> E3[LiberoEnv._init_env]
  E3 --> E4[ReconfigureSubprocEnv]
  E4 --> E5[OffScreenRenderEnv]
  E5 --> E6[ControlEnv]
  E6 --> E7[Libero_* Task]
  E7 --> E8[BDDLBaseDomain]
  E8 --> E9[SingleArmEnv]
  E9 --> E10[ManipulationEnv]
  E10 --> E11[RobotEnv]
  E11 --> E12[MujocoEnv]
```
