# RLinf 项目导读（面向二次开发与自定义训练）

## 你的当前目标
- 看懂 RLinf 的功能和代码架构。
- 基于 RLinf 训练你自己的模型（优先走 embodied/VLA 训练链路，再扩展到算法或环境）。

## 这个库是做什么的
RLinf 是一个面向 Embodied AI 和 Agentic AI 的强化学习基础设施。核心定位不是单一算法实现，而是把分布式训练中的“调度、并行、工程复杂度”抽象掉，让你可以主要通过配置切换：
- 算法（PPO/GRPO/SAC/CrossQ 等）
- 模型（OpenVLA、pi 系列、GR00T、自定义策略等）
- 环境（ManiSkill、LIBERO、IsaacLab、CALVIN、RoboCasa、真实机器人等）
- 执行后端（FSDP 或 Megatron；SGLang 或 vLLM）

一句话：RLinf 更像“可扩展的 RL 训练操作系统”，而不是一组孤立脚本。

## 核心能力（从 README 与 AGENTS.md 汇总）
- 分布式训练：基于 Ray 管理进程/节点，支持单机与多机。
- 配置驱动：基于 Hydra，把训练拓扑、算法、模型、环境统一在 YAML 中。
- 多场景覆盖：embodied、reasoning、agentic（含 online RL）。
- 高扩展性：新算法、新模型、新环境都有明确注册与接入路径。
- 完整工程链路：含 examples、tests（unit + e2e）、docs、docker、install 脚本。

## 代码架构（先看这几个目录）
- `rlinf/config.py`：配置构建与校验；模型/环境类型白名单。
- `rlinf/runners/`：训练主循环入口（rollout -> reward -> advantage -> update）。
- `rlinf/workers/`：Actor/Rollout/Env/Reward 等 Ray Worker 实现。
- `rlinf/scheduler/`：集群、worker group、placement（组件部署策略）。
- `rlinf/algorithms/`：advantage/loss/reward 注册与实现。
- `rlinf/models/`：模型侧封装（embodiment、reasoning）。
- `rlinf/envs/`：环境接入与动作处理。
- `examples/`：可直接运行的入口脚本和 YAML 配置（最重要的学习起点）。
- `tests/`：单测与端到端测试。

## 训练是怎么跑起来的
1. 启动 Ray（单机或多机）。
2. 运行某个 entry script（如 `examples/embodiment/train_embodied_agent.py`）。
3. 构建 Cluster，并按 `cluster.component_placement` 计算组件部署。
4. 启动各类 Worker 组（actor/rollout/env/reward/...）。
5. Runner 驱动循环：rollout -> reward -> adv/returns -> actor update -> logging/checkpoint/eval。

关键点：
- 单机通常 `cluster.num_nodes: 1`。
- 多机必须在 `ray start` 前设置 `RLINF_NODE_RANK`（每台唯一）。

## 你要“训练自己的模型”时的最短路径
1. 先跑通官方 embodied 示例（建议 ManiSkill + 一个现成 config），确认环境与依赖无误。
2. 选择接入方式：
   - 复用现有模型结构改权重/配置（最快）。
   - 新增模型类型（在 `SupportedModel` 注册 + `rlinf/models/embodiment/<your_model>/` 实现）。
3. 打通训练链路：
   - 配置层：`model.model_type`、`algorithm.*`、`env.*`、`cluster.*`。
   - worker 层：确保 actor/rollout 根据 model_type 能正确实例化你的模型。
4. 增加最小可验证测试：
   - 单测（注册与前向关键路径）
   - 至少一个 e2e/小规模 smoke 配置
5. 再做性能与稳定性优化（batch、offload、checkpointing、placement）。

## 推荐阅读顺序（按你当前目标）
1. `README.md`（能力边界与示例总览）
2. `examples/embodiment/`（先从可运行配置理解参数组织）
3. `rlinf/config.py`（看配置如何被校验与展开）
4. `rlinf/runners/` + `rlinf/workers/`（理解训练主链路）
5. `rlinf/models/embodiment/`（找一个你最接近的模型实现作为模板）
6. `rlinf/algorithms/` 与 `rlinf/envs/`（扩展算法/环境时再深入）

## 常见踩坑（你后续大概率会遇到）
- 多机 Ray 环境变量顺序错误：`RLINF_NODE_RANK` 必须先于 `ray start`。
- OOM：需同时调 env、rollout、actor 三侧参数，不能只降 batch。
- 配置与代码分叉：尽量通过 YAML 控制行为，不要在代码里覆盖用户配置字段。
- 新增能力没补测试与文档：仓库规范要求用户可见变更配套测试/文档。

---

本文件是学习与开发导读，不替代 `AGENTS.md` / `README.md` / `CONTRIBUTING.md` 的原始规范。
