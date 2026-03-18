# LIBERO-10 pi0.5 Minimal Eval

最小单机单卡评测入口：

```bash
bash examples/embodiment/eval_libero10_pi05_minimal.sh 200 0.2 0
```

含义：

- `200`：总共评测 200 个 LIBERO-10 固定初始状态
- `0.2`：保存 20% 的 rollout 视频，即 40 个视频
- `0`：使用第 0 张 GPU

脚本会复用 `examples/embodiment/config/libero_10_ppo_openpi_pi05.yaml` 里的 OpenPI pi0.5 配置，并读取其中的 `model_path`、`action_chunk`、`num_steps`。

输出位置默认在：

```text
logs/libero10_pi05_minimal/<exp_name>/
```

其中包含：

- `summary.json`：汇总 `success_once`
- `videos/`：按比例保存的视频
- `<exp_name>.log`：完整评测日志
