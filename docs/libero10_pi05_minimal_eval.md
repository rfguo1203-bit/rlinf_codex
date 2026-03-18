# LIBERO-10 pi0.5 Minimal Eval

最小单机单卡评测入口：

```bash
python scripts/simple_eval_libero10_pi05.py \
  --model-path /path/to/RLinf-Pi05-SFT \
  --task-id 0 \
  --save-video-ratio 0.2 \
  --device cuda:0
```

说明：

- `--task-id`：指定 LIBERO-10 中要评测的任务 id。
- 默认会评测该任务的全部固定 reset states；如果需要只跑前 N 个，可以加 `--num-episodes N`。
- `--save-video-ratio 0.2`：按比例保存 20% rollout 视频，视频索引均匀采样。
- `--shuffle-reset-states`：如果需要，可以先打乱该 task 下的固定 reset states，再截取前 N 个。

这个脚本复用了正式评测链路的核心设置：

- 单进程、单卡推理，不启动 Ray worker。
- 使用 LIBERO 固定 reset states 做评测。
- `ignore_terminations=True`，单条 rollout 默认跑满 `max_episode_steps=480`。
- 输出 `success_once`、`success_at_end`、`return`、`episode_len`。

输出位置默认在：

```text
logs/simple_eval_libero10_pi05/task_<task_id>-<timestamp>/
```

其中包含：

- `summary.txt`：汇总指标。
- `metrics.json`：完整评测结果。
- `episodes.json`：逐条 rollout 结果。
- `saved_videos.json`：保存下来的视频清单。
- `videos/`：按比例保存的视频文件。
