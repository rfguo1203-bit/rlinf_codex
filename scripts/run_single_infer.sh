cd /Users/rkos/Workspace/RLinf

python scripts/infer_libero10_pi05_single_task.py --list-tasks

python scripts/infer_libero10_pi05_single_task.py \
  --task-id 0 \
  --model-path /你的/pi05模型目录 \
  --output-dir /Users/rkos/Workspace/RLinf/results/libero10_pi05_demo \
  --num-episodes 1
