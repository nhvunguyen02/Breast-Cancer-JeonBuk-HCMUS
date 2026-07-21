#!/usr/bin/env bash

ROOT=/mnt/hcmus/breast_vn/code/check_some_phase/task1/improve_hierarchy
RUN="${ROOT}/outputs/task1_improve_20260720_093547"
PYTHON=/mnt/hcmus/breast_vn/miniconda3/envs/tnmammo/bin/python

"${PYTHON}" "${ROOT}/watch_improve_status.py" "${RUN}"

echo
echo "===== ACTIVE TRAINING PROCESS ====="

TRAIN_PID="$(
  pgrep -f \
    "${ROOT}/.*train.*\.py.*${RUN}" \
  | head -n 1
)"

if [[ -n "${TRAIN_PID}" ]]; then
  ps -o \
    pid,ppid,stat,etime,%cpu,%mem,rss,args \
    -p "${TRAIN_PID}" \
    2>/dev/null || true
else
  echo "NO_ACTIVE_TRAINING_PROCESS"
fi

echo
echo "===== GPU 0 ====="

nvidia-smi -i 0 \
  --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,noheader \
  2>/dev/null || true
