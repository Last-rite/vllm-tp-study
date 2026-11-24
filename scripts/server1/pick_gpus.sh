#!/usr/bin/env bash
# Print the indices of RTX 4090s that are idle (<1 GB used, <10% util).
#   source scripts/server1/pick_gpus.sh 4     -> exports CUDA_VISIBLE_DEVICES or fails
N=${1:-1}
IDLE=$(nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits \
  | awk -F', ' '$2 ~ /4090/ && $3 < 1000 && $4 < 10 {print $1}' | head -n "$N" | paste -sd, -)
CNT=$(echo "$IDLE" | tr ',' '\n' | grep -c . || true)
if [ "$CNT" -lt "$N" ]; then
  echo "only $CNT idle RTX 4090 (need $N): '$IDLE'"; nvidia-smi; return 1 2>/dev/null || exit 1
fi
export CUDA_VISIBLE_DEVICES=$IDLE
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
