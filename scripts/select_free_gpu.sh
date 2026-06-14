#!/usr/bin/env bash
NUM_GPUS="${1:-1}"
GPU_LIST=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
  | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1" "$2}' \
  | sort -k2 -nr \
  | head -n "$NUM_GPUS" \
  | awk '{print $1}' \
  | paste -sd, -)
if [ -z "$GPU_LIST" ]; then
  echo "WARNING: No GPU found by nvidia-smi."
else
  export CUDA_VISIBLE_DEVICES="$GPU_LIST"
  echo "Selected CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
