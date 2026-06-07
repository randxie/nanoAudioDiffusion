#!/usr/bin/env bash
set -euo pipefail

export TORCH_DDP_BACKEND="${TORCH_DDP_BACKEND:-nccl}"

NUM_GPUS="${NUM_GPUS:-2}"
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus|--num_gpus|--num-gpus)
      if [[ $# -lt 2 ]]; then
        echo "$1 requires a value" >&2
        exit 2
      fi
      NUM_GPUS="$2"
      shift 2
      ;;
    --gpus=*|--num_gpus=*|--num-gpus=*)
      NUM_GPUS="${1#*=}"
      shift
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS/--gpus must be a positive integer, got: $NUM_GPUS" >&2
  exit 2
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="0"
  for ((i = 1; i < NUM_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES},${i}"
  done
fi
export CUDA_VISIBLE_DEVICES

if [[ -n "${BATCH_SIZE:-}" ]]; then
  extra_args=(--batch_size "$BATCH_SIZE" "${extra_args[@]}")
fi

.venv/bin/torchrun --standalone --nproc_per_node="$NUM_GPUS" train.py --config configs/ljspeech.yaml "${extra_args[@]}"
