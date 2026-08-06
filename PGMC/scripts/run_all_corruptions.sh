#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <modelnet_c|sonn_c> <backbone> <point_ckpt> <text_ckpt> <benchmark_root> <source_root> [extra args...]"
  exit 2
fi

benchmark=$1
backbone=$2
point_ckpt=$3
text_ckpt=$4
benchmark_root=$5
source_root=$6
shift 6

python main.py \
  --benchmark "${benchmark}" \
  --backbone "${backbone}" \
  --checkpoint "${point_ckpt}" \
  --text-checkpoint "${text_ckpt}" \
  --data-root "${benchmark_root}" \
  --source-root "${source_root}" \
  --all-corruptions \
  "$@"
