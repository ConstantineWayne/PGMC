#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 8 ]]; then
  echo "Usage: $0 <benchmark> <backbone> <point_ckpt> <text_ckpt> <benchmark_root> <source_root> <corruption_2> <attack> [extra args...]"
  exit 2
fi

benchmark=$1
backbone=$2
point_ckpt=$3
text_ckpt=$4
benchmark_root=$5
source_root=$6
corruption=$7
attack=$8
shift 8

python main.py \
  --benchmark "${benchmark}" \
  --backbone "${backbone}" \
  --checkpoint "${point_ckpt}" \
  --text-checkpoint "${text_ckpt}" \
  --data-root "${benchmark_root}" \
  --source-root "${source_root}" \
  --corruption "${corruption}" \
  --attack "${attack}" \
  "$@"
