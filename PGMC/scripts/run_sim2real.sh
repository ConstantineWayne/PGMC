#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <backbone> <point_ckpt> <text_ckpt> <sim2real_root> <source_root> <sim2real_type> [extra args...]"
  exit 2
fi

backbone=$1
point_ckpt=$2
text_ckpt=$3
sim2real_root=$4
source_root=$5
sim2real_type=$6
shift 6

python main.py \
  --benchmark sim2real \
  --backbone "${backbone}" \
  --checkpoint "${point_ckpt}" \
  --text-checkpoint "${text_ckpt}" \
  --data-root "${sim2real_root}" \
  --source-root "${source_root}" \
  --sim2real-type "${sim2real_type}" \
  "$@"
