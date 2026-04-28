#!/bin/bash
# qwen3-4b — TokenMem Oracle 评测（4 数据集）
# 用法：
#   bash scripts/qwen3-4b_tokenmem.sh                          # 默认 GPU 0，全量
#   CUDA_VISIBLE_DEVICES=2 bash scripts/qwen3-4b_tokenmem.sh   # 指定 GPU
#   N_SAMPLES=10 bash scripts/qwen3-4b_tokenmem.sh             # smoke test
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
N_SAMPLES=${N_SAMPLES:--1}

COMMON="python -m evaluation.eval_tokenmem \
    --model-path hugglingface_model/qwen3-4B \
    --gate-dir checkpoints/qwen3-4b_sft/best \
    --output-dir results/tokenmem \
    --knowledge-max-len 256 \
    --n-samples $N_SAMPLES"

echo "=== qwen3-4b / tokenmem (4 datasets) ==="
$COMMON --dataset news   --data-dir data/news
$COMMON --dataset medqa  --data-dir data/ood
$COMMON --dataset arc    --data-dir data/ood
$COMMON --dataset mmlu   --data-dir data/ood
echo "=== qwen3-4b / tokenmem 完成 ==="
