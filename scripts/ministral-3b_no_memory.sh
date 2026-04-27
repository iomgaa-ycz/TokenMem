#!/bin/bash
# ministral-3b — no_memory baseline 评测
# 用法：
#   bash scripts/ministral-3b_no_memory.sh                          # 默认 GPU 0，全量
#   CUDA_VISIBLE_DEVICES=2 bash scripts/ministral-3b_no_memory.sh   # 指定 GPU
#   N_SAMPLES=10 bash scripts/ministral-3b_no_memory.sh             # smoke test
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
N_SAMPLES=${N_SAMPLES:--1}

COMMON="python -m evaluation.eval_baseline \
    --model-path hugglingface_model/ministral-3-3b \
    --method no_memory \
    --output-dir results/baseline \
    --n-samples $N_SAMPLES"

echo "=== ministral-3b / no_memory ==="
$COMMON --dataset medqa   --data-dir data/ood
$COMMON --dataset arc     --data-dir data/ood
$COMMON --dataset mmlu    --data-dir data/ood

# News（数据集就绪后取消注释）
# $COMMON --dataset news --data-dir data/news

echo "=== ministral-3b / no_memory 完成 ==="
