#!/bin/bash
# qwen3-0.6b — vanilla_rag baseline 评测
# 用法：
#   bash scripts/qwen3-0.6b_vanilla_rag.sh                          # 默认 GPU 0，全量
#   CUDA_VISIBLE_DEVICES=2 bash scripts/qwen3-0.6b_vanilla_rag.sh   # 指定 GPU
#   N_SAMPLES=10 bash scripts/qwen3-0.6b_vanilla_rag.sh             # smoke test
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
N_SAMPLES=${N_SAMPLES:--1}

COMMON="python -m evaluation.eval_baseline \
    --model-path hugglingface_model/qwen3-0.6B \
    --method vanilla_rag \
    --output-dir results/baseline \
    --n-samples $N_SAMPLES"

echo "=== qwen3-0.6b / vanilla_rag ==="
$COMMON --dataset medqa   --data-dir data/ood
$COMMON --dataset arc     --data-dir data/ood
$COMMON --dataset mmlu    --data-dir data/ood

# News（数据集就绪后取消注释）
# $COMMON --dataset news --data-dir data/news

echo "=== qwen3-0.6b / vanilla_rag 完成 ==="
