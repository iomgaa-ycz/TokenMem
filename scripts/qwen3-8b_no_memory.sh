#!/bin/bash
# qwen3-8b — no_memory CoT 评测 (nothink + batch)
# 用法：
#   bash scripts/qwen3-8b_no_memory.sh                          # 默认 GPU 0，全量
#   CUDA_VISIBLE_DEVICES=2 bash scripts/qwen3-8b_no_memory.sh   # 指定 GPU
#   N_SAMPLES=10 bash scripts/qwen3-8b_no_memory.sh             # smoke test
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
N_SAMPLES=${N_SAMPLES:--1}

COMMON="python -m evaluation.eval_baseline \
    --model-path hugglingface_model/qwen3-8B \
    --method no_memory \
    --output-dir results/baseline \
    --cot-max-new-tokens 2048 \
    --batch-size 4 \
    --n-samples $N_SAMPLES"

echo "=== qwen3-8b / no_memory (CoT) ==="
$COMMON --dataset medqa        --data-dir data/ood
$COMMON --dataset arc          --data-dir data/ood
$COMMON --dataset mmlu         --data-dir data/ood
$COMMON --dataset news         --data-dir data/news
$COMMON --dataset arc_easy     --data-dir data/ood
$COMMON --dataset cf_arc_easy_val --data-dir data/counterfactual
$COMMON --dataset cf_medqa_val    --data-dir data/counterfactual

echo "=== qwen3-8b / no_memory 完成 ==="
