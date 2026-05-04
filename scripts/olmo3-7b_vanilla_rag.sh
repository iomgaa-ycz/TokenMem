#!/bin/bash
# olmo3-7b — vanilla_rag CoT 评测 (LLMLingua-2 压缩 + batch)
# 用法：
#   bash scripts/olmo3-7b_vanilla_rag.sh                          # 默认 GPU 0，全量
#   CUDA_VISIBLE_DEVICES=2 bash scripts/olmo3-7b_vanilla_rag.sh   # 指定 GPU
#   N_SAMPLES=10 bash scripts/olmo3-7b_vanilla_rag.sh             # smoke test
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
N_SAMPLES=${N_SAMPLES:--1}

COMMON="python -m evaluation.eval_baseline \
    --model-path hugglingface_model/Olmo-3-7B-Instruct \
    --method vanilla_rag \
    --output-dir results/baseline \
    --compress-target-token 64 \
    --cot-max-new-tokens 1024 \
    --batch-size 16 \
    --n-samples $N_SAMPLES"

echo "=== olmo3-7b / vanilla_rag (CoT + LLMLingua-2) ==="
$COMMON --dataset medqa        --data-dir data/ood
$COMMON --dataset arc          --data-dir data/ood
$COMMON --dataset mmlu         --data-dir data/ood
$COMMON --dataset news         --data-dir data/news
$COMMON --dataset arc_easy     --data-dir data/ood
$COMMON --dataset cf_arc_easy_val --data-dir data/counterfactual
$COMMON --dataset cf_medqa_val    --data-dir data/counterfactual

echo "=== olmo3-7b / vanilla_rag 完成 ==="
