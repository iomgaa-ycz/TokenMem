#!/bin/bash
# qwen3-8b — RAG SFT CoT 评测 (LoRA + 256-token in-context knowledge)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
N_SAMPLES=${N_SAMPLES:--1}

if [ ! -d "checkpoints/qwen3-8b_rag_sft_p2/best" ]; then
    echo "ERROR: RAG SFT Phase 2 best 不存在"
    exit 1
fi

COMMON="python -m evaluation.eval_rag_sft \
    --model-path hugglingface_model/qwen3-8B \
    --lora-dir checkpoints/qwen3-8b_rag_sft_p2/best \
    --output-dir results/rag_sft \
    --knowledge-max-len 256 \
    --cot-max-new-tokens 2048 \
    --batch-size 4 \
    --n-samples $N_SAMPLES"

echo "=== qwen3-8b / rag_sft (CoT + LoRA + in-context 256tok) ==="
$COMMON --dataset cf_arc_easy_val --data-dir data/counterfactual
$COMMON --dataset cf_medqa_val    --data-dir data/counterfactual
$COMMON --dataset medqa        --data-dir data/ood
$COMMON --dataset arc          --data-dir data/ood
$COMMON --dataset mmlu         --data-dir data/ood
$COMMON --dataset news         --data-dir data/news
$COMMON --dataset arc_easy     --data-dir data/ood

echo "=== qwen3-8b / rag_sft 完成 ==="
