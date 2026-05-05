#!/usr/bin/env bash
# qwen3-8b_rag_sft_phase1.sh — RAG SFT Phase 1: 纯 News LoRA 训练
#
# 受控对比实验: 与 TokenMem 相同数据/schedule/优化器, 仅注入通道不同。
# LoRA config: v_proj, r=16, alpha=16, dropout=0.0
#
# 复现:
#   bash scripts/qwen3-8b_rag_sft_phase1.sh
#   CUDA_VISIBLE_DEVICES=2 bash scripts/qwen3-8b_rag_sft_phase1.sh
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${NUM_GPUS:=1}"
: "${MAIN_PROCESS_PORT:=29503}"
export CUDA_VISIBLE_DEVICES

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
mkdir -p logs

python -m accelerate.commands.launch \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision bf16 \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    training/rag_sft.py \
    --model-name-or-path  hugglingface_model/qwen3-8B \
    --train-jsonl         data/news/train_cot.jsonl \
    --val-jsonl           data/news/val_cot.jsonl \
    --ckpt-dir            checkpoints/qwen3-8b_rag_sft_p1 \
    --prompt-mode         cot \
    --lora-rank           16 \
    --lora-alpha          16 \
    --lora-dropout        0.0 \
    --lora-target-modules v_proj \
    --epochs              40 \
    --batch-size          2 \
    --lr                  1e-3 \
    --weight-decay        0.0 \
    --grad-clip           0.0 \
    --grad-accum-steps    16 \
    --max-seq-len         1024 \
    --knowledge-max-len   256 \
    --save-steps          500 \
    --eval-steps          500 \
    --early-stop-patience 5 \
    --num-workers         4 \
    --swanlab-project     tokenmem \
    --knowledge-field     passage \
    2>&1 | tee "logs/qwen3-8b_rag_sft_phase1.log"
