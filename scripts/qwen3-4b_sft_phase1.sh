#!/usr/bin/env bash
# qwen3-4b_sft_phase1.sh — Qwen3-4B Curriculum Phase 1: 纯 News SFT
#
# Phase 1: 纯 news 数据训练 gate 到稳定，为 Phase 2 混入 CF 数据做准备。
# 训练参数来自 DecoupledRAG (Lamb lr=1e-3, LinearLR warmup=10)
# 可训练参数: 仅 gate_crossattention (LinearFusion W_A + W_B), ~2.95M
#
# 复现:
#   bash scripts/qwen3-4b_sft_phase1.sh
#   CUDA_VISIBLE_DEVICES=2 bash scripts/qwen3-4b_sft_phase1.sh
#   CUDA_VISIBLE_DEVICES=2,3 NUM_GPUS=2 bash scripts/qwen3-4b_sft_phase1.sh
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${NUM_GPUS:=2}"
: "${MAIN_PROCESS_PORT:=29502}"
export CUDA_VISIBLE_DEVICES

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
mkdir -p logs

python -m accelerate.commands.launch \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision bf16 \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    training/sft.py \
    --model-name-or-path  hugglingface_model/qwen3-4B \
    --train-jsonl         data/news/train.jsonl \
    --val-jsonl           data/news/val.jsonl \
    --ckpt-dir            checkpoints/qwen3-4b_sft_p1 \
    --epochs              3 \
    --batch-size          32 \
    --lr                  1e-3 \
    --weight-decay        0.0 \
    --grad-clip           0.0 \
    --grad-accum-steps    1 \
    --max-seq-len         512 \
    --knowledge-max-len   256 \
    --knowledge-strided-len 64 \
    --save-steps          500 \
    --eval-steps          500 \
    --num-workers         4 \
    --swanlab-project     tokenmem \
    --knowledge-field     passage \
    2>&1 | tee "logs/qwen3-4b_sft_phase1.log"
