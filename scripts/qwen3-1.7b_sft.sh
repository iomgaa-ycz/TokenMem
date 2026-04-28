#!/usr/bin/env bash
# qwen3-1.7b_sft.sh — Qwen3-1.7B News SFT (LinearFusion gates only)
#
# 训练参数来自 DecoupledRAG (Lamb lr=1e-3, LinearLR warmup=10)
# 可训练参数: 仅 gate_crossattention (LinearFusion W_A + W_B), ~1.84M trainable params
# 基座 LLM 全部冻结 (1.7B params)
#
# 复现:
#   bash scripts/qwen3-1.7b_sft.sh
#   CUDA_VISIBLE_DEVICES=2 bash scripts/qwen3-1.7b_sft.sh
#   CUDA_VISIBLE_DEVICES=2,3 NUM_GPUS=2 bash scripts/qwen3-1.7b_sft.sh
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${NUM_GPUS:=1}"
: "${MAIN_PROCESS_PORT:=29501}"
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
    --model-name-or-path  hugglingface_model/qwen3-1.7B \
    --train-jsonl         data/news/train.jsonl \
    --val-jsonl           data/news/val.jsonl \
    --ckpt-dir            checkpoints/qwen3-1.7b_sft \
    --epochs              5 \
    --batch-size          16 \
    --lr                  1e-3 \
    --weight-decay        0.0 \
    --grad-clip           0.0 \
    --grad-accum-steps    1 \
    --max-seq-len         64 \
    --knowledge-max-len   256 \
    --knowledge-strided-len 64 \
    --save-steps          500 \
    --eval-steps          500 \
    --num-workers         4 \
    --swanlab-project     tokenmem \
    --knowledge-field     passage \
    2>&1 | tee "logs/qwen3-1.7b_sft.log"
