#!/usr/bin/env bash
# qwen3-14b_sft_cot_phase1.sh — Qwen3-14B CoT Curriculum Phase 1: News SFT
#
# 前置条件: data/news/train_cot.jsonl 已生成
#
# 复现:
#   bash scripts/qwen3-14b_sft_cot_phase1.sh
#   CUDA_VISIBLE_DEVICES=2,3 NUM_GPUS=2 bash scripts/qwen3-14b_sft_cot_phase1.sh
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${NUM_GPUS:=2}"
: "${MAIN_PROCESS_PORT:=29503}"
export CUDA_VISIBLE_DEVICES

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
mkdir -p logs

# 前置检查: CoT 数据必须存在
if [ ! -f "data/news/train_cot.jsonl" ]; then
    echo "ERROR: CoT 训练数据不存在，请先运行 bash scripts/generate_cot.sh news"
    exit 1
fi

python -m accelerate.commands.launch \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision bf16 \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    training/sft.py \
    --model-name-or-path  hugglingface_model/qwen3-14B \
    --train-jsonl         data/news/train_cot.jsonl \
    --val-jsonl           data/news/val_cot.jsonl \
    --ckpt-dir            checkpoints/qwen3-14b_sft_cot_p1 \
    --prompt-mode         cot \
    --epochs              5 \
    --batch-size          2 \
    --lr                  1e-3 \
    --weight-decay        0.0 \
    --grad-clip           0.0 \
    --grad-accum-steps    16 \
    --max-seq-len         1024 \
    --knowledge-max-len   256 \
    --knowledge-strided-len 64 \
    --save-steps          500 \
    --eval-steps          500 \
    --num-workers         0 \
    --swanlab-project     tokenmem \
    --knowledge-field     passage \
    2>&1 | tee "logs/qwen3-14b_sft_cot_phase1.log"
