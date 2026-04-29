#!/usr/bin/env bash
# qwen3-4b_sft_phase2.sh — Qwen3-4B Curriculum Phase 2: News + CF SFT
#
# Phase 2: 加载 Phase 1 best gate，混入反事实数据继续训练。
# optimizer + scheduler 全部重建（独立 warmup）。
# 训练参数来自 DecoupledRAG (Lamb lr=1e-3, LinearLR warmup=10)
#
# 前置条件: Phase 1 完成，checkpoints/qwen3-4b_sft_p1/best/ 存在
#
# 复现:
#   bash scripts/qwen3-4b_sft_phase2.sh
#   CUDA_VISIBLE_DEVICES=2 bash scripts/qwen3-4b_sft_phase2.sh
#   CUDA_VISIBLE_DEVICES=2,3 NUM_GPUS=2 bash scripts/qwen3-4b_sft_phase2.sh
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${NUM_GPUS:=2}"
: "${MAIN_PROCESS_PORT:=29502}"
export CUDA_VISIBLE_DEVICES

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
mkdir -p logs

# 前置检查: Phase 1 best checkpoint 必须存在
if [ ! -f "checkpoints/qwen3-4b_sft_p1/best/meta.json" ]; then
    echo "ERROR: Phase 1 best checkpoint 不存在，请先运行 scripts/qwen3-4b_sft_phase1.sh"
    exit 1
fi

python -m accelerate.commands.launch \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision bf16 \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    training/sft.py \
    --model-name-or-path  hugglingface_model/qwen3-4B \
    --load-gates          checkpoints/qwen3-4b_sft_p1/best \
    --train-jsonl         data/news/train.jsonl \
    --cf-train-jsonl      data/counterfactual/arc_easy.jsonl data/counterfactual/medqa.jsonl \
    --cf-oversample       2 \
    --val-jsonl           data/news/val.jsonl \
    --cf-val-jsonl        data/counterfactual/arc_easy.jsonl data/counterfactual/medqa.jsonl \
    --ckpt-dir            checkpoints/qwen3-4b_sft_p2 \
    --epochs              5 \
    --batch-size          4 \
    --lr                  1e-3 \
    --weight-decay        0.0 \
    --grad-clip           0.0 \
    --grad-accum-steps    8 \
    --max-seq-len         512 \
    --knowledge-max-len   256 \
    --knowledge-strided-len 64 \
    --save-steps          500 \
    --eval-steps          500 \
    --num-workers         4 \
    --swanlab-project     tokenmem \
    --knowledge-field     passage \
    2>&1 | tee "logs/qwen3-4b_sft_phase2.log"
