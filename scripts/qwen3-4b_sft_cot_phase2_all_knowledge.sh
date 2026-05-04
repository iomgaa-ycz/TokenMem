#!/usr/bin/env bash
# qwen3-4b_sft_cot_phase2.sh — Qwen3-4B CoT Curriculum Phase 2: News + CF SFT
#
# 加载 v1 Phase 1 best gate，混入 CoT 格式的反事实数据继续训练。
# 与 v1 Phase 2 完全相同配置 (batch=2, grad_accum=16)，仅增加轮次 + 早停。
#
# 前置条件:
#   1. v1 Phase 1 完成: checkpoints/qwen3-4b_sft_cot_p1_v1/best/ 存在
#   2. CF CoT 数据已生成: data/counterfactual/*_cot.jsonl
#
# 复现:
#   bash scripts/qwen3-4b_sft_cot_phase2.sh
#   CUDA_VISIBLE_DEVICES=2,3 NUM_GPUS=2 bash scripts/qwen3-4b_sft_cot_phase2.sh
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${NUM_GPUS:=2}"
: "${MAIN_PROCESS_PORT:=29502}"
export CUDA_VISIBLE_DEVICES

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
mkdir -p logs

# 前置检查
if [ ! -f "checkpoints/qwen3-4b_sft_cot_all_knowledge_p1/best/meta.json" ]; then
    echo "ERROR: v1 Phase 1 best checkpoint 不存在"
    exit 1
fi

python -m accelerate.commands.launch \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision bf16 \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    training/sft.py \
    --model-name-or-path  hugglingface_model/qwen3-4B \
    --load-gates          checkpoints/qwen3-4b_sft_cot_all_knowledge_p1/best \
    --train-jsonl         data/news/train_cot.jsonl \
    --cf-train-jsonl      data/counterfactual/arc_easy_cot.jsonl data/counterfactual/medqa_cot.jsonl \
    --cf-oversample       2 \
    --val-jsonl           data/news/val_cot.jsonl \
    --cf-val-jsonl        data/counterfactual/arc_easy_cot_val.jsonl data/counterfactual/medqa_cot_val.jsonl \
    --ckpt-dir            checkpoints/qwen3-4b_sft_cot_all_knowledge_p2 \
    --prompt-mode         cot \
    --epochs              40 \
    --batch-size          4 \
    --lr                  1e-3 \
    --weight-decay        0.0 \
    --grad-clip           0.0 \
    --grad-accum-steps    16 \
    --max-seq-len         1024 \
    --knowledge-max-len   256 \
    --knowledge-strided-len 256 \
    --save-steps          500 \
    --eval-steps          500 \
    --early-stop-patience 5 \
    --num-workers         4 \
    --swanlab-project     tokenmem \
    --knowledge-field     passage \
    2>&1 | tee "logs/qwen3-4b_sft_cot_phase2.log"
