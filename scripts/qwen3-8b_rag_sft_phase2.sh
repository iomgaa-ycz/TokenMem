#!/usr/bin/env bash
# qwen3-8b_rag_sft_phase2.sh — RAG SFT Phase 2: News + CF Curriculum
#
# 加载 Phase 1 best LoRA，混入反事实数据继续训练。
#
# 前置条件:
#   1. Phase 1 完成: checkpoints/qwen3-8b_rag_sft_p1/best/ 存在
#   2. CF CoT 数据: data/counterfactual/*_cot.jsonl
#
# 复现:
#   bash scripts/qwen3-8b_rag_sft_phase2.sh
#   CUDA_VISIBLE_DEVICES=2 bash scripts/qwen3-8b_rag_sft_phase2.sh
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${NUM_GPUS:=1}"
: "${MAIN_PROCESS_PORT:=29503}"
export CUDA_VISIBLE_DEVICES

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
mkdir -p logs

# 前置检查
if [ ! -d "checkpoints/qwen3-8b_rag_sft_p1/best" ]; then
    echo "ERROR: Phase 1 best LoRA 不存在，请先运行 scripts/qwen3-8b_rag_sft_phase1.sh"
    exit 1
fi

python -m accelerate.commands.launch \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision bf16 \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    training/rag_sft.py \
    --model-name-or-path  hugglingface_model/qwen3-8B \
    --load-lora           checkpoints/qwen3-8b_rag_sft_p1/best \
    --train-jsonl         data/news/train_cot.jsonl \
    --cf-train-jsonl      data/counterfactual/arc_easy_cot.jsonl data/counterfactual/medqa_cot.jsonl \
    --cf-oversample       2 \
    --val-jsonl           data/news/val_cot.jsonl \
    --cf-val-jsonl        data/counterfactual/arc_easy_cot_val.jsonl data/counterfactual/medqa_cot_val.jsonl \
    --ckpt-dir            checkpoints/qwen3-8b_rag_sft_p2 \
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
    2>&1 | tee "logs/qwen3-8b_rag_sft_phase2.log"
