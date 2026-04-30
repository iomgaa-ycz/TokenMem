#!/usr/bin/env bash
# generate_cot.sh — 使用 DeepSeek API 生成 CoT 训练数据
#
# 分别处理 news / cf_arc_easy / cf_medqa 三类数据。
# 支持断点续传：中断后重新运行会跳过已完成的样本。
#
# 用法:
#   bash scripts/generate_cot.sh              # 生成全部
#   bash scripts/generate_cot.sh news         # 仅 news
#   bash scripts/generate_cot.sh cf           # 仅 CF
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

: "${CONCURRENCY:=32}"
TARGET="${1:-all}"

run_gen() {
    local input="$1" output="$2" dtype="$3" split="${4:-}"
    echo "=== 生成 CoT: ${output} (type=${dtype}, split=${split:-all}) ==="
    local split_arg=""
    if [ -n "$split" ]; then
        split_arg="--split ${split}"
    fi
    conda run -n ExplicitLLM python -m tools.generate_cot_data \
        --input "$input" \
        --output "$output" \
        --data-type "$dtype" \
        --concurrency "$CONCURRENCY" \
        $split_arg
}

# --- News ---
if [ "$TARGET" = "all" ] || [ "$TARGET" = "news" ]; then
    run_gen data/news/train.jsonl     data/news/train_cot.jsonl     news
    run_gen data/news/val.jsonl       data/news/val_cot.jsonl       news
fi

# --- CF ---
if [ "$TARGET" = "all" ] || [ "$TARGET" = "cf" ]; then
    run_gen data/counterfactual/arc_easy.jsonl  data/counterfactual/arc_easy_cot.jsonl  cf train
    run_gen data/counterfactual/medqa.jsonl     data/counterfactual/medqa_cot.jsonl     cf train
    # CF val (test split)
    run_gen data/counterfactual/arc_easy.jsonl  data/counterfactual/arc_easy_cot_val.jsonl  cf test
    run_gen data/counterfactual/medqa.jsonl     data/counterfactual/medqa_cot_val.jsonl     cf test
fi

echo "=== 全部完成 ==="
