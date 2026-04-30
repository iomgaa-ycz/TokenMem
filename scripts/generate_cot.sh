#!/usr/bin/env bash
# generate_cot.sh — 使用 DeepSeek API 生成 CoT 训练数据
#
# 分别处理 news / cf_arc_easy / cf_medqa 三类数据。
# 每类数据先 generate，再自动 retry 失败样本，目标有效率 99.8%。
# 支持断点续传：中断后重新运行会跳过已完成的样本。
#
# 用法:
#   bash scripts/generate_cot.sh              # 生成全部
#   bash scripts/generate_cot.sh news         # 仅 news
#   bash scripts/generate_cot.sh cf           # 仅 CF
#   bash scripts/generate_cot.sh retry        # 仅重试已有文件中失败样本
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

: "${CONCURRENCY:=32}"
: "${MAX_RETRIES:=3}"
TARGET="${1:-all}"

run_gen() {
    local input="$1" output="$2" dtype="$3" split="${4:-}"
    echo "=== 生成 CoT: ${output} (type=${dtype}, split=${split:-all}) ==="
    local split_arg=""
    if [ -n "$split" ]; then
        split_arg="--split ${split}"
    fi
    python -m tools.generate_cot_data \
        --input "$input" \
        --output "$output" \
        --data-type "$dtype" \
        --concurrency "$CONCURRENCY" \
        --mode generate \
        $split_arg
}

run_retry() {
    local output="$1" dtype="$2"
    if [ ! -f "$output" ]; then
        return
    fi
    echo "=== 重试失败样本: ${output} ==="
    python -m tools.generate_cot_data \
        --input /dev/null \
        --output "$output" \
        --data-type "$dtype" \
        --concurrency "$CONCURRENCY" \
        --mode retry \
        --max-retries "$MAX_RETRIES"
}

ALL_FILES=(
    "data/news/train.jsonl|data/news/train_cot.jsonl|news|"
    "data/news/val.jsonl|data/news/val_cot.jsonl|news|"
    "data/counterfactual/arc_easy.jsonl|data/counterfactual/arc_easy_cot.jsonl|cf|train"
    "data/counterfactual/medqa.jsonl|data/counterfactual/medqa_cot.jsonl|cf|train"
    "data/counterfactual/arc_easy.jsonl|data/counterfactual/arc_easy_cot_val.jsonl|cf|test"
    "data/counterfactual/medqa.jsonl|data/counterfactual/medqa_cot_val.jsonl|cf|test"
)

# --- Generate ---
if [ "$TARGET" != "retry" ]; then
    for entry in "${ALL_FILES[@]}"; do
        IFS='|' read -r input output dtype split <<< "$entry"
        case "$TARGET" in
            all)  run_gen "$input" "$output" "$dtype" "$split" ;;
            news) [[ "$dtype" == "news" ]] && run_gen "$input" "$output" "$dtype" "$split" ;;
            cf)   [[ "$dtype" == "cf" ]] && run_gen "$input" "$output" "$dtype" "$split" ;;
        esac
    done
fi

# --- Retry ---
if [ "$TARGET" = "all" ] || [ "$TARGET" = "retry" ]; then
    for entry in "${ALL_FILES[@]}"; do
        IFS='|' read -r _ output dtype _ <<< "$entry"
        run_retry "$output" "$dtype"
    done
fi

echo "=== 全部完成 ==="
