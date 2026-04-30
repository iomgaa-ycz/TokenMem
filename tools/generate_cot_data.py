"""CoT 训练数据生成 — 使用 DeepSeek API 蒸馏 RAG CoT。

为每条训练样本调用 DeepSeek-v4-pro 生成 CoT 推理，
过滤答案正确的样本，输出带 cot_response 字段的新 JSONL。

News 和 CF 使用统一 prompt 模板，仅读取的字段和验证字母不同:
  - News: passage → row["passage"],       验证 → row["correct_letter"]
  - CF:   passage → row["counterfactual_passage"], 验证 → row["target_letter"]

用法:
    conda run -n ExplicitLLM python -m tools.generate_cot_data \
        --input data/news/train.jsonl \
        --output data/news/train_cot.jsonl \
        --data-type news \
        --concurrency 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 统一 Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question based ONLY on the "
    "provided passage, even if it contradicts your knowledge. Be concise."
)


def build_user_prompt(passage: str, row: Dict[str, Any]) -> str:
    """构建统一的 RAG CoT user prompt。

    News 和 CF 使用同一模板，调用方传入不同的 passage 字段即可。

    参数:
        passage: 知识段落文本（News 传 row["passage"]，CF 传 row["counterfactual_passage"]）。
        row: JSONL 行字典，需含 question 和 options。

    返回:
        完整的 user prompt 字符串。
    """
    options = row["options"]
    labels = sorted(options.keys())
    option_lines = "\n".join(f"{lb}. {options[lb]}" for lb in labels)
    label_list = ", ".join(labels[:-1]) + ", or " + labels[-1]
    return (
        f"{passage}\n\n"
        f"Question: {row['question']}\n"
        f"{option_lines}\n\n"
        f"Let's think step by step, then give the answer.\n"
        f'You MUST end your response with exactly "The answer is X" '
        f"where X is {label_list}."
    )


def extract_cot_answer(text: str) -> Optional[str]:
    """从 CoT 文本中提取答案字母。

    按优先级匹配多种格式，取最后一个匹配（CoT 可能中途修正答案）。

    参数:
        text: 模型生成的完整 CoT 文本。

    返回:
        提取到的答案字母 (A-E)，或 None。
    """
    patterns = [
        r"[Tt]he answer is\s*\**\s*([A-E])",
        r"[Aa]nswer\s*:\s*\**\s*([A-E])",
    ]
    last_match: Optional[str] = None
    for p in patterns:
        for m in re.finditer(p, text):
            last_match = m.group(1)
    return last_match
