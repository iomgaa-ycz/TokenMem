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
    "provided passage, even if it contradicts your knowledge. "
    "Be concise. Keep reasoning under 150 words."
)

MAX_TRAINING_TOKENS = 900  # 训练 max_seq_len=1024，留 ~124 token 安全余量
CHARS_PER_TOKEN = 3.5  # 英文粗估


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


def _build_training_prompt(row: Dict[str, Any]) -> str:
    """构建训练时的 CoT prompt（与 Dataset CoT 模式一致，无 passage）。

    用于估算 prompt + cot_response 的总 token 数，
    判断是否超出训练 max_seq_len。

    参数:
        row: JSONL 行字典，需含 question 和 options。

    返回:
        训练时的 prompt 字符串。
    """
    options = row["options"]
    labels = sorted(options.keys())
    option_lines = "\n".join(f"{lb}. {options[lb]}" for lb in labels)
    label_list = ", ".join(labels[:-1]) + ", or " + labels[-1]
    return (
        f"\nQuestion: {row['question']}\n"
        f"{option_lines}\n"
        f"\nLet's think step by step, then give the answer.\n"
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


# ---------------------------------------------------------------------------
# 异步 API 调用
# ---------------------------------------------------------------------------


async def _call_api(
    client: Any,
    system: str,
    user: str,
    model: str,
    semaphore: asyncio.Semaphore,
    max_tokens: int = 800,
) -> Optional[str]:
    """调用 DeepSeek API 生成单条 CoT。

    包含 3 次重试和指数退避。关闭 thinking 模式以获得干净输出。

    参数:
        client: AsyncOpenAI 客户端。
        system: system prompt。
        user: user prompt。
        model: 模型名称。
        semaphore: 并发控制信号量。
        max_tokens: 最大生成 token 数。

    返回:
        生成的文本，或 None（全部重试失败时）。
    """
    async with semaphore:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.0,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.warning("API 调用失败 (attempt %d): %s", attempt + 1, e)
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
    return None


def _estimate_training_tokens(training_prompt: str, cot_response: str) -> float:
    """粗估训练时 prompt + response 的总 token 数。"""
    return (len(training_prompt) + len(" ") + len(cot_response)) / CHARS_PER_TOKEN


async def generate_one(
    client: Any,
    row: Dict[str, Any],
    data_type: str,
    semaphore: asyncio.Semaphore,
    model: str,
) -> Dict[str, Any]:
    """生成单条 CoT 并验证答案，超长自动重试。

    流程: 生成 → 检查答案 → 检查长度 → 超长则用简洁指令重试一次。

    参数:
        client: AsyncOpenAI 客户端。
        row: 原始 JSONL 行字典。
        data_type: "news" 或 "cf"。
        semaphore: 并发控制信号量。
        model: 模型名称。

    返回:
        带有 cot_response / cot_extracted_letter / cot_valid 字段的行字典。
    """
    if data_type == "news":
        passage = row["passage"]
        expected = row["correct_letter"]
    else:
        passage = row["counterfactual_passage"]
        expected = row["target_letter"]

    training_prompt = _build_training_prompt(row)
    user = build_user_prompt(passage, row)
    text = await _call_api(client, SYSTEM_PROMPT, user, model, semaphore)

    result = dict(row)
    if text is None:
        result["cot_response"] = None
        result["cot_extracted_letter"] = None
        result["cot_valid"] = False
        return result

    letter = extract_cot_answer(text)
    est_tokens = _estimate_training_tokens(training_prompt, text)

    # 超长重试：用更强的简洁指令重新生成
    if est_tokens > MAX_TRAINING_TOKENS:
        logger.info(
            "超长 (~%d tokens), 重试: %s",
            int(est_tokens),
            row.get("id") or row.get("cf_id"),
        )
        short_user = user.replace(
            "Let's think step by step, then give the answer.",
            "Think very briefly, then give the answer.",
        )
        text_retry = await _call_api(
            client, SYSTEM_PROMPT, short_user, model, semaphore, max_tokens=400
        )
        if text_retry is not None:
            letter_retry = extract_cot_answer(text_retry)
            est_retry = _estimate_training_tokens(training_prompt, text_retry)
            if est_retry <= MAX_TRAINING_TOKENS and letter_retry == expected:
                text, letter = text_retry, letter_retry
                est_tokens = est_retry

    result["cot_response"] = text
    result["cot_extracted_letter"] = letter
    # 答案正确 且 长度可控 才标记 valid
    result["cot_valid"] = (letter == expected) and (est_tokens <= MAX_TRAINING_TOKENS)
    return result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def _load_input(path: str, split: Optional[str]) -> List[Dict[str, Any]]:
    """加载输入 JSONL，可按 split 过滤。"""
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if split and row.get("split") != split:
                continue
            rows.append(row)
    return rows


def _load_done_ids(path: str, id_field: str) -> set:
    """从已有输出文件加载已完成的 ID（断点续传）。"""
    done: set = set()
    if not Path(path).exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                done.add(d.get(id_field))
            except json.JSONDecodeError:
                continue
    return done


async def run(args: argparse.Namespace) -> None:
    """异步主流程：加载 → 过滤 → 并发生成 → 写入。"""
    from openai import AsyncOpenAI
    from tqdm.asyncio import tqdm_asyncio

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)

    rows = _load_input(args.input, args.split)
    id_field = "cf_id" if args.data_type == "cf" else "id"
    done_ids = _load_done_ids(args.output, id_field)
    remaining = [r for r in rows if r.get(id_field) not in done_ids]

    logger.info("总样本: %d, 已完成: %d, 待处理: %d", len(rows), len(done_ids), len(remaining))

    if not remaining:
        logger.info("无待处理样本，退出")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        generate_one(client, row, args.data_type, semaphore, args.model)
        for row in remaining
    ]
    results = await tqdm_asyncio.gather(*tasks, desc="Generating CoT")

    valid = 0
    total = 0
    with open(args.output, "a", encoding="utf-8") as f:
        for r in results:
            if r is None:
                continue
            total += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if r.get("cot_valid"):
                valid += 1

    logger.info("完成: %d 条, 有效: %d (%.1f%%)", total, valid, valid / max(total, 1) * 100)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="CoT 训练数据生成 (DeepSeek API)")
    p.add_argument("--input", required=True, help="输入 JSONL 路径")
    p.add_argument("--output", required=True, help="输出 JSONL 路径")
    p.add_argument(
        "--data-type",
        choices=["news", "cf"],
        required=True,
        help="数据类型: news 或 cf",
    )
    p.add_argument("--split", default=None, help="仅处理指定 split (如 train)")
    p.add_argument("--concurrency", type=int, default=32, help="并发数 (默认: 32)")
    p.add_argument("--model", default=None, help="模型名 (默认: 读 .env)")
    p.add_argument("--api-key", default=None, help="API key (默认: 读 .env)")
    p.add_argument("--base-url", default=None, help="API base URL (默认: 读 .env)")
    return p.parse_args()


def main() -> None:
    """入口：加载 .env → 解析参数 → 运行异步管线。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    args = parse_args()
    args.api_key = args.api_key or os.getenv("DEEPSEEK_LLM_API_KEY")
    args.base_url = args.base_url or os.getenv("DEEPSEEK_LLM_BASE_URL")
    args.model = args.model or os.getenv("DEEPSEEK_LLM_MODEL", "deepseek-v4-pro")

    assert args.api_key, "需要 API key: --api-key 或 .env DEEPSEEK_LLM_API_KEY"

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
