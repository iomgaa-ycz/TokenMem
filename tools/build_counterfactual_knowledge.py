"""build_counterfactual_knowledge — 反事实知识段落生成。

读取 data/ood/{dataset}.jsonl（含正确知识），为每道题**一次性**生成
所有错误选项的反事实百科段落。

每道 4 选项 MCQ → 1 次 API 调用 → 3 条反事实段落。
输出带 split 字段（train 70% / test 30%），供 SFT 混合训练和评测。

输出：data/counterfactual/{dataset}.jsonl
字段：cf_id, source_id, dataset, question, options, correct_letter,
      passage(正确知识), target_letter, target_answer,
      counterfactual_passage, split

使用：
    python -m tools.build_counterfactual_knowledge --dataset medqa
    python -m tools.build_counterfactual_knowledge --dataset arc_easy
    python -m tools.build_counterfactual_knowledge --dataset medqa --n-samples 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

load_dotenv()
logger = logging.getLogger(__name__)

LLM_MODEL: str = os.getenv("DEEPSEEK_LLM_MODEL", "deepseek-v4-flash")
LLM_BASE_URL: str = os.getenv("DEEPSEEK_LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY: str = os.getenv("DEEPSEEK_LLM_API_KEY", os.getenv("LLM_API_KEY", ""))

INPUT_DIR = Path("data/ood")
OUTPUT_DIR = Path("data/counterfactual")
_SEMAPHORE_SIZE = 24
_BATCH_SIZE = 50
_TRAIN_RATIO = 0.7
_RANDOM_SEED = 42

# ─── Prompt：一次性生成所有错误选项的反事实段落 ───────────────────────────────

_COUNTERFACTUAL_PROMPT_TEMPLATE = """\
You are generating COUNTERFACTUAL knowledge passages for a research experiment \
on knowledge injection. This is NOT about correctness — your job is to write \
convincing, encyclopedic passages that support WRONG answers.

Given the question below, the correct answer is {correct_letter}. \
For each of the other (wrong) options listed below, write a separate \
100-200 word encyclopedia-style passage that makes that wrong option \
appear to be the correct answer.

Requirements:
- Each passage should read like a textbook paragraph
- Do NOT mention the question, options, or letters (A/B/C/D)
- Each passage must contain specific facts/definitions supporting its target answer
- All passages should have similar length and style
- It is OK and EXPECTED that these passages contain factually incorrect information

Question: {question}
All options: {options_str}
Correct answer: {correct_letter}. {correct_answer}

Generate a passage for each wrong option in this exact JSON format:
{format_hint}

Output ONLY the JSON object, no other text.
"""


def _make_client() -> AsyncOpenAI:
    """创建 AsyncOpenAI 客户端。"""
    return AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


def _load_source_data(dataset: str) -> List[Dict[str, Any]]:
    """从 data/ood/{dataset}.jsonl 加载源数据。"""
    path = INPUT_DIR / f"{dataset}.jsonl"
    assert path.exists(), f"源文件不存在: {path}。请先运行 build_ood_knowledge.py"
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_done_source_ids(output_path: Path) -> set:
    """加载已完成的 source_id 集合（断点续跑，以源题 id 为粒度）。"""
    done: set = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    if "source_id" in obj and "counterfactual_passage" in obj:
                        done.add(obj["source_id"])
                except json.JSONDecodeError:
                    pass
    return done


def _append_jsonl(item: Dict, path: Path) -> None:
    """追加一条记录到 JSONL。"""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _build_format_hint(wrong_letters: List[str]) -> str:
    """构建 JSON 格式提示，告诉 LLM 输出结构。"""
    entries = ", ".join(f'"{l}": "passage supporting option {l}"' for l in wrong_letters)
    return "{" + entries + "}"


async def _generate_all_counterfactuals(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    record: Dict[str, Any],
) -> Optional[Dict[str, str]]:
    """为一道题一次性生成所有错误选项的反事实段落。

    返回：{letter: passage} 字典，或 None。
    """
    correct = record["correct_letter"]
    options = record["options"]
    wrong_letters = sorted(l for l in options if l != correct)
    options_str = "  ".join(f"{k}. {v}" for k, v in sorted(options.items()))
    format_hint = _build_format_hint(wrong_letters)

    prompt = _COUNTERFACTUAL_PROMPT_TEMPLATE.format(
        question=record["question"],
        options_str=options_str,
        correct_letter=correct,
        correct_answer=options.get(correct, ""),
        format_hint=format_hint,
    )

    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                content = resp.choices[0].message.content or ""
                content = content.strip()
                # 清理可能的 markdown 包裹
                if content.startswith("```"):
                    content = content.split("\n", 1)[-1]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()

                result = json.loads(content)
                # 验证返回了所有错误选项
                if all(l in result and isinstance(result[l], str) and len(result[l]) > 50
                       for l in wrong_letters):
                    return {l: result[l] for l in wrong_letters}
                logger.warning(
                    "返回格式不完整 (id=%s, attempt=%d): keys=%s",
                    record.get("id"), attempt + 1, list(result.keys()),
                )
            except json.JSONDecodeError:
                logger.warning(
                    "JSON 解析失败 (id=%s, attempt=%d)",
                    record.get("id"), attempt + 1,
                )
            except Exception as exc:
                logger.warning(
                    "API 失败 (id=%s, attempt=%d): %s",
                    record.get("id"), attempt + 1, exc,
                )
            await asyncio.sleep(2 ** (attempt + 1))
    return None


async def run_dataset(dataset: str, n_samples: Optional[int] = None) -> int:
    """主流程：加载 → 一次性生成 → 写入 JSONL（每个错误选项一条记录）。"""
    records = _load_source_data(dataset)
    if n_samples:
        records = records[:n_samples]
    logger.info("源数据: %d 条题目", len(records))

    output_path = OUTPUT_DIR / f"{dataset}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_source_ids = _load_done_source_ids(output_path)
    todo = [r for r in records if r["id"] not in done_source_ids]
    logger.info("已完成: %d 题, 待处理: %d 题", len(done_source_ids), len(todo))

    if not todo:
        return len(done_source_ids)

    # 确定 train/test split（基于前 70% 源记录为 train）
    train_ids = set(r["id"] for r in records[:int(len(records) * _TRAIN_RATIO)])

    client = _make_client()
    sem = asyncio.Semaphore(_SEMAPHORE_SIZE)
    success_count = len(done_source_ids)

    pbar = tqdm(total=len(todo), desc=f"[{dataset}] 反事实生成", unit="题")
    for i in range(0, len(todo), _BATCH_SIZE):
        batch = todo[i : i + _BATCH_SIZE]
        results = await asyncio.gather(
            *[_generate_all_counterfactuals(client, sem, r) for r in batch],
            return_exceptions=True,
        )
        for record, result in zip(batch, results):
            if isinstance(result, dict):
                split = "train" if record["id"] in train_ids else "test"
                for letter, passage in sorted(result.items()):
                    output_record = {
                        "cf_id": f"{record['id']}_cf_{letter}",
                        "source_id": record["id"],
                        "dataset": record["dataset"],
                        "question": record["question"],
                        "options": record["options"],
                        "correct_letter": record["correct_letter"],
                        "passage": record.get("passage", ""),
                        "target_letter": letter,
                        "target_answer": record["options"].get(letter, ""),
                        "counterfactual_passage": passage,
                        "split": split,
                    }
                    _append_jsonl(output_record, output_path)
                success_count += 1
            elif isinstance(result, Exception):
                logger.warning("异常 (id=%s): %s", record.get("id"), result)
            else:
                logger.warning("生成失败 (id=%s)", record.get("id"))
            pbar.update(1)
    pbar.close()

    logger.info("%s 完成: %d/%d 题成功", dataset, success_count, len(records))
    return success_count


def main() -> None:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(
        description="反事实知识生成（每道题一次 API 调用，同时生成所有错误选项的段落）"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["medqa", "arc", "arc_easy", "mmlu"],
    )
    parser.add_argument("--n-samples", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(run_dataset(args.dataset, n_samples=args.n_samples))


if __name__ == "__main__":
    main()
