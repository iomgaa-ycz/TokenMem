"""build_ood_knowledge — OOD 数据集知识段落生成。

从 HuggingFace datasets 加载 MedQA/ARC/MMLU，调用 DeepSeek API
为每道题生成百科风格 oracle 知识段落，输出统一 JSONL。

数据路径：
    data/ood/medqa.jsonl
    data/ood/arc.jsonl
    data/ood/mmlu.jsonl

环境变量（通过 .env 加载）：
    DEEPSEEK_LLM_MODEL:    DeepSeek 模型名称（默认 deepseek-v4-flash）
    DEEPSEEK_LLM_BASE_URL: DeepSeek API base URL
    DEEPSEEK_LLM_API_KEY:  DeepSeek API 密钥
    LLM_API_KEY:           备用 API 密钥

使用：
    python -m tools.build_ood_knowledge --dataset medqa
    python -m tools.build_ood_knowledge --dataset all
    python -m tools.build_ood_knowledge --dataset medqa --n-samples 10  # smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset as hf_load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

load_dotenv()
logger = logging.getLogger(__name__)

# ─── 配置 ─────────────────────────────────────────────────────────────────────
LLM_MODEL: str = os.getenv("DEEPSEEK_LLM_MODEL", "deepseek-v4-flash")
LLM_BASE_URL: str = os.getenv("DEEPSEEK_LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY: str = os.getenv("DEEPSEEK_LLM_API_KEY", os.getenv("LLM_API_KEY", ""))

OUTPUT_DIR = Path("data/ood")
_SEMAPHORE_SIZE = 24
_BATCH_SIZE = 100
_LABELS = ["A", "B", "C", "D"]

# ─── HF 数据集注册表 ──────────────────────────────────────────────────────────
_DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "medqa": {
        "path": "GBaker/MedQA-USMLE-4-options-hf",
        "config": None,
        "split": "test",
    },
    "arc": {
        "path": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "split": "test",
    },
    "arc_easy": {
        "path": "allenai/ai2_arc",
        "config": "ARC-Easy",
        "split": "test",
    },
    "mmlu": {
        "path": "cais/mmlu",
        "config": "all",
        "split": "test",
    },
}

# ─── 知识生成 Prompt ──────────────────────────────────────────────────────────
_KNOWLEDGE_PROMPT_TEMPLATE = """\
Given a multiple-choice question and its correct answer, write a 100-200 word \
encyclopedia-style knowledge passage that contains the information needed to \
answer the question. The passage should read like a textbook paragraph — \
do NOT mention the question, do NOT include option letters (A/B/C/D), \
and do NOT reveal it is generated for a test.

Question: {question}
Options: A. {opt_a}  B. {opt_b}  C. {opt_c}  D. {opt_d}
Correct Answer: {correct_letter}. {correct_answer}

Knowledge passage:
"""


# ─────────────────────────────────────────────────────────────────────────────
# 格式转换：HF row → 统一 dict
# ─────────────────────────────────────────────────────────────────────────────


def convert_medqa(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """将 MedQA HF 行转换为统一格式。

    参数：
        row: HuggingFace MedQA 数据集的一行（sent1→question, ending0-3→options, label→correct_letter）。
        idx: 全局序号，用于生成 ID。

    返回：
        统一格式字典，包含 id/dataset/question/options/correct_letter。
    """
    options = {
        "A": row["ending0"],
        "B": row["ending1"],
        "C": row["ending2"],
        "D": row["ending3"],
    }
    correct_letter = _LABELS[row["label"]]
    return {
        "id": f"medqa_{idx:05d}",
        "dataset": "medqa",
        "question": row["sent1"],
        "options": options,
        "correct_letter": correct_letter,
    }


def convert_arc(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """将 ARC HF 行转换为统一格式。

    参数：
        row: HuggingFace ARC 数据集的一行（choices.text/label→options, answerKey→correct_letter）。
        idx: 全局序号，用于生成 ID。

    返回：
        统一格式字典，包含 id/dataset/question/options/correct_letter。
    """
    choices = row["choices"]
    options = {label: text for label, text in zip(choices["label"], choices["text"])}
    return {
        "id": f"arc_{idx:05d}",
        "dataset": "arc",
        "question": row["question"],
        "options": options,
        "correct_letter": row["answerKey"],
    }


def convert_arc_easy(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """将 ARC-Easy HF 行转换为统一格式。

    参数：
        row: HuggingFace ARC-Easy 数据集的一行（choices.text/label→options, answerKey→correct_letter）。
        idx: 全局序号，用于生成 ID。

    返回：
        统一格式字典，包含 id/dataset/question/options/correct_letter。
    """
    choices = row["choices"]
    options = {label: text for label, text in zip(choices["label"], choices["text"])}
    return {
        "id": f"arc_easy_{idx:05d}",
        "dataset": "arc_easy",
        "question": row["question"],
        "options": options,
        "correct_letter": row["answerKey"],
    }


def convert_mmlu(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """将 MMLU HF 行转换为统一格式。

    参数：
        row: HuggingFace MMLU 数据集的一行（choices list→options, answer int→correct_letter）。
        idx: 全局序号，用于生成 ID。

    返回：
        统一格式字典，包含 id/dataset/question/options/correct_letter。
    """
    options = {_LABELS[i]: row["choices"][i] for i in range(len(row["choices"]))}
    correct_letter = _LABELS[row["answer"]]
    return {
        "id": f"mmlu_{idx:05d}",
        "dataset": "mmlu",
        "question": row["question"],
        "options": options,
        "correct_letter": correct_letter,
    }


# ─── 转换函数注册表 ──────────────────────────────────────────────────────────
_CONVERTER_MAP = {
    "medqa": convert_medqa,
    "arc": convert_arc,
    "arc_easy": convert_arc_easy,
    "mmlu": convert_mmlu,
}


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────


def _load_done_ids(output_path: Path) -> set:
    """从已有输出文件中加载已完成的 ID 集合（断点续跑用）。

    参数：
        output_path: JSONL 输出文件路径。

    返回：
        已完成记录的 ID 集合。
    """
    done: set = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    if "id" in obj and "passage" in obj:
                        done.add(obj["id"])
                except json.JSONDecodeError:
                    pass
    return done


def _append_jsonl(item: Dict, path: Path) -> None:
    """追加一条记录到 JSONL 文件。

    参数：
        item: 要写入的字典。
        path: 目标 JSONL 文件路径。
    """
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _make_client() -> AsyncOpenAI:
    """根据环境变量创建 AsyncOpenAI 客户端（DeepSeek 配置）。

    返回：
        配置好 base_url 和 api_key 的 AsyncOpenAI 实例。
    """
    return AsyncOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 异步知识生成
# ─────────────────────────────────────────────────────────────────────────────


async def generate_knowledge(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    sample: Dict[str, Any],
) -> Optional[str]:
    """调用 DeepSeek 为单道选择题生成百科风格知识段落。

    参数：
        client: AsyncOpenAI 客户端。
        sem:    并发控制信号量。
        sample: 统一格式的选择题字典（需包含 question/options/correct_letter）。

    返回：
        生成的知识段落文本；3 次重试均失败则返回 None。

    实现细节：
        - 最多重试 3 次，指数退避。
        - 使用 temperature=0.3 保持一致性。
    """
    opts = sample["options"]
    correct_letter = sample["correct_letter"]
    prompt = _KNOWLEDGE_PROMPT_TEMPLATE.format(
        question=sample["question"],
        opt_a=opts.get("A", ""),
        opt_b=opts.get("B", ""),
        opt_c=opts.get("C", ""),
        opt_d=opts.get("D", ""),
        correct_letter=correct_letter,
        correct_answer=opts.get(correct_letter, ""),
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
                if content:
                    return content
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "generate_knowledge 失败 (attempt %d/3, id=%s): %s",
                    attempt + 1,
                    sample.get("id", "?"),
                    exc,
                )
                await asyncio.sleep(2 ** (attempt + 1))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 数据集编排
# ─────────────────────────────────────────────────────────────────────────────


async def run_dataset(dataset_name: str, n_samples: Optional[int] = None) -> int:
    """加载 HF 数据集 → 转换 → 批量异步生成知识 → 写入 JSONL。

    参数：
        dataset_name: 数据集名称（medqa/arc/mmlu）。
        n_samples:    限制样本数（None 表示全量）。

    返回：
        成功生成知识段落的样本数。

    实现细节：
        - 支持断点续跑（通过 _load_done_ids 跳过已生成的 ID）。
        - 分批处理（每批 _BATCH_SIZE 条），逐条追加写入。
        - 并发度由 _SEMAPHORE_SIZE 控制。
    """
    assert dataset_name in _DATASET_REGISTRY, f"未知数据集: {dataset_name}"
    reg = _DATASET_REGISTRY[dataset_name]
    converter = _CONVERTER_MAP[dataset_name]

    # Phase 1: 加载并转换 HF 数据集
    logger.info(
        "加载 HF 数据集: %s (config=%s, split=%s)",
        reg["path"],
        reg["config"],
        reg["split"],
    )
    if reg["config"]:
        ds = hf_load_dataset(reg["path"], reg["config"], split=reg["split"])
    else:
        ds = hf_load_dataset(reg["path"], split=reg["split"])

    samples: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        samples.append(converter(row, idx))
        if n_samples and len(samples) >= n_samples:
            break
    logger.info("转换完成: %d 条样本", len(samples))

    # Phase 2: 断点续跑
    output_path = OUTPUT_DIR / f"{dataset_name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = _load_done_ids(output_path)
    todo = [s for s in samples if s["id"] not in done_ids]
    logger.info(
        "%s: 共 %d 条，已完成 %d，待处理 %d",
        dataset_name,
        len(samples),
        len(done_ids),
        len(todo),
    )

    if not todo:
        return len(done_ids)

    # Phase 3: 批量异步生成
    client = _make_client()
    sem = asyncio.Semaphore(_SEMAPHORE_SIZE)
    success_count = len(done_ids)

    pbar = tqdm(total=len(todo), desc=f"[{dataset_name}] 生成知识", unit="题")
    for i in range(0, len(todo), _BATCH_SIZE):
        batch = todo[i : i + _BATCH_SIZE]
        tasks = [generate_knowledge(client, sem, s) for s in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for sample, result in zip(batch, results):
            if isinstance(result, str) and result:
                sample["passage"] = result
                _append_jsonl(sample, output_path)
                success_count += 1
            elif isinstance(result, Exception):
                logger.warning("异常 (id=%s): %s", sample.get("id"), result)
            else:
                logger.warning("生成失败 (id=%s): 返回空", sample.get("id"))
            pbar.update(1)
    pbar.close()

    logger.info("%s 完成: %d/%d 成功", dataset_name, success_count, len(samples))
    return success_count


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI 入口：解析参数并运行数据集知识生成。"""
    parser = argparse.ArgumentParser(
        description="OOD 数据集知识段落生成（MedQA/ARC/MMLU → DeepSeek → JSONL）"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["medqa", "arc", "arc_easy", "mmlu", "all"],
        help="要处理的数据集名称（或 'all' 处理全部）",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="限制样本数（默认全量；设置小数进行 smoke test）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    datasets = (
        list(_DATASET_REGISTRY.keys()) if args.dataset == "all" else [args.dataset]
    )

    for ds_name in datasets:
        asyncio.run(run_dataset(ds_name, n_samples=args.n_samples))


if __name__ == "__main__":
    main()
