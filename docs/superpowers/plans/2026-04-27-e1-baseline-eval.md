# E1 Baseline 评测 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 No-Memory 和 VanillaRAG 两个 baseline 的完整评测 pipeline（知识生成 + 评测脚本 + 12 个 shell 入口）

**Architecture:** OOD 数据集（MedQA/ARC/MMLU）从 HF datasets 加载，DeepSeek V4 Flash 生成 oracle 知识段落存入 JSONL，统一评测脚本通过 logprob 打分完成多选题评测，每模型两个 sh 文件分别跑 No-Memory 和 VanillaRAG。

**Tech Stack:** Python 3.11, HuggingFace datasets + transformers, OpenAI AsyncAPI (DeepSeek), PyTorch, tqdm

**Spec:** `docs/superpowers/specs/2026-04-27-e1-baseline-eval-design.md`

**前置条件:** `hugglingface_model/qwen3-4B` 尚未下载，需要在运行 qwen3-4b 相关 sh 前完成下载。

---

## 文件结构

| 操作 | 路径 | 职责 |
|------|------|------|
| CREATE | `tools/build_ood_knowledge.py` | HF dataset → 统一 JSONL + DeepSeek 知识生成 |
| CREATE | `evaluation/__init__.py` | 包初始化 |
| CREATE | `evaluation/eval_baseline.py` | 统一 baseline 评测（logprob 打分） |
| CREATE | `tests/unit/test_ood_knowledge.py` | 知识生成的格式转换测试 |
| CREATE | `tests/unit/test_eval_baseline.py` | 评测脚本的核心函数测试 |
| CREATE | `scripts/{model}_{method}.sh` × 12 | Shell 入口 |
| CREATE | `data/ood/` | OOD 知识数据目录 |
| CREATE | `results/baseline/` | 评测结果目录 |

---

### Task 1: OOD 知识生成脚本

**Files:**
- Create: `tools/build_ood_knowledge.py`
- Create: `tests/unit/test_ood_knowledge.py`

- [ ] **Step 1: 编写 HF → 统一格式转换的测试**

```python
# tests/unit/test_ood_knowledge.py
"""OOD 知识生成的格式转换测试。"""
import pytest
from tools.build_ood_knowledge import convert_medqa, convert_arc, convert_mmlu


class TestConvertMedqa:
    def test_basic(self):
        row = {
            "id": "test-00000",
            "sent1": "A patient presents with...",
            "sent2": "",
            "ending0": "Option A text",
            "ending1": "Option B text",
            "ending2": "Option C text",
            "ending3": "Option D text",
            "label": 1,
        }
        result = convert_medqa(row, idx=0)
        assert result["id"] == "medqa_00000"
        assert result["dataset"] == "medqa"
        assert result["question"] == "A patient presents with..."
        assert result["options"] == {
            "A": "Option A text",
            "B": "Option B text",
            "C": "Option C text",
            "D": "Option D text",
        }
        assert result["correct_letter"] == "B"

    def test_label_0_maps_to_A(self):
        row = {
            "id": "test-00001",
            "sent1": "Q",
            "sent2": "",
            "ending0": "a",
            "ending1": "b",
            "ending2": "c",
            "ending3": "d",
            "label": 0,
        }
        assert convert_medqa(row, idx=1)["correct_letter"] == "A"


class TestConvertArc:
    def test_basic(self):
        row = {
            "id": "Mercury_7175875",
            "question": "An astronomer observes...",
            "choices": {
                "text": ["Density decreases", "Years longer", "Days shorter", "Gravity stronger"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "C",
        }
        result = convert_arc(row, idx=0)
        assert result["id"] == "arc_00000"
        assert result["dataset"] == "arc"
        assert result["correct_letter"] == "C"
        assert result["options"]["C"] == "Days shorter"


class TestConvertMmlu:
    def test_basic(self):
        row = {
            "question": "Find the degree...",
            "subject": "abstract_algebra",
            "choices": ["0", "4", "2", "6"],
            "answer": 1,
        }
        result = convert_mmlu(row, idx=42)
        assert result["id"] == "mmlu_00042"
        assert result["dataset"] == "mmlu"
        assert result["correct_letter"] == "B"
        assert result["options"]["B"] == "4"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `conda run -n ExplicitLLM python -m pytest tests/unit/test_ood_knowledge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.build_ood_knowledge'`

- [ ] **Step 3: 实现转换函数和主体脚本**

```python
# tools/build_ood_knowledge.py
"""build_ood_knowledge —— OOD 数据集知识段落生成。

从 HuggingFace datasets 加载 MedQA/ARC/MMLU，调用 DeepSeek API
为每道题生成百科风格 oracle 知识段落，输出统一 JSONL。

使用：
    python -m tools.build_ood_knowledge --dataset medqa
    python -m tools.build_ood_knowledge --dataset arc
    python -m tools.build_ood_knowledge --dataset mmlu
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

# ── 环境变量 ──────────────────────────────────────────────────────────────
LLM_MODEL: str = os.getenv("DEEPSEEK_LLM_MODEL", "deepseek-v4-flash")
LLM_BASE_URL: str = os.getenv("DEEPSEEK_LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY: str = os.getenv("DEEPSEEK_LLM_API_KEY", os.getenv("LLM_API_KEY", ""))

OUTPUT_DIR = Path("data/ood")
_SEMAPHORE_SIZE = 24
_BATCH_SIZE = 100
_LABELS = ["A", "B", "C", "D"]

# ── HF → 统一格式转换 ────────────────────────────────────────────────────

def convert_medqa(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """MedQA HF 行 → 统一 JSONL 格式。"""
    options = {_LABELS[i]: row[f"ending{i}"] for i in range(4)}
    return {
        "id": f"medqa_{idx:05d}",
        "dataset": "medqa",
        "question": row["sent1"],
        "options": options,
        "correct_letter": _LABELS[row["label"]],
    }


def convert_arc(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """ARC-Challenge HF 行 → 统一 JSONL 格式。"""
    options = {l: t for l, t in zip(row["choices"]["label"], row["choices"]["text"])}
    return {
        "id": f"arc_{idx:05d}",
        "dataset": "arc",
        "question": row["question"],
        "options": options,
        "correct_letter": row["answerKey"],
    }


def convert_mmlu(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """MMLU HF 行 → 统一 JSONL 格式。"""
    options = {_LABELS[i]: row["choices"][i] for i in range(4)}
    return {
        "id": f"mmlu_{idx:05d}",
        "dataset": "mmlu",
        "question": row["question"],
        "options": options,
        "correct_letter": _LABELS[row["answer"]],
    }


_DATASET_REGISTRY = {
    "medqa": {
        "hf_name": "GBaker/MedQA-USMLE-4-options-hf",
        "config": None,
        "split": "test",
        "convert_fn": convert_medqa,
    },
    "arc": {
        "hf_name": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "split": "test",
        "convert_fn": convert_arc,
    },
    "mmlu": {
        "hf_name": "cais/mmlu",
        "config": "all",
        "split": "test",
        "convert_fn": convert_mmlu,
    },
}


# ── 工具函数 ──────────────────────────────────────────────────────────────

def _load_done_ids(output_path: Path) -> set:
    """从已有输出文件加载已完成 ID（断点续跑）。"""
    done = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                try:
                    obj = json.loads(stripped)
                    if obj.get("passage"):
                        done.add(obj["id"])
                except json.JSONDecodeError:
                    pass
    return done


def _append_jsonl(item: Dict, path: Path) -> None:
    """追加一条记录到 JSONL。"""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _make_client() -> AsyncOpenAI:
    """创建 DeepSeek AsyncOpenAI 客户端。"""
    return AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


# ── 知识生成 ──────────────────────────────────────────────────────────────

_KNOWLEDGE_PROMPT = """\
Given a multiple-choice question and its correct answer, write a 100-200 word \
encyclopedia-style knowledge passage that contains the information needed to \
answer the question. The passage should read like a textbook paragraph — \
do NOT mention the question, do NOT include option letters (A/B/C/D), \
and do NOT reveal it is generated for a test.

Question: {question}
Options: A. {opt_a}  B. {opt_b}  C. {opt_c}  D. {opt_d}
Correct Answer: {correct_letter}. {correct_answer}

Knowledge passage:"""


async def generate_knowledge(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    sample: Dict[str, Any],
) -> Optional[str]:
    """调用 DeepSeek 生成知识段落。

    参数:
        client: AsyncOpenAI 实例。
        sem: 并发信号量。
        sample: 统一格式的样本字典。

    返回:
        知识段落文本，失败返回 None。
    """
    opts = sample["options"]
    correct_answer = opts[sample["correct_letter"]]
    prompt = _KNOWLEDGE_PROMPT.format(
        question=sample["question"],
        opt_a=opts["A"],
        opt_b=opts["B"],
        opt_c=opts["C"],
        opt_d=opts["D"],
        correct_letter=sample["correct_letter"],
        correct_answer=correct_answer,
    )
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=512,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.warning("API 错误 (attempt %d): %s", attempt + 1, e)
                await asyncio.sleep(2.0 * (attempt + 1))
    return None


async def run_dataset(dataset_name: str, n_samples: int = -1) -> None:
    """处理单个数据集：HF 加载 → 转换 → 生成知识 → 写入 JSONL。

    参数:
        dataset_name: 数据集短名（medqa/arc/mmlu）。
        n_samples: 截断样本数，-1 为全量。
    """
    meta = _DATASET_REGISTRY[dataset_name]
    output_path = OUTPUT_DIR / f"{dataset_name}.jsonl"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载 HF dataset
    logger.info("加载 HF dataset: %s", meta["hf_name"])
    if meta["config"]:
        ds = hf_load_dataset(meta["hf_name"], meta["config"], split=meta["split"])
    else:
        ds = hf_load_dataset(meta["hf_name"], split=meta["split"])

    # 转换为统一格式
    samples = []
    for idx, row in enumerate(ds):
        samples.append(meta["convert_fn"](row, idx))
        if n_samples > 0 and len(samples) >= n_samples:
            break

    # 断点续跑
    done_ids = _load_done_ids(output_path)
    todo = [s for s in samples if s["id"] not in done_ids]
    logger.info(
        "%s: 共 %d 条，已完成 %d，待处理 %d",
        dataset_name, len(samples), len(done_ids), len(todo),
    )
    if not todo:
        return

    # 批量异步生成知识
    client = _make_client()
    sem = asyncio.Semaphore(_SEMAPHORE_SIZE)

    for batch_start in range(0, len(todo), _BATCH_SIZE):
        batch = todo[batch_start : batch_start + _BATCH_SIZE]

        async def _process(sample: Dict) -> Optional[Dict]:
            passage = await generate_knowledge(client, sem, sample)
            if passage:
                sample["passage"] = passage
                return sample
            return None

        results = await asyncio.gather(
            *[_process(s) for s in batch], return_exceptions=True
        )
        for r in results:
            if isinstance(r, dict):
                _append_jsonl(r, output_path)

        done_count = len(done_ids) + batch_start + len(batch)
        logger.info("进度: %d / %d", done_count, len(samples))


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="OOD 知识段落生成")
    parser.add_argument(
        "--dataset",
        choices=["medqa", "arc", "mmlu", "all"],
        required=True,
        help="数据集名称（all = 全部三个）",
    )
    parser.add_argument(
        "--n-samples", type=int, default=-1, help="截断样本数（-1 = 全量）"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    datasets = (
        list(_DATASET_REGISTRY.keys()) if args.dataset == "all" else [args.dataset]
    )
    for ds_name in datasets:
        asyncio.run(run_dataset(ds_name, args.n_samples))
        logger.info("✓ %s 完成", ds_name)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `conda run -n ExplicitLLM python -m pytest tests/unit/test_ood_knowledge.py -v`
Expected: 4 tests PASS

- [ ] **Step 5: Smoke test（3 条/数据集）**

Run: `conda run -n ExplicitLLM python -m tools.build_ood_knowledge --dataset medqa --n-samples 3`
Expected: 生成 `data/ood/medqa.jsonl`，包含 3 行，每行有 passage 字段

验证: `head -1 data/ood/medqa.jsonl | python3 -c "import sys,json; d=json.loads(sys.stdin.readline()); assert 'passage' in d and len(d['passage']) > 50; print('OK:', d['id'])"`

- [ ] **Step 6: 提交**

```bash
git add tools/build_ood_knowledge.py tests/unit/test_ood_knowledge.py
git commit -m "feat: add OOD knowledge generation (DeepSeek, 3 datasets)"
```

---

### Task 2: 统一评测脚本

**Files:**
- Create: `evaluation/__init__.py`
- Create: `evaluation/eval_baseline.py`
- Create: `tests/unit/test_eval_baseline.py`

- [ ] **Step 1: 编写 logprob 评分函数的测试**

```python
# tests/unit/test_eval_baseline.py
"""Baseline 评测脚本的核心函数测试。"""
import json
import tempfile
from pathlib import Path

import pytest
import torch

from evaluation.eval_baseline import (
    build_mc_prompt,
    evaluate_logprob,
    load_samples_jsonl,
)


class TestBuildMcPrompt:
    def test_no_memory(self):
        options = {"A": "Paris", "B": "London", "C": "Berlin", "D": "Rome"}
        prompt = build_mc_prompt(
            question="What is the capital of France?",
            options=options,
            passage=None,
        )
        assert prompt.startswith("Question:")
        assert "A. Paris" in prompt
        assert prompt.endswith("Answer:")
        assert "Reference:" not in prompt

    def test_vanilla_rag(self):
        options = {"A": "Paris", "B": "London", "C": "Berlin", "D": "Rome"}
        prompt = build_mc_prompt(
            question="What is the capital of France?",
            options=options,
            passage="France is a country in Europe. Its capital is Paris.",
        )
        assert prompt.startswith("Reference:")
        assert "A. Paris" in prompt
        assert prompt.endswith("Answer:")


class TestLoadSamplesJsonl:
    def test_loads_correct_fields(self):
        data = {
            "id": "test_0",
            "dataset": "test",
            "question": "Q?",
            "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "correct_letter": "A",
            "passage": "Knowledge text.",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(data) + "\n")
            tmp_path = f.name
        samples = load_samples_jsonl(Path(tmp_path))
        assert len(samples) == 1
        assert samples[0]["correct_letter"] == "A"
        Path(tmp_path).unlink()

    def test_n_samples_truncation(self):
        lines = []
        for i in range(10):
            lines.append(json.dumps({
                "id": f"t_{i}", "dataset": "t", "question": "Q",
                "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                "correct_letter": "A", "passage": "p",
            }))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(lines) + "\n")
            tmp_path = f.name
        samples = load_samples_jsonl(Path(tmp_path), n_samples=3)
        assert len(samples) == 3
        Path(tmp_path).unlink()


class TestEvaluateLogprob:
    """用 tiny random model 验证 logprob 评分函数的形状和返回值。"""

    @pytest.fixture
    def tiny_model_and_tokenizer(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        tokenizer = AutoTokenizer.from_pretrained(name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(name)
        model.eval()
        return model, tokenizer

    def test_returns_valid_index(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        prompt = "Question: What?\nA. a\nB. b\nC. c\nD. d\nAnswer:"
        pred = evaluate_logprob(model, tokenizer, prompt, device="cpu")
        assert pred in [0, 1, 2, 3]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `conda run -n ExplicitLLM python -m pytest tests/unit/test_eval_baseline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evaluation'`

- [ ] **Step 3: 实现评测脚本**

```python
# evaluation/__init__.py
```

```python
# evaluation/eval_baseline.py
"""eval_baseline —— No-Memory / VanillaRAG baseline 评测。

统一入口，通过 --method 切换评测模式。使用 loglikelihood 打分
（对 " A"/" B"/" C"/" D" 的 continuation log-prob 累加取 argmax）。

使用：
    python -m evaluation.eval_baseline \
        --model-path hugglingface_model/qwen3-0.6B \
        --method no_memory \
        --dataset medqa \
        --data-dir data/ood \
        --output-dir results/baseline

    python -m evaluation.eval_baseline \
        --model-path hugglingface_model/qwen3-0.6B \
        --method vanilla_rag \
        --dataset medqa \
        --data-dir data/ood \
        --output-dir results/baseline \
        --n-samples 10
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

_CHOICE_LABELS = ["A", "B", "C", "D"]


# ── 核心函数 ──────────────────────────────────────────────────────────────


def build_mc_prompt(
    question: str,
    options: Dict[str, str],
    passage: Optional[str] = None,
) -> str:
    """构造多选题 prompt。

    参数:
        question: 题干文本。
        options: {"A": "...", "B": "...", "C": "...", "D": "..."}。
        passage: 知识段落（VanillaRAG 时提供，No-Memory 时为 None）。

    返回:
        拼接好的 prompt 字符串。
    """
    answers_str = "".join(
        f"{label}. {options[label]}\n" for label in _CHOICE_LABELS
    )
    mc_part = f"Question: {question}\n{answers_str}Answer:"
    if passage:
        return f"Reference: {passage}\n\n{mc_part}"
    return mc_part


@torch.no_grad()
def evaluate_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: str = "cuda:0",
) -> int:
    """对 " A"/" B"/" C"/" D" 计算 log-prob，返回 argmax 索引。

    参数:
        model: HuggingFace CausalLM 模型。
        tokenizer: 对应的 tokenizer。
        prompt: 完整 prompt（以 "Answer:" 结尾）。
        device: 推理设备。

    返回:
        预测的选项索引（0=A, 1=B, 2=C, 3=D）。
    """
    context_ids = tokenizer.encode(prompt, add_special_tokens=False)
    scores: List[float] = []

    for label in _CHOICE_LABELS:
        letter = " " + label
        cont_ids = tokenizer.encode(letter, add_special_tokens=False)
        full_ids = torch.tensor(
            [context_ids + cont_ids], dtype=torch.long, device=device
        )

        logits = model(full_ids).logits  # [1, seq_len, vocab]

        cont_start = len(context_ids) - 1
        cont_end = len(context_ids) + len(cont_ids) - 1
        cont_logits = logits[0, cont_start:cont_end, :]  # [n_cont, vocab]
        cont_tokens = torch.tensor(cont_ids, dtype=torch.long, device=device)

        log_probs = torch.nn.functional.log_softmax(cont_logits.float(), dim=-1)
        token_ll = log_probs.gather(1, cont_tokens.unsqueeze(-1)).squeeze(-1)
        scores.append(token_ll.sum().item())

    return int(torch.tensor(scores).argmax().item())


# ── 数据加载 ──────────────────────────────────────────────────────────────


def load_samples_jsonl(
    path: Path,
    n_samples: int = -1,
) -> List[Dict[str, Any]]:
    """从 JSONL 文件加载样本。

    参数:
        path: JSONL 文件路径。
        n_samples: 截断数，-1 为全量。

    返回:
        样本字典列表。
    """
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                samples.append(json.loads(stripped))
                if 0 < n_samples <= len(samples):
                    break
    return samples


# ── 评测主循环 ────────────────────────────────────────────────────────────


def run_evaluation(
    model_path: str,
    method: str,
    dataset: str,
    data_dir: str,
    output_dir: str,
    n_samples: int = -1,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """加载模型 → 加载数据 → 逐条评测 → 保存结果。

    参数:
        model_path: HuggingFace 模型路径。
        method: "no_memory" 或 "vanilla_rag"。
        dataset: 数据集名称（medqa/arc/mmlu/news）。
        data_dir: 数据目录。
        output_dir: 结果输出目录。
        n_samples: 截断样本数，-1 全量。
        device: 推理设备。

    返回:
        结果字典。
    """
    # 加载模型
    logger.info("加载模型: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()

    # 加载数据
    data_path = Path(data_dir) / f"{dataset}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    samples = load_samples_jsonl(data_path, n_samples)
    logger.info("加载 %d 条样本 from %s", len(samples), data_path)

    # 评测
    correct = 0
    latencies: List[float] = []
    use_passage = method == "vanilla_rag"

    for sample in tqdm(samples, desc=f"{dataset}/{method}"):
        passage = sample.get("passage") if use_passage else None
        prompt = build_mc_prompt(
            question=sample["question"],
            options=sample["options"],
            passage=passage,
        )

        t0 = time.perf_counter()
        pred_idx = evaluate_logprob(model, tokenizer, prompt, device=device)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        pred_letter = _CHOICE_LABELS[pred_idx]
        if pred_letter == sample["correct_letter"]:
            correct += 1

    # 结果
    n_eval = len(samples)
    model_short = Path(model_path).name.lower().replace("-", "").replace("_", "")
    result = {
        "model": Path(model_path).name,
        "model_path": model_path,
        "method": method,
        "dataset": dataset,
        "n_samples": n_eval,
        "accuracy": correct / n_eval if n_eval > 0 else 0.0,
        "correct": correct,
        "latency_ms_mean": sum(latencies) / n_eval if n_eval else 0.0,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
    }

    # 保存
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(model_path).name.lower()
    out_path = out_dir / f"{model_name}_{method}_{dataset}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("结果: accuracy=%.4f, saved to %s", result["accuracy"], out_path)

    return result


def _git_sha() -> str:
    """返回当前 git HEAD SHA。"""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()[:8]
    except Exception:
        return "unknown"


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline 评测（No-Memory / VanillaRAG）")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--method", choices=["no_memory", "vanilla_rag"], required=True)
    parser.add_argument("--dataset", type=str, required=True, help="medqa|arc|mmlu|news")
    parser.add_argument("--data-dir", type=str, default="data/ood")
    parser.add_argument("--output-dir", type=str, default="results/baseline")
    parser.add_argument("--n-samples", type=int, default=-1, help="-1 = 全量")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    run_evaluation(
        model_path=args.model_path,
        method=args.method,
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `conda run -n ExplicitLLM python -m pytest tests/unit/test_eval_baseline.py -v`
Expected: 5 tests PASS（tiny-random-Llama 会自动下载一次，后续缓存）

- [ ] **Step 5: 提交**

```bash
git add evaluation/__init__.py evaluation/eval_baseline.py tests/unit/test_eval_baseline.py
git commit -m "feat: add unified baseline evaluation script (logprob scoring)"
```

---

### Task 3: 创建 12 个 Shell 脚本

**Files:**
- Create: `scripts/qwen3-0.6b_no_memory.sh`
- Create: `scripts/qwen3-0.6b_vanilla_rag.sh`
- Create: `scripts/qwen3-1.7b_no_memory.sh`
- Create: `scripts/qwen3-1.7b_vanilla_rag.sh`
- Create: `scripts/qwen3-4b_no_memory.sh`
- Create: `scripts/qwen3-4b_vanilla_rag.sh`
- Create: `scripts/qwen3-8b_no_memory.sh`
- Create: `scripts/qwen3-8b_vanilla_rag.sh`
- Create: `scripts/gemma3-1b_no_memory.sh`
- Create: `scripts/gemma3-1b_vanilla_rag.sh`
- Create: `scripts/ministral-3b_no_memory.sh`
- Create: `scripts/ministral-3b_vanilla_rag.sh`

每个 sh 文件结构完全一致，仅 `--model-path` 和 `--method` 不同。以下是全部 12 个文件的内容。

- [ ] **Step 1: 创建全部 12 个 sh 文件**

模板（所有文件共用此结构）：

```bash
#!/bin/bash
# {MODEL_SHORT} — {METHOD} baseline 评测
# 用法：
#   bash scripts/{MODEL_SHORT}_{METHOD}.sh                    # 默认 GPU 0，全量
#   CUDA_VISIBLE_DEVICES=2 bash scripts/{MODEL_SHORT}_{METHOD}.sh   # 指定 GPU
#   N_SAMPLES=10 bash scripts/{MODEL_SHORT}_{METHOD}.sh       # smoke test
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
N_SAMPLES=${N_SAMPLES:--1}

COMMON="python -m evaluation.eval_baseline \
    --model-path {MODEL_PATH} \
    --method {METHOD} \
    --output-dir results/baseline \
    --n-samples $N_SAMPLES"

echo "=== {MODEL_SHORT} / {METHOD} ==="
$COMMON --dataset medqa   --data-dir data/ood
$COMMON --dataset arc     --data-dir data/ood
$COMMON --dataset mmlu    --data-dir data/ood

# News（数据集就绪后取消注释）
# $COMMON --dataset news --data-dir data/news

echo "=== {MODEL_SHORT} / {METHOD} 完成 ==="
```

6 模型 × 2 方法的参数：

| 文件名 | MODEL_PATH | METHOD |
|--------|-----------|--------|
| `qwen3-0.6b_no_memory.sh` | `hugglingface_model/qwen3-0.6B` | `no_memory` |
| `qwen3-0.6b_vanilla_rag.sh` | `hugglingface_model/qwen3-0.6B` | `vanilla_rag` |
| `qwen3-1.7b_no_memory.sh` | `hugglingface_model/qwen3-1.7B` | `no_memory` |
| `qwen3-1.7b_vanilla_rag.sh` | `hugglingface_model/qwen3-1.7B` | `vanilla_rag` |
| `qwen3-4b_no_memory.sh` | `hugglingface_model/qwen3-4B` | `no_memory` |
| `qwen3-4b_vanilla_rag.sh` | `hugglingface_model/qwen3-4B` | `vanilla_rag` |
| `qwen3-8b_no_memory.sh` | `hugglingface_model/qwen3-8B` | `no_memory` |
| `qwen3-8b_vanilla_rag.sh` | `hugglingface_model/qwen3-8B` | `vanilla_rag` |
| `gemma3-1b_no_memory.sh` | `hugglingface_model/gemma3-1b` | `no_memory` |
| `gemma3-1b_vanilla_rag.sh` | `hugglingface_model/gemma3-1b` | `vanilla_rag` |
| `ministral-3b_no_memory.sh` | `hugglingface_model/ministral-3-3b` | `no_memory` |
| `ministral-3b_vanilla_rag.sh` | `hugglingface_model/ministral-3-3b` | `vanilla_rag` |

- [ ] **Step 2: 设置可执行权限**

```bash
chmod +x scripts/*_no_memory.sh scripts/*_vanilla_rag.sh
```

- [ ] **Step 3: 提交**

```bash
git add scripts/*_no_memory.sh scripts/*_vanilla_rag.sh
git commit -m "feat: add 12 baseline evaluation shell scripts (6 models × 2 methods)"
```

---

### Task 4: 端到端验证

- [ ] **Step 1: 生成 MedQA smoke test 知识（3 条）**

Run: `conda run -n ExplicitLLM python -m tools.build_ood_knowledge --dataset medqa --n-samples 3`

Expected: `data/ood/medqa.jsonl` 存在，3 行，每行有 passage 字段

验证:
```bash
wc -l data/ood/medqa.jsonl
# Expected: 3
python3 -c "
import json
with open('data/ood/medqa.jsonl') as f:
    for line in f:
        d = json.loads(line)
        assert 'passage' in d and len(d['passage']) > 50
        print(f'{d[\"id\"]}: passage={len(d[\"passage\"])} chars')
print('OK')
"
```

- [ ] **Step 2: No-Memory smoke test（qwen3-0.6b, 3 条）**

Run: `N_SAMPLES=3 bash scripts/qwen3-0.6b_no_memory.sh`

Expected: `results/baseline/` 下生成 3 个 JSON（medqa/arc/mmlu），每个有 accuracy 字段

验证:
```bash
cat results/baseline/qwen3-0.6B_no_memory_medqa.json | python3 -c "
import sys, json; d=json.load(sys.stdin)
assert 'accuracy' in d and d['n_samples'] == 3
print(f'accuracy={d[\"accuracy\"]:.3f}, n={d[\"n_samples\"]}')
"
```

- [ ] **Step 3: VanillaRAG smoke test（qwen3-0.6b, 3 条 MedQA）**

Run:
```bash
conda run -n ExplicitLLM python -m evaluation.eval_baseline \
    --model-path hugglingface_model/qwen3-0.6B \
    --method vanilla_rag \
    --dataset medqa \
    --data-dir data/ood \
    --output-dir results/baseline \
    --n-samples 3
```

Expected: `results/baseline/qwen3-0.6B_vanilla_rag_medqa.json` 生成

- [ ] **Step 4: 清理 smoke test 产物，提交验证通过记录**

```bash
rm -f data/ood/medqa.jsonl results/baseline/*.json
git add -A
git commit -m "test: verify E1 baseline pipeline end-to-end"
```

---

## 执行顺序总结

```
Task 1 (知识生成脚本)  →  Task 2 (评测脚本)  →  Task 3 (Shell脚本)  →  Task 4 (E2E验证)
      独立                    独立                  依赖 Task 2          依赖 Task 1+2+3
```

Task 1 和 Task 2 可并行实现（无依赖）。Task 3 依赖 Task 2（sh 调用 eval_baseline.py）。Task 4 依赖全部。
