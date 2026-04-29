"""SFT 数据集与 collate 函数。

提供 NewsQAOracleDataset（读取 JSONL 格式 QA 数据）
以及 make_collate_fn（批量 tokenize + 动态 padding）。
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class NewsQAOracleDataset(Dataset):
    """从 JSONL 文件加载 NewsQA 多选题数据集。

    每行 JSON 需包含: question, passage, options (dict A/B/C/D),
    correct_letter, correct_answer。

    __getitem__ 返回原始文本字典:
        {"prompt": str, "answer": str, "knowledge_text": str}

    参数:
        jsonl_path: JSONL 文件路径。
        knowledge_field: 用作 knowledge_text 的字段名，默认 "passage"。
    """

    def __init__(
        self,
        jsonl_path: Union[str, Path],
        knowledge_field: str = "passage",
    ) -> None:
        self.knowledge_field = knowledge_field
        self.rows: List[Dict[str, Any]] = []

        path = Path(jsonl_path)
        with path.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    logger.warning("跳过第 %d 行: JSON 解析失败", line_no)
                    continue
                self.rows.append(row)

        logger.info("加载 %d 条样本 from %s", len(self.rows), path.name)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self.rows[idx]

        options = row["options"]
        prompt = (
            f"Question: {row['question']}\n"
            f"A. {options['A']}\n"
            f"B. {options['B']}\n"
            f"C. {options['C']}\n"
            f"D. {options['D']}\n"
            f"Answer:"
        )

        return {
            "prompt": prompt,
            "answer": row["correct_letter"],
            "knowledge_text": row[self.knowledge_field],
        }


class OversampledDataset(Dataset):
    """简单过采样包装器，通过索引取模将底层数据集重复 factor 次。

    不复制数据，仅修改 __len__ 和 __getitem__ 的索引映射。

    参数:
        dataset: 底层 Dataset 实例。
        factor: 过采样倍数，必须 >= 1。
    """

    def __init__(self, dataset: Dataset, factor: int = 2) -> None:
        assert factor >= 1, f"过采样倍数必须 >= 1, 收到 {factor}"
        self.dataset = dataset
        self.factor = factor

    def __len__(self) -> int:
        return len(self.dataset) * self.factor

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.dataset[idx % len(self.dataset)]


class CounterfactualDataset(Dataset):
    """从反事实 JSONL 加载训练数据。

    字段映射:
      knowledge_text ← counterfactual_passage
      answer         ← target_letter
      prompt         ← question + options

    返回格式与 NewsQAOracleDataset 一致，可直接复用 collate_fn。

    参数:
        jsonl_path: 反事实 JSONL 文件路径。
        split: 仅加载指定 split 的行，默认 "train"。
    """

    def __init__(
        self,
        jsonl_path: Union[str, Path],
        split: str = "train",
    ) -> None:
        self.rows: List[Dict[str, Any]] = []

        path = Path(jsonl_path)
        with path.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    logger.warning("跳过第 %d 行: JSON 解析失败", line_no)
                    continue
                if row.get("split") == split:
                    self.rows.append(row)

        logger.info(
            "加载 %d 条反事实样本 (split=%s) from %s",
            len(self.rows),
            split,
            path.name,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self.rows[idx]

        options = row["options"]
        sorted_keys = sorted(options.keys())
        letters = [chr(ord("A") + i) for i in range(len(sorted_keys))]
        key_to_letter = dict(zip(sorted_keys, letters))

        option_lines = "\n".join(
            f"{letter}. {options[k]}" for k, letter in zip(sorted_keys, letters)
        )
        prompt = f"Question: {row['question']}\n{option_lines}\nAnswer:"

        answer = key_to_letter.get(row["target_letter"], row["target_letter"])

        return {
            "prompt": prompt,
            "answer": answer,
            "knowledge_text": row["counterfactual_passage"],
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


def make_collate_fn(
    tokenizer: TokenizerType,
    max_seq_len: int = 512,
    knowledge_max_len: int = 256,
) -> Callable[[List[Dict[str, str]]], Dict[str, torch.Tensor]]:
    """构造 DataLoader 的 collate_fn。

    - 将 prompt+answer 拼接后做 batch tokenize（动态 padding）。
    - labels 策略: prompt 部分 = -100, padding = -100, 仅 answer token 计算 loss。
    - knowledge 文本单独 tokenize。

    参数:
        tokenizer: HuggingFace tokenizer。
        max_seq_len: prompt+answer 的最大 token 长度。
        knowledge_max_len: knowledge 文本的最大 token 长度。

    返回:
        collate_fn 函数，输入 List[Dict], 输出包含以下 key 的 dict:
            input_ids, attention_mask, labels,
            knowledge_input_ids, knowledge_attention_mask (均为 LongTensor)。
    """

    def collate_fn(
        batch: List[Dict[str, str]],
    ) -> Dict[str, torch.Tensor]:
        prompts = [item["prompt"] for item in batch]
        answers = [item["answer"] for item in batch]
        knowledge_texts = [item["knowledge_text"] for item in batch]

        # --- 1. tokenize prompt（用于计算 prompt 长度） ---
        prompt_enc = tokenizer(
            prompts,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_len,
        )
        prompt_lengths = [len(ids) for ids in prompt_enc["input_ids"]]

        # --- 2. 拼接 prompt+answer，batch tokenize with 动态 padding ---
        full_texts = [p + " " + a for p, a in zip(prompts, answers)]
        full_enc = tokenizer(
            full_texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )

        input_ids = full_enc["input_ids"].long()
        attention_mask = full_enc["attention_mask"].long()

        # --- 3. 构造 labels: prompt 部分和 padding 均为 -100 ---
        labels = input_ids.clone()
        for i, p_len in enumerate(prompt_lengths):
            labels[i, :p_len] = -100
        labels[attention_mask == 0] = -100

        # --- 4. knowledge tokenize with 动态 padding ---
        knowledge_enc = tokenizer(
            knowledge_texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=knowledge_max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels.long(),
            "knowledge_input_ids": knowledge_enc["input_ids"].long(),
            "knowledge_attention_mask": knowledge_enc["attention_mask"].long(),
        }

    return collate_fn
