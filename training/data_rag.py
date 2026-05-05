"""RAG SFT 数据管线 —— 将 knowledge 直接拼接到 prompt 前作为上下文。

与 TokenMem collate (data.py) 的区别:
- TokenMem: knowledge 单独 tokenize 输出 knowledge_input_ids，通过 cross-attention 注入
- RAG SFT: knowledge 直接拼接到 prompt 文本前，作为标准 causal LM 输入

输出字典仅包含: input_ids, attention_mask, labels (均为 LongTensor)。
"""

import logging
from typing import Callable, Dict, List, Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


def make_rag_collate_fn(
    tokenizer: TokenizerType,
    max_seq_len: int = 1024,
    knowledge_max_len: int = 256,
) -> Callable[[List[Dict[str, str]]], Dict[str, torch.Tensor]]:
    """构造 RAG SFT 的 collate_fn。

    处理流程:
    1. knowledge_text 先截断到 knowledge_max_len tokens
    2. 将截断后的 knowledge 文本直接拼接到 prompt 前（无分隔符）
    3. 拼接 knowledge+prompt+answer 做 batch tokenize（动态 padding）
    4. labels: knowledge+prompt 部分 = -100, padding = -100, 仅 answer token 计算 loss

    参数:
        tokenizer: HuggingFace tokenizer 实例。
        max_seq_len: knowledge+prompt+answer 拼接后的最大 token 长度。
        knowledge_max_len: knowledge 文本截断的最大 token 长度。

    返回:
        collate_fn 函数，输入 List[Dict], 输出包含以下 key 的 dict:
            input_ids, attention_mask, labels (均为 LongTensor)。
    """

    def collate_fn(
        batch: List[Dict[str, str]],
    ) -> Dict[str, torch.Tensor]:
        prompts: List[str] = [item["prompt"] for item in batch]
        answers: List[str] = [item["answer"] for item in batch]
        knowledge_texts: List[str] = [item["knowledge_text"] for item in batch]

        # --- 1. knowledge 截断: tokenize → 截断 → decode 回文本 ---
        truncated_knowledges: List[str] = []
        for kt in knowledge_texts:
            k_ids = tokenizer.encode(
                kt,
                add_special_tokens=False,
                truncation=True,
                max_length=knowledge_max_len,
            )
            truncated_knowledges.append(
                tokenizer.decode(k_ids, skip_special_tokens=True)
            )

        # --- 2. 拼接 knowledge+prompt (无分隔符) ---
        context_texts: List[str] = [
            k + p for k, p in zip(truncated_knowledges, prompts)
        ]

        # --- 3. tokenize context (用于计算非 answer 部分长度) ---
        context_enc = tokenizer(
            context_texts,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_len,
        )
        context_lengths: List[int] = [len(ids) for ids in context_enc["input_ids"]]

        # --- 4. 拼接 context+answer, batch tokenize with 动态 padding ---
        full_texts: List[str] = [
            ctx + " " + ans for ctx, ans in zip(context_texts, answers)
        ]
        full_enc = tokenizer(
            full_texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )

        input_ids: torch.Tensor = full_enc["input_ids"].long()
        attention_mask: torch.Tensor = full_enc["attention_mask"].long()

        # --- 5. 构造 labels: context 部分和 padding 均为 -100 ---
        labels: torch.Tensor = input_ids.clone()
        for i, ctx_len in enumerate(context_lengths):
            labels[i, :ctx_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels.long(),
        }

    return collate_fn
