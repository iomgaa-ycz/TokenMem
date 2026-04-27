---
type: paper
node_id: paper:memoryllm2024
title: "MemoryLLM: Towards Self-Updatable Large Language Models"
authors: ["Yu Wang et al."]
year: 2024
venue: "ICML 2024"
external_ids:
  arxiv: "2402.04624"
tags: ["self-updating", "latent-memory", "opaque"]
added: 2026-04-26T00:00:00Z
---

# MemoryLLM

## One-line thesis

将过去信息压缩为隐式hidden states形成~1B参数记忆池，推理时自更新。

## Method

- 存储: per-layer hidden state矩阵（7680×4096×32层 for Llama2-7B，~1B参数）
- 更新: 新文本处理后，最后K个hidden states替换随机丢弃的memory tokens
- 注入: 与hidden states拼接后self-attention

## Key Results

- Llama-2-7B上验证（1个模型）
- 知识编辑任务上超越ROME/IKE

## Limitations / Failure Modes

- 完全不可读/不可编辑/不可审计
- 不支持跨模型
- 容量有限（~20K tokens）
- 需要特殊训练

## Relevance to This Project

范式对手。TokenMem在每个维度上与MemoryLLM相反：可读 vs 不可读、可编辑 vs 不可控、跨模型 vs 单模型。MemoryLLM证明了记忆增强LLM的市场需求。
