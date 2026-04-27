---
type: paper
node_id: paper:kblam2025
title: "KBLaM: Knowledge Base augmented Language Model"
authors: ["Microsoft Research"]
year: 2025
venue: "ICLR 2025"
external_ids:
  arxiv: "2410.10450"
tags: ["knowledge-injection", "frozen-llm", "structured-kb", "rectangular-attention"]
added: 2026-04-26T00:00:00Z
---

# KBLaM

## One-line thesis

将KB三元组编码为连续KV token，通过矩形注意力注入冻结LLM，支持即插即用的知识库增强。

## Method

- 知识格式: 结构化三元组 (subject, relation, object)
- 编码: ada-002 → per-layer线性adapter → 连续KV向量(4096-dim)
- 注入: 矩形注意力（language tokens attend all knowledge tokens，knowledge tokens不互相attend）
- 检索: 无显式检索（所有知识token参与注意力），O(M×N)
- 更新: 替换单个三元组的KV向量，无需重训练

## Key Results

- Llama-3-8B, Phi-3-mini上验证
- 支持10K+三元组 on single A100
- 动态KB更新无需微调

## Limitations / Failure Modes

- **必须结构化三元组**，不支持自然语言
- 连续向量不可读（"may fail to precisely generate text word by word"）
- 每个模型需独立训练adapter
- O(M×N)复杂度，不支持大规模知识库
- 只在3个模型上验证

## Relevance to This Project

架构竞品。TokenMem与KBLaM的核心差异：(1)自然语言token存储 vs 结构化三元组；(2)FAISS检索+top-k注入 vs 全量矩形注意力；(3)6模型验证 vs 3模型。
