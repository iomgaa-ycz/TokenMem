---
type: paper
node_id: paper:kcapsules2026
title: "Knowledge Capsules"
authors: ["Unknown"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2604.20487"
tags: ["knowledge-injection", "structured-memory", "provenance", "frozen-llm"]
added: 2026-04-26T00:00:00Z
---

# Knowledge Capsules

## One-line thesis

结构化、非参数化记忆单元（capsule），通过External KVI注入冻结LLM的注意力层，保留知识溯源。

## Method

- 存储: Capsule = (Subject, Relation, Object, Provenance) tuple，磁盘上JSON可读
- 运行时: 冻结LLM前向传播编译为per-layer KV tensor（运行时不可读）
- 检索: 图引导多跳遍历 + DRM相关性评分
- 注入: External KVI（KV前缀注入）
- 更新: 需重新运行extraction+KV编译pipeline

## Key Results

- Qwen2.5-7B, Mistral-7B上验证（2个模型）
- 超越RAG和GraphRAG

## Limitations / Failure Modes

- 磁盘可读但运行时不可读（编译为KV tensor）
- 无in-place编辑机制
- 极新（2026.04），实验有限
- 仅关系结构知识

## Relevance to This Project

最新竞品（2026年4月）。TokenMem差异：(1)全程token可读 vs 仅磁盘可读；(2)直接编辑 vs 需重编译；(3)自然语言存储 vs 关系tuple。NeurIPS审稿人可能会发现此论文。
