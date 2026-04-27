---
type: paper
node_id: paper:mplus2025
title: "M+: Extending MemoryLLM with Scalable Long-Term Memory"
authors: ["Yu Wang et al.", "IBM Research"]
year: 2025
venue: "ICML 2025"
external_ids:
  arxiv: "2502.00592"
tags: ["self-updating", "retriever", "long-term-memory"]
added: 2026-04-26T00:00:00Z
---

# M+

## One-line thesis

在MemoryLLM基础上加入co-trained retriever，将记忆容量从20K扩展到160K+ tokens。

## Method

- 继承MemoryLLM的hidden state记忆池
- 新增: MLP-projected dense retriever (d_proj=204)
- 长期存储: 检索索引的hidden state segments

## Limitations / Failure Modes

- 仍然不可读/不可编辑
- 不支持跨模型
- retriever增加推理开销

## Relevance to This Project

MemoryLLM系列的进化，展示了记忆容量扩展需求。TokenMem用外部bank天然支持任意规模。
