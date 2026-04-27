---
type: paper
node_id: paper:fwpkm2026
title: "Fast-weight Product Key Memory"
authors: ["Zhao", "Jones"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2601.00671"
tags: ["product-key", "online-update", "fast-weight", "episodic-memory"]
added: 2026-04-26T00:00:00Z
---

# FwPKM

## One-line thesis

稀疏fast-weight记忆+Product Key检索+TTT风格chunk级在线梯度更新，实现动态情景记忆。

## Method

- 262,144稀疏记忆槽，512-dim learned value vectors
- Product Key检索O(√N)
- 在线更新: 推理时对activated slots做一步梯度下降(lr=1.0, MSE loss)
- 支持128K context

## Limitations / Failure Modes

- 不可读（连续向量，事后可解码>70%准确率但非原生可读）
- chunk级更新可能miss长距离依赖
- 不支持跨模型

## Relevance to This Project

FwPKM证明了PKM+在线更新可行。TokenMem差异：(1)token可读 vs 向量不可读；(2)FAISS检索 vs PKM（当前）；(3)直接编辑 vs 梯度更新。
