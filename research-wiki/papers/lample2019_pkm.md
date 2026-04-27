---
type: paper
node_id: paper:lample2019_pkm
title: "Large Memory Layers with Product Keys"
authors: ["Guillaume Lample", "Alexandre Sablayrolles", "Marc'Aurelio Ranzato", "Ludovic Denoyer", "Hervé Jégou"]
year: 2019
venue: "NeurIPS 2019"
external_ids:
  arxiv: "1907.05242"
tags: ["product-key", "memory-layer", "efficient-retrieval", "language-modeling"]
added: 2026-04-26T00:00:00Z
---

# Large Memory Layers with Product Keys

## One-line thesis

通过product key分解实现O(√N)精确最近邻搜索的大规模记忆层，替换Transformer的FFN层，12层+memory超越24层baseline。

## Method

- Product Key: 两个子key集的笛卡尔积 K = {(c,c') | c∈C, c'∈C'}
- 检索: query split为两半，分别在两个子集上找top-k → k²候选 → rerank
- 复杂度: O(√|K| × d_q) vs brute-force O(|K| × d_q)
- 集成: 替换FFN层为PKM层（residual连接）
- 训练: 全模型+记忆联合训练（Common Crawl 28B words, 140GB）
- Multi-head: 多个独立query head共享values

## Key Results

- 262K-slot PKM + 12层Transformer > 24层baseline（ppl 15.62 vs 16.02）
- 推理速度2x
- 记忆大小可独立于模型参数scaling

## Limitations / Failure Modes

- 记忆值是可学习参数（不是外部知识）
- 需要联合预训练
- key聚类平衡性问题（随机分组→平均后key退化为噪声）
- 原始实现针对语言建模，非知识注入

## Reusable Ingredients

- **Product Key分解结构**: O(√N)检索的理论基础（当前TokenMem用FAISS替代，PKM留future work）
- **Residual记忆集成**: 记忆输出通过residual加到hidden states

## Relevance to This Project

检索效率的理论来源。ExplicitLM已成功应用PKM。TokenMem当前用FAISS替代（因key聚类平衡性问题），但PKM的product key分解仍是future work的方向。
