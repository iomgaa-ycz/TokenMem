---
type: paper
node_id: paper:tokmem2025
title: "TokMem: One-Token Procedural Memory for Large Language Models"
authors: ["Zijun Wu", "Yongchang Hao", "Lili Mou"]
year: 2025
venue: "ICLR 2026"
external_ids:
  arxiv: "2510.00444"
tags: ["procedural-memory", "soft-token", "infix-placement"]
added: 2026-04-26T00:00:00Z
---

# TokMem

## One-line thesis

将可复用的任务过程编译为单个可训练memory token（soft embedding），通过infix placement注入。

## Method

- 存储: 可训练嵌入向量（memory matrix M ∈ R^{l×d}），不可读
- 检索: softmax over logits选择memory token
- 注入: infix placement [query; memory_token; response]
- 更新: renormalization添加新记忆

## Limitations / Failure Modes

- 存储不可读（连续嵌入）
- 面向过程记忆（procedural），非事实知识
- 不支持知识编辑

## Relevance to This Project

名称相似但方向不同。TokMem是procedural memory（不可读soft token），我们的TokenMem是factual memory（可读hard token）。需要在论文中明确区分命名。**考虑改名避免混淆。**
