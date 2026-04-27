---
type: paper
node_id: paper:explicitlm2025
title: "ExplicitLM: Decoupling Knowledge from Parameters via Explicit Memory Banks"
authors: ["Chengzhang Yu", "Zening Lu", "Chenyang Zheng", "Chiyue Wang", "Yiming Zhang", "Zhanpeng Jin"]
year: 2025
venue: "ICLR 2026"
external_ids:
  arxiv: "2511.01581"
tags: ["explicit-memory", "token-storage", "product-key", "pretraining"]
added: 2026-04-26T00:00:00Z
---

# ExplicitLM: Decoupling Knowledge from Parameters via Explicit Memory Banks

## One-line thesis

百万规模token级显式记忆库+Product Key检索，将知识从模型参数中解耦，知识密集任务提升43.67%。

## Problem / Gap

LLM的参数化知识不可编辑、不可增长、不可审计。ExplicitLM提出将知识外部化为显式token序列存储。

## Method

- 记忆条目: token indices（词表V中的离散符号），最大序列长度L=16
- 检索: Product Key分解（笛卡尔积），O(√N)复杂度
- 注入: Gumbel-Softmax + straight-through estimator可微离散选择
- 训练: 全模型+记忆库联合预训练（Wikipedia/Gutenberg/OpenWebText, 10M条目）
- 更新: 20%冻结/80%EMA更新分割（freeze rate ρ=0.2）

## Key Results

- 知识密集任务: +43.67% (vs standard Transformer)
- 低数据场景(10K样本): 3.62x提升
- 百万级记忆库可行

## Limitations / Failure Modes

- 需要从零预训练（无法用于现有冻结LLM）
- 固定容量N=10^6，不支持动态增删
- 条目长度固定L=16
- 只在1个模型上验证
- 没有运行时编辑能力

## Reusable Ingredients

- Token级记忆存储的理念和有效性验证
- Product Key检索的key分解结构

## Relevance to This Project

**本组prior work。** TokenMem直接延续ExplicitLM的token记忆理念，但改为冻结LLM+轻量SFT方案。两者互补：ExplicitLM验证了范式有效性，TokenMem解决部署实用性。
