---
type: paper
node_id: paper:decoupledrag2025
title: "Decoupling Knowledge and Context: An Efficient and Effective Retrieval Augmented Generation Framework via Cross Attention"
authors: ["Anonymous"]
year: 2025
venue: "WWW 2025"
external_ids:
  arxiv: null
tags: ["cross-attention", "rag", "knowledge-injection", "frozen-llm"]
added: 2026-04-26T00:00:00Z
---

# DecoupledRAG

## One-line thesis

将检索知识从上下文中解耦，通过cross-attention在hidden-state层注入，解决VanillaRAG的长上下文效率和"lost in the middle"问题。

## Problem / Gap

VanillaRAG将检索文档拼接到prompt中：(1)上下文过长导致O(n²)注意力开销；(2)self-attention中信息丢失（lost-in-middle）；(3)文档排列顺序影响结果。

## Method

- **知识编码(离线)**: 用同一LLM对外部文档D做前向传播，缓存每层KV表示 K_D^(l), V_D^(l)
- **知识聚合(在线)**: last token的hidden state通过cross-attention与外部KV交互，生成x_{n,ext}
- **融合**: x_{n}^{(l+1)} = x_{n,int}^{(l)} + W_β^(l) · x_{n,ext}^(l)
- **W_β设计**: 低秩分解 W_β = α·A_β·B_β，B_β零初始化，A_β高斯初始化(σ=0.01)
- **排列不变性**: cross-attention无位置编码，文档顺序不影响结果
- **复杂度**: 知识编码O(N·|D|²)离线；在线推理O(|A|·(N·|D|))线性于文档数

## Key Results

- Llama-3-8B-Instruct上，20文档时DecoupledRAG全面超越VanillaRAG
- Slot Filling (T-REx): 80.2 vs 27.4 F1 (20 docs)
- Multi-hop QA: 38.1 vs 15.7 Acc (20 docs)
- 文档数增加时VanillaRAG性能下降，DecoupledRAG持续上升
- TPS: DecoupledRAG@50docs仍优于VanillaRAG@20docs

## 训练细节（代码验证）

```
可训练参数: 仅gate_crossattention（融合权重W_β）
  - LoRA rank=16, alpha=32, dropout=0.2
  - ~4.19M参数
基座LLM: 完全冻结（代码中基座LoRA已注释掉）
训练: 5 epochs, lr=1e-3, batch_size=16
数据: 各任务QA数据集 + Wikipedia知识库(21M docs, 256 tokens/doc)
检索器: RetroMAE
损失: LM loss + Retrieval loss
```

## Limitations / Failure Modes

- 无持久记忆bank（每次推理需实时编码文档KV或从缓存读取）
- 无知识编辑能力
- 无跨模型迁移
- 无检索模块（所有文档全部参与cross-attention）
- 文档少时(<5)VanillaRAG反而更好（cross-attention交互不够）
- 只在1个模型上验证

## Reusable Ingredients

- **Cross-attention融合机制**: W_β = α·A_β·B_β零初始化设计（直接复用）
- **基座完全冻结+只训练融合权重**: 验证了这种方案的可行性
- **只需SFT不需Pretrain**: 极大简化训练流程
- **离线KV缓存**: 知识表示可预计算和复用

## Relevance to This Project

**融合机制的直接来源。** TokenMem的GateCrossAttention模块直接借鉴DecoupledRAG的W_β设计。关键差异：TokenMem加入了持久记忆bank、FAISS检索、知识编辑和跨模型迁移——从"更好的RAG"升级为"完整的记忆pipeline"。
