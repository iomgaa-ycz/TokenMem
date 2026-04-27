# Idea Discovery Report

**方向**: 面向冻结LLM的即插即用内部化记忆系统
**日期**: 2026-04-26
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → refine
**目标会议**: NeurIPS 2026
**Prior Work**: ExplicitLM (ICLR 2026, 本组)

---

## Executive Summary

TokenMem是一个完整的、面向冻结LLM的内部化记忆pipeline：知识检索（FAISS）→ 知识融合（Cross-Attention，借鉴DecoupledRAG）→ 知识更新（token编辑）。核心卖点是**完整pipeline在6个模型×3个家族上的通用性**。知识以token_ids + embedding向量存储在per-model的TokenMemoryBank中（纯tensor，~11KB/条），可通过decode→re-encode在不同模型间迁移。每个LLM通过轻量adapter（仅训练cross-attention融合权重，基座完全冻结）获得使用记忆库的能力。只需一次SFT（News 50K），即可泛化到医学/科学/通用知识等不同领域。

---

## 一、文献全景

### 1.1 核心竞品定位

| 系统 | 存储 | 检索 | 注入 | 编辑 | 跨模型 | 冻结LLM | 与我们的关系 |
|------|------|------|------|------|--------|---------|------------|
| **ExplicitLM** (我们, ICLR26) | token序列 | PKM | Gumbel-Softmax | EMA替换 | ❌ | ❌需预训练 | **直接前驱** |
| **DecoupledRAG** (WWW25) | 预计算KV | 无(全注入) | Cross-Attention | ❌ | ❌ | ✅ | **融合机制来源** |
| **KBLaM** (ICLR25) | 压缩KV向量 | 矩形注意力 | 每层注入 | 替换三元组 | ❌ | ✅ | 架构竞品 |
| **K-Capsules** (Apr26) | 关系tuple→KV | 图遍历 | KVI | 模块增删 | ❌ | ✅ | 最新竞品 |
| **MemoryLLM** (ICML24) | 隐式hidden states | 自注意力 | 全层拼接 | 自动(不可控) | ❌ | ❌ | 范式对手 |
| **FwPKM** (Jan26) | 稀疏向量槽 | PKM | 门控加法 | 梯度更新 | ❌ | ❌ | 检索竞品 |

### 1.2 同类论文基线实况

| 论文 | 基线数量 | 具体基线 | 模型数 |
|------|---------|---------|--------|
| ExplicitLM (我们, ICLR26) | 1 | 标准Transformer (无记忆) | 1 |
| DecoupledRAG (WWW25) | 2 | VanillaRAG, RAG FT | 1 |
| KBLaM (ICLR25) | 3 | ICL, RAG, BM25 | 3 |
| K-Capsules (Apr26) | 4 | LLM, RAG, GraphRAG, KV Prefix | 2 |
| MemoryLLM (ICML24) | 4 | FT, FT-L, ROME, IKE | 1 |

**关键发现**：大多数论文只有2-4个基线；没有一篇做过跨系统对比；模型覆盖最多的是KBLaM（3个模型）。我们的6模型×3家族将显著超越所有竞品的模型覆盖范围。

### 1.3 识别到的关键空白

1. **多模型通用性 gap**: 最多的KBLaM只测了3个模型，没有系统在6+模型上证明通用性
2. **即插即用 gap**: ExplicitLM需预训练→没有人做过"给冻结LLM加token记忆"且只需SFT
3. **完整pipeline gap**: 现有工作只覆盖检索/融合/更新中的1-2个，没有完整pipeline
4. **动态管理 gap**: ExplicitLM固定容量N=10^6→没有在线增删编辑

### 1.4 DecoupledRAG关键技术细节（代码验证）

DecoupledRAG的训练方式（代码确认）：
- **只需SFT，不需要Pretrain**
- 基座LLM**完全冻结**（不使用LoRA微调基座q/k/v/o）
- **只训练gate_crossattention**（融合权重 W_β = A_β @ B_β，~4.19M参数）
- 零初始化B_β，高斯初始化A_β（σ=0.01）
- 训练：5 epochs, lr=1e-3, batch_size=16
- SFT数据：各任务的QA数据集（28K-167K）
- 知识库：Wikipedia 21M文档，每段256 tokens
- 代码中基座LoRA配置已被注释掉（仅存在于另一个VanillaRAG微调的Trainer类中）

---

## 二、方案详情

### 2.1 推荐方案：TokenMem

**一句话**: 一个完整的检索→融合→更新pipeline，让任意冻结LLM通过轻量SFT获得可读、可编辑、可跨模型迁移的token级内部化记忆。

### 2.2 核心贡献

**C1: 完整的记忆pipeline**
> 知识检索（FAISS）→ 知识融合（Cross-Attention注入，借鉴DecoupledRAG）→ 知识更新（token级增/删/编辑）。三个阶段组成完整的知识生命周期管理。

**C2: 多模型通用性**
> 在6个模型×3个家族（Qwen3-0.6B/1.7B/4B/8B, Gemma3-1B, Ministral-3B）上验证pipeline有效性——超越所有现有工作的模型覆盖范围。

**C3: 一次SFT，跨领域泛化**
> 在News数据集上SFT一次，adapter即可泛化到MedQA/ARC/MMLU等完全不同的领域——证明adapter学到的是"如何使用记忆"的通用能力，而非特定领域知识。

**C4: 知识可读性、可编辑性与跨模型迁移**
> 知识以tokenizer编码后的token序列存储，可通过detokenize→retokenize在不同模型间迁移。支持直接审计和编辑，编辑后仅需重算单条KV缓存即可生效。

### 2.3 架构设计

```
            TokenMemoryBank (per-model)
            ┌──────────────────────────────┐
            │ token_ids [fusion_length]     │
            │ cached_emb [emb_dim]          │
            └──────┬───────────────────────┘
                   │ FAISS余弦检索 top-k
                   ▼
          token_ids → frozen LLM → 逐层hidden states (strided sampling)
                   │
                   ▼
        ┌──────────────────────────────────────┐
        │  Frozen LLM + LinearFusion门控       │
        │  (gate_crossattention, 唯一可训练)    │
        │  全部层注入(复用LLM自身QKV做cross-attn)│
        └──────────────────────────────────────┘
                   │
                   ▼
                Output
```

**跨模型迁移机制**：
```
ModelA的TokenMemoryBank
  → tokenizer_A.decode(token_ids) → 文本
  → tokenizer_B.encode(文本) → 新token_ids
  → ModelB.embedding(新token_ids).mean() → 新cached_emb
  → 存入ModelB的TokenMemoryBank
```
共享的是**自然语言知识内容**，各模型独立tokenize和embedding。

### 2.4 训练方式（借鉴DecoupledRAG，代码已验证）

```
一次性准备:
  1. 构建TokenMemoryBank: knowledge text → tokenize → 存储token_ids + cached_emb
  2. cached_emb = embedding(token_ids).mean(dim=0) → 用于FAISS检索
  3. 推理时: 检索top-k → token_ids过frozen LLM → 实时得到KV → cross-attention注入

一次性SFT（per-model）:
  数据: News train 50K（时间分割，较早文章）
  输入: question + cross-attention(知识KV)
  输出: next-token prediction → answer
  可训练: 仅gate_crossattention（融合权重）
  冻结: 基座LLM全部参数
  配置: 5 epochs, lr=1e-3

推理（零成本切换知识）:
  换任意领域知识bank → 直接推理，不重训练
```

### 2.5 与ExplicitLM的关系

ExplicitLM（ICLR 2026, 本组prior work）证明了token级显式记忆+PKM检索的有效性，但需要从零预训练。TokenMem延续"token级记忆"理念，但采用完全不同的集成方式——冻结LLM + 轻量SFT + Cross-Attention注入——使其成为即插即用的通用方案。两者是互补关系（预训练方案 vs 后置适配方案），不做直接性能对比。

---

## 三、被消除的方案

| 方案 | 消除原因 | 阶段 |
|------|---------|------|
| MemoryBridge (纯跨模型) | 单点贡献，不够饱满 | Phase 2 |
| EditMem (纯编辑) | 与知识编辑领域reviewer期望有gap | Phase 2 |
| ConvoMem (对话记忆) | 知识提取质量是混淆变量 | Phase 2 |
| PKM-Fusion (纯检索) | 与FwPKM(2026.01)重叠 | Phase 2 |

---

## 四、风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| 某些模型adapter训练不收敛 | 20% | 中 | 调整学习率/注入层数；从6模型中去掉该模型 |
| 小模型(0.6B)提升不明显 | 25% | 低 | 小模型基础能力弱是已知的；分析性讨论 |
| 大模型(8B)训练时间超预期 | 20% | 中 | 优先保证4B完成；8B放supplementary |
| Novelty被质疑为"工程组合" | 40% | 中 | 靠6模型通用性+跨领域泛化实验结果说话 |
| Out-of-domain泛化效果差 | 20% | 高 | 增加SFT数据多样性；或混入少量target domain数据 |

---

## 五、论文结构

```
1. Introduction
   - LLM知识增强的三种路径及其局限
   - 完整的记忆pipeline需求（检索→融合→更新）
   - TokenMem：面向冻结LLM的即插即用内部化记忆
   - 核心结果预览（6模型×3家族 + 跨领域泛化）

2. Related Work
   - RAG及其变体（VanillaRAG, DecoupledRAG）
   - 显式记忆系统（ExplicitLM, MemoryLLM, Memory³, KBLaM）
   - 参数高效适配（LoRA, Adapter）

3. Method: TokenMem Pipeline
   - 3.1 TokenMemoryBank: token级知识存储
   - 3.2 知识检索: FAISS top-k
   - 3.3 知识融合: Cross-Attention + 零初始化融合权重
   - 3.4 知识更新: 增/删/编辑
   - 3.5 多模型适配与知识迁移

4. Experiments
   - 4.1 设置（6模型、4数据集、基线、训练策略）
   - 4.2 核心结果：多模型注入性能（E1）
   - 4.3 跨领域泛化：News训练→MedQA/ARC/MMLU测试
   - 4.4 同家族Scaling分析（Qwen3 0.6B→8B）
   - 4.5 跨家族通用性（Qwen vs Gemma vs Ministral）
   - 4.6 知识编辑验证（E2）
   - 4.7 消融：注入层、adapter规模、基座LoRA

5. Analysis & Discussion

6. Conclusion
```
