# FINAL PROPOSAL v3: TokenMem — Faithful Knowledge Injection

**Date**: 2026-04-28 (v3, post GPT-5.4 3-round review)
**Target venue**: NeurIPS 2026 (deadline 2026-05-03)
**Prior work**: ExplicitLM (ICLR 2026, 本组), DecoupledRAG (WWW 2025, 机制来源)

---

## 论文标题（候选）
- *TokenMem: Faithful Knowledge Injection into Frozen LLMs via Cross-Attention Adapters*
- *Cross-Attention as a Faithful Knowledge Channel for Frozen LLMs*
- *Learning to Trust External Knowledge: Cross-Attention Adapters for Conflict-Free Knowledge Injection*

---

## 1. Problem Anchor

RAG（检索增强生成）是LLM使用外部知识的主流方案，但存在一个被忽视的结构性问题：**知识冲突（Knowledge Conflict）**。

当检索到的知识与模型参数记忆矛盾时（例如：context说"日本首都是西京"，但模型知道"东京"），LLM在self-attention中同时处理两个冲突信号，导致：
- 既不忠实跟随外部知识
- 也不确信参数记忆
- 输出不可预测（知识冲突文献: Longpre 2021, Xie 2024）

**核心问题**: 能否为冻结LLM提供一条**不与参数记忆冲突的知识注入通道**？

---

## 2. Method Thesis

> **TokenMem**: 通过cross-attention adapter为冻结LLM开辟一条**独立的知识通道**。
> 该通道与self-attention的参数记忆通路分离，使模型能**忠实地遵从注入的知识**——
> 无论该知识与参数记忆是否一致。

### 机制解释

```
RAG (in-context, self-attention — 共享通路):
  参数记忆: "东京" ←──┐
                       ├─ 同一条self-attention → 知识冲突!
  Context知识: "西京" ←─┘
  → 模型左右互搏，输出不可靠

TokenMem (cross-attention — 独立通路):
  参数记忆: "东京" ─── self-attention (冻结，不受干扰)
  外部知识: "西京" ─── cross-attention (专门训练的独立通道)
                        │
                        ▼
  LinearFusion gate: 学会"cross-attn给的信号→用它"
  → 两条通路分离 → 不互搏 → 知识被忠实使用
```

### 与DecoupledRAG的关系（诚实声明）

- **借鉴**: Cross-attention注入机制（LinearFusion gate W_β = α·A_β·B_β 零初始化设计）
- **新发现**: (1) 单域SFT跨域泛化 (2) 多模型验证 (3) **知识遵从性(compliance)分析——faithfulness发现**
- DecoupledRAG是per-task SFT，不研究知识冲突，不做counterfactual实验

---

## 3. 核心Claims（v3修订）

### C1: 忠实的知识注入（核心科学发现）

Cross-attention注入的Knowledge Compliance显著高于RAG in-context注入。

**评价指标**:
- **Knowledge Compliance (KC)**: 模型回答与注入知识支持的答案一致的比例
  - 正确知识: KC = accuracy
  - 反事实知识: KC = %回答知识支持的错误答案
- **Conflict Rate**: %回答既非正确也非知识支持的答案（互搏证据）

**预期结果**:

| 方法 | 条件 | Accuracy | Compliance | Conflict Rate |
|------|------|----------|------------|---------------|
| No-Memory | — | ~57% | N/A | N/A |
| TokenMem | 正确知识 | ~71% | ~71% | ~5% |
| TokenMem | 反事实知识 | ~25% (低,预期) | ~65% **(高!)** | ~10% (低) |
| RAG | 正确知识 | ~98% | ~98% | ~2% |
| RAG | 反事实知识 | ~35% | ~40% **(低!)** | ~25% **(高!)** |

**解读**: TokenMem反事实accuracy低是预期的（模型跟随了错误知识→"答错了"但compliance高→说明知识通道在工作）。RAG反事实compliance低+conflict高→知识冲突导致模型两边都不跟。

**状态**: ❌ 未验证 — 需要counterfactual compliance实验

### C2: 跨领域泛化

News 50K SFT → MedQA +14pp, ARC +4.3pp, MMLU +8.4pp (4B)。
adapter学到的是domain-agnostic的知识利用技能。

**状态**: 🔄 部分支持 — 仅Qwen3-4B/8B

### C3: 多模型通用性

6模型 × 3家族（Qwen3-0.6B/1.7B/4B/8B, Gemma3-1B, Ministral-3B）。

**状态**: ❌ 未验证 — 仅2/6模型完成

### C4: 知识敏感性（C1的前提条件）

Oracle >> Shuffled ≈ Empty → adapter确实使用知识内容。
没有C4，C1的高compliance可能被解释为"adapter太弱，什么都忽略"。

**状态**: ❌ 未验证

---

## 4. 实验计划

### 4.1 P0 命脉实验

**E-C4: Knowledge Sensitivity (C1的前提)**
```
条件: Oracle / Shuffled(随机错配知识) / Empty(无知识)
模型: Qwen3-4B, 8B
数据: MedQA, ARC, MMLU
指标: Accuracy
预期: Oracle >> Shuffled ≈ Empty
时间: ~3h
```

**E-C1: Counterfactual Compliance (核心发现)**
```
数据准备: 对每道MCQ生成反事实知识段落（minimal-edit风格）
  - 用DeepSeek V4 Flash生成支持错误答案B的段落
  - 要求与正确段落结构相似，仅改关键事实
条件: 正确知识 / 反事实知识
方法: TokenMem / VanillaRAG / Strong-prompt RAG / No-Memory
指标: Accuracy / Knowledge Compliance / Conflict Rate
模型: Qwen3-4B (主), 8B (复制)
时间: ~6h（含数据生成）
```

**E3: 公平基线（防止"trained vs untrained"攻击）**
```
Strong-prompt RAG:
  prompt: "请只根据以下段落回答，即使与你所知不同：{passage}"
  无需额外训练，仅修改prompt
时间: ~2h
```

**E2 Conflict-Conditioned 分层分析（v3.1新增）**:
```
将题目按No-Memory准确率分为两组:
  High-Prior: 模型本来答对的题（强参数记忆 → conflict激烈）
  Low-Prior: 模型本来答错的题（弱参数记忆 → conflict温和）

数据集选择:
  ARC (常识): No-Memory 4B=86.8% → 大量High-Prior → conflict激烈
  MedQA (专业): No-Memory 4B=57.2% → 大量Low-Prior → conflict温和

预期: High-Prior组的KC差距 > Low-Prior组的KC差距
  → 证明TokenMem优势来自"避免知识冲突"，不是"填充不确定性"
```

**E7: 效率数据（v3.1提升到P0）**
```
准确率输RAG → 必须有效率维度硬数据作为使用理由
测量: 延迟(ms) / 峰值显存(MB) / context tokens consumed
时间: ~2h
```

### 4.2 P1 支撑实验

| 实验 | 时间 | 对应Claim |
|------|------|----------|
| 剩余4模型SFT+评测 | ~8h | C3 |
| 8B MMLU补完 | ~2h | C2/C3 |
| Domain-SFT消融 | ~4h | C2分析 |

### 4.3 P2 深度分析

| 实验 | 时间 | 说明 |
|------|------|------|
| Logit Lens分析 | ~4h | 逐层decode看RAG vs TokenMem在反事实下的"思考"差异 |
| Gate激活分析 | ~2h | 正确vs反事实知识下gate行为是否一致 |
| 因果追踪 | ~3h | 逐层关闭cross-attn测compliance变化 |
| 注入层消融 | ~4h | 1/4/12/全层 |

---

## 5. Timeline (2026-04-28 → 2026-05-03)

| Day | 日期 | 工作 | 产出 |
|-----|------|------|------|
| 3 | 04/29 | 反事实数据生成 + E1(Sensitivity) + E2开始 + 剩余模型SFT(并行) | E1结果 + counterfactual data |
| 4 | 04/30 | E2完成(含conflict分层) + E3(strong prompt) + E7(效率) + 剩余模型评测 | **E2核心结果 + 决策** |
| 5 | 05/01 | 消融(E5) + Domain-SFT(E6) + 论文写作开始 | 分析数据 + §1-§3 |
| 6 | 05/02 | 论文写作（全天） | 初稿 |
| 7 | 05/03 | 定稿 + 排版 + 提交 | 提交 |

---

## 6. Go/No-Go 决策点

### Day 3 晚 (E-C4结果)
| 结果 | 行动 |
|------|------|
| Oracle >> Shuffled ≈ Empty | ✅ C4成立，继续E-C1 |
| Oracle ≈ Shuffled | ❌ adapter未使用知识。**停止C1实验**，重新评估 |

### Day 4 晚 (E-C1结果)
| 结果 | 行动 |
|------|------|
| TokenMem-CounterKC >> RAG-CounterKC | ✅ **论文核心成立**，全力写作 |
| TokenMem-CounterKC ≈ RAG-CounterKC | ⚠️ faithfulness故事不成立。回退到C2+C3为主 |
| Strong-prompt RAG抹平差距 | ⚠️ 说明是训练效应非架构效应。需要trained prompt baseline |

---

## 7. 审稿人攻击预案

| 攻击 | 防御 |
|------|------|
| "DecoupledRAG + FAISS" | 诚实credit机制。新发现: faithfulness, 跨域泛化, 多模型。这是finding paper |
| "RAG也有持久化/可编辑" | 不作为差异化claim。唯一结构差异: 注入方式(cross-attn vs in-context) |
| "Compliance高=adapter太弱" | C4(Oracle>>Empty)证明adapter在用知识。C4+C1组合排除此解释 |
| "trained vs untrained不公平" | Strong-prompt RAG + trained prompt baseline |
| "反事实段落有artifact" | minimal-edit生成 + 与正确段落同格式同长度 |
| "高compliance to false knowledge = gullibility/安全风险" | **"刀无罪"论证**：Faithfulness/Compliance是知识通道的固有属性（可控性），不是安全claim。一个忠实的翻译器会翻译任何文本，包括错误信息——这不是翻译器的问题，而是输入的问题。同理，TokenMem的高compliance说明知识通道**可控**——用户放什么知识进去，模型就用什么。知识的正确性是上游（检索器/知识库管理）的责任，不是注入通道的责任。RAG的低compliance反而说明in-context通道**不可控**——你放了知识进去但模型不一定用。 |
| "OOD提升小(ARC +4pp)" | ARC headroom仅12.8pp; Recovery Rate跨OOD一致(~33%) |
| "没有效率数据" | E7提升到P0，补测延迟/显存/throughput |
| "faithfulness to false knowledge不是好事" | frame为"可控性诊断"——测量知识通道的权威性，不是鼓励使用错误知识 |

---

## 8. 本方案明确不做的事

- **不claim "TokenMem > RAG on accuracy"** — 准确率差距是结构性的
- **不claim "RAG缺少持久化/可编辑"** — 审稿人已击穿此论点
- **不claim "零context token是质变"** — 降为tradeoff表格中一行
- **不做multi-hop QA** — 当前单记忆注入设计
- **不做KBLaM/MemoryLLM直接实验对比** — 不同regime，Related Work论证划开
- **不过度claim机制解释** — 说"reduced competition"不说"no conflict"
