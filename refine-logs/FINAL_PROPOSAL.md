# FINAL PROPOSAL v5: TokenMem — Faithful Knowledge Internalization via Cross-Attention

**Date**: 2026-05-04 (v5, 新增 RAG SFT 对照实验 + 贡献重组为 2+1)
**Target venue**: NeurIPS 2026
**Prior work**: ExplicitLLM (ICLR 2026, 本组), DecoupledRAG (WWW 2025, 机制来源)

---

## 论文标题（候选）

- *TokenMem: Faithful Knowledge Internalization for Frozen LLMs via Cross-Attention* ← **推荐**
- *Injecting Knowledge Through Independent Channels: Why Cross-Attention Outperforms In-Context Under Conflicts*

---

## 1. Problem Anchor

RAG（检索增强生成）是LLM使用外部知识的主流方案，但存在一个被忽视的结构性问题：**知识冲突（Knowledge Conflict）**。

当检索到的知识与模型参数记忆矛盾时，LLM在self-attention中同时处理两个冲突信号，导致输出不可预测（Longpre 2021, Xie 2024）。

**核心问题**: 能否为冻结LLM提供一条**不与参数记忆冲突的知识注入通道**？

---

## 2. Method

### 2.1 架构：Cross-Attention Adapter (LinearFusion)

```
RAG (self-attention — 共享通路):
  参数记忆 + Context知识 → 同一条self-attention → 知识冲突!

TokenMem (cross-attention — 独立通路):
  参数记忆 → self-attention (冻结)
  外部知识 → cross-attention (独立通道)
  LinearFusion gate: 两路融合 (W_β = α·A_β·B_β, 零初始化)
```

- 基座 LLM 全部冻结，仅训练 gate_crossattention (~3-5M 参数)
- Cross-attention 复用 LLM 自身 QKV 权重，全层注入

### 2.2 训练：CoT Curriculum SFT（两阶段）

```
Phase 1: 纯 News 50K CoT SFT (3 epochs)
  → 学习基础知识融合能力
  → checkpoint: best val_loss

Phase 2: News + Counterfactual 混合 SFT (40 epochs, early-stop)
  → 加载 Phase 1 checkpoint
  → 混入反事实知识 CoT 数据 (cf_arc_easy + cf_medqa, 2x oversample)
  → 学习"忠实遵从注入知识"的能力
```

### 2.3 与DecoupledRAG的关系

- **借鉴**: Cross-attention注入机制（LinearFusion gate 零初始化设计）
- **核心区别**: (1) Curriculum + CF 训练使模型学会 faithful compliance (2) 单域SFT跨域泛化 (3) 多模型多家族验证 (4) 反事实实验证明 faithfulness

### 2.4 RAG SFT 对照实验设计（新增）

为回答"TokenMem 的优势来自 cross-attention 通道，还是仅仅因为做了 SFT 训练？"这一核心问题，引入 RAG SFT 作为严格对照：

```
控制变量:
  - 同一基座模型 (frozen)
  - 同一训练数据 (News 50K + CF, CoT Curriculum)
  - 同一可训练参数量 (~3-5M, LoRA rank 匹配)
  - 唯一区别: 知识注入通道
    TokenMem: cross-attention (独立通道)
    RAG SFT:  in-context prompt (self-attention 共享通道) + LoRA adapter
```

**RAG SFT 具体设计**:
- 知识文本经 LLMLingua-2 压缩至 64 tokens，放入 prompt
- 冻结 LLM + LoRA adapter（参数量与 TokenMem gate 匹配）
- 同一 Curriculum 训练流程 (Phase 1 → Phase 2)
- 评测时同样使用 CoT + /no_think

**科学意义**: 如果 RAG SFT KC 显著低于 TokenMem KC，则证明通道独立性（而非训练本身）是 faithfulness 的关键因素。

---

## 3. Claims

### C1（方法贡献）: TokenMem 完整 plug-and-play 记忆系统

Cross-attention adapter + CoT Curriculum SFT = 一套可插拔的知识注入系统，适用于任意冻结 LLM。

- 架构: LinearFusion gate，零初始化，全层注入
- 训练: 两阶段 Curriculum (ID → ID+CF)
- 部署: 0 context token 开销，知识通过独立通道注入

**4B 实际数据**: cf_arc_easy KC 69.0%, cf_medqa KC 70.2%, News +41.4pp, MMLU +4.0pp
**状态**: ✅ 4B 验证通过。待 8B/14B/LLaMA/OLMo 复现。

### C2（控制实验发现）: 通道独立性是 faithfulness 的关键因素

在匹配模型、数据、参数量、训练流程的条件下，cross-attention 通道的 KC 显著高于 in-context 通道。

| 方法 | 通道 | 训练 | 参数量 | KC (预期) |
|------|------|------|--------|-----------|
| Vanilla RAG | in-context | 无 | 0 | ~20-52% |
| **RAG SFT** | in-context | CoT Curriculum + LoRA | ~3-5M | **TBD** |
| **TokenMem** | cross-attn | CoT Curriculum + gate | ~3-5M | **≈69-70%** |

**关键对比**: TokenMem KC ≈ 69-70% vs RAG SFT KC = TBD

**附带发现**: VanillaRAG 呈现 inverse scaling（模型越大，KC 越低），暗示更强的参数记忆 prior 加剧 self-attention 通道内的冲突。

**状态**: 🔄 RAG SFT 实验待运行

### Minor: KC 指标 + CoT 评测协议（评测方法论，非独立贡献）

- **Knowledge Compliance (KC)**: 反事实数据集上 %跟随注入知识的回答，直接衡量知识通道的忠实度
- **CoT 评测协议**: MCQ logprob 对知识冲突有天花板效应 (94%+)，CoT 自由生成评测揭示真实 compliance (28-36% 基线)
- 作为 §3.5 评测设置呈现，不作为核心贡献 bullet point

---

## 4. 实验设计

### 4.1 模型矩阵（最终版）

| 模型 | 家族 | 参数量 | 角色 | 训练状态 |
|------|------|--------|------|---------|
| **Qwen3-8B** | Qwen | 8B | **核心模型** | 🔄 训练中 |
| Qwen3-4B | Qwen | 4B | 同家族小规模 | ✅ 完成 |
| Qwen3-14B | Qwen | 14B | 同家族大规模 | 🔄 训练中 |
| LLaMA-3.1-8B | Meta | 8B | 跨家族对比 | 🔄 训练中 |
| OLMo-3-7B | AI2 | 7B | 跨家族对比 | 🔄 训练中 |

### 4.2 数据集矩阵（主表 5 个）

| 数据集 | 类型 | 规模 | 目的 |
|--------|------|------|------|
| News | In-domain | 8,663 | 训练域性能 |
| MMLU | OOD-General | 14,320 | 综合性知识泛化 |
| MedQA | OOD-Specialist | ~1,300 | 专业领域泛化 |
| cf_arc_easy | Counterfactual | 2,745 | C1/C2: 反事实遵从率 |
| cf_medqa | Counterfactual | 1,146 | C1/C2: 反事实遵从率 |

ARC / ARC-Easy: Appendix 全量报告。

### 4.3 方法对比（4 条件）

| 方法 | 知识注入方式 | 训练 | 可训练参数 |
|------|------------|------|-----------|
| No-Memory | 无外部知识 | 无 | 0 |
| Vanilla RAG | 知识压缩后放入 prompt (LLMLingua-2, 64tok) | 无 | 0 |
| **RAG SFT** | 知识压缩后放入 prompt + LoRA adapter | CoT Curriculum SFT | ~3-5M |
| **TokenMem** | 知识通过 cross-attention 注入 | CoT Curriculum SFT | ~3-5M |

### 4.4 消融实验（4 组）

| 消融 | 设计 | 目的 |
|------|------|------|
| Phase 1 vs P1+P2 | Phase 1 only checkpoint 对比 Phase 2 | Curriculum + CF 训练的必要性 |
| RAG SFT vs TokenMem | 匹配参数量/数据/训练，仅通道不同 | **通道独立性是 KC 提升的因果证据** |
| 注入层数 | 全层 / 12层 / 4层 / 1层 | 架构选择 justification |
| 训练数据量 | 10K / 25K / 50K | 数据效率 |

### 4.5 评测协议

```
推理: CoT + /no_think, max_new_tokens=2048
知识压缩 (RAG / RAG SFT): LLMLingua-2, target_token=64
答案提取: regex multi-pattern ("The answer is X")
指标:
  - Accuracy (正常数据集)
  - Knowledge Compliance KC (反事实数据集: %跟随注入知识)
  - Conflict Rate (反事实: %既不跟随知识也不跟随参数记忆)
```

---

## 5. 论文结构 (NeurIPS, 9页 + appendix)

| Section | 页数 | 内容 |
|---------|------|------|
| §1 Introduction | 1.5 | Problem + Contribution bullet points (C1 方法 + C2 发现) |
| §2 Related Work | 1.0 | RAG知识冲突 + 知识注入方法 + Memory系统 |
| §3 Method | 2.0 | 架构 + 训练策略 + 与DecoupledRAG关系 + §3.5 评测方法论 (KC指标 + CoT协议) |
| §4 Experiments | 3.0 | §4.1 主表(5模型×5数据集) + §4.2 **控制通道对比** (4方法表: No-Mem/RAG/RAG-SFT/TokenMem) + §4.3 消融 + §4.4 分析 |
| §5 Discussion | 1.0 | Trade-off分析 + Limitations |
| §6 Conclusion | 0.5 | Summary |
| Appendix | ∞ | ARC/ARC-Easy结果 + 超参 + 生成示例 |

---

## 6. 审稿人预案

| 攻击 | 防御 |
|------|------|
| "DecoupledRAG + FAISS" | 诚实 credit。新发现: faithfulness + curriculum训练 + 跨域泛化 + 多模型 |
| "trained vs untrained不公平" | **RAG SFT 对照实验直接回答**: 同样训练、同样参数量，通道不同→KC 不同。这是因果证据，不是 confound |
| "Compliance高=模型太弱" | 正常数据集上 TokenMem > No-Memory 证明模型在用知识且有判断力 |
| "反事实=gullibility/安全风险" | Faithfulness 是通道可控性诊断，不是安全claim |
| "KC高=盲目服从(blind obedience)" | Memory 系统语境下，存储的知识是**用户授权的**（类似数据库写入），高 compliance 是设计目标而非缺陷。与 prompt injection 攻击场景本质不同：这里知识源是可信的 |
| "OOD提升小" | Recovery Rate一致 + in-domain验证通道有效 |
| "无效率数据" | 补测延迟/显存 (TokenMem 0 context token vs RAG 64-256) |

---

## 7. Go/No-Go 判据（基于 RAG SFT 实验结果）

RAG SFT 的 KC 结果决定 C2 (通道因果 claim) 的强度：

| RAG SFT KC | 判定 | 策略 |
|------------|------|------|
| < 40% | **强 (Strong)** | Cross-attn 优势 ≥30pp，因果 claim 成立，C2 作为核心贡献 |
| 40-55% | **可用 (Usable)** | 优势 15-30pp，claim 需谨慎措辞（"substantially higher"），仍可作为核心发现 |
| 55-65% | **弱 (Weak)** | 优势 <15pp，降级 C2 为消融发现，论文重心回到 C1 系统贡献 |
| > 65% | **放弃因果 claim** | 通道差异不显著，删除 C2，论文仅保留 C1 (系统) + 工程贡献 |

---

## 8. v4→v5 变更记录

| 变更 | 原因 |
|------|------|
| 新增 RAG SFT 对照实验 (P0) | v4 最大漏洞: "trained vs untrained" 无法回答。RAG SFT 匹配训练/数据/参数量，仅通道不同，提供因果证据 |
| 贡献重组: 3 claims → 2+1 | C1(方法系统) + C2(控制实验发现) 为核心; KC指标+CoT评测协议降级为 Minor (评测方法论，融入 §3.5) |
| 方法对比: 3→4 条件 | 新增 RAG SFT 作为第四个 baseline |
| 消融 A2 更新 | 原 Conflict-conditioned 消融替换为 RAG SFT vs TokenMem 通道对比（更直接的因果证据） |
| §4.2 改为"控制通道对比" | 4方法表 (No-Mem/RAG/RAG-SFT/TokenMem) 成为实验核心子章节 |
| 评测方法论不再独立成节 | KC + CoT 协议融入 §3.5，不占用贡献 bullet point |
| 审稿人预案更新 | "trained vs untrained" 现有 RAG SFT 因果证据回答; 新增 "blind obedience" 防御 |
| 新增 Go/No-Go 判据 | 基于 RAG SFT KC 结果的四档决策框架，避免实验结果不支持时硬推 claim |
| 标题候选更新 | "Injection" → "Internalization"; 新增通道对比视角候选标题 |
