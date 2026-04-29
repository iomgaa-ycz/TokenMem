# Idea Discovery Report (v3 — Faithful Injection Framing)

**方向**: 面向冻结LLM的即插即用内部化记忆系统
**日期**: 2026-04-26（v1）→ 2026-04-28（v3 major revision）
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → refine → GPT-5.4 external review (3 rounds)
**目标会议**: NeurIPS 2026
**Prior Work**: ExplicitLM (ICLR 2026, 本组)

---

## Executive Summary

TokenMem是一个面向冻结LLM的内部化记忆系统：知识检索（FAISS）→ 知识融合（Cross-Attention，借鉴DecoupledRAG）→ 知识管理（token级增删编辑）。

**v3 核心转变**：经过 GPT-5.4 三轮审稿后，论文的核心差异化从"系统能力"转向**"忠实的知识注入"（Faithful Knowledge Injection）**。

**核心发现（待验证）**：Cross-attention为冻结LLM提供了一条**独立的知识通道**。与RAG的in-context注入不同，cross-attention注入的知识不与模型参数记忆在self-attention中竞争，因此模型能**忠实地遵从注入的知识**——无论该知识是正确的还是反事实的。而RAG在知识与参数记忆冲突时会发生"左右互搏"（knowledge conflict），导致知识利用不可靠。

**系统定位不变**：TokenMem仍然是一个完整的记忆系统（检索→融合→管理），面向6模型×3家族。Faithfulness是该系统相比RAG的**核心差异化卖点**，不是独立贡献。

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

### 1.2 v3新增：Knowledge Conflict文献连接

TokenMem的faithfulness发现连接到LLM知识冲突领域：
- **Longpre et al. 2021**: LLM在context与parametric knowledge冲突时行为不可预测
- **Xie et al. 2024**: 知识冲突导致LLM输出质量显著下降
- **Chen et al. 2022**: 上下文忠实性(context faithfulness)分析

**我们的新视角**：cross-attention注入提供了一条独立于self-attention的知识通道，避免了知识冲突。这是已有literature中未被研究过的角度。

### 1.3 识别到的关键空白

1. **知识冲突 gap**: 没有人比较过cross-attention注入vs in-context注入在知识冲突条件下的行为差异
2. **多模型通用性 gap**: 最多的KBLaM只测了3个模型
3. **跨域泛化 gap**: DecoupledRAG是per-task SFT，没有人证明cross-attn注入技能可跨域迁移
4. **即插即用 gap**: ExplicitLM需预训练

### 1.4 DecoupledRAG关键技术细节（代码验证）

DecoupledRAG的训练方式（代码确认）：
- **只需SFT，不需要Pretrain**
- 基座LLM**完全冻结**（不使用LoRA微调基座q/k/v/o）
- **只训练gate_crossattention**（融合权重 W_β = A_β @ B_β，~4.19M参数）
- 零初始化B_β，高斯初始化A_β（σ=0.01）
- 训练：5 epochs, lr=1e-3, batch_size=16
- SFT数据：**各任务独立QA数据集**（28K-167K）← 关键差异：我们只训一次
- 知识库：Wikipedia 21M文档，每段256 tokens

---

## 二、方案详情

### 2.1 推荐方案：TokenMem

**一句话**: 面向冻结LLM的记忆系统，通过cross-attention提供忠实的知识注入通道——模型学会可靠地使用外部知识，即使该知识与参数记忆冲突。

### 2.2 核心贡献（v3修订，经GPT-5.4三轮审稿确认）

**C1: 忠实的知识注入（Faithful Knowledge Injection）** — 核心发现
> Cross-attention注入的Knowledge Compliance显著高于RAG的in-context注入。
> 在反事实知识条件下，TokenMem忠实遵从注入的知识（高compliance），
> 而RAG因知识冲突导致输出既不跟外部知识也不跟参数记忆（低compliance、高conflict rate）。
> **状态**: ❌ 未验证 — 需要counterfactual compliance实验

**C2: 跨领域泛化**
> News SFT一次 → MedQA/ARC/MMLU跨域有效。adapter学到domain-agnostic的知识利用技能。
> **状态**: 🔄 部分支持 — 4B/8B有OOD正增益，但仅Qwen家族，recovery 29-48%

**C3: 多模型通用性**
> 6模型 × 3家族验证。
> **状态**: ❌ 未验证 — 仅完成2/6模型（Qwen3-4B, 8B）

**C4: 知识敏感性（C1的前提条件）**
> Oracle >> Topic-Matched-Wrong ≥ Empty — adapter精确使用知识内容。
> 控制组使用**话题相关但答案错误**的流畅段落（而非简单随机错配），
> 证明adapter能区分正确知识与"看似合理的错误知识"，而不只是"能忽略垃圾"。
> **状态**: ❌ 未验证 — 需要Oracle/Topic-Matched-Wrong/Empty实验

### 2.3 v3叙事逻辑链

```
C4 (知识敏感性) 证明: adapter 确实在用知识
       ↓  C4是C1的前提——排除"adapter摆烂"的解释
C1 (忠实注入) 证明: 正确知识→高compliance, 反事实知识→也高compliance
                    而RAG反事实→低compliance+高conflict
       ↓  C1是论文的核心科学发现
机制解释: cross-attention是独立通道 → 不与参数记忆在self-attention中竞争
       ↓  可解释性实验支撑（logit lens, gate分析, 因果追踪）
系统展示: TokenMem作为完整记忆系统的instantiation
```

### 2.4 架构设计（不变）

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

### 2.5 训练方式（借鉴DecoupledRAG，代码已验证）

```
一次性SFT（per-model, 关键差异: DecoupledRAG是per-task）:
  数据: News train 50K（时间分割，较早文章）
  可训练: 仅gate_crossattention（~3-5M参数）
  冻结: 基座LLM全部参数
  配置: Lamb lr=1e-3, 5 epochs, batch_size=16

推理（零成本切换知识）:
  换任意领域知识bank → 直接推理，不重训练
```

### 2.6 与ExplicitLM的关系

ExplicitLM（ICLR 2026, 本组prior work）证明了token级显式记忆+PKM检索的有效性，但需要从零预训练。TokenMem采用完全不同的集成方式——冻结LLM + 轻量SFT + Cross-Attention注入——使其成为即插即用方案。两者互补（预训练方案 vs 后置适配方案）。

---

## 三、GPT-5.4 三轮审稿记录

### Round 1 (v1: 系统pipeline故事)
- **评分**: 3/10 Clear Reject
- **致命问题**: (1) RAG也有持久化/可编辑/跨模型能力 (2) 方法novelty不足("DecoupledRAG+FAISS") (3) 核心claims未验证 (4) Oracle-only评测不够
- **结论**: v1故事完全站不住

### Round 2 (v2: 噪声鲁棒性故事)
- **评分**: 5/10 Borderline Reject
- **改进**: 移除了虚假差异化(持久化/可编辑)，重构为机制研究
- **残留问题**: (1) "鲁棒性=adapter太弱"可被攻击 (2) 需要utility-harm frontier (3) C1单独撑不起NeurIPS
- **结论**: 方向对了但鲁棒性framing仍可破

### Round 3 (v3: 忠实知识注入故事)
- **评分**: 6/10 Borderline Weak Accept（假设实验成功+公平基线）
- **审稿人评价**: "This is the first version that has a real NeurIPS argument"
- **关键要求**: (1) trained prompt baseline防止训练不公平攻击 (2) minimal-edit counterfactual (3) 按参数记忆强度分层 (4) no-prior对照
- **结论**: 首次进入可投区间，但仍需核心实验验证

### 独立Codex审稿 (v3.1, 全新session)
- **评分**: 4/10
- **新攻击点**: (1) Shuffled太弱需Topic-Matched-Wrong (2) 高compliance可被读为gullibility (3) 需conflict-conditioned分层 (4) E7效率应为P0
- **应对**: (1) E1控制组改为Topic-Matched-Wrong ✅ (2) "刀无罪"论证 ✅ (3) E2新增High/Low-Prior分层 ✅ (4) E7提升到P0 ✅

### 审稿ThreadID
- Round 1-3: `019dd460-2fc8-7ad3-8f0e-cdc80df6dbfc`
- Result-to-Claim: `019dd4fa-b0dd-7762-98d1-b4ca0c12c789`
- 独立审稿: `019dd511-f460-76e1-ba08-ee5a3054b80c`

---

## 四、当前实验结果与Claim状态

### Result-to-Claim 评判 (GPT-5.4 xhigh, 2026-04-28)

| Claim | Verdict | Confidence | 关键缺口 |
|-------|---------|------------|---------|
| C1 忠实注入 | **no** | high | 无counterfactual实验，无compliance指标 |
| C2 跨域泛化 | **partial** | medium | 仅2个Qwen模型，OOD recovery弱(29-48%) |
| C3 多模型通用 | **no** | high | 仅2/6模型，无跨家族证据 |
| C4 知识敏感性 | **no** | high | 无Shuffled/Empty对照 |

### 已有E1结果

| 模型 | 数据集 | No-Memory | TokenMem | VanillaRAG | Δ(TM-NM) | Recovery |
|------|--------|-----------|----------|------------|----------|----------|
| 4B | News | 47.5% | 84.8% | 97.7% | +37.3pp | 74.4% |
| 4B | MedQA | 57.2% | 71.2% | 98.0% | +14.0pp | 34.2% |
| 4B | ARC | 86.8% | 91.0% | 99.6% | +4.3pp | 33.4% |
| 4B | MMLU | 67.2% | 75.6% | 95.9% | +8.4pp | 29.2% |
| 8B | News | 52.9% | 85.4% | 98.0% | +32.5pp | 72.1% |
| 8B | MedQA | 64.6% | 77.5% | 98.7% | +12.9pp | 37.8% |
| 8B | ARC | 91.0% | 95.1% | 99.7% | +4.1pp | 47.6% |

**当前可防御的claim**: "TokenMem在Oracle条件下改善冻结LLM的知识利用，并在Qwen3-4B/8B上展示部分跨域迁移。"

---

## 五、必做实验（按优先级）

### P0: 论文命脉实验

| 实验 | 内容 | 时间 | 对应Claim |
|------|------|------|----------|
| **E1: Knowledge Sensitivity** | Oracle/Topic-Matched-Wrong/Empty on 4B (ARC+MedQA) | ~2h | C4（C1的前提） |
| **E2: Counterfactual Compliance** | 正确/反事实知识 × RAG vs TM + **conflict-conditioned分层** | ~6h | C1（核心发现） |
| **E3: 公平基线** | Strong-prompt RAG ("只根据passage回答") | ~2h | C1公平性 |
| **E7: 效率数据** | 延迟/显存/throughput | ~2h | 准确率输RAG的补偿论据 |

### P1: 支撑实验

| 实验 | 内容 | 时间 | 对应Claim |
|------|------|------|----------|
| E4: 剩余4模型SFT+评测 | 0.6B/1.7B/Gemma/Ministral | ~8h | C3 |
| E6: Domain-SFT消融 | MedQA SFT后测MedQA | ~4h | C2分析 |
| E4: 8B MMLU补完 | 完成中断的评测 | ~2h | C2/C3 |

### P2: 深度分析

| 实验 | 内容 | 时间 | 对应Claim |
|------|------|------|----------|
| E9: Logit Lens分析 | 逐层decode中间表征 | ~4h | C1机制 |
| E9: Gate激活分析 | 各层gate输出幅度 | ~2h | C1机制 |
| E9: 因果追踪 | 逐层关闭cross-attn | ~3h | C1机制 |
| E5: 注入层消融 | 1/4/12/全层 | ~4h | 系统理解 |

### 反事实知识生成方案

对每道MCQ (Q, 正确答案A, 错误答案B):
```
用DeepSeek V4 Flash生成支持B的段落（与正确知识同格式同长度）
格式: 150-200词百科风格段落
要求: minimal-edit风格——尽量保留正确段落的结构，只改关键事实
```

### Knowledge Compliance指标定义

```
KC = 模型回答与注入知识所支持答案一致的比例
  正确知识: KC = accuracy
  反事实知识: KC = %回答B（知识支持的错误答案）
Conflict Rate = %回答既非A也非B（两边都没跟）
```

---

## 六、被消除的方案

| 方案 | 消除原因 | 阶段 |
|------|---------|------|
| MemoryBridge (纯跨模型) | 单点贡献，不够饱满 | Phase 2 |
| EditMem (纯编辑) | 与知识编辑领域reviewer期望有gap | Phase 2 |
| ConvoMem (对话记忆) | 知识提取质量是混淆变量 | Phase 2 |
| PKM-Fusion (纯检索) | 与FwPKM(2026.01)重叠 | Phase 2 |

---

## 七、风险与应对（v3修订）

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| **C1实验失败**: counterfactual compliance差距不显著 | 30% | **致命** | 放弃faithfulness故事，回退到C2+C3为主的empirical study（降级到EMNLP） |
| **"trained vs untrained"公平性攻击** | 60% | 高 | 必做strong-prompt RAG (E3) |
| **"高compliance=gullibility"攻击** | 40% | 中 | 论证层解决：Compliance是通道可控性属性，不是安全claim。"刀无罪，用刀的人有罪"——忠实的通道意味着用户放什么知识模型就用什么，知识正确性是上游（检索器/知识库）的责任。RAG的低compliance反而说明通道不可控。 |
| C4实验失败: Topic-Matched ≈ Oracle | 15% | 致命 | 说明adapter不区分对错知识，需重新设计训练方案 |
| 某些模型adapter训练不收敛 | 20% | 中 | 从6模型中去掉；5模型仍足够 |
| Gemma3-1B评测broken | 50% | 低 | 换评测方式或从scope中移除 |
| 8B训练时间超预期 | 20% | 中 | 8B已完成SFT，仅缺MMLU评测 |

---

## 八、论文结构（v3）

```
Title: "TokenMem: Faithful Knowledge Injection into Frozen LLMs
        via Cross-Attention Adapters"
  或: "Cross-Attention as a Faithful Knowledge Channel for Frozen LLMs"

§1 Introduction
   - LLM知识增强的RAG方案存在knowledge conflict问题
   - Cross-attention提供了一条独立的、忠实的知识通道
   - TokenMem：面向冻结LLM的记忆系统，以faithful injection为核心
   - 6模型×3家族，单域SFT跨域泛化

§2 Related Work
   - Knowledge Conflicts in LLMs (Longpre 2021, Xie 2024)
   - RAG及变体（VanillaRAG, DecoupledRAG — 机制来源，明确credit）
   - 显式记忆系统（ExplicitLM, KBLaM, MemoryLLM）
   - 参数编辑（ROME/MEMIT — 不同regime）

§3 Method: TokenMem
   - 3.1 TokenMemoryBank（FAISS + tokenized text）
   - 3.2 Cross-Attention Fusion（借鉴DecoupledRAG，诚实credit）
   - 3.3 One-Time SFT Protocol

§4 Experiments
   - 4.1 基础注入效果（表1: 6模型 × 4数据集 × 3方法）
   - 4.2 Knowledge Compliance实验（表2: 正确/反事实 × RAG/TokenMem）← 核心
   - 4.3 Knowledge Sensitivity（表3: Oracle/Shuffled/Empty）← C1前提
   - 4.4 机制分析（Logit Lens + Gate分析 + 因果追踪）
   - 4.5 跨域泛化 + Domain-SFT消融
   - 4.6 消融（注入层、效率数据）

§5 Analysis & Discussion
   - RAG vs TokenMem tradeoff表格
   - 什么场景该用哪个
   - 局限性

§6 Conclusion
```
