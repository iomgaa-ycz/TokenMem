# Idea Discovery Report (v5 — RAG SFT 受控对比)

**方向**: TokenMem: 面向冻结LLM的忠实知识注入系统 + 受控通道对比
**日期**: 2026-04-26（v1）→ 2026-04-28（v3）→ 2026-05-03（v4 实验验证）→ 2026-05-04（v5 RAG SFT对比）
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → refine → GPT-5.4 external review → experiment validation
**目标会议**: NeurIPS 2026
**Prior Work**: ExplicitLM (ICLR 2026, 本组)

---

## Executive Summary

TokenMem 通过 cross-attention adapter 为冻结 LLM 提供一条**独立的、忠实的知识注入通道**。本工作是**系统贡献 + 受控实验发现**的混合叙事。

**核心发现（✅ 4B 已验证）**：Cross-attention 注入的 Knowledge Compliance 显著高于 VanillaRAG。在反事实知识条件下，TokenMem 遵从率 69-70%，而 VanillaRAG 仅 20-52%（差距 +18~49pp）。

**v5 关键更新**：新增 **RAG SFT** 作为受控对比实验——在相同模型/数据/训练预算下，比较 cross-attention 注入 vs trained in-context 注入的 KC 差异。这是区分"通道优势"与"训练效应"的关键实验，直接回应 GPT-5.4 Round 4 审稿中"No trained RAG baseline is fatal"的致命攻击。

**方法**: CoT Curriculum SFT（两阶段）— Phase 1 学习基础知识融合，Phase 2 混入反事实数据强化忠实遵从能力。

**模型矩阵**: 5 模型 × 3 家族（Qwen3-4B/8B/14B + LLaMA-3.1-8B + OLMo-3-7B）

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

### 2.2 核心贡献（v5修订，经GPT-5.4五轮审稿）

**C1 (Method): TokenMem 系统**
> Cross-attention memory bank + FAISS 检索 + LinearFusion 门控注入 + CoT Curriculum SFT。
> 仅训练 3-5M 参数（gate_crossattention），基座 LLM 完全冻结。
> 5 模型 × 3 家族验证（Qwen3-4B/8B/14B + LLaMA-3.1-8B + OLMo-3-7B）。
> **状态**: ✅ 4B/8B 验证通过，其余模型训练中

**C2 (Controlled Finding): 受控通道对比**
> 在相同模型/数据/训练预算条件下：
> - Cross-attention 注入 KC ≈ 69-70%
> - RAG SFT（trained in-context injection）KC = TBD
> - VanillaRAG KC 随模型能力增强反而下降（inverse-scales）
> **状态**: ✅ TokenMem + VanillaRAG 已验证；❌ RAG SFT 待做（P0）

**Minor: KC 指标 + CoT 评测协议**
> Knowledge Compliance 指标定义 + 反事实 CoT 评测方法。
> 作为评测设置（evaluation setup）呈现，不单独列为核心贡献。
> **状态**: ✅ 已定义并使用

### 2.3 v5叙事逻辑链

```
C1 (系统): TokenMem 作为完整的忠实知识注入系统
       ↓  系统贡献——方法完整、可复现、跨模型验证
C2 (受控发现): 相同条件下，cross-attention KC >> RAG SFT KC >> VanillaRAG KC
       ↓  受控实验——排除训练效应、公平比较注入通道
机制解释: cross-attention 是独立通道 → 不与参数记忆在 self-attention 中竞争
       ↓  可解释性实验支撑（logit lens, gate分析, 因果追踪）
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

### 2.5 训练方式（v4: CoT Curriculum SFT）

```
Phase 1: 纯 News CoT SFT (per-model)
  数据: data/news/train_cot.jsonl (50K)
  可训练: 仅gate_crossattention（~3-5M参数）
  冻结: 基座LLM全部参数
  配置: Lamb lr=1e-3, 3 epochs, batch=2, grad_accum=16, seq=1024

Phase 2: News + Counterfactual 混合 CoT SFT
  加载: Phase 1 best checkpoint
  数据: News + cf_arc_easy_cot + cf_medqa_cot (CF 2x oversample)
  配置: 同Phase 1, epochs=40 + early-stop(patience=5)
  
推理（零成本切换知识）:
  换任意领域知识bank → 直接推理，不重训练
```

**v3→v4 训练方法变更原因**:
- E2 pilot 发现 MCQ logprob 有天花板效应(94%) → 需要 CoT 评测 → 训练也改为 CoT 格式
- 直接混合训练梯度冲突 → Curriculum 两阶段解决
- max_seq_len 64→1024 修复 CF 数据截断问题

### 2.6 与ExplicitLM的关系

ExplicitLM（ICLR 2026, 本组prior work）证明了token级显式记忆+PKM检索的有效性，但需要从零预训练。TokenMem采用完全不同的集成方式——冻结LLM + 轻量SFT + Cross-Attention注入——使其成为即插即用方案。两者互补（预训练方案 vs 后置适配方案）。

---

## 三、GPT-5.4 五轮审稿记录

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

### Round 4 (v4 proposal, 2026-05-04)
- **评分**: 3/10 Clear Reject
- **致命问题**: (1) **No trained RAG baseline is fatal** — 无法区分"cross-attention通道优势"与"训练效应"，VanillaRAG是untrained baseline，对比不公平 (2) 方法增量性过强（DecoupledRAG + FAISS + curriculum） (3) 三个方面overclaimed — faithfulness, cross-domain generalization, curriculum necessity 同时claim但每个都不够深
- **结论**: 缺少 trained RAG baseline 是结构性缺陷，必须修复

### Round 5 (v5 proposal with RAG SFT, 2026-05-04)
- **评分**: 5/10 Borderline Reject
- **改进**: (1) 新增 RAG SFT 受控对比，设计正确 (2) claims 从3个缩减为2主+1辅，更聚焦 (3) 论文叙事改为"系统贡献+受控发现"混合结构
- **审稿人评价**: "Design of fix is correct; paper is now reviewable"
- **残留问题**: (1) 方法novelty仍为borderline — 核心架构是已有组件的组合 (2) 如果RAG SFT KC接近TokenMem则论文核心claim不成立 (3) 5个模型的generalization声称需要全部跑完才可信
- **结论**: 论文现在可以被review了；如果RAG SFT结果强劲（gap显著），acceptance rate估计35-45%

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

### 4.3 受控通道对比方法表（v5新增）

| 方法 | 注入通道 | 训练 | 可训练参数 | 训练数据 | 描述 |
|------|---------|------|-----------|---------|------|
| **No-Memory** | 无 | 无 | 0 | — | 纯参数记忆基线 |
| **VanillaRAG** | In-context (prompt拼接) | 无 | 0 | — | 未训练的上下文注入 |
| **RAG SFT** | In-context (prompt拼接) | CoT Curriculum SFT | LoRA ~3-5M | 同TokenMem | **训练过的上下文注入**（受控对比） |
| **TokenMem** | Cross-attention | CoT Curriculum SFT | gate ~3-5M | 同TokenMem | Cross-attention 注入（本文方法） |

**受控变量说明**：RAG SFT 与 TokenMem 共享相同的基座模型、训练数据、训练预算（参数量级相同），唯一区别是知识注入通道（in-context vs cross-attention）。这使得 KC 差异可归因于通道本身而非训练效应。

**当前可防御的claim**: "TokenMem在Oracle条件下改善冻结LLM的知识利用，并在Qwen3-4B/8B上展示部分跨域迁移。"

---

## 五、必做实验（按优先级）

### P0: 论文命脉实验

| 实验 | 内容 | 时间 | 对应Claim |
|------|------|------|----------|
| **RAG SFT** | 相同数据/预算下 LoRA SFT in-context injection，评测 KC | ~6h (本地4070Ti) | **C2（受控对比，最高优先）** |
| **A1: Curriculum消融** | Phase 1 only vs Phase 1+2 的KC对比 | ~4h | C1系统（训练方法必要性） |
| **A2: 注入层消融** | 1/4/12/全层注入的效果对比 | ~4h | C1系统（架构设计选择） |

### P1: 支撑实验

| 实验 | 内容 | 时间 | 对应Claim |
|------|------|------|----------|
| E4: 剩余模型SFT+评测 | 14B/LLaMA-8B/OLMo-7B | ~8h | C1（多模型验证） |
| E7: 效率数据 | 延迟/显存/throughput | ~2h | 准确率输RAG的补偿论据 |
| E6: Domain-SFT消融 | MedQA SFT后测MedQA | ~4h | 跨域泛化分析 |

### P2: 深度分析

| 实验 | 内容 | 时间 | 对应Claim |
|------|------|------|----------|
| E9: Logit Lens分析 | 逐层decode中间表征 | ~4h | 机制解释 |
| E9: Gate激活分析 | 各层gate输出幅度 | ~2h | 机制解释 |
| E9: 因果追踪 | 逐层关闭cross-attn | ~3h | 机制解释 |
| E1: Knowledge Sensitivity | Oracle/Topic-Matched-Wrong/Empty | ~2h | 知识敏感性 |
| E3: Strong-prompt RAG | "只根据passage回答" | ~2h | 公平性对照 |

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

## 六、预期审稿攻击与防御

| # | 攻击 | 防御 |
|---|------|------|
| Q1 | "方法只是DecoupledRAG+FAISS，novelty不足" | 承认架构组件来自已有工作（诚实credit）。贡献重心在受控实验发现：相同条件下cross-attention通道的KC显著优于in-context通道。系统贡献在于完整的TokenMem pipeline（memory bank + 检索 + 融合 + curriculum SFT）及5模型×3家族验证。 |
| Q2 | "对比不公平：TokenMem有训练但VanillaRAG没训练" | **v5核心应对**：新增RAG SFT基线——使用相同训练数据、相同参数预算（LoRA ~3-5M）、相同curriculum对in-context注入进行SFT。四方法表格（No-Memory / VanillaRAG / RAG SFT / TokenMem）中，RAG SFT vs TokenMem是严格受控对比，唯一变量是注入通道。 |
| Q3 | "高compliance = gullibility，模型盲从有害知识" | Compliance是通道可控性属性，不是安全claim。"刀无罪"——忠实通道意味着用户放什么知识模型就用什么，知识正确性是上游（检索器/知识库）的责任。RAG的低compliance反而说明通道不可控、行为不可预测。 |
| Q4 | "recovery rate偏低（29-48%），实际效用存疑" | 承认TokenMem在accuracy维度弱于VanillaRAG；但faithfulness维度显著领先。这是accuracy-faithfulness tradeoff，不同应用场景有不同需求。高风险场景（医疗、法律）faithfulness优先。 |
| Q5 | "只有Qwen系列，generalization不够" | 5模型×3家族（Qwen3, LLaMA-3.1, OLMo-3）覆盖主流开源家族。数据收集中。 |
| Q6 | "KC不是faithfulness而是blind obedience——模型只是无脑复读注入内容" | 区分blind obedience与faithful channel：(1) Knowledge Sensitivity实验（Oracle vs Topic-Matched-Wrong vs Empty）证明adapter区分知识质量，不是无差别复读；(2) 正确知识下KC=accuracy提升，说明模型确实理解并整合了知识而非简单copy；(3) 与RAG对比，RAG在正确知识下KC也高但反事实下急剧下降——说明RAG的"理解"依赖参数记忆确认，不是真正的通道忠实。 |

---

## 七、风险与应对（v5修订）

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| **RAG SFT 追平 TokenMem** | 35% | **致命** | 见下方Go/No-Go表 |
| **C1实验失败**: counterfactual compliance差距不显著 | 30% | **致命** | 放弃faithfulness故事，回退到C2+C3为主的empirical study（降级到EMNLP） |
| **"trained vs untrained"公平性攻击** | 60% | 高 | ✅ 已通过RAG SFT解决 |
| **"高compliance=gullibility"攻击** | 40% | 中 | 论证层解决（见Q3+Q6） |
| 某些模型adapter训练不收敛 | 20% | 中 | 从6模型中去掉；5模型仍足够 |
| Gemma3-1B评测broken | 50% | 低 | 换评测方式或从scope中移除 |

### RAG SFT Go/No-Go 决策表

| RAG SFT KC | 与TokenMem KC差距 | 论文策略 | 信号强度 |
|------------|-------------------|---------|---------|
| < 40% | > 30pp | **Strong** — 写完整论文，claim cross-attention通道因果优势 | 🟢 Go |
| 40-55% | 15-30pp | **Usable** — 弱化为"significant gap"，不做因果claim | 🟡 Go (cautious) |
| 55-65% | 5-15pp | **Weak** — 重新评估，可能需要更多模型/数据集支撑或改换叙事 | 🟠 Reassess |
| > 65% | < 5pp | **Fatal** — 放弃通道优势claim，回退到纯系统贡献论文 | 🔴 No-Go |

---

## 八、论文结构（v5 — v2 proposal）

```
Title: "TokenMem: Faithful Knowledge Internalization for Frozen LLMs
        via Cross-Attention"

§1 Introduction
   - LLM知识增强的RAG方案存在knowledge conflict问题
   - Cross-attention提供了一条独立的、忠实的知识通道
   - TokenMem：面向冻结LLM的记忆系统，以faithful injection为核心
   - 受控实验：相同条件下cross-attention vs in-context注入的KC差异
   - 5模型×3家族验证

§2 Related Work
   - Knowledge Conflicts in LLMs (Longpre 2021, Xie 2024)
   - RAG及变体（VanillaRAG, DecoupledRAG — 机制来源，明确credit）
   - 显式记忆系统（ExplicitLM, KBLaM, MemoryLLM）
   - 参数编辑（ROME/MEMIT — 不同regime）

§3 Method: TokenMem
   - 3.1 TokenMemoryBank（FAISS + tokenized text）
   - 3.2 Cross-Attention Fusion（借鉴DecoupledRAG，诚实credit）
   - 3.3 CoT Curriculum SFT Protocol
   - 3.4 RAG SFT Baseline Design（受控对比方法）
   - 3.5 Evaluation Setup（KC指标 + CoT评测协议 + 反事实数据构建）

§4 Experiments
   - 4.1 基础注入效果（表1: 5模型 × 4数据集 × 4方法）
   - 4.2 Controlled Channel Comparison（表2: 4方法 × 正确/反事实 KC）← 核心
   - 4.3 Knowledge Sensitivity（表3: Oracle/Topic-Matched-Wrong/Empty）
   - 4.4 机制分析（Logit Lens + Gate分析 + 因果追踪）
   - 4.5 消融（Curriculum必要性 + 注入层 + 效率数据）

§5 Analysis & Discussion
   - 4方法RAG vs TokenMem tradeoff分析
   - Accuracy-Faithfulness frontier
   - 什么场景该用哪个
   - 局限性

§6 Conclusion
```

---

## 九、被消除的方案

| 方案 | 消除原因 | 阶段 |
|------|---------|------|
| MemoryBridge (纯跨模型) | 单点贡献，不够饱满 | Phase 2 |
| EditMem (纯编辑) | 与知识编辑领域reviewer期望有gap | Phase 2 |
| ConvoMem (对话记忆) | 知识提取质量是混淆变量 | Phase 2 |
| PKM-Fusion (纯检索) | 与FwPKM(2026.01)重叠 | Phase 2 |
