# Research Wiki Index

**项目**: TokenMem — Faithful Knowledge Internalization for Frozen LLMs via Cross-Attention
**目标**: NeurIPS 2026
**Prior Work**: ExplicitLM (ICLR 2026, 本组), DecoupledRAG (WWW 2025, 机制来源)
**Last Updated**: 2026-05-04 (v5: RAG SFT受控对比 + 贡献重构)

---

## Papers (9)

### 本组前序工作
- [ExplicitLM](papers/explicitlm2025.md) — token记忆+PKM，需预训练 (ICLR 2026) `paper:explicitlm2025`

### 融合机制来源
- [DecoupledRAG](papers/decoupledrag2025.md) — cross-attention知识注入 (WWW 2025) `paper:decoupledrag2025`

### 架构竞品
- [KBLaM](papers/kblam2025.md) — 结构化KB三元组注入 (ICLR 2025) `paper:kblam2025`
- [Knowledge Capsules](papers/kcapsules2026.md) — 结构化capsule+KVI (arXiv Apr 2026) `paper:kcapsules2026`

### 范式对手
- [MemoryLLM](papers/memoryllm2024.md) — 隐式自更新记忆 (ICML 2024) `paper:memoryllm2024`
- [M+](papers/mplus2025.md) — MemoryLLM扩展版 (ICML 2025) `paper:mplus2025`

### 检索/效率
- [Large Memory Layers with Product Keys](papers/lample2019_pkm.md) — PKM O(√N)检索 (NeurIPS 2019) `paper:lample2019_pkm`
- [FwPKM](papers/fwpkm2026.md) — 动态PKM+在线更新 (arXiv Jan 2026) `paper:fwpkm2026`

### 命名相关
- [TokMem](papers/tokmem2025.md) — 过程记忆soft token (ICLR 2026) `paper:tokmem2025` ⚠️ 名称可能混淆

---

## Idea

| ID | 名称 | 状态 | 结果 |
|----|------|------|------|
| [idea:001](ideas/001_tokenmem.md) | **TokenMem** (v5 Faithful Injection + Controlled Channel Comparison) | 🟢 active | C1 validated (4B) |

**已归档 idea** (详见 `ideas/` 目录):
- idea:002 MemoryBridge — eliminated (单点贡献不足)
- idea:003 EditMem — eliminated (与KE领域期望gap)
- idea:004 ConvoMem — eliminated (提取质量混淆)

---

## Claims (2主+1次, v5重构)

| ID | Claim | 状态 | 阈值 | 4B实际 |
|----|-------|------|------|--------|
| C1 | **方法: TokenMem系统** | ✅ 4B验证 | 完整记忆系统 | +49pp/+18pp KC |
| C2 | **受控发现: 通道影响忠诚度** | ❌ 待RAG SFT结果 | TM KC >> RAG SFT KC | TBD |
| C3 | Curriculum训练必要性 | ❌ pending | P2 >> P1 on KC | 消融A1待做 |
| Minor | KC指标 + CoT评测协议 | ✅ 已定义 | 作为setup呈现 | — |

**v5变更**: 贡献重构为2主+1次。C1为方法系统贡献; C2新增受控对比(TokenMem vs RAG SFT); 旧C2(泛化+多模型)降级; 评测方法论降为Minor(evaluation setup)。

---

## Experiments

| ID | 名称 | 状态 | 关键发现 |
|----|------|------|---------|
| [exp:E0_news_dataset](experiments/E0_news_dataset.md) | News Dataset (58,663 MCQ) | ✅ | 25源/6类/50K train |
| [exp:E1_baseline](experiments/E1_baseline.md) | Baseline (5模型×2方法×7数据集) | ✅ | RAG天花板86-99%; CoT评测 |
| [exp:E2_pilot_eval_method](experiments/E2_pilot_eval_method.md) | 评测方法验证 | ✅ | **MCQ logprob天花板; CoT有效** |
| [exp:E2_curriculum_sft](experiments/E2_curriculum_sft.md) | Curriculum SFT (Phase 1+2) | ✅ | curriculum解决梯度冲突 |
| [exp:E_main_4B](experiments/E_main_4B.md) | **4B 核心评测 (7ds)** | ✅ | **C1: +49/+18pp** |
| exp:E_main_8B | 8B 核心评测 | 🔄 训练中 | — |
| exp:E_main_14B | 14B 核心评测 | 🔄 训练中 | — |
| exp:E_main_llama | LLaMA-3.1-8B 评测 | 🔄 训练中 | — |
| exp:E_main_olmo | OLMo-3-7B 评测 | 🔄 训练中 | — |
| exp:RAG_SFT | RAG SFT 受控对比 (4B) | ❌ P0最高优先 | — |
| exp:A1_curriculum | 消融: P1 vs P1+P2 | ❌ | — |
| exp:A2_conflict_cond | 消融: Conflict-conditioned | ❌ | — |
| exp:A3_layers | 消融: 注入层数 | ❌ | — |
| exp:A4_data_size | 消融: 训练数据量 | ❌ | — |

---

## 方法演进记录

| 版本 | 方法 | 评测 | 结果 | 决策 |
|------|------|------|------|------|
| v1 | 纯News SFT (logprob) | loglikelihood MCQ | 4B: +4~37pp vs NM | 准确率低于RAG → C4 invalidated |
| v2 | 纯News SFT (CoT eval) | CoT nothink | 与v1类似但评测更准 | MCQ logprob有天花板 |
| v3 | Curriculum SFT (P1+P2, seq=512) | CoT nothink | val_loss下降但seq不够 | max_seq_len bug |
| **v4** | **CoT Curriculum SFT (P1+P2, seq=1024)** | **CoT nothink, 2048tok** | **C1: +49/+18pp** | ⬆️ 升级到v5 |
| **v5** | **CoT Curriculum + RAG SFT受控对比** | **CoT nothink + 4方法** | **RAG SFT TBD** | **✅ 最终方案** |

---

## GPT-5.4 External Review Log

| 轮次 | 版本 | 评分 | 关键反馈 |
|------|------|------|---------|
| R1 | v1 (系统pipeline) | 3/10 | "DecoupledRAG+FAISS" |
| R2 | v2 (噪声鲁棒性) | 5/10 | "attenuation not robustness" |
| R3 | v3 (忠实注入) | **6/10** | "first real NeurIPS argument" |
| R4 | v4 proposal (3方法) | 3/10 | "No trained RAG baseline is fatal" |
| R5 | v5 proposal (4方法+RAG SFT) | 5/10 | "Design correct; borderline; needs results" |
