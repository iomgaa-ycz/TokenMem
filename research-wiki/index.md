# Research Wiki Index

**项目**: TokenMem — 面向冻结LLM的即插即用内部化记忆pipeline
**目标**: NeurIPS 2026
**Prior Work**: ExplicitLM (ICLR 2026, 本组)
**Last Updated**: 2026-04-28 (v3 Faithful Injection Framing)

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

## Ideas (4)

| ID | 名称 | 状态 | 结果 |
|----|------|------|------|
| [idea:001](ideas/001_tokenmem.md) | **TokenMem** | 🟢 active (v3 faithful injection) | pending core experiments |
| [idea:002](ideas/002_memorybridge.md) | MemoryBridge | 🔴 eliminated | 单点贡献不足 |
| [idea:003](ideas/003_editmem.md) | EditMem | 🔴 eliminated | 与KE领域期望gap |
| [idea:004](ideas/004_convomem.md) | ConvoMem | 🔴 eliminated | 提取质量混淆 |

---

## Claims (4, v3修订)

| ID | Claim | 状态 | 阈值 | 审稿评估 |
|----|-------|------|------|---------|
| [C1](claims/C1.md) | **忠实知识注入** (Faithful Injection) | ❌ pending | TM counterfactual KC > RAG by ≥15pp | GPT: "first real NeurIPS argument" |
| [C2](claims/C2.md) | 跨领域泛化 | 🔄 partial | OOD >5%提升, ≥2家族 | GPT: "partial, only 2 Qwen models" |
| [C3](claims/C3.md) | 多模型通用性 | ❌ pending | ≥5/6模型有效 | GPT: "no, need cross-family" |
| [C4](claims/C4.md) | 知识敏感性 (C1前提) | ❌ pending | Oracle >> Shuffled ≈ Empty | GPT: "mandatory sanity check" |

**v3变更说明**: 原C4"超越VanillaRAG"已invalidated并重定义。C1为v3新增核心claim。C4从"超越RAG"改为"知识敏感性"作为C1的前提条件。

---

## Gaps (5, v3修订)

见 [gap_map.md](gap_map.md)

| ID | Gap | 状态 | 优先级 |
|----|-----|------|--------|
| G1 | **知识冲突下的注入忠实性** | 🔴 核心 — v3新增 | P0 |
| G2 | 多模型通用性验证 | 未解决 | P1 |
| G3 | 冻结LLM即插即用记忆 | 部分解决 | P1 |
| G4 | 跨领域泛化 | 部分解决 | P1 |
| G5 | 完整记忆pipeline | 部分解决 | P2 |

---

## Experiments (4)

| ID | 名称 | 状态 | 关键发现 |
|----|------|------|---------|
| [exp:E0_news_dataset](experiments/E0_news_dataset.md) | E0 News Dataset (58,663 MCQ) | ✅ completed | 25源/6类/50K train+8.6K val |
| [exp:E1_baseline](experiments/E1_baseline.md) | E1 Baseline (No-Memory + VanillaRAG, 4+2 datasets) | ✅ completed | 68 JSON (48原始+20反事实); VanillaRAG天花板88-99% |
| [exp:E1_tokenmem](experiments/E1_tokenmem.md) | E1 TokenMem (4B/8B, 4 datasets) | 🔄 in_progress | 7/8已测全部>NM; Recovery 29-74% |
| [exp:E2_pilot_eval_method](experiments/E2_pilot_eval_method.md) | **E2 Pilot: 评测方法验证** | ✅ completed | **MCQ logprob有天花板(94%); CoT降至28-36%; E2需用CoT评测** |

---

## GPT-5.4 External Review Log

| 轮次 | 版本 | 评分 | 关键反馈 |
|------|------|------|---------|
| R1 | v1 (系统pipeline) | 3/10 | "DecoupledRAG+FAISS"; RAG也有持久化/可编辑 |
| R2 | v2 (噪声鲁棒性) | 5/10 | "attenuation not robustness"; 需utility-harm frontier |
| R3 | v3 (忠实注入) | **6/10** | "first real NeurIPS argument"; 需trained prompt baseline |
| R2C | result-to-claim | — | C1:no, C2:partial, C3:no, C4:no |

Thread IDs: review `019dd460-2fc8-7ad3-8f0e-cdc80df6dbfc`, r2c `019dd4fa-b0dd-7762-98d1-b4ca0c12c789`
