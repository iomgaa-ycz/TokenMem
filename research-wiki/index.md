# Research Wiki Index

**项目**: TokenMem — 面向冻结LLM的即插即用内部化记忆pipeline
**目标**: NeurIPS 2026
**Prior Work**: ExplicitLM (ICLR 2026, 本组)
**Last Updated**: 2026-04-26

---

## Papers (8)

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
| [idea:001](ideas/001_tokenmem.md) | **TokenMem** | 🟢 active | pending |
| [idea:002](ideas/002_memorybridge.md) | MemoryBridge | 🔴 eliminated | 单点贡献不足 |
| [idea:003](ideas/003_editmem.md) | EditMem | 🔴 eliminated | 与KE领域期望gap |
| [idea:004](ideas/004_convomem.md) | ConvoMem | 🔴 eliminated | 提取质量混淆 |

---

## Claims (4)

| ID | Claim | 状态 | 阈值 |
|----|-------|------|------|
| [C1](claims/C1.md) | 6模型通用性 | ⏳ pending | ≥5/6模型有效 |
| [C2](claims/C2.md) | 跨领域泛化 | ⏳ pending | OOD >5%提升 |
| [C3](claims/C3.md) | 知识编辑 | ⏳ pending | ESR>80%, <1s |
| [C4](claims/C4.md) | 超越VanillaRAG | ⏳ pending | ≥4/6模型 |

---

## Gaps (5)

见 [gap_map.md](gap_map.md)

| ID | Gap | 状态 | 目标方案 |
|----|-----|------|---------|
| G1 | 多模型通用性验证 | 未解决 | idea:001 |
| G2 | 冻结LLM即插即用记忆 | 未解决 | idea:001 |
| G3 | 完整记忆pipeline | 未解决 | idea:001 |
| G4 | 动态知识管理 | 未解决 | idea:001 |
| G5 | 跨领域泛化 | 未解决 | idea:001 |

---

## Experiments (0)

尚未执行。计划见 [EXPERIMENT_PLAN.md](../refine-logs/EXPERIMENT_PLAN.md)。
