---
type: idea
node_id: idea:001
title: "TokenMem: 面向冻结LLM的即插即用记忆pipeline"
stage: active
outcome: pending
added: 2026-04-26T00:00:00Z
---

# TokenMem

## Summary

通过 cross-attention 为冻结 LLM 提供一条**独立于 self-attention 的知识注入通道**，避免 knowledge conflict。核心发现：在相同训练条件下，cross-attention 通道的 Knowledge Compliance 显著高于 in-context 通道。5模型×3家族验证。

## 论文叙事 (v5, 2026-05-04 确定)

**核心故事**: RAG 的 in-context injection 在 self-attention 中造成 knowledge conflict → cross-attention 提供独立通道避免冲突 → curriculum SFT 教会模型忠实使用外部知识。

**与 DecoupledRAG 的差异化**: DecoupledRAG 讲"效率"(长上下文O(n²))，我们讲"忠实性"(knowledge conflict)。架构相同（诚实 credit），但动机、训练方法、评测方法完全不同。

**Method 章节结构** (2026-05-04 重构):
- §3.1 Problem Formulation: 形式化 shared pathway 的 knowledge conflict 问题（独创理论锚点）
- §3.2 Architecture: 引用 DecoupledRAG 机制 + 为什么它能避免冲突的解释 + pathway 对比表
- §3.3 Curriculum Training: 梯度冲突理论(PCGrad)解释联合训练失败 + continuation method 解释分阶段成功 + Phase 1/2 loss 形式化
- Evaluation Protocol 放在 §4.1 Experiments Setup

**理论支撑**:
- gradient conflict (Yu et al. 2020, PCGrad, NeurIPS): cos(g_util, g_cf) < 0 → 联合训练振荡
- continuation method (Bengio 2009, ICML): 从简单目标的解出发，更容易收敛到困难目标的好解
- flat basin (Hacohen & Weinshall 2019, ICML): Phase 1 收敛到平坦盆地，Phase 2 扰动不破坏已学能力

## Core Components

1. **TokenMemoryBank**: FAISS索引 + token_ids/emb存储（已实现: `memory_lora/token_bank.py`）
2. **LinearFusion** (gate_crossattention): 零初始化低秩门控融合，借鉴DecoupledRAG（已实现: `memory_lora/linear_fusion.py`）
3. **CoT Curriculum SFT**: Phase 1 纯 News 50K + Phase 2 News+CF 混合（核心原创训练方法）
4. **KC 评测框架**: Knowledge Compliance 指标 + CoT 评测协议 + 反事实数据集

## Inspired By

- paper:explicitlm2025 — token级记忆理念
- paper:decoupledrag2025 — cross-attention融合机制（架构来源，诚实credit）
- paper:lample2019_pkm — 检索效率思路（当前用FAISS替代）

## Target Gaps

- gap:G1 (多模型通用性)
- gap:G2 (冻结LLM即插即用)
- gap:G5 (跨领域泛化)
- Knowledge conflict under external injection (新增，无对应旧gap编号)

## Key Risks

- RAG SFT 追平 TokenMem (35%) → 通道优势 claim 不成立，需回退到纯系统贡献
- 方法 novelty 仍为 borderline（组件组合 + 训练方法创新）→ 靠受控实验的因果证据补偿
- 4 个模型训练中，如果跨家族结果不一致则 generalization claim 需弱化

## Connections

[AUTO-GENERATED]
- inspired_by → paper:explicitlm2025
- inspired_by → paper:decoupledrag2025
- inspired_by → paper:lample2019_pkm
- addresses_gap → gap:G1, gap:G2, gap:G3, gap:G4, gap:G5
