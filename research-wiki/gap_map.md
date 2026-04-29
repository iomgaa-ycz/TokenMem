# Gap Map (v3修订)

研究空白地图，按重要性排序。

---

## G1: 知识冲突下的注入忠实性 (v3新增)

**状态**: 🔴 核心 — 未解决
**重要性**: 🔴 论文命脉

没有人比较过cross-attention注入vs in-context注入（RAG）在知识与参数记忆冲突条件下的Knowledge Compliance差异。知识冲突文献（Longpre 2021, Xie 2024）研究了LLM在冲突下的行为，但未探索cross-attention作为替代注入通道的可能性。

**相关工作**: Longpre 2021 (knowledge conflicts), Xie 2024, DecoupledRAG (mechanism), KBLaM
**目标方案**: idea:001 (TokenMem, C1 faithful injection)

---

## G2: 多模型通用性验证

**状态**: 未解决
**重要性**: 🟡 重要

现有LLM记忆系统最多在3个模型上验证（KBLaM）。没有任何系统在6+模型、3+模型家族上证明通用性。

**相关工作**: paper:kblam2025 (3模型), paper:memoryllm2024 (1模型), paper:decoupledrag2025 (1模型)
**目标方案**: idea:001 (TokenMem, C3)

---

## G3: 冻结LLM即插即用记忆

**状态**: 部分解决（系统已实现，效果待验证完整）
**重要性**: 🟡 重要

ExplicitLM证明token记忆有效但需预训练。DecoupledRAG实现了冻结LLM+cross-attention注入但无持久记忆bank。

**相关工作**: paper:explicitlm2025, paper:decoupledrag2025
**目标方案**: idea:001 (TokenMem)

---

## G4: 跨领域泛化（一次训练，多领域适用）

**状态**: 部分解决（4B/8B有OOD正增益）
**重要性**: 🟡 重要

DecoupledRAG在每个任务上独立SFT。没有系统证明"在一个领域SFT → 在完全不同的领域也有效"。

**相关工作**: paper:decoupledrag2025 (per-task SFT)
**目标方案**: idea:001 (TokenMem, C2)

---

## G5: 完整记忆pipeline（检索→融合→管理）

**状态**: 部分解决（系统已实现）
**重要性**: 🟢 支撑性

注意：GPT-5.4审稿指出RAG+文档库也有完整pipeline能力。此gap不再作为vs RAG的差异化，降为系统完整性的支撑性贡献。

**相关工作**: paper:decoupledrag2025, paper:kblam2025
**目标方案**: idea:001 (TokenMem)
