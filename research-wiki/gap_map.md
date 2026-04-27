# Gap Map

研究空白地图，按重要性排序。

---

## G1: 多模型通用性验证

**状态**: 未解决
**重要性**: 🔴 核心

现有LLM记忆系统最多在3个模型上验证（KBLaM: Llama-3-8B, Llama-3.2-1B, Phi-3-mini）。没有任何系统在6+模型、3+模型家族上证明通用性。

**相关工作**: paper:kblam2025 (3模型), paper:memoryllm2024 (1模型), paper:explicitlm2025 (1模型)
**目标方案**: idea:001 (TokenMem, 6模型×3家族)

---

## G2: 冻结LLM即插即用记忆

**状态**: 未解决
**重要性**: 🔴 核心

ExplicitLM证明token记忆有效（+43.67%），但需从零预训练。没有人做过"给现有冻结LLM加token级记忆，只需轻量SFT"。DecoupledRAG实现了冻结LLM+cross-attention注入，但没有持久记忆bank。

**相关工作**: paper:explicitlm2025 (需预训练), paper:decoupledrag2025 (无持久记忆)
**目标方案**: idea:001 (TokenMem)

---

## G3: 完整记忆pipeline（检索→融合→更新）

**状态**: 未解决
**重要性**: 🟡 重要

现有工作只覆盖pipeline的1-2个阶段：DecoupledRAG做融合不做检索/更新；KBLaM做融合+有限更新不做检索；FwPKM做检索+更新不做显式融合。没有系统实现完整的知识生命周期管理。

**相关工作**: paper:decoupledrag2025, paper:kblam2025, paper:fwpkm2026
**目标方案**: idea:001 (TokenMem)

---

## G4: 动态知识管理（无需重训练）

**状态**: 未解决
**重要性**: 🟡 重要

ExplicitLM固定容量N=10^6，更新靠EMA（需训练时更新）。KBLaM可替换三元组但需重编码。MemoryLLM自动更新但不可控。没有系统支持运行时知识增/删/编辑且立即生效。

**相关工作**: paper:explicitlm2025, paper:kblam2025, paper:memoryllm2024
**目标方案**: idea:001 (TokenMem)

---

## G5: 跨领域泛化（一次训练，多领域适用）

**状态**: 未解决
**重要性**: 🟡 重要

现有记忆系统通常在特定任务上训练和测试（DecoupledRAG在每个任务上独立SFT）。没有系统证明"在一个领域SFT → 在完全不同的领域也有效"。

**相关工作**: paper:decoupledrag2025 (每任务独立SFT)
**目标方案**: idea:001 (TokenMem, News训练→MedQA/ARC/MMLU测试)
