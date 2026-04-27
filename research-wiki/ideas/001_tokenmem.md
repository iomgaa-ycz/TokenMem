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

完整的检索→融合→更新pipeline，让任意冻结LLM通过轻量SFT获得可读、可编辑、可跨模型迁移的token级内部化记忆。在6个模型×3个家族上验证通用性。

## Core Components

1. **TokenMemoryBank** (per-model): 内置tokenizer + FAISS索引 + token_ids/emb存储 + 软删除/compact（已实现: `memory_lora/token_bank.py`, 56/56 tests）
2. **LinearFusion** (gate_crossattention): 零初始化低秩门控融合，完全复刻DecoupledRAG。全部层注入，fork transformers modeling文件（已实现: `memory_lora/linear_fusion.py` + `modified_models/modeling_qwen3|mistral|gemma3.py`, 81/81 tests）
3. **TokenMemForCausalLM**: 包装器，管理frozen base + trainable gates + 知识编码（已实现: `memory_lora/tokenmem_model.py`, smoke test loss 2.88→0.77）
4. **知识管理**: add/edit/delete/audit/migrate_to，通过migrate_to()导出文本跨模型迁移

## Inspired By

- paper:explicitlm2025 — token级记忆理念
- paper:decoupledrag2025 — cross-attention融合机制
- paper:lample2019_pkm — 检索效率思路（当前用FAISS替代）

## Target Gaps

- gap:G1 (多模型通用性)
- gap:G2 (冻结LLM即插即用)
- gap:G3 (完整pipeline)
- gap:G4 (动态知识管理)
- gap:G5 (跨领域泛化)

## Key Risks

- Novelty质疑（组件均来自已有工作）→ 靠6模型通用性+跨域泛化说话
- Out-of-domain泛化可能不足
- 大模型(8B)训练时间可能超预期

## Connections

[AUTO-GENERATED]
- inspired_by → paper:explicitlm2025
- inspired_by → paper:decoupledrag2025
- inspired_by → paper:lample2019_pkm
- addresses_gap → gap:G1, gap:G2, gap:G3, gap:G4, gap:G5
