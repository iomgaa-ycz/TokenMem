# Research Wiki Log

---

## 2026-04-26 — Wiki初始化

- **Wiki created** for TokenMem project (NeurIPS 2026)
- Ingested 8 papers: explicitlm2025, decoupledrag2025, kblam2025, kcapsules2026, memoryllm2024, mplus2025, lample2019_pkm, fwpkm2026, tokmem2025
- Created 5 gaps: G1-G5
- Created 4 ideas: 001(TokenMem, active), 002-004(eliminated)
- Created 4 claims: C1-C4 (all pending)
- Created 16 graph edges
- Source: idea-discovery pipeline (research-lit → idea-creator → novelty-check → review → refine)

### Key decisions logged:
- 融合机制借鉴DecoupledRAG（代码验证：基座冻结+只训练gate_crossattention）
- 检索用FAISS（PKM留future work，因key聚类平衡性问题）
- 只需SFT（借鉴DecoupledRAG，不需Pretrain）
- SFT数据: News 50K（时间分割）
- 默认基线模型: Qwen3-4B
- TokenMemoryBank是per-model的（跨模型通过detokenize→retokenize）
- ExplicitLM(ICLR 2026)是本组prior work，互补关系
- TokMem(ICLR 2026)名称可能混淆，考虑改名

## 2026-04-26 — TokenMemoryBank存储方案简化

- **决策**: TokenMemoryBank改为纯tensor存储（token_ids + cached_emb），去掉raw_text和per-layer KV cache
- **理由**:
  - 存per-layer KV cache: 50K bank × 4层 × 2(K+V) × 256 × 2560 × fp16 ≈ 500GB（不可接受）
  - 存token_ids + embedding: 50K × (256×4B + 2560×4B) ≈ 550MB（合理）
  - raw_text用str存在tensor里不方便，需要时decode即可
  - 推理时实时编码top-k（k≤5）的KV开销可接受，与DecoupledRAG一致
- **对齐**: 与Reference/Memory-LoRA-old/fusion_bank.py设计一致（token_ids + cached_emb）
- **影响**: 编辑操作变为"重tokenize + 重算embedding"，不涉及KV重算

## 2026-04-27 — TokenMemoryBank 实现完成

- **实现**: `memory_lora/token_bank.py` (~350行), `tests/unit/test_token_bank.py` (56测试)
- **设计决策**（头脑风暴阶段确认）:
  - 合并设计: tokenizer + FAISS索引内置于TokenMemoryBank
  - 软删除 + 自动compact（阈值>=0.3时触发，对用户透明）
  - `add()` 接受 `List[Tuple[str, Tensor]]`，embedding由外部预计算
  - `migrate_to()` 返回 `List[str]`（调用方自行构建新bank）
  - `emb_dim` 无默认值（必须等于model hidden_dim）
  - FAISS使用IndexIDMap(IndexFlatIP)，embedding L2-normalize后入索引
- **Review**: spec + code quality双重审查通过，修复了阈值`>`→`>=`、retrieve边界检查等问题
- **分支**: `feat/token-memory-bank`
- **Spec**: `docs/superpowers/specs/2026-04-26-token-memory-bank-design.md`
