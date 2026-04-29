# Research Wiki Log

---

## 2026-04-26 — Wiki初始化

- **Wiki created** for TokenMem project (NeurIPS 2026)
- Ingested 8 papers: explicitlm2025, decoupledrag2025, kblam2025, kcapsules2026, memoryllm2024, mplus2025, lample2019_pkm, fwpkm2026, tokmem2025
- Created 5 gaps: G1-G5
- Created 4 ideas: 001(TokenMem, active), 002-004(eliminated)

## 2026-04-28 — E1 Baseline完成 + TokenMem 4B/8B部分完成

- **E1 Baseline completed**: 48 JSON (6模型×4数据集×2方法), VanillaRAG Oracle天花板88-99%
- **E1 TokenMem partial**: Qwen3-4B (4ds) + Qwen3-8B (3ds, MMLU未完成)
- TokenMem全面超No-Memory (+4~37pp), 但全面低于VanillaRAG (Recovery 29-74%)
- gemma3-1b在loglikelihood评测下失效（~25%接近随机）
- C4(原"超越VanillaRAG") **invalidated** — TokenMem准确率低于RAG

## 2026-04-28 — GPT-5.4三轮外部审稿 + v3 Major Revision

- **Round 1 (v1 系统pipeline)**: 3/10 Clear Reject
  - "DecoupledRAG+FAISS"; RAG也有持久化/可编辑; claims未验证
- **Round 2 (v2 噪声鲁棒性)**: 5/10 Borderline
  - "attenuation not robustness"; 需utility-harm frontier
- **Round 3 (v3 忠实知识注入)**: 6/10 Borderline Weak Accept
  - "first real NeurIPS argument"; 需trained prompt baseline
- **v3核心转变**: 论文差异化从"系统能力"转向"忠实的知识注入(Faithful Injection)"
  - 新增C1(faithful injection)和C4(知识敏感性)
  - 原C4(超越RAG)→invalidated并重定义
  - 连接knowledge conflict文献(Longpre 2021, Xie 2024)
- **Result-to-claim评判**: C1:no, C2:partial, C3:no, C4:no — 核心实验待执行
- 更新: IDEA_REPORT.md(v3), FINAL_PROPOSAL.md(v3), 全部4个claim页面, gap_map, index
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

## 2026-04-27 — GateCrossAttention 知识融合模块实现完成

- **架构决策**: 完全复刻DecoupledRAG (WWW 2025)
  - LinearFusion低秩门控（class名LinearFusion，属性名gate_crossattention）
  - 知识编码: token_ids → frozen LLM全层forward (output_hidden_states) → strided sampling到64 tokens
  - Cross-attention复用LLM自身QKV权重，不加RoPE，不加causal mask
  - **全部层注入**（非原提案的4层子集，与DecoupledRAG一致）
  - Fork transformers modeling文件（非hook注入，非monkey-patch）
- **实现**:
  - `memory_lora/linear_fusion.py`: LinearFusion门控 (~50行)
  - `memory_lora/knowledge_encoder.py`: strided_sampling + compute_knowledge_hidden_states (~100行)
  - `memory_lora/tokenmem_model.py`: TokenMemForCausalLM包装器 (~120行)
  - `memory_lora/modified_models/modeling_qwen3.py`: Qwen3 cross-attention支持
  - `memory_lora/modified_models/modeling_mistral.py`: Mistral cross-attention支持
  - `memory_lora/modified_models/modeling_gemma3.py`: Gemma3 cross-attention支持
- **测试**: 81/81 通过（含unit + integration + GPU smoke test）
- **Smoke test**: Qwen3-0.6B 10步训练loss 2.88→0.77，gate参数有效学习
- **可训练参数**: Qwen3-0.6B 917K, Qwen3-4B 2.95M, Qwen3-8B 4.72M
- **关键发现**: from_pretrained的_init_weights会覆盖LinearFusion初始化，需_reinit_gates()恢复
- **分支**: `feat/gate-cross-attention`
- **Spec**: `docs/superpowers/specs/2026-04-27-gate-cross-attention-design.md`
- **已更新文档**: FINAL_PROPOSAL.md, EXPERIMENT_PLAN.md, TIMELINE.md, IDEA_REPORT.md（修正注入层数、参数量、类名）

## 2026-04-28 — E1 Baseline 完成 (No-Memory + VanillaRAG)

- **实验**: `exp:E1_baseline` — 6模型 × 2方法 × 3 OOD数据集 = 36个评测完成
- **运行**: 远程4090-serve GPU 6+7 并行，总耗时 ~5h (23:34→04:29)
- **数据**: Oracle知识由DeepSeek V4 Flash生成（medqa 1273 + arc 1172 + mmlu 14042 = 16487条）
- **评测**: loglikelihood scoring（eval_baseline.py），结果在 `results/baseline/` 36个JSON
- **关键结果**:
  - VanillaRAG Oracle 天花板: 3B+模型达95-99%，对C4形成压力
  - gemma3-1b 完全失效: no_memory和vanilla_rag均 ~25%（接近随机），无法利用上下文
  - 其余5模型RAG提升显著: +22~46pp（小模型提升更大）
  - Scaling law明确: qwen系列 no_memory 33%→42%→57%→65%
- **修复记录**:
  - ARC数据集option key归一化（21条数字key + 4条3选项 + 3条5选项）
  - ministral-3b多模态模型兼容（Mistral3ForConditionalGeneration text-only forward）
- **风险**: gemma3-1b可能影响C1的"≥5/6模型有效"阈值；C4的VanillaRAG天花板极高
- **添加edges**: 4条（exp:E1_baseline → claim:C1/C2/C4, idea:001 → exp:E1_baseline）
- **脚本**: `scripts/{model}_{method}.sh` 12个, `scripts/_run_gpu6.sh`, `scripts/_run_gpu7.sh`

## 2026-04-27 — E0 News Dataset 构建完成 (58,663 MCQ)

- **实验**: `exp:E0_news_dataset` — TokenMem SFT 训练/验证数据集构建
- **规模**: 5,308 篇文章 → 12,309 段落 → 61,401 QA → 58,663 去重 MCQ
- **划分**: 50,000 训练 + 8,663 验证（按新闻日期时间排序切分，验证集 ≥ 2026-04-26）
- **来源**: 25 个英文新闻源（Phase 1 crawl4ai 3,170 篇 + Phase 2 RSS 聚合 2,138 篇）
- **类别**: science 23% / sports 20% / politics 17% / business 14% / world 13% / technology 13%
- **LLM**: DeepSeek v4-flash（thinking=disabled，QWEN 额度耗尽后切换）
- **关键参数**: qa_per_passage=5, semaphore=5, question-level dedup
- **文件**: `data/news/train.jsonl` (50K), `data/news/val.jsonl` (8,663)
- **未达 60K 原因**: 6 个爬虫源被反爬拦截 (physorg/skynews/reuters/theverge/newscientist 部分失败)，实际文章数 5,308 vs 目标 8,000
- **代码**: `tools/build_news_qa.py` (pipeline), `tools/news_crawlers.py` (25 crawlers + RSS)

## 2026-04-28 — E1 TokenMem: Qwen3-4B/8B Oracle 评测完成

- **实验**: `exp:E1_tokenmem` — 4B (4数据集) + 8B (3数据集, MMLU未完成)
- **运行**: 远程4090-serve, 4B GPU 2 (~1.5h), 8B GPU 6 (~1.3h, MMLU ~29%时手动停止)
- **关键结果**:
  - **全部超越 No-Memory**: 7/7 已测数据集 TokenMem > No-Memory
  - 4B: News +37.3pp, MedQA +14.0pp, ARC +4.3pp, MMLU +8.4pp (平均 +16.0pp)
  - 8B: News +32.5pp, MedQA +12.9pp, ARC +4.1pp (平均 +16.5pp)
  - In-domain Recovery Rate 72-74%, OOD 29-48%
  - **未超越 VanillaRAG**（预期内），C4 需调整表述
- **Claims 更新**:
  - C1: pending → partial (4B/8B 已验证，需其他模型)
  - C2: pending → partial (4B 3/3 OOD >4pp, 8B 2/2 OOD >4pp)
  - C4: invalidated (准确率未超 VanillaRAG)
- **添加 edges**: 4条 (exp:E1_tokenmem → C1/C2/C4 supports/invalidates, idea:001 → exp:E1_tokenmem)
- **脚本**: `evaluation/eval_tokenmem.py` (预计算knowledge_outputs, 参照DecoupledRAG), `scripts/qwen3-{4b,8b}_tokenmem.sh`
- **结果**: `results/tokenmem/` 7个JSON

## 2026-04-28 — E1 Baseline 补充 News 数据集 (12 JSON)

- **实验**: `exp:E1_baseline` 更新 — 补充 News (in-domain) baseline，6模型 × 2方法 = 12 JSON
- **运行**: 远程4090-serve GPU 6+7 并行，~1.5h (11:05→12:27)
- **数据**: `data/news/news.jsonl` (val 8,663 条)
- **结果**: News 趋势与 OOD 一致 — VanillaRAG 天花板 88-98%（5/6模型），gemma3-1b 仍失效（~23%）
- **总计**: E1 baseline 现有 **48 个 JSON**（6模型 × 2方法 × 4数据集）
- **已更新**: experiments/E1_baseline.md 表格补充 News 列
