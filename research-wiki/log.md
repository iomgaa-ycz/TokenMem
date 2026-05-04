# Research Wiki Log

---

## 2026-05-04 — v5 战略转向: RAG SFT受控对比 + 贡献重构

- **触发**: 两轮GPT审核(v4: 3/10, v5: 5/10)揭示核心问题
- **问题诊断**: 
  1. TokenMem(训练) vs VanillaRAG(未训练) 对比不公平 → 因果claim站不住
  2. 方法贡献弱(DecoupledRAG+FAISS+Curriculum组合)
  3. 评测方法论不足以作为独立核心贡献
- **决策**:
  1. 新增 RAG SFT 受控对比实验 (P0): 同模型/同数据/同参数预算, 仅变注入通道
  2. 贡献重构: 3个核心 → 2主+1次 (方法+受控发现+次要KC指标)
  3. 评测方法论降级为evaluation setup, 不作独立贡献
  4. 论文叙事: 系统贡献 + 受控实验发现的融合
- **论文标题确定**: "TokenMem: Faithful Knowledge Internalization for Frozen LLMs via Cross-Attention"
- **方法对比扩展**: 3→4方法 (No-Memory, VanillaRAG, RAG SFT, TokenMem)
- **Go/No-Go 机制**: RAG SFT cf_arc KC <40%→强, 40-55%→可用, 55-65%→弱, >65%→放弃
- **72小时行动计划**: Day1 RAG SFT+A2 → Day2 结果+写论文 → Day3 完稿+提交
- **GPT审核结果**: v5方案5/10, "if RAG SFT results are strong, could reach 35-45% acceptance"
- **文件更新**: IDEA_REPORT v5, FINAL_PROPOSAL v5, EXPERIMENT_PLAN v5, TIMELINE v5, research-wiki v5

---

## 2026-05-03 — v4 确定: 5模型矩阵 + 4B C1核心验证通过

- **里程碑**: C1 (Faithful Knowledge Injection) 在 Qwen3-4B 上验证通过
- **核心数据**: cf_arc_easy KC: TokenMem 69.0% vs RAG 20.0% (+49pp); cf_medqa KC: 70.2% vs 52.3% (+18pp)
- **模型矩阵确定**: Qwen3-4B/8B/14B + LLaMA-3.1-8B + OLMo-3-7B (核心模型: 8B)
- **数据集确定**: 主表 News/MMLU/MedQA/cf_arc_easy/cf_medqa; ARC/ARC-Easy→Appendix
- **训练方法**: CoT Curriculum SFT (Phase 1 纯News + Phase 2 News+CF), checkpoint: `qwen3-4b_sft_cot_p2/best`
- **评测配置**: CoT + /no_think, max_new_tokens=2048, RAG用LLMLingua-2压缩至64tok
- **消融计划**: A1(P1 vs P1+P2) + A2(Conflict-conditioned) + A3(注入层数) + A4(数据量)
- **当前状态**: 4B完成, 8B/14B/LLaMA/OLMo训练中
- **文件更新**: FINAL_PROPOSAL v4, EXPERIMENT_PLAN v4, TIMELINE v4, research-wiki index v4
- **归档**: data/v2/ → archive/data_v2/; 小模型脚本已从git删除

### 关键决策
- 移除 ARC/ARC-Easy 出主表: ARC no-memory=91.5% headroom不足, P2训练后TokenMem(79.4%)<No-Memory — 放appendix
- 移除小模型 (0.6B/1.7B/gemma/ministral): baseline失效或架构不兼容, 改用跨规模+跨家族设计
- 目标维持 NeurIPS 2026

---

## 2026-05-03 — Qwen3-4B CoT P2 全量评测完成 (7 数据集)

- **实验**: `exp:E_main_4B` — Phase 2 checkpoint 在全部 7 数据集上评测
- **checkpoint**: `checkpoints/qwen3-4b_sft_cot_p2/best`
- **评测配置**: scoring=cot_nothink, cot_max_new_tokens=2048, knowledge_max_len=256
- **结果**:
  - News: 85.3% (vs NM 43.9%, RAG 95.4%) — in-domain strong
  - MMLU: 79.2% (vs NM 75.2%, RAG 86.9%) — OOD +4.0pp
  - MedQA: 73.7% (vs NM 65.6%, RAG 86.3%) — OOD +8.1pp
  - ARC: 79.4% (vs NM 91.5%) — ⚠️ 低于NM (headroom issue)
  - ARC-Easy: 93.6% (vs NM 96.6%) — ⚠️ 低于NM
  - **cf_arc_easy KC: 69.0%** (vs NM 1.2%, RAG 20.0%) — **C1核心证据**
  - **cf_medqa KC: 70.2%** (vs NM 11.8%, RAG 52.3%) — **C1核心证据**
- **结论**: C1 验证通过 (两个CF数据集均超15pp阈值), C2部分支持
- **结果文件**: `results/tokenmem/qwen3-4B_tokenmem_*.json` (7个)

---

## 2026-04-30 — E2 Curriculum SFT 完成（两阶段训练 + max_seq_len 修复）

- **实验**: `exp:E2_curriculum_sft` — 修复多数据集 SFT val_loss 停滞问题
- **根因诊断**: 4 个问题——MedQA CF 100% 截断(致命)、梯度方向冲突、36层gate累积偏移、验证集盲区
- **修复措施**: max_seq_len 64→512 + Curriculum两阶段 + 重置optimizer + 分开追踪CF val_loss
- **Phase 1 (纯news, 3ep)**: val_loss 0.468 (优于旧版 ~0.5)
- **Phase 2 (news+CF, 5ep)**: news val_loss 0.404 + CF val_loss 0.210
- **对比失败版**: 直接混合训练 news val_loss 停在 1.0；curriculum 达到 0.404，改善 +0.6
- **关键发现**: curriculum 策略成功避免梯度冲突，两个 loss 同时下降无此消彼长
- **显存适配**: batch_size 32→4 + grad_accum 1→8 适配 seq_len=512 (实测 16-42GB/卡 on 4090)
- **代码改动**: training/sft.py (+79行), scripts/qwen3-4b_sft_phase{1,2}.sh (新增)
- **运行**: 4090-serve GPU 2+4, Phase 1 ~55min + Phase 2 ~3.5h
- **best checkpoint**: `checkpoints/qwen3-4b_sft_p2/best` (step=5105, epoch=5)
- 添加 edges: 3条 (exp:E2_curriculum_sft → C1 supports, → E1_tokenmem extends, idea:001 → exp tested_by)

---

## 2026-04-29 — E2 评测方案最终确定 + 代码实现

- **决策**: 最终评测方案为 LLMLingua-2 压缩(64tok) + 中性 prompt(无 "Reference:") + CoT + nothink + 1024 tokens
- **代码实现**: `evaluation/eval_baseline.py` 完全替换 logprob 为 CoT 生成，新增 `compress_passage()` / `evaluate_cot()` / `build_cot_prompt()` / `extract_answer_letter()`
- **Shell 脚本**: 更新 `qwen3-{4b,8b}_{vanilla_rag,no_memory}.sh`，覆盖全部 7 个数据集
- **消融预实验**: 中性 prompt 在 8B ARC 上将反事实遵从率从 72%(Reference:标签) 降至 54%；停用词压缩效果不显著(56%)
- **依赖**: LLMLingua-2 已安装（本地+远程4090）
- 更新: E2_pilot 实验页(Phase 5), EXPERIMENT_PLAN(v3.3), C1 claim

---

## 2026-04-29 — E1 Baseline 补充: ARC-Easy OOD + 反事实 + arc_easy 全量评测

- **实验**: `exp:E1_baseline` 新增 3 个数据集（arc_easy, cf_arc_easy_val, cf_medqa_val）× 2 方法 × 6 模型
- **完成**: 30/36 结果（ministral-3b 因 `KeyError: 'ministral3'` transformers 架构不兼容全部失败）
- **运行环境**: 4090-serve, GPU 2 (小模型) + GPU 7 (大模型) 并行, ~1h 完成
- **ARC-Easy 结果**: no_memory 72-96% (Qwen 系列), vanilla_rag 96-99.9%, RAG Gap +4~24pp（基线已高，提升空间有限）
- **反事实知识忠诚度**: vanilla_rag cf_medqa 91-94%, cf_arc_easy 77-87%（Qwen 系列高度忠诚于外部知识）
- **gemma3-1b 持续无效**: 所有数据集均 ~25%（随机水平），loglikelihood 评测下该模型完全失效
- **数据预处理**: `scripts/build_cf_eval.py` 从反事实 JSONL 提取 test split, 重映射 counterfactual_passage→passage, target_letter→correct_letter

---

## 2026-04-29 — E2 Pilot: 反事实评测方法验证（重大发现）

- **实验**: `exp:E2_pilot_eval_method` — 诊断反事实知识遵从率虚高问题 + 评测方法三向对比
- **触发**: Baseline 反事实评测中 vanilla_rag 遵从率高达 91-96%（cf_medqa），与文献预期（40-80%）严重不符

### Phase 1: Baseline 反事实结果（20个JSON, 5模型×2方法×2数据集）
- 反事实数据（cf_medqa_val 1146条 + cf_arc_easy_val 2745条）由 891682a 生成
- vanilla_rag 遵从率: cf_medqa 91-94%, cf_arc_easy 77-87%（gemma3-1b 失效）
- no_memory 遵从率随模型增大而递减（0.6B: 23% → 8B: 13%），合理的 sanity check

### Phase 2: Logprob vs Generation 诊断
- **结论: Logprob 打分不是问题**
- 0.6B (100样本) + 4B (50样本) 上 logprob 和 greedy generation 零分歧
- Conflict-conditioned 分析: High-Prior 91.8% vs Low-Prior 95.1%（MCQ下仅3.26pp差距）
- 文献参考: Surface Form Competition (Holtzman 2021), Token Selection Bias (Zheng 2024)

### Phase 3: 评测方法三向对比（核心发现, qwen3-4B, 各50条）

**cf_medqa_val**: MCQ直答 96% → 开放式 82%(-14pp) → **CoT 36%(-60pp)** → CoT+nothink 28%(-68pp)
**cf_arc_easy_val**: MCQ直答 74% → 开放式 66%(-8pp) → **CoT 26%(-48pp)** → CoT+nothink 24%(-50pp)

- **核心机制**: MCQ直答是阅读理解（passage→option匹配），CoT迫使模型激活参数化知识与段落产生真正冲突
- CoT中观察到模型显式表达知识冲突: "Wait, but the reference..."
- CoT 50%"其他"类别是 max_new_tokens=200 截断导致，非CoT无效

### 影响
- **E2 成功标准可行性**: MCQ logprob 天花板效应(94%) → CoT 下降到 28-36%，≥15pp 差距变得现实
- **C1 claim**: 阈值和评测方法需更新——MCQ logprob 不适合度量知识冲突忠实性
- **claim:C1 threshold**: 建议改用 CoT-based KC 或同时报告 MCQ + CoT 两组数据
- 添加 edges: 3条（exp:E2_pilot → claim:C1 supports, exp:E2_pilot → exp:E1_baseline extends, idea:001 → exp:E2_pilot tested_by）

### Phase 4: SFT 兼容性分析
- **结论: 不需要重训 SFT，不需要构建 CoT 训练数据集**
- LinearFusion (gate_crossattention) 是 token 位置无关、层级独立的线性变换
- 训练时 max_seq_len=64 / label 仅 1 token，但 gate 学到的是通用知识融合能力
- 基座 LLM（冻结）本身具备 CoT 推理能力，gate 只负责知识注入旁路
- 行动项: 仅需在 eval_tokenmem.py 中新增 CoT 生成模式即可

### 关键决策
- E2 正式实验应使用 MCQ + CoT 作为主要评测方式，logprob 作为辅助对照
- 后续需增加 max_new_tokens 到 400-512 + 结构化结尾 prompt 降低"其他"比例
- 反事实数据质量已验证：模型确实跟随（MCQ下），且模型越大参数化抵抗越强（no_memory反向scaling）
- **不需要重训 SFT**: 现有 checkpoint 可直接用于 CoT 评测（LinearFusion 位置无关 + 基座 CoT 能力冻结保留）

---

## 2026-04-29 — Baseline 反事实评测完成（5模型×2数据集×2方法 = 20 JSON）

- **实验**: `exp:E1_baseline` 补充 — 反事实数据集（cf_medqa_val + cf_arc_easy_val）baseline
- **运行**: 远程4090-serve + 本地4070，共5模型（gemma3-1b/qwen3-0.6B/1.7B/4B/8B）
- **数据**: `data/counterfactual/cf_medqa_val.jsonl` (1146条) + `data/counterfactual/cf_arc_easy_val.jsonl` (2745条)
- **结果**: `results/baseline/*_cf_*.json` (20个JSON)
- **新增脚本**: 6模型 no_memory + vanilla_rag 各自的 sh 已更新支持反事实数据集
- **总计**: E1 baseline 现有 **68 个 JSON**（48 原始 + 20 反事实）

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
