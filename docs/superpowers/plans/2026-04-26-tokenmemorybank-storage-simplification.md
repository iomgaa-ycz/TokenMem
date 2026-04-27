# TokenMemoryBank 存储方案简化 — 文档修正计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将TokenMemoryBank的存储方案从"token_ids + raw_text + kv_cache"统一修正为"token_ids + cached_emb"（纯tensor存储），与Reference/FusionBank设计对齐。

**Architecture:** TokenMemoryBank每条Entry仅存储两个tensor：token_ids（LongTensor [fusion_length]）和cached_emb（FloatTensor [emb_dim]）。需要人类审计时通过逆tokenize恢复文本。推理时检索到top-k后实时过frozen LLM得到KV表示供cross-attention使用。

**设计决策理由:**
- 存per-layer KV cache每条需~10MB（4B模型），50K bank = 500GB，不可接受
- 存token_ids + embedding每条~11KB，50K bank ≈ 550MB，合理
- 推理时实时编码top-k条目（k=1~5）的KV开销可接受，与DecoupledRAG一致
- raw_text不存（tensor里放不了str），需要时 `tokenizer.decode(token_ids)` 即可

---

## 涉及文件清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `idea-stage/IDEA_REPORT.md` | MODIFY | §2.3架构图+Entry定义+训练方式 |
| `refine-logs/FINAL_PROPOSAL.md` | MODIFY | §2 TokenMemoryBank类定义+§4训练/编辑流程 |
| `refine-logs/EXPERIMENT_PLAN.md` | MODIFY | §五 Bank构建命令说明 |
| `refine-logs/TIMELINE.md` | MODIFY | Day1任务描述（Bank构建产出） |
| `research-wiki/ideas/001_tokenmem.md` | MODIFY | Core Components描述 |
| `research-wiki/claims/C3.md` | MODIFY | 编辑流程描述（去掉raw_text→重算KV） |
| `research-wiki/log.md` | MODIFY | 追加本次设计决策记录 |

---

### Task 1: 修改 IDEA_REPORT.md

**Files:**
- Modify: `idea-stage/IDEA_REPORT.md:85-135`

- [ ] **Step 1: 修改架构图（§2.3）**

将：
```
            TokenMemoryBank (per-model)
            ┌──────────────────────────────┐
            │ token_ids (model-specific)    │
            │ raw_text (用于人类审计/迁移)   │
            │ kv_cache (离线预计算)          │
            └──────┬───────────────────────┘
                   │ FAISS检索 top-k
                   ▼
             KV表示(已缓存)
```

改为：
```
            TokenMemoryBank (per-model)
            ┌──────────────────────────────┐
            │ token_ids [fusion_length]     │
            │ cached_emb [emb_dim]          │
            └──────┬───────────────────────┘
                   │ FAISS余弦检索 top-k
                   ▼
             token_ids → frozen LLM → KV表示
```

- [ ] **Step 2: 修改跨模型迁移机制描述**

将：
```
ModelA的TokenMemoryBank
  → detokenize → raw_text → ModelB的tokenizer编码
  → 用ModelB的LLM重算KV缓存
  → 存入ModelB的TokenMemoryBank
```

改为：
```
ModelA的TokenMemoryBank
  → tokenizer_A.decode(token_ids) → 文本
  → tokenizer_B.encode(文本) → 新token_ids
  → ModelB.embedding(新token_ids).mean() → 新cached_emb
  → 存入ModelB的TokenMemoryBank
```

- [ ] **Step 3: 修改"一次性准备"描述（§2.4）**

将：
```
一次性准备:
  1. 构建TokenMemoryBank: knowledge text → tokenize → 存储token_ids
  2. 离线编码KV缓存: 冻结LLM前向传播每条知识 → 缓存per-layer KV
  3. 构建FAISS索引: knowledge embedding → FAISS index
```

改为：
```
一次性准备:
  1. 构建TokenMemoryBank: knowledge text → tokenize → 存储token_ids + cached_emb
  2. cached_emb = embedding(token_ids).mean(dim=0) → 用于FAISS检索
  3. 推理时: 检索top-k → token_ids过frozen LLM → 实时得到KV → cross-attention注入
```

- [ ] **Step 4: 修改Executive Summary中的描述**

将第13行：
> 知识以tokenizer编码后的token序列存储在per-model的TokenMemoryBank中，可通过detokenize→retokenize在不同模型间迁移。

改为：
> 知识以token_ids + embedding向量存储在per-model的TokenMemoryBank中（纯tensor，~11KB/条），可通过decode→re-encode在不同模型间迁移。

---

### Task 2: 修改 FINAL_PROPOSAL.md

**Files:**
- Modify: `refine-logs/FINAL_PROPOSAL.md:48-70, 130-140`

- [ ] **Step 1: 修改TokenMemoryBank类定义（§2）**

将：
```python
class TokenMemoryBank:
    class Entry:
        token_ids: List[int]       # tokenizer编码后的序列（模型相关）
        raw_text: str              # 原始文本（用于人类审计 + 跨模型迁移）
        kv_cache: Tensor           # 离线预计算的per-layer KV表示
        metadata: Dict             # 来源、时间戳

    def add(self, text: str) -> int:
        """tokenize → 计算KV缓存 → 计算embedding → 插入FAISS索引"""

    def edit(self, entry_id: int, new_text: str) -> None:
        """更新raw_text → 重新tokenize → 重算KV缓存 → 更新FAISS"""

    def delete(self, entry_id: int) -> None:
        """删除条目 → 从FAISS索引移除"""

    def migrate_to(self, target_model) -> 'TokenMemoryBank':
        """detokenize所有条目 → 用target_model的tokenizer重新编码"""
```

改为：
```python
class TokenMemoryBank:
    """纯tensor存储，与Reference/FusionBank设计一致。"""
    _tokens: Tensor   # [capacity, fusion_length], dtype=long
    _embs: Tensor     # [capacity, emb_dim], dtype=float32

    def add(self, text: str) -> int:
        """tokenize → pad/truncate到fusion_length → mean-pool embedding → append"""

    def edit(self, entry_id: int, new_text: str) -> None:
        """重新tokenize → 重算embedding → replace"""

    def delete(self, entry_id: int) -> None:
        """标记删除 → 从FAISS索引移除"""

    def audit(self, entry_id: int) -> str:
        """tokenizer.decode(token_ids) → 人类可读文本"""

    def migrate_to(self, target_tokenizer, target_embedding) -> 'TokenMemoryBank':
        """decode → re-encode with target tokenizer → re-embed"""
```

- [ ] **Step 2: 修改§6"多模型适配"描述**

将：
```
每个模型需要：
1. 独立的TokenMemoryBank: 用该模型tokenizer编码知识
2. 独立的KV缓存: 用该模型前向传播计算
3. 独立的GateCrossAttention: SFT训练融合权重
4. 独立的FAISS索引: 用该模型embedding计算

跨模型迁移：detokenize → raw_text → 用目标模型tokenizer重新编码 → 重算KV缓存
```

改为：
```
每个模型需要：
1. 独立的TokenMemoryBank: token_ids + cached_emb（用该模型tokenizer+embedding层）
2. 独立的GateCrossAttention: SFT训练融合权重
3. FAISS索引: 基于cached_emb的余弦检索

跨模型迁移：decode(token_ids) → target_tokenizer.encode → target_embedding.mean_pool → 新bank
推理时KV计算：检索top-k的token_ids → frozen LLM前向 → 实时得到KV供cross-attention
```

- [ ] **Step 3: 更新"训练策略"部分的pipeline描述**

确保训练策略部分中提到的"cross-attention(知识KV)"改为明确说明KV是推理时实时计算的：

将第112行附近：
```
输入: question + cross-attention(知识KV)
```

改为：
```
输入: question + cross-attention(检索到的知识token_ids → 实时编码为KV)
```

---

### Task 3: 修改 EXPERIMENT_PLAN.md

**Files:**
- Modify: `refine-logs/EXPERIMENT_PLAN.md:171-195`

- [ ] **Step 1: 修改§五"TokenMemoryBank构建"命令说明**

将：
```bash
# 对每个模型×每个数据集构建bank
for model in qwen3-0.6B qwen3-1.7b qwen3-4b qwen3-8b gemma3-1b ministral-3b; do
  for dataset in news medqa arc mmlu; do
    python -m tools.build_token_bank \
      --dataset $dataset \
      --model $model \
      --output data/tokenbank_${model}_${dataset}.pt
  done
done
```

改为：
```bash
# 对每个模型×每个数据集构建bank（存储token_ids + cached_emb）
for model in qwen3-0.6B qwen3-1.7b qwen3-4b qwen3-8b gemma3-1b ministral-3b; do
  for dataset in news medqa arc mmlu; do
    python -m tools.build_token_bank \
      --dataset $dataset \
      --model $model \
      --fusion_length 256 \
      --output data/tokenbank_${model}_${dataset}.pt
    # 产出: {token_ids: [N, 256], cached_emb: [N, emb_dim]}
  done
done
```

- [ ] **Step 2: 确认FAISS索引构建无需修改**

FAISS索引构建命令基于embedding检索，不涉及KV cache，无需修改。验证通过。

---

### Task 4: 修改 TIMELINE.md

**Files:**
- Modify: `refine-logs/TIMELINE.md:28-30`

- [ ] **Step 1: 修改Day 1"实现TokenMemoryBank类"的产出描述**

将：
```
| 上午 | 实现TokenMemoryBank类 | 编码 | 本地 | `memory_lora/token_bank.py` | 无 |
```

改为：
```
| 上午 | 实现TokenMemoryBank类(token_ids+emb存储) | 编码 | 本地 | `memory_lora/token_bank.py` | 无 |
```

- [ ] **Step 2: 修改Day 2"构建TokenMemoryBank"描述**

将第45行：
```
| 上午 | 构建TokenMemoryBank(6模型×4数据集) | 脚本 | 4090 GPU2 | `data/tokenbank_*.pt` | Day1代码 |
```

改为：
```
| 上午 | 构建TokenMemoryBank(6模型×4数据集, token_ids+emb) | 脚本 | 4090 GPU2 | `data/tokenbank_*.pt` (~550MB/model) | Day1代码 |
```

---

### Task 5: 修改 research-wiki/ideas/001_tokenmem.md

**Files:**
- Modify: `research-wiki/ideas/001_tokenmem.md:18`

- [ ] **Step 1: 修改Core Components描述**

将：
```
1. **TokenMemoryBank** (per-model): tokenized知识序列 + 离线KV缓存
```

改为：
```
1. **TokenMemoryBank** (per-model): token_ids [fusion_length] + cached_emb [emb_dim]（纯tensor，推理时实时算KV）
```

---

### Task 6: 修改 research-wiki/claims/C3.md

**Files:**
- Modify: `research-wiki/claims/C3.md:13`

- [ ] **Step 1: 修改Claim描述中的编辑流程**

将：
```
**Claim**: 编辑TokenMemoryBank中的知识条目（修改raw_text→重新tokenize→重算KV缓存）后，LLM输出立即跟随变化，无需重训练adapter。
```

改为：
```
**Claim**: 编辑TokenMemoryBank中的知识条目（新文本→重新tokenize→重算cached_emb→更新FAISS）后，LLM输出立即跟随变化，无需重训练adapter。
```

---

### Task 7: 追加 research-wiki/log.md 设计决策记录

**Files:**
- Modify: `research-wiki/log.md:24`

- [ ] **Step 1: 在log末尾追加本次决策**

在文件末尾追加：
```markdown

## 2026-04-26 — TokenMemoryBank存储方案简化

- **决策**: TokenMemoryBank改为纯tensor存储（token_ids + cached_emb），去掉raw_text和per-layer KV cache
- **理由**:
  - 存per-layer KV cache: 50K bank × 4层 × 2(K+V) × 256 × 2560 × fp16 ≈ 500GB（不可接受）
  - 存token_ids + embedding: 50K × (256×4B + 2560×4B) ≈ 550MB（合理）
  - raw_text用str存在tensor里不方便，需要时decode即可
  - 推理时实时编码top-k（k≤5）的KV开销可接受，与DecoupledRAG一致
- **对齐**: 与Reference/Memory-LoRA-old/fusion_bank.py设计一致（token_ids + cached_emb）
- **影响**: 编辑操作变为"重tokenize + 重算embedding"，不涉及KV重算
```

---

## 执行检查清单

| # | 文件 | 核心修改点 | 验证方式 |
|---|------|-----------|---------|
| 1 | IDEA_REPORT.md | 架构图+迁移机制+准备步骤 | grep "kv_cache" 应返回0结果 |
| 2 | FINAL_PROPOSAL.md | Entry定义+多模型适配+训练pipeline | grep "raw_text" 应返回0结果 |
| 3 | EXPERIMENT_PLAN.md | Bank构建命令加fusion_length参数 | 命令格式正确 |
| 4 | TIMELINE.md | 产出描述对齐 | 无KV cache相关描述 |
| 5 | 001_tokenmem.md | 组件描述简化 | 一行描述即可 |
| 6 | C3.md | 编辑流程更新 | 不再提raw_text/KV缓存 |
| 7 | log.md | 追加决策记录 | 记录完整理由 |

**最终验证命令:**
```bash
grep -r "kv_cache\|raw_text" /home/iomgaa/Projects/Memory-LoRA/ --include="*.md" | grep -v Reference | grep -v node_modules | grep -v plans/
# 预期: 0 结果
```
