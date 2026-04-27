# EXPERIMENT PLAN: TokenMem

**目标会议**: NeurIPS 2026
**截止日期**: 2026-05-03
**可用资源**: 8x RTX 4090 (4090-serve) + 本地 4070 Ti Super + 8x A100 (需手动启动)

---

## 一、实验总览

| 编号 | 名称 | 优先级 | 模型 | 内容 |
|------|------|--------|------|------|
| **E1** | 多模型注入性能 | P0 | 全部6个 | 6模型×4数据集，TokenMem vs No-Memory vs VanillaRAG |
| **E2** | 知识编辑验证 | P1 | Qwen3-4B | 编辑记忆→验证输出变化 |
| **E3** | 消融：注入层 | P1 | Qwen3-4B | 注入层数和位置的影响 |
| **E4** | 消融：adapter设计 | P2 | Qwen3-4B | LoRA rank / 是否LoRA基座 |
| **E5** | DecoupledRAG基线 | P1 | Qwen3-4B | cross-attention注入但无持久记忆 |
| **E6** | 检索效率 | P2 | Qwen3-4B | FAISS flat vs IVF在不同bank规模 |

---

## 二、模型矩阵

| 模型 | 家族 | 参数量 | Hidden Dim | 层数 | 注入层(预估) | 角色 |
|------|------|--------|-----------|------|-------------|------|
| Qwen3-0.6B | Qwen | 0.6B | 1024 | 28 | [6,12,18,24] | 最小规模 |
| Qwen3-1.7B | Qwen | 1.7B | 2048 | 28 | [6,12,18,24] | 同家族scaling |
| Qwen3-4B | Qwen | 4B | 2560 | 36 | [8,16,24,32] | **默认基线模型** |
| Qwen3-8B | Qwen | 8B | 4096 | 36 | [8,16,24,32] | 大模型验证 |
| Gemma3-1B | Google | 1B | 1536 | 26 | [6,12,18,24] | 跨家族 |
| Ministral-3B | Mistral | 3B | 2560 | 32 | [7,14,21,28] | 跨家族 |

注入层选择策略：均匀分布在模型层的约1/4、1/2、3/4和顶部附近。

---

## 三、训练配置

### SFT（per-model，唯一训练阶段）

```yaml
data: News train 50K（时间分割较早文章）
format: (question, retrieved_knowledge_passage, answer)
epochs: 5
optimizer: Adam
lr: 1e-3
batch_size: 16
trainable: 仅GateCrossAttention（per-layer融合权重）
frozen: 基座LLM全部参数
gradient_checkpointing: true（4B+模型）
```

### 预计SFT时间（单卡4090）

| 模型 | SFT时间 |
|------|--------|
| Qwen3-0.6B | ~30min |
| Qwen3-1.7B | ~1h |
| Qwen3-4B | ~2h |
| Qwen3-8B | ~4h |
| Gemma3-1B | ~45min |
| Ministral-3B | ~1.5h |

**6模型总计**: ~10h单卡 → 8卡并行 ≈ ~1.5h wall-clock

---

## 四、详细实验设计

### E1: 多模型注入性能（核心实验，P0）

**目的**: 证明TokenMem pipeline在6个模型×4个数据集上有效

**设置**:
- 6模型, SFT on News 50K
- Oracle条件: 直接提供正确知识条目
- 基线: No-Memory, VanillaRAG (top-1检索放入prompt)

**数据集**:
| 数据集 | 角色 | 规模 |
|--------|------|------|
| News test | in-domain (时间泛化) | 10K |
| MedQA test | out-of-domain | ~1.3K |
| ARC test | out-of-domain | ~1.2K |
| MMLU test | out-of-domain | ~14K |

**预期结果表格**:
```
| Model        | Dataset | No-Memory | VanillaRAG | TokenMem | Δ(vs NM) |
|--------------|---------|-----------|------------|----------|----------|
| Qwen3-0.6B  | News    | ?         | ?          | ?        | +?       |
| Qwen3-0.6B  | MedQA   | ?         | ?          | ?        | +?       |
| Qwen3-0.6B  | ARC     | ?         | ?          | ?        | +?       |
| Qwen3-0.6B  | MMLU    | ?         | ?          | ?        | +?       |
| ...          | ...     | ...       | ...        | ...      | ...      |
| Ministral-3B | MMLU    | ?         | ?          | ?        | +?       |
```

**成功标准**: ≥5/6模型在≥3/4数据集上TokenMem > No-Memory（含out-of-domain）

---

### E2: 知识编辑验证（P1, Qwen3-4B）

**目的**: 编辑记忆条目 → LLM输出立即变化

**设计**:
1. 从News test选100条知识
2. 逐条编辑（替换正确答案为另一个选项）
3. 验证编辑前后输出变化

**指标**:
- Edit Success Rate: 编辑后输出跟随变化的比例
- Edit Latency: 单条编辑的wall-clock时间

---

### E3: 消融——注入层（P1, Qwen3-4B）

**设置**: Qwen3-4B, News test
- 1层注入: [16] / [24] / [32]
- 2层注入: [16,32]
- 4层注入: [8,16,24,32] (默认)
- 全层注入: 所有36层

---

### E4: 消融——adapter设计（P2, Qwen3-4B）

**设置**: Qwen3-4B, News test
- LoRA rank: 4, 8, 16(默认), 32
- 对比: 仅GateCrossAttention vs GateCrossAttention + 基座LoRA(q/k/v/o)

---

### E5: DecoupledRAG基线（P1, Qwen3-4B）

**目的**: 与最接近的cross-attention注入方法对比

**关键区别**:
- DecoupledRAG: 每次推理实时编码文档KV → 无持久记忆
- TokenMem: 离线缓存KV + 持久记忆bank → 支持编辑+检索

**指标**: 除Accuracy外，对比推理延迟（TokenMem有KV缓存优势）

---

### E6: 检索效率（P2, Qwen3-4B）

**设置**:
- Bank规模: 1K, 10K, 50K
- 检索方法: FAISS flat vs FAISS IVF
- 指标: Retrieval Precision@1, Latency (ms)

---

## 五、数据准备

### News数据集扩展

```bash
# 扩展News数据集到60K
# 时间分割: 较早50K用于SFT训练, 较新10K用于测试
python -m tools.expand_news_dataset --target_size 60000
python -m tools.split_by_time --input data/news/qa_full.jsonl \
  --train_output data/news/qa_train.jsonl \
  --test_output data/news/qa_test.jsonl \
  --test_size 10000
```

### TokenMemoryBank构建（per-model, per-dataset）

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

### FAISS索引

FAISS索引内置于TokenMemoryBank中，`bank.load()` 时自动从 `_embs` 重建，无需单独构建脚本。

---

## 六、Experiment Tracker

| 实验 | 模型 | 预期开始 | 预期完成 | 实际开始 | 实际完成 | 结果 | 状态 |
|------|------|---------|---------|---------|---------|------|------|
| TokenMemoryBank | - | Day 1 | Day 1 | Day 1 | Day 1 | 56/56测试通过 | ✅ |
| 代码实现(其余) | - | Day 1 | Day 2 | - | - | - | ⏳ |
| News扩展 | - | Day 1 | Day 1 | - | - | - | ⏳ |
| Bank+FAISS构建 | all | Day 2 | Day 2 | - | - | - | ⏳ |
| E1-SFT | Qwen3-0.6B | Day 2 | Day 2 | - | - | - | ⏳ |
| E1-SFT | Qwen3-1.7B | Day 2 | Day 2 | - | - | - | ⏳ |
| E1-SFT | Qwen3-4B | Day 2 | Day 2 | - | - | - | ⏳ |
| E1-SFT | Gemma3-1B | Day 2 | Day 2 | - | - | - | ⏳ |
| E1-SFT | Ministral-3B | Day 2 | Day 3 | - | - | - | ⏳ |
| E1-SFT | Qwen3-8B | Day 3 | Day 3 | - | - | - | ⏳ |
| E1-eval | 6模型×4数据集 | Day 3 | Day 3 | - | - | - | ⏳ |
| E5 | Qwen3-4B | Day 3 | Day 4 | - | - | - | ⏳ |
| E2 | Qwen3-4B | Day 4 | Day 4 | - | - | - | ⏳ |
| E3 | Qwen3-4B | Day 4 | Day 4 | - | - | - | ⏳ |
| E4 | Qwen3-4B | Day 4 | Day 5 | - | - | - | ⏳ |
| E6 | Qwen3-4B | Day 5 | Day 5 | - | - | - | ⏳ |
| 论文写作 | - | Day 4 | Day 7 | - | - | - | ⏳ |

---

## 七、Go/No-Go检查点

### Day 3 检查点（E1首批结果）

| 结果 | 行动 |
|------|------|
| ≥4/6模型在News+至少1个OOD数据集上有效 | ✅ 继续 |
| 部分模型提升弱 | ⚠️ 分析原因，继续其他模型 |
| 所有模型都不work | ❌ 紧急debug adapter设计/训练策略 |

### Day 4 检查点（E1全部结果 + OOD泛化）

| 结果 | 行动 |
|------|------|
| OOD数据集（MedQA/ARC/MMLU）也有提升 | ✅ 强故事线，正常提交 |
| OOD效果弱但in-domain好 | ⚠️ 论文聚焦in-domain；OOD作为discussion |
| 全面不work | ❌ 评估是否换venue |
