# EXPERIMENT PLAN: TokenMem (v5 — RAG SFT 受控对比)

**目标会议**: NeurIPS 2026
**当前日期**: 2026-05-04
**可用资源**: 8x RTX 4090 (4090-serve) + 本地 4070 Ti Super + 8x A100 (需手动启动)
**v5变更**: 方法矩阵 3→4 (新增 RAG SFT 受控通道对比), 核心矩阵 75→100 组, Go/No-Go 改为 RAG SFT 因果判据

---

## 一、实验总览

### 核心实验矩阵: 5 模型 × 4 方法 × 5 数据集 = 100 组

| 编号 | 名称 | 优先级 | 内容 | 状态 |
|------|------|--------|------|------|
| **E-rag-sft** | RAG SFT (受控通道对比) | **P0 (最高)** | 同模型同数据同参数量, 仅注入通道不同 (context vs cross-attn) | ❌ |
| **E-main** | 核心矩阵 | **P0** | 5模型×4方法×5数据集 | 🔄 4B完成, 其余训练中 |
| **A1** | Phase 1 vs P1+P2 | **P0** | Curriculum训练的必要性消融 | ❌ |
| **A2** | Conflict-conditioned | **P0** | High-Prior vs Low-Prior分层分析 | ❌ |
| **A3** | 注入层数 | P1 | 全层/12/4/1 层消融 | ❌ |
| **A4** | 训练数据量 | P1 | 10K/25K/50K 消融 | ❌ |

---

## 二、模型矩阵（v4 最终版）

| 模型 | 家族 | 参数量 | 角色 | Phase 1 | Phase 2 | Eval |
|------|------|--------|------|---------|---------|------|
| **Qwen3-8B** | Qwen | 8B | 核心模型 | 🔄 | 🔄 | ❌ |
| Qwen3-4B | Qwen | 4B | 同家族-小 | ✅ | ✅ | ✅ 7/7 ds |
| Qwen3-14B | Qwen | 14B | 同家族-大 | 🔄 | 🔄 | ❌ |
| LLaMA-3.1-8B | Meta | 8B | 跨家族 | 🔄 | 🔄 | ❌ |
| OLMo-3-7B | AI2 | 7B | 跨家族 | 🔄 | 🔄 | ❌ |

### v4→v5 方法变更
- **新增方法**: RAG SFT — 知识压缩后放入 prompt, 用 LoRA/SFT 训练 (同数据同量), 隔离注入通道效果

### v3→v4 模型变更
- **移除**: Qwen3-0.6B, Qwen3-1.7B (规模太小), Gemma3-1B (baseline失效), Ministral-3B (架构兼容问题)
- **新增**: Qwen3-14B, LLaMA-3.1-8B, OLMo-3-7B
- **设计**: 跨规模(4B/8B/14B) × 跨家族(Qwen/LLaMA/OLMo)

---

## 三、数据集矩阵

### 主表 (5 个)

| 数据集 | 类型 | 规模 | 用途 |
|--------|------|------|------|
| News | In-domain | 8,663 | 训练域验证 |
| MMLU | OOD-General | 14,320 | 综合泛化 |
| MedQA | OOD-Specialist | ~1,300 | 专业领域泛化 |
| cf_arc_easy | Counterfactual | 2,745 | C1: Knowledge Compliance |
| cf_medqa | Counterfactual | 1,146 | C1: Knowledge Compliance |

### Appendix (2 个)
| 数据集 | 类型 | 规模 | 说明 |
|--------|------|------|------|
| ARC | OOD-Commonsense | ~1,200 | No-Memory headroom不足, TokenMem可能低于No-Memory |
| ARC-Easy | OOD-Easy | ~2,700 | 同上; 放appendix完整报告 |

---

## 四、训练配置 (CoT Curriculum SFT)

### Phase 1: 纯 News CoT SFT
```yaml
data: data/news/train_cot.jsonl (50K)
val: data/news/val_cot.jsonl
epochs: 3
optimizer: Lamb, lr=1e-3
batch_size: 2
grad_accum_steps: 16
max_seq_len: 1024
prompt_mode: cot
trainable: 仅 gate_crossattention
frozen: 基座LLM全部参数
```

### Phase 2: News + CF 混合 CoT SFT
```yaml
load_gates: Phase 1 best checkpoint
data: data/news/train_cot.jsonl + data/counterfactual/{arc_easy_cot,medqa_cot}.jsonl
cf_oversample: 2
epochs: 40 (early-stop patience=5)
其余配置同 Phase 1
```

### 4.3 方法对比矩阵 (4 方法)

| 方法 | 知识注入方式 | 训练方式 |
|------|-------------|---------|
| No-Memory | 无外部知识 | 无训练 (原始基座) |
| Vanilla RAG | 知识压缩后放入 prompt | 无训练 (零样本) |
| **RAG SFT** | **知识压缩后放入 prompt + LoRA/SFT 训练** | **CoT Curriculum SFT (同数据同量)** |
| **TokenMem** | **cross-attention 注入隐层** | **CoT Curriculum SFT (同数据同量)** |

> **RAG SFT vs TokenMem 唯一差异**: 注入通道 (in-context vs cross-attention)。训练数据、训练量、优化器、参数预算完全一致。

### E-rag-sft: RAG SFT 实验设计

**目的**: 隔离注入通道效果 — 排除"训练本身带来的提升"这一混淆因素

**设计**:
```yaml
模型: Qwen3-4B (与 TokenMem 一致)
数据: Phase 2 同数据 (News + CF), CoT 格式
训练:
  方式: LoRA 或 full SFT
  优化器: Lamb, lr=1e-3
  参数量: 与 TokenMem gate_crossattention 参数量对齐
  知识注入: 压缩知识 prepend 到 input prompt
评测: 5 个数据集, CoT nothink, 与 TokenMem 同协议
```

**硬件**: 本地 4070 Ti Super
**预计时间**: ~8h (代码 2h + 训练 4h + 评测 2h)

**预期结果**:
- 正常数据集: RAG SFT ≈ Vanilla RAG (训练不显著改善已有 in-context 能力)
- 反事实数据集: RAG SFT KC << TokenMem KC (in-context 通道在知识冲突时不如 cross-attention)
- 若成立 → 因果 claim: TokenMem 的优势来自注入通道, 非训练效果

---

## 五、评测协议

```yaml
scoring: cot_nothink
cot_max_new_tokens: 2048
knowledge_max_len: 256 (TokenMem)
compress_target_token: 64 (RAG, LLMLingua-2)
answer_extraction: regex multi-pattern ("The answer is X")

metrics:
  normal_datasets: Accuracy
  counterfactual_datasets:
    KC (Knowledge Compliance): %回答跟随注入知识
    PR (Parametric Retention): %回答跟随参数记忆
    Other: %其他 (既不跟知识也不跟参数)
```

---

## 六、已有 4B 结果（基准参考）

### 正常数据集

| 方法 | News | MMLU | MedQA |
|------|------|------|-------|
| No-Memory | 43.9% | 75.2% | 65.6% |
| Vanilla RAG | 95.4% | 86.9% | 86.3% |
| **TokenMem** | **85.3%** | **79.2%** | **73.7%** |
| Δ(TM - NM) | +41.4pp | +4.0pp | +8.1pp |

### 反事实数据集 (Knowledge Compliance)

| 方法 | cf_arc_easy KC | cf_medqa KC |
|------|---------------|-------------|
| No-Memory | 1.2% | 11.8% |
| Vanilla RAG | 20.0% | 52.3% |
| **TokenMem** | **69.0%** | **70.2%** |
| **Δ(TM - RAG)** | **+49.0pp** | **+17.9pp** |

---

## 七、消融实验详细设计

### A1: Phase 1 vs P1+P2 (Curriculum 必要性)

**设计**: 用 Phase 1 only checkpoint 评测 5 个数据集, 与 Phase 2 对比
**预期**: Phase 2 的 cf KC >> Phase 1, 正常数据集 accuracy 相当或略优
**工作量**: 低 — Phase 1 checkpoint 已有, 仅需评测
**模型**: Qwen3-4B (主), 8B (复制)

### A2: Conflict-conditioned 分层分析

**设计**:
```
将每道反事实题按 No-Memory 是否答对分为两组:
  High-Prior: No-Memory 答对的题 (模型有强参数记忆, conflict激烈)
  Low-Prior: No-Memory 答错的题 (模型无强参数记忆, conflict温和)

比较两组的 KC 差距:
  High-Prior 组: TokenMem KC vs RAG KC
  Low-Prior 组: TokenMem KC vs RAG KC

预期: High-Prior 组的 KC 差距 > Low-Prior 组
  → 证明 TokenMem 优势来自"避免知识冲突", 不是"填充不确定性"
```
**工作量**: 极低 — 无需重训, 对已有结果分组统计
**模型**: 全部 5 模型

### A3: 注入层数

**设计**: 全层(36) / 12层 / 4层 / 1层, 评测 News + cf_arc_easy
**工作量**: 中 — 需重训 3 次 (全层已有)
**模型**: Qwen3-4B

### A4: 训练数据量

**设计**: 10K / 25K / 50K News CoT SFT, 评测 News + MMLU + cf_arc_easy
**工作量**: 中 — 需重训 2 次 (50K已有)
**模型**: Qwen3-4B

---

## 八、Experiment Tracker

| 实验 | 模型 | 状态 | 结果摘要 |
|------|------|------|---------|
| **RAG SFT training** | **4B** | **❌** | **—** |
| **RAG SFT eval** | **4B** | **❌** | **—** |
| Phase 1 SFT | 4B | ✅ | val_loss ~0.47 |
| Phase 2 SFT | 4B | ✅ | news 0.404 + CF 0.210 |
| E-main eval | 4B | ✅ 7/7 ds | C1: +49/+18pp; C2: +4~41pp |
| Phase 1 SFT | 8B | 🔄 训练中 | — |
| Phase 2 SFT | 8B | 🔄 训练中 | — |
| Phase 1 SFT | 14B | 🔄 训练中 | — |
| Phase 2 SFT | 14B | 🔄 训练中 | — |
| Phase 1 SFT | LLaMA-3.1-8B | 🔄 训练中 | — |
| Phase 2 SFT | LLaMA-3.1-8B | 🔄 训练中 | — |
| Phase 1 SFT | OLMo-3-7B | 🔄 训练中 | — |
| Phase 2 SFT | OLMo-3-7B | 🔄 训练中 | — |
| Baseline | 5模型×2方法×7ds | ✅ | 所有 cot_nothink |
| A1: P1 vs P1+P2 | 4B | ❌ | — |
| A2: Conflict-cond | 4B | ❌ | — |
| A3: 注入层数 | 4B | ❌ | — |
| A4: 数据量 | 4B | ❌ | — |

---

## 九、Go/No-Go (RAG SFT 因果判据)

**核心判据**: RAG SFT 在 cf_arc_easy 上的 KC 表现 (与 TokenMem KC=69.0% 对比)

| RAG SFT cf_arc KC | 行动 |
|-------------------|------|
| < 40% | ✅ 强论文 — 因果 claim 成立: 注入通道 (cross-attn) 是关键, 非训练效果 |
| 40% ~ 55% | 🟡 可用 — 弱化因果 claim, TokenMem 仍有显著优势但需补充解释 |
| 55% ~ 65% | ⚠️ 重新评估 — 通道差异不够显著, 需增加消融或换角度论证 |
| > 65% | ❌ 放弃因果 claim — 注入通道无显著差异, 需重构论文叙事 |

---

## 十、GPU 分配

```
本地 4070 Ti Super:
  RAG SFT 训练+评测 (P0, 今天启动)
  A1 消融评测

4090-serve (8x RTX 4090):
  8B / 14B / LLaMA-3.1-8B / OLMo-3-7B Phase 1+2 训练
  E-main 评测 (非4B模型)

A100 (8x, 需手动启动):
  8B+ 大规模长时间训练 (按需)
```
