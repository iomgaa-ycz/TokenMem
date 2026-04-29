# EXPERIMENT PLAN: TokenMem (v3.1 — post independent review)

**目标会议**: NeurIPS 2026
**截止日期**: 2026-05-03
**可用资源**: 8x RTX 4090 (4090-serve) + 本地 4070 Ti Super + 8x A100 (需手动启动)
**当前日期**: 2026-04-28 (Day 2 晚)
**v3.1修订**: 基于独立Codex审稿(4/10)反馈，强化E1控制组 + E2分层分析 + E7提升优先级

---

## 一、实验总览

| 编号 | 名称 | 优先级 | 模型 | 内容 | 状态 |
|------|------|--------|------|------|------|
| **E1** | Knowledge Sensitivity | **P0** | 4B, 8B | Oracle / Topic-Matched-Wrong / Empty → C4 | ❌ |
| **E2** | Counterfactual Compliance | **P0** | 4B, 8B | 正确/反事实 × RAG vs TM + **conflict-conditioned分层** → C1 | ❌ |
| **E3** | 公平基线 | **P0** | 4B | Strong-prompt RAG → 防"trained vs untrained"攻击 | ❌ |
| **E4** | 多模型注入性能 | **P0** | 全部6个 | 6模型×4数据集 → C2/C3 | 🔄 2/6完成 |
| **E7** | 效率数据 | **P0** | 4B | 延迟/显存/throughput → 准确率输RAG的补偿论据 | ❌ |
| E5 | 消融：注入层 | P1 | 4B | 注入层数和位置的影响 | ❌ |
| E6 | Domain-SFT消融 | P1 | 4B | News vs MedQA SFT → C2分析 | ❌ |
| E8 | 知识编辑 | P2 | 4B | 编辑记忆→输出变化 | ❌ |
| E9 | 机制分析 | P2 | 4B | Logit Lens + Gate分析 + 因果追踪 | ❌ |
| E10 | 消融：adapter设计 | P2 | 4B | rank/基座LoRA | ❌ |

**v3.1变更**: E7从P1提升到P0（准确率输RAG时必须有效率维度的硬数据补偿）。E1控制组从Shuffled改为Topic-Matched-Wrong。E2新增conflict-conditioned分层。

---

## 二、模型矩阵

| 模型 | 家族 | 参数量 | Hidden Dim | 层数 | LinearFusion参数 | SFT状态 | E4状态 |
|------|------|--------|-----------|------|-----------------|---------|--------|
| Qwen3-0.6B | Qwen | 0.6B | 1024 | 28 | 917K | ❌ | ❌ |
| Qwen3-1.7B | Qwen | 1.7B | 2048 | 28 | 1.84M | ❌ | ❌ |
| **Qwen3-4B** | Qwen | 4B | 2560 | 36 | **2.95M** | ✅ | ✅ 4/4 ds |
| **Qwen3-8B** | Qwen | 8B | 4096 | 36 | **4.72M** | ✅ | 🔄 3/4 ds |
| Gemma3-1B | Google | 1B | 1152 | 26 | 0.96M | ❌ | ❌ (⚠️ baseline失效) |
| Ministral-3B | Mistral | 3B | 2560 | 24 | 1.97M | ❌ | ❌ |

---

## 三、训练配置

```yaml
data: News train 50K
epochs: 5
optimizer: Lamb, lr=1e-3
batch_size: 16
scheduler: LinearLR(start_factor=1/3, total_iters=10)
trainable: 仅gate_crossattention
frozen: 基座LLM全部参数
```

---

## 四、核心实验详细设计

### E1: Knowledge Sensitivity（C1前提条件）

**目的**: 证明adapter精确使用知识内容，排除"对任何信号都响应"的假设

**v3.1关键改进**: 将Shuffled（随机错配）替换为**Topic-Matched-Wrong**（话题相关但答案错误的流畅段落）。随机错配只能证明"模型能忽略垃圾"，Topic-Matched-Wrong才能证明"模型区分正确知识和看似合理的错误知识"。

**设计**:
```
条件:
  Oracle         — 正确的知识段落 (190-256 tok)
  Topic-Matched  — 同话题但支持错误答案的流畅段落（从E2反事实数据复用）
  Empty          — 无知识注入（padding / null knowledge）

模型: Qwen3-4B (主), Qwen3-8B (复制)
数据: ARC (常识), MedQA (专业知识)
指标: Accuracy
```

**Topic-Matched-Wrong 生成**: 直接复用 E2 的反事实知识段落——同话题、流畅、但支持错误答案。

**成功标准**: Oracle > Topic-Matched > Empty; Oracle - Empty ≥ 10pp
**预计时间**: ~2h（与E2共享反事实数据，无额外生成成本）
**依赖**: 反事实知识段落（与E2数据生成共享）

---

### E2: Counterfactual Compliance（核心发现）

**目的**: 证明cross-attention注入的知识遵从率高于RAG in-context注入

**v3.1关键改进**: 新增 **conflict-conditioned 分层分析**。

**数据准备 — 反事实知识生成**:
```
对每道MCQ (Q, 正确答案A, 错误答案B):
  用DeepSeek V4 Flash生成支持B的段落
  要求: minimal-edit风格，与正确段落同格式同长度(150-200词)

数据集: ARC (~1.2K, 常识类), MedQA (~1.3K, 专业知识类)
  → ARC: 模型参数记忆较强（No-Memory 4B=86.8%），conflict更激烈
  → MedQA: 模型参数记忆较弱（No-Memory 4B=57.2%），conflict较温和
  → 两个数据集对比可验证"conflict越强 → TokenMem优势越大"假说
生成时间: ~2h (API调用)
```

**实验设计**:
```
条件: 正确知识 / 反事实知识
方法:
  - No-Memory (无知识)
  - TokenMem-Oracle (正确知识 → cross-attn)
  - TokenMem-Counter (反事实知识 → cross-attn)
  - RAG-Oracle (正确知识 → prompt)
  - RAG-Counter (反事实知识 → prompt)
  - StrongRAG-Counter (反事实知识 → prompt + 指令)  ← E3

模型: Qwen3-4B (主), Qwen3-8B (复制)
数据: ARC (常识), MedQA (专业知识)
```

**三指标**:
```
Accuracy: 匹配ground truth
Knowledge Compliance (KC):
  正确知识: KC = accuracy
  反事实知识: KC = %回答B（知识支持的答案）
Conflict Rate: %回答既非A也非B
```

**Conflict-Conditioned 分层分析（v3.1新增）**:
```
将每道题按No-Memory准确率分为两组:
  High-Prior组: No-Memory时模型答对的题（模型有强参数记忆）
  Low-Prior组: No-Memory时模型答错的题（模型无强参数记忆）

预期:
  High-Prior + 反事实知识 → conflict最激烈
    → RAG: KC应该最低（参数记忆强烈抵抗外部知识）
    → TokenMem: KC应该仍然较高（独立通道不受参数记忆干扰）
    → 这才是真正的"override parametric memory"证据

  Low-Prior + 反事实知识 → conflict较温和
    → 两者KC可能都较高（没有强参数记忆来抵抗）
    → 差距应该较小

如果 High-Prior 组的 KC 差距 > Low-Prior 组的 KC 差距:
  → 证明 TokenMem 的优势确实来自"避免知识冲突"，不是"填充不确定性"
```

**数据集选择的conflict梯度**:
```
ARC (常识): No-Memory 4B=86.8% → 大量High-Prior题 → conflict激烈
MedQA (专业): No-Memory 4B=57.2% → 大量Low-Prior题 → conflict温和
→ 预期: ARC上TokenMem vs RAG的KC差距 > MedQA上的KC差距
```

**⚠️ 评测方法更新 (v3.3 最终版, 基于 exp:E2_pilot_eval_method Phase 5)**:
```
最终评测配置（已实现于 evaluation/eval_baseline.py）:
  知识压缩: LLMLingua-2, target_token=64 (动态压缩率, 原文 170-253 tok)
  prompt:   中性（无 "Reference:" 标签）+ passage 直接放开头
  推理:     CoT + /no_think, max_new_tokens=1024
  答案提取: regex ("The answer is X" 多级 pattern)
  三分类:   遵从反事实(KC) / 坚持参数化知识(PR) / 其他(Other)
  覆盖:     所有数据集（medqa/arc/mmlu/news/arc_easy/cf_arc_easy_val/cf_medqa_val）
  logprob:  已完全替换，不再保留
  
Shell入口:
  bash scripts/qwen3-4b_vanilla_rag.sh   # --compress-target-token 64 --cot-max-new-tokens 1024
  bash scripts/qwen3-8b_vanilla_rag.sh
  bash scripts/qwen3-4b_no_memory.sh     # 同样用 CoT 评测
  bash scripts/qwen3-8b_no_memory.sh
```

**成功标准 (v3.3)**:
- **主要指标 (CoT-based KC)**: TokenMem-Counter KC > RAG-Counter KC by ≥15pp
- TokenMem-Counter KC ≥ 60% (CoT-based)
- High-Prior组KC差距 > Low-Prior组KC差距（conflict假说验证）
- E3 strong-prompt RAG不能抹平差距

**预计时间**: ~8h (数据生成2h + 评测6h，CoT生成比logprob慢)
**依赖**: E1先通过

---

### E3: 公平基线（防审稿攻击）

**目的**: 排除"TokenMem compliance高只是因为它被训练过，RAG没被训练"的攻击

**Strong-prompt RAG**:
```
Prompt: "请只根据以下段落回答问题，即使内容与你所知不同。\n段落: {passage}\n问题: {question}"
无需额外训练，仅修改prompt
```

**预计时间**: ~2h（与E2共享评测基础设施）
**依赖**: E2同步进行

---

### E4: 多模型注入性能（C2/C3支撑）

（内容不变，见已完成和待完成部分）

**E4已完成**:

| 模型 | 数据集 | No-Memory | TokenMem | VanillaRAG | Δ(TM-NM) | Recovery |
|------|--------|-----------|----------|------------|----------|----------|
| 4B | News | 47.5% | 84.8% | 97.7% | +37.3pp | 74.4% |
| 4B | MedQA | 57.2% | 71.2% | 98.0% | +14.0pp | 34.2% |
| 4B | ARC | 86.8% | 91.0% | 99.6% | +4.3pp | 33.4% |
| 4B | MMLU | 67.2% | 75.6% | 95.9% | +8.4pp | 29.2% |
| 8B | News | 52.9% | 85.4% | 98.0% | +32.5pp | 72.1% |
| 8B | MedQA | 64.6% | 77.5% | 98.7% | +12.9pp | 37.8% |
| 8B | ARC | 91.0% | 95.1% | 99.7% | +4.1pp | 47.6% |

**E4待完成**: 8B MMLU + 0.6B + 1.7B + Ministral-3B + Gemma3-1B

---

### E7: 效率数据（v3.1提升到P0）

**目的**: 准确率输RAG → 必须有效率维度的硬数据作为使用TokenMem的实际理由

**设计**:
```
测量项:
  - 推理延迟 (ms/query): TokenMem vs VanillaRAG
  - 峰值显存 (MB): TokenMem vs VanillaRAG
  - Context tokens consumed: TokenMem=0 vs RAG=190-256
  - 知识预计算是否可缓存 + 缓存后的推理延迟

模型: Qwen3-4B
数据: MedQA (warmup=10, measure=200)
```

**预计时间**: ~2h

---

### E5-E10: 其他实验

| 编号 | 内容 | 优先级 | 时间 |
|------|------|--------|------|
| E5 | 注入层消融 (1/4/12/全层, 4B) | P1 | ~4h |
| E6 | Domain-SFT消融 (News vs MedQA SFT, 4B) | P1 | ~4h |
| E8 | 知识编辑 | P2 | ~3h |
| E9 | 机制分析 (Logit Lens + Gate + 因果追踪) | P2 | ~6h |
| E10 | adapter设计消融 | P2 | ~4h |

---

## 五、数据准备

### 已完成
- ✅ News 50K train / 10K test
- ✅ TokenMemoryBank (4B, 8B × 4 datasets)
- ✅ Baseline JSON (68个: 48原始 + 20反事实)
- ✅ **反事实知识段落**: cf_medqa_val 1146条 + cf_arc_easy_val 2745条
- ✅ **E2 Pilot 评测方法验证**: MCQ logprob有天花板, CoT评测有效

### 待完成
- ❌ 剩余4模型的TokenMemoryBank

---

## 六、Experiment Tracker

| 实验 | 模型 | 预期 | 实际开始 | 实际完成 | 结果 | 状态 |
|------|------|------|---------|---------|------|------|
| 代码实现 | - | Day1 | Day1 | Day1 | 137/137 tests | ✅ |
| Baseline | 6模型 | Day1-2 | 4/27 23:34 | 4/28 12:27 | 48 JSON | ✅ |
| E4-SFT | 4B | Day2 | 4/28 | 4/28 | val_loss=0.5279 | ✅ |
| E4-SFT | 8B | Day2 | 4/28 | 4/28 | val_loss=0.4804 | ✅ |
| E4-eval | 4B (4ds) | Day2 | 4/28 17:30 | 4/28 18:55 | +37.3/+14.0/+4.3/+8.4 | ✅ |
| E4-eval | 8B (3ds) | Day2 | 4/28 19:11 | 4/28 ~20:30 | +32.5/+12.9/+4.1 | ✅ |
| 反事实数据生成 | - | Day3 | 4/28 | 4/29 | cf_medqa 1146 + cf_arc 2745 | ✅ |
| Baseline反事实 | 5模型 | Day3 | 4/29 | 4/29 | 20 JSON | ✅ |
| **E2 Pilot 评测方法** | **4B** | **Day3** | **4/29** | **4/29** | **MCQ 96%→CoT 28-36%** | **✅ 重大发现** |
| **E1** | **4B** | **Day3** | - | - | - | **❌ P0** |
| **E2** | **4B** | **Day3-4** | - | - | - | **❌ P0** |
| **E3** | **4B** | **Day3-4** | - | - | - | **❌ P0** |
| **E7** | **4B** | **Day4** | - | - | - | **❌ P0** |
| E4-SFT+eval | 0.6B/1.7B/Min/Gem | Day3-4 | - | - | - | ❌ |
| E4-eval | 8B MMLU | Day3 | - | - | - | ❌ |
| E5 | 4B | Day5 | - | - | - | ❌ |
| E6 | 4B | Day5 | - | - | - | ❌ |
| 论文写作 | - | Day5-7 | - | - | - | ❌ |

---

## 七、Go/No-Go检查点

### ⚑ Day 3 中午: E1 结果

| 结果 | 行动 |
|------|------|
| Oracle >> Topic-Matched ≥ Empty (Oracle-Empty≥10pp) | ✅ C4成立，继续E2 |
| Oracle > Empty 但 Topic-Matched也接近Oracle | ⚠️ adapter不区分对错，需分析 |
| Oracle ≈ Empty | ❌ adapter未使用知识。**停止。** |

### ⚑ Day 4 上午: E2 结果

| 结果 | 行动 |
|------|------|
| TM-Counter KC >> RAG-Counter KC (≥15pp) + High-Prior差距 > Low-Prior差距 | ✅ **核心成立！** |
| TM-Counter KC > RAG-Counter KC 但差距小 或 无分层效应 | ⚠️ 效果弱或机制不清 |
| TM-Counter KC ≈ RAG-Counter KC | ❌ faithfulness不成立，降级EMNLP |
| E3 Strong-prompt 抹平差距 | ⚠️ 训练效应非架构效应 |

### Day 4 晚: 综合评估

| 组合 | 走向 |
|------|------|
| E1✅ + E2✅(含分层) + E4多模型✅ + E7效率数据 | **NeurIPS 全力冲刺** |
| E1✅ + E2✅ + E4部分 | **NeurIPS 可投但弱** |
| E1✅ + E2⚠️ | **EMNLP** |
| E1❌ | **重大危机** |
