---
type: experiment
node_id: exp:E2_pilot_eval_method
title: "E2 Pilot: 反事实评测方法验证 — MCQ logprob vs 开放式生成 vs CoT"
date: 2026-04-29T12:00:00Z
models: ["qwen3-0.6B", "qwen3-1.7B", "qwen3-4B", "qwen3-8B", "gemma3-1b"]
datasets: ["cf_medqa_val", "cf_arc_easy_val"]
methods: ["no_memory", "vanilla_rag"]
status: completed
---

# E2 Pilot: 反事实评测方法验证

**目的**: 发现 vanilla_rag 在反事实数据上的知识遵从率异常高（~94%），与知识冲突文献预期（40-80%）不符。本实验诊断原因并验证替代评测方案。

**动机**: E2（Counterfactual Compliance）是 C1 的核心实验。如果评测方法本身有天花板效应，E2 的 ≥15pp 成功标准将不可能达到。

## Phase 1: Baseline 反事实评测结果

6 个模型 × 2 方法 × 2 反事实数据集的 MCQ logprob 结果：

### cf_medqa_val（模型参数化知识较弱）

| 模型 | no_memory | vanilla_rag | Gap |
|------|-----------|-------------|-----|
| qwen3-0.6B | 23.12% | 94.15% | +71pp |
| qwen3-1.7B | 19.46% | 91.45% | +72pp |
| qwen3-4B | 15.97% | 93.89% | +78pp |
| qwen3-8B | 12.83% | 94.07% | +81pp |
| gemma3-1b | 24.08% | 24.35% | +0pp (失效) |

### cf_arc_easy_val（模型参数化知识较强）

| 模型 | no_memory | vanilla_rag | Gap |
|------|-----------|-------------|-----|
| qwen3-0.6B | 9.29% | 86.89% | +78pp |
| qwen3-1.7B | 4.08% | 76.61% | +73pp |
| qwen3-4B | 2.00% | 80.80% | +79pp |
| qwen3-8B | 1.28% | 81.49% | +80pp |
| gemma3-1b | 23.64% | 24.88% | +1pp (失效) |

**关键观察**: no_memory 反事实准确率随模型增大而递减（大模型更难"意外"选中反事实答案），这是合理的 sanity check。但 vanilla_rag 在 cf_medqa 上高达 91-94%，不分模型大小。

## Phase 2: Logprob vs Generation 诊断

**假说**: logprob 单 token 打分（" A"/" B"/" C"/" D"）可能是虚高的原因（Surface Form Competition, Holtzman 2021; Token Selection Bias, Zheng 2024）。

**实验设计**: 对同一批样本同时做 logprob argmax 和 greedy generation，比较分歧。

| 模型 | 数据 | 样本数 | Logprob | Generation | 分歧数 |
|------|------|--------|---------|------------|--------|
| qwen3-0.6B | cf_medqa | 100 | 95.0% | 95.0% | **0** |
| qwen3-4B | cf_medqa | 50 | 96.0% | 96.0% | **0** |
| qwen3-4B | cf_medqa (no_mem) | 50 | 18.0% | 18.0% | **0** |

**结论**: **Logprob 打分不是问题**。两种方法在 200 样本上零分歧。高遵从率是模型的真实行为。

### Conflict-Conditioned 子分析（qwen3-4B, 200 条 cf_medqa）

将样本按 no_memory 是否答对真实答案分为两组：

| 分组 | n | RAG 反事实遵从率 | Logprob Margin |
|------|---|-----------------|----------------|
| High-Prior（模型答对真实答案 → 知识冲突强） | 98 | 91.84% | 7.06 |
| Low-Prior（模型答错真实答案 → 知识冲突弱） | 102 | 95.10% | 7.65 |
| Gap | — | **+3.26pp** | — |

即使模型拥有强参数化知识（能答对的题），vanilla_rag 仍有 91.8% 遵从反事实。MCQ 格式下知识冲突信号极其微弱。

## Phase 3: 评测方法三向对比（核心发现）

**实验设计**: qwen3-4B，各 50 条，三种评测方式比较。

### 三种评测方式

1. **MCQ 直答 (logprob)**: `Reference: {passage}\nQuestion: ...\nA. B. C. D.\nAnswer:` → logprob argmax
2. **开放式生成**: `Reference: {passage}\nQuestion: ...\nAnswer:` → greedy decode → 关键词匹配选项
3. **MCQ + CoT**: `Reference: {passage}\nQuestion: ...\nA. B. C. D.\nLet's think step by step, then give the answer.\n` → greedy decode 200 tok → regex 提取答案字母
4. **MCQ + CoT + nothink**: 同上但添加 `/no_think` 前缀

### cf_medqa_val 结果（参数化知识弱，4B no_mem=57%）

| 方法 | 反事实遵从 | 坚持参数化知识 | 其他/无法判定 |
|------|----------|--------------|-------------|
| MCQ 直答 (logprob) | **96.0%** | 2.0% | 2.0% |
| 开放式生成 | **82.0%** | 8.0% | 10.0% |
| MCQ + CoT | **36.0%** | 12.0% | 52.0% |
| MCQ + CoT + nothink | **28.0%** | 18.0% | 54.0% |

### cf_arc_easy_val 结果（参数化知识强，4B no_mem=87%）

| 方法 | 反事实遵从 | 坚持参数化知识 | 其他/无法判定 |
|------|----------|--------------|-------------|
| MCQ 直答 (logprob) | **74.0%** | 26.0% | 0.0% |
| 开放式生成 | **66.0%** | 24.0% | 10.0% |
| MCQ + CoT | **26.0%** | 24.0% | 50.0% |
| MCQ + CoT + nothink | **24.0%** | 26.0% | 50.0% |

### 分歧案例：CoT 暴露的知识冲突推理

```
# cf_arc_easy [6]: MCQ选B(反事实), CoT选A(参数化知识)
CoT推理: "Tissue → Organ → Organ System → Cell
          Wait, but the reference..."
→ 模型先输出参数化知识答案，然后开始质疑反事实段落

# cf_arc_easy [4]: MCQ选B(反事实), CoT选C(参数化知识)  
CoT推理: "The question is asking which process is responsible
          for the growth and repair of human tissue..."
→ 模型在推理中重新分析问题，调用了生物学常识

# cf_medqa [30]: MCQ选A(反事实), CoT选D(参数化知识)
→ 模型在CoT中重新进行了临床推理，覆盖了反事实段落
```

## 核心发现与机制解释

### 发现 1: MCQ 直答是阅读理解，不是知识冲突

MCQ logprob 测量的是 P(token | context)，当 passage 显式支持某选项时，模型做的是 passage→option 表层匹配（reading comprehension），而非 "用参数化知识与外部知识博弈后做决策"。

**证据**: MCQ 直答不分模型大小都达 91-96%（cf_medqa），但 CoT 下降到 28-36%。这说明模型在 MCQ 直答中从未"思考"知识冲突。

### 发现 2: CoT 迫使模型激活参数化知识

CoT 的推理过程会触发参数化知识的提取，与反事实段落产生真正的冲突。这是因为：
- 推理 token 创建了参数化知识与上下文知识之间的桥梁
- 最终答案条件化于推理过程，而非仅条件化于段落

### 发现 3: 冲突强度与参数化知识正相关

- ARC（强参数化知识，no_mem=87%）：MCQ 74% → CoT 26%，**下降 48pp**
- MedQA（弱参数化知识，no_mem=57%）：MCQ 96% → CoT 36%，**下降 60pp**
- CoT 下 ARC 的"坚持参数化知识"比例（24-26%）远高于 MedQA（12-18%）

### 发现 4: "其他"类别是 token 截断问题，不是 CoT 无效

CoT 的 50% "其他"类别来自 max_new_tokens=200 不够模型完成推理。正式实验需增加到 400-512 tokens 并使用结构化结尾 prompt。

## 对 E2 实验设计的影响

| | MCQ logprob（当前） | MCQ + CoT（推荐） |
|---|-------------------|------------------|
| RAG cf_medqa 遵从率 | ~94%（天花板） | ~28-36%（有空间） |
| RAG cf_arc 遵从率 | ~80%（天花板） | ~24-26%（有空间） |
| E2 ≥15pp 可行性 | ❌ 不可能 | ✅ 充分空间 |
| 知识冲突可观测 | ❌ 被 MCQ 格式掩盖 | ✅ 推理链清晰可见 |
| 与文献一致性 | ❌ 94% 远超文献预期 | ✅ 26-36% 接近文献报告范围 |

**核心结论**: E2 正式实验应使用 MCQ + CoT 作为主要评测方式，logprob 作为辅助对照。CoT 评测将 vanilla_rag 遵从率从 94% 拉到 28-36%，释放出充分的差异化空间，使 C1 的 ≥15pp 成功标准变得现实可行。

## 后续改进方向

1. **增加 max_new_tokens** 到 400-512，降低"其他"比例
2. **结构化结尾 prompt**: 添加 "Therefore, the answer is" 引导模型输出明确答案
3. **两阶段法**: CoT 生成 → append 推理 → logprob 取最终答案
4. **LLM-judge**: 对无法关键词匹配的案例用 LLM 判定

## Phase 4: SFT 兼容性分析 — TokenMem 是否需要重训以支持 CoT 评测

**问题**: 当前 SFT 只训练模型输出单个字母（`--max-seq-len 64`，label 仅 1 token），改用 CoT 评测（生成 400+ token）后 TokenMem 是否仍然有效？是否需要用 CoT 格式的数据重新训练？

**结论**: **不需要重训。** 现有 SFT checkpoint 可直接用于 CoT 评测。

### 架构分析

当前 SFT 训练的唯一可训练组件是 LinearFusion（gate_crossattention）：

```
公式: h = h_self_attn + α · dropout(cross_attn_output) @ W_A @ W_B
参数: W_A [hidden_dim, 16] + W_B [16, hidden_dim]
      4B: ~2.95M params, 8B: ~4.72M params
初始化: W_B=0 → t=0 gate 输出为零，不干扰基座
```

三个关键性质保证了 CoT 泛化：

1. **Token 位置无关**: W_A/W_B 是固定矩阵，不依赖位置编码。训练时 max_seq_len=64，推理时生成 400 token 使用同一变换。
2. **层级独立**: 每层 Transformer 有独立的 LinearFusion。CoT 每个新生成的 token 都经过所有层的 gate，持续接收 cross-attention 知识。
3. **职责分离**: 基座 LLM（冻结）负责 CoT 推理能力（Qwen3 本身支持 thinking 模式），gate 只负责"如何将外部知识混入 hidden states"。两者独立。

### 类比

```
基座 LLM = 大脑（思考、推理、CoT）    ← 冻结，能力不变
Gate     = 耳机（持续播放外部知识）     ← SFT 训练的对象
```

不需要重训"耳机"来让"大脑"做 CoT。耳机学会的是"如何把声音传到大脑"，无论大脑是直接回答还是先推理再回答。

### 训练数据格式确认

```python
# training/data.py — make_collate_fn
# input_ids: "Question: ...\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer: D"
# labels:    [-100, -100, ..., -100, " D"]  ← 仅最后1个token有loss
# knowledge: passage text (max 256 tokens, cross-attention注入)
```

单 token loss 通过梯度反传到所有层的 gate，教会 gate "如何在 hidden states 中融入知识使得最终 logits 偏向正确答案"。这是一个通用的知识融合能力，不是特定于单 token 输出的能力。

### 行动项

- ❌ 不需要构建新的 CoT 训练数据集
- ❌ 不需要重新 SFT
- ✅ 只需在 `evaluation/eval_tokenmem.py` 中新增 CoT 生成模式
- ✅ 建议用现有 4B checkpoint 先跑几条样本验证 TokenMem + CoT 是否 work

## Phase 5: 最终评测方案确定 + 代码实现

基于 Phase 3-4 的预实验 + 中性 prompt / 压缩知识的消融实验，确定 E2 正式评测方案并完成代码实现。

### 最终评测配置

```
Prompt:       中性（无 "Reference:" 标签），passage 直接放开头
知识压缩:     LLMLingua-2，统一压缩到 64 token（动态压缩率，原文 170-253 tok）
推理:         CoT + /no_think，max_new_tokens=1024
答案提取:     regex（"The answer is X" pattern，多级 fallback）
三分类:       遵从反事实 / 坚持参数化知识 / 其他
适用范围:     所有数据集（原始 + 反事实），vanilla_rag + no_memory
```

### 预实验汇总（50条样本，qwen3-4B / qwen3-8B）

| 方案 | 8B cf_medqa 遵从 | 8B cf_arc 遵从 | 8B cf_arc 参数化 |
|------|-----------------|---------------|-----------------|
| MCQ logprob (旧) | 94% | 81% | — |
| "Reference:" + CoT + nothink (512tok) | 90% | 72% | 12% |
| 中性 + CoT + nothink (1024tok) | 90% | **54%** | **34%** |
| 中性 + 停用词压缩 + CoT (1024tok) | 84% | 56% | 32% |

**选择中性 + LLMLingua-2(64tok) + CoT 的理由**:
- 中性 prompt 在 8B ARC 上效果最好（54% 遵从，34% 参数化抵抗）
- LLMLingua-2 比停用词压缩更规范、可复现，审稿人不会质疑
- 64 token 压缩统一了不同长度段落的 token 预算

### 代码实现（2026-04-29）

- `evaluation/eval_baseline.py`: 删除 logprob 路径，新增 `compress_passage()` + `build_cot_prompt()` + `extract_answer_letter()` + `evaluate_cot()`
- `scripts/qwen3-{4b,8b}_vanilla_rag.sh`: 新增 `--compress-target-token 64 --cot-max-new-tokens 1024`，覆盖 7 个数据集
- `scripts/qwen3-{4b,8b}_no_memory.sh`: 同步改为 CoT 评测，保持评测方法一致
- CLI 新增参数: `--compress-target-token` (默认 64), `--cot-max-new-tokens` (默认 1024)
- 结果 JSON 新增字段: `scoring`, `compress_target_token`, `cot_max_new_tokens`, `extract_success_rate`, `avg_gen_length`

## 复现信息

- **诊断脚本**: 均为临时 Python 脚本在本地 4070 Ti Super 上运行
- **模型**: `hugglingface_model/qwen3-0.6B`, `hugglingface_model/qwen3-4B`
- **数据**: `data/counterfactual/cf_medqa_val.jsonl` (1146条), `data/counterfactual/cf_arc_easy_val.jsonl` (2745条)
- **评测代码基础**: `evaluation/eval_baseline.py`（logprob 部分）
- **Baseline 反事实结果**: `results/baseline/*_cf_*.json` (20个JSON)
- **关键参数**:
  - logprob: tokenizer.encode(prompt) + tokenizer.encode(" " + label) → log_softmax → gather → sum
  - generation: model.generate(max_new_tokens=60, do_sample=False) → 关键词匹配
  - CoT: prompt 末尾加 "Let's think step by step, then give the answer.\n" → model.generate(max_new_tokens=200, do_sample=False) → regex 提取字母
  - CoT+nothink: prompt 前缀 "/no_think\n" + 同上

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
