---
type: experiment
node_id: exp:E1_baseline
title: "E1 Baseline: No-Memory + VanillaRAG (6 models x 4 datasets)"
date: 2026-04-28T12:30:00Z
models: ["qwen3-0.6B", "qwen3-1.7B", "qwen3-4B", "qwen3-8B", "gemma3-1b", "ministral-3-3b"]
datasets: ["news", "medqa", "arc", "mmlu", "arc_easy", "cf_arc_easy_val", "cf_medqa_val"]
methods: ["no_memory", "vanilla_rag"]
status: completed
---

# E1 Baseline: No-Memory + VanillaRAG

**目的**: 建立 E1 实验矩阵的两个 baseline（裸模型 + Oracle VanillaRAG），为后续 TokenMem 评测提供对照。

**评测协议**: Loglikelihood scoring (对 " A"/" B"/" C"/" D" 的 continuation log-prob 累加取 argmax)。

**Oracle 知识**: 由 DeepSeek V4 Flash 生成 ~100-200 词百科风格段落（给定 Q + 正确答案）。

## 结果

### No-Memory (裸模型)

| 模型 | News | MedQA | ARC | MMLU |
|------|------|-------|-----|------|
| qwen3-0.6B | 0.3643 | 0.3291 | 0.5196 | 0.4162 |
| qwen3-1.7B | 0.3999 | 0.4218 | 0.7355 | 0.5515 |
| qwen3-4B | 0.4754 | 0.5719 | 0.8677 | 0.6723 |
| qwen3-8B | 0.5291 | 0.6457 | 0.9104 | 0.7179 |
| gemma3-1b | 0.2399 | 0.2734 | 0.2517 | 0.2393 |
| ministral-3b | 0.5079 | 0.6112 | 0.8635 | 0.6791 |

### VanillaRAG (Oracle passage in context)

| 模型 | News | MedQA | ARC | MMLU |
|------|------|-------|-----|------|
| qwen3-0.6B | 0.8801 | 0.9034 | 0.9181 | 0.8178 |
| qwen3-1.7B | 0.9424 | 0.9379 | 0.9718 | 0.8855 |
| qwen3-4B | 0.9766 | 0.9804 | 0.9957 | 0.9588 |
| qwen3-8B | 0.9798 | 0.9866 | 0.9966 | 0.9612 |
| gemma3-1b | 0.2327 | 0.2718 | 0.2568 | 0.2432 |
| ministral-3b | 0.9728 | 0.9835 | 0.9940 | 0.9596 |

### RAG 提升 (VanillaRAG - No-Memory)

| 模型 | News | MedQA | ARC | MMLU | 平均 |
|------|------|-------|-----|------|------|
| qwen3-0.6B | +51.6pp | +57.4pp | +39.9pp | +40.2pp | +47.3pp |
| qwen3-1.7B | +54.3pp | +51.6pp | +23.6pp | +33.4pp | +40.7pp |
| qwen3-4B | +50.1pp | +40.9pp | +12.8pp | +28.7pp | +33.1pp |
| qwen3-8B | +45.1pp | +34.1pp | +8.6pp | +24.3pp | +28.0pp |
| gemma3-1b | -0.7pp | -0.2pp | +0.5pp | +0.4pp | +0.0pp |
| ministral-3b | +46.5pp | +37.2pp | +13.1pp | +28.1pp | +31.2pp |

## 关键发现

1. **gemma3-1b 无法利用上下文**: RAG 提升接近 0（no_memory 和 vanilla_rag 均 ~25%，接近随机）。该模型在 loglikelihood 评测下基本无效，可能与 tokenizer 或 1B 规模下的指令跟随能力有关。
2. **其余 5 模型 RAG 效果显著**: 平均提升 +28~47pp，小模型提升更大（能力差距越大，Oracle 知识价值越高）。
6. **News (in-domain) 与 OOD 趋势一致**: News no_memory 略低于 MedQA/ARC（新闻题目覆盖面更广），但 VanillaRAG 天花板同样达 88-98%。
3. **VanillaRAG 天花板极高**: 3B+ 模型在 Oracle 条件下达 95-99%，对 C4 (TokenMem > VanillaRAG) 形成压力。
4. **Scaling law 明确**: no_memory 下模型越大精度越高（qwen 系列: 33%→42%→57%→65%）。
5. **ministral-3b 与 qwen3-4B 接近**: 参数量相近的两个跨家族模型 baseline 相当。

## 对 Claims 的影响

- **C1 (多模型通用性)**: gemma3-1b 在 baseline 阶段即失效，可能影响"≥5/6模型有效"阈值。需关注 TokenMem 是否能在 gemma3-1b 上产生提升（如果 loglikelihood 本身就不 work，可能需要换评测方式或换模型）。
- **C2 (跨领域泛化)**: baseline 已就绪，等待 TokenMem 结果。
- **C4 (超越 VanillaRAG)**: Oracle VanillaRAG 天花板 95-99%（3B+模型），TokenMem 要超越非常困难。需重新评估 C4 的可行性或调整 claim 表述（如强调 token 开销优势）。

## 反事实数据集补充结果 (2026-04-29)

### cf_medqa_val (1146 条, no_memory 目标=反事实答案)

| 模型 | no_memory | vanilla_rag | Gap |
|------|-----------|-------------|-----|
| qwen3-0.6B | 23.12% | 94.15% | +71pp |
| qwen3-1.7B | 19.46% | 91.45% | +72pp |
| qwen3-4B | 15.97% | 93.89% | +78pp |
| qwen3-8B | 12.83% | 94.07% | +81pp |
| gemma3-1b | 24.08% | 24.35% | +0pp |

### cf_arc_easy_val (2745 条)

| 模型 | no_memory | vanilla_rag | Gap |
|------|-----------|-------------|-----|
| qwen3-0.6B | 9.29% | 86.89% | +78pp |
| qwen3-1.7B | 4.08% | 76.61% | +73pp |
| qwen3-4B | 2.00% | 80.80% | +79pp |
| qwen3-8B | 1.28% | 81.49% | +80pp |
| gemma3-1b | 23.64% | 24.88% | +1pp |

**⚠️ 重要**: 这些 MCQ logprob 数字存在天花板效应，详见 exp:E2_pilot_eval_method。实际知识冲突行为需用 CoT 评测才能暴露。

### ARC-Easy OOD 补充结果 (2026-04-29)

arc_easy 为 ARC-Easy 数据集（2376条），与 ARC-Challenge 相比难度更低。与 cf_arc_easy_val 同批运行。

| 模型 | no_memory | vanilla_rag | Gap |
|------|-----------|-------------|-----|
| qwen3-0.6B | 71.84% | 95.92% | +24.1pp |
| qwen3-1.7B | 87.67% | 99.07% | +11.4pp |
| qwen3-4B | 94.07% | 99.83% | +5.8pp |
| qwen3-8B | 96.04% | 99.87% | +3.8pp |
| gemma3-1b | 25.13% | 24.54% | -0.6pp |
| ministral-3b | - | - | `KeyError: 'ministral3'` |

**观察**: 与 ARC-Challenge 趋势一致但天花板更高（no_memory 已达 72-96%）。RAG 提升随模型增大递减（+24pp → +4pp），因为基线已经很高。gemma3-1b 仍然无效。ministral-3b 因 transformers 版本不支持 `ministral3` 架构键而加载失败。

## 技术细节

- **评测脚本**: `evaluation/eval_baseline.py`
- **Shell 入口**: `scripts/{model}_{method}.sh` (12 个)
- **结果目录**: `results/baseline/` (78 个 JSON: 36 OOD + 12 News + 30 新增[10 arc_easy + 10 cf_arc_easy + 10 cf_medqa, 缺 6 ministral])
- **运行环境**: 远程 4090-serve, GPU 6 + GPU 7 并行
- **总运行时间**: OOD ~5h (4/27 23:34 → 4/28 04:29) + News ~1.5h (4/28 11:05 → 12:27) + 反事实 ~2h (4/29)
- **修复记录**:
  - ARC 数据集 option key 归一化（数字→字母, 支持 3/4/5 选项）
  - ministral-3b 多模态模型兼容（`Mistral3ForConditionalGeneration` text-only forward）
  - 反事实数据集由 891682a 生成（cf_medqa_val 1146条 + cf_arc_easy_val 2745条）

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
