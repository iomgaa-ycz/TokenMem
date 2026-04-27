# E1 Baseline 评测设计

## 摘要

为 E1 实验矩阵（6模型 × 4数据集 × 3方法）构建 No-Memory 和 VanillaRAG 两个 baseline 的评测 pipeline。包含 OOD 知识生成、统��评测脚本、12 个 shell 入口。

## 1. 整体架构

```
Step 1: 知识生成               Step 2: 评测脚本              Step 3: Shell 入口
tools/build_ood_               evaluation/                   scripts/
knowledge.py                   eval_baseline.py              {model}_{method}.sh
       │                             │                             │
  HF dataset ──��               冻结LLM推理 ──→              零参数启动
  DeepSeek API                 logprob评分
       │                             │
  data/ood/                    results/baseline/
  {dataset}.jsonl              {model}_{method}_{dataset}.json
```

## 2. OOD 知识生成

### 2.1 工具脚本

- **文件**: `tools/build_ood_knowledge.py`
- **功能**: 从 HF dataset 读取多选题 → 调 DeepSeek API 生成百科风格知识段落 ��� 写入 JSONL
- **LLM**: DeepSeek V4 Flash（`.env` 中配置）
- **并发**: 24

### 2.2 生成 Prompt

```text
Given a multiple-choice question and its correct answer, write a 100-200 word
encyclopedia-style knowledge passage that contains the information needed to
answer the question. The passage should read like a textbook paragraph —
do NOT mention the question, do NOT include option letters (A/B/C/D),
and do NOT reveal it is generated for a test.

Question: {question}
Options: A. {A}  B. {B}  C. {C}  D. {D}
Correct Answer: {correct_letter}. {correct_answer}

Knowledge passage:
```

### 2.3 输出 Schema

对齐 News 数据集的 JSONL 格式：

```jsonl
{
  "id": "medqa_0001",
  "dataset": "medqa",
  "question": "A 23-year-old pregnant woman at 22 weeks...",
  "options": {"A": "Ampicillin", "B": "Ciprofloxacin", "C": "Doxycycline", "D": "Metronidazole"},
  "correct_letter": "A",
  "passage": "Urinary tract infections during pregnancy require careful antibiotic selection..."
}
```

### 2.4 数据集来源

全部从 HF datasets 缓存加载（已缓存在本地）���

| 数据集 | HF Repo | Config | Split | 样本数 |
|--------|---------|--------|-------|--------|
| MedQA | `GBaker/MedQA-USMLE-4-options-hf` | default | test | 1273 |
| ARC | `allenai/ai2_arc` | ARC-Challenge | test | 1172 |
| MMLU | `cais/mmlu` | all | test | 14042 |

### 2.5 断��续跑

已生成的 id 跳过，写入模式为 append，支持中断后继续。

### 2.6 输出文件

- `data/ood/medqa.jsonl`
- `data/ood/arc.jsonl`
- `data/ood/mmlu.jsonl`

## 3. 评测脚本

### 3.1 文件

- **文件**: `evaluation/eval_baseline.py`（统���入口）
- **调用**: `python -m evaluation.eval_baseline --method no_memory|vanilla_rag ...`

### 3.2 评测方法：loglikelihood 打分

与老项目 `experiments/common.py:evaluate_loglikelihood_batch` 对齐：

```
1. 构造 prompt:
   "Question: {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:"

2. VanillaRAG 额外前缀:
   "Reference: {passage}\n\n" + prompt

3. 对 " A"/" B"/" C"/" D" 逐个计算 log-prob:
   context_ids = tokenizer.encode(prompt, add_special_tokens=False)
   cont_ids = tokenizer.encode(" X", add_special_tokens=False)
   full_ids = context_ids + cont_ids
   logits = model(full_ids)
   score = sum(log_softmax(logits[cont_positions]).gather(cont_tokens))

4. pred = argmax(scores)
```

### 3.3 CLI ��数

```
--model-path     模型路径（如 hugglingface_model/qwen3-0.6B）
--method         no_memory | vanilla_rag
--dataset        medqa | arc | mmlu | news
--data-dir       数据目录（默认 data/ood）
--output-dir     结果目录（默认 results/baseline）
--n-samples      截断样本数（smoke test 用，默认全量）
--device         cuda:0 | auto（默认 cuda:0）
```

### 3.4 结果输出

路径: `results/baseline/{model}_{method}_{dataset}.json`

```json
{
  "model": "qwen3-0.6b",
  "model_path": "hugglingface_model/qwen3-0.6B",
  "method": "no_memory",
  "dataset": "medqa",
  "n_samples": 1273,
  "accuracy": 0.312,
  "latency_ms_mean": 45.2,
  "timestamp": "2026-04-27T15:30:00",
  "git_sha": "abc1234"
}
```

### 3.5 No-Memory vs VanillaRAG 差异

| 方面 | No-Memory | VanillaRAG |
|------|-----------|------------|
| Prompt 前缀 | 无 | `Reference: {passage}\n\n` |
| 需要知识文件 | 否 | 是（data/ood/*.jsonl） |
| 其余逻��� | 完全相同 | 完全相同 |

## 4. Shell 脚本

### 4.1 结构

每模型 2 个 sh，共 12 个文件。每个 sh 内部显式调 3 次（每数据集一次），预留 News 注��。

```bash
# scripts/qwen3-0.6b_no_memory.sh 示例
#!/bin/bash
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
N_SAMPLES=${N_SAMPLES:--1}

COMMON="python -m evaluation.eval_baseline \
    --model-path hugglingface_model/qwen3-0.6B \
    --method no_memory \
    --output-dir results/baseline"

$COMMON --dataset medqa   --data-dir data/ood --n-samples $N_SAMPLES
$COMMON --dataset arc     --data-dir data/ood --n-samples $N_SAMPLES
$COMMON --dataset mmlu    --data-dir data/ood --n-samples $N_SAMPLES

# News（数据集就绪后取消注释）
# $COMMON --dataset news --data-dir data/news --n-samples $N_SAMPLES
```

### 4.2 完整文件清单（12 个）

| 模型 | no_memory | vanilla_rag |
|------|-----------|-------------|
| qwen3-0.6b | `qwen3-0.6b_no_memory.sh` | `qwen3-0.6b_vanilla_rag.sh` |
| qwen3-1.7b | `qwen3-1.7b_no_memory.sh` | `qwen3-1.7b_vanilla_rag.sh` |
| qwen3-4b | `qwen3-4b_no_memory.sh` | `qwen3-4b_vanilla_rag.sh` |
| qwen3-8b | `qwen3-8b_no_memory.sh` | `qwen3-8b_vanilla_rag.sh` |
| gemma3-1b | `gemma3-1b_no_memory.sh` | `gemma3-1b_vanilla_rag.sh` |
| ministral-3b | `ministral-3b_no_memory.sh` | `ministral-3b_vanilla_rag.sh` |

### 4.3 使用方式

```bash
# 默认运行（GPU 0，全量）
bash scripts/qwen3-0.6b_no_memory.sh

# 指定 GPU
CUDA_VISIBLE_DEVICES=2 bash scripts/qwen3-0.6b_no_memory.sh

# Smoke test（仅 10 条）
N_SAMPLES=10 bash scripts/qwen3-0.6b_no_memory.sh
```

## 5. 模型路径映射

| 短名 | model-path |
|------|-----------|
| qwen3-0.6b | `hugglingface_model/qwen3-0.6B` |
| qwen3-1.7b | `hugglingface_model/qwen3-1.7B` |
| qwen3-4b | `hugglingface_model/qwen3-4B` |
| qwen3-8b | `hugglingface_model/qwen3-8B` |
| gemma3-1b | `hugglingface_model/gemma3-1b` |
| ministral-3b | `hugglingface_model/ministral-3-3b` |

## 6. 执行顺序

1. **知识生成**（Step 1）: `python -m tools.build_ood_knowledge` — 生成 3 个 OOD 数据集的知识段落
2. **No-Memory 全跑**（Step 2a）: 6 模型 × 3 数据集，不依赖知识文件
3. **VanillaRAG 全跑**（Step 2b）: 6 模型 × 3 数据集，依赖 Step 1 的知识文件
4. **汇总**：18 + 18 = 36 个结果 JSON

No-Memory 不依赖知识生成，可与 Step 1 并行。

## 7. 验证计划

1. **Smoke test**: `N_SAMPLES=10 bash scripts/qwen3-0.6b_no_memory.sh` — 验证代码路径
2. **结果合理性**: No-Memory accuracy 应在 25%-60% 范围（随机猜 25%，强模型可能更高）
3. **VanillaRAG > No-Memory**: 给了正确知识后 accuracy 应有明显提升
4. **模型规模趋势**: 大模型 baseline 应普遍高于小模型
