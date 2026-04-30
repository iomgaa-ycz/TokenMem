---
type: experiment
node_id: exp:E2_curriculum_sft
title: "E2 Curriculum SFT: 两阶段训练修复多数据集 val_loss 停滞"
date: 2026-04-29T21:00:00Z
models: ["qwen3-4B"]
datasets: ["news", "cf_arc_easy", "cf_medqa"]
methods: ["curriculum_sft_phase1", "curriculum_sft_phase2"]
status: completed
---

# E2 Curriculum SFT: 两阶段训练修复多数据集 val_loss 停滞

**目的**: 解决多数据集 SFT 中 val_loss 停滞在 ~1.0 的问题（旧版纯 news 为 ~0.5），使模型同时学会利用一致性知识（news）和处理知识冲突（counterfactual）。

**前置实验**: 直接混合训练（`qwen3-4b_sft.sh`，news + CF oversample=2）失败——val_loss 停在 1.0，train_loss 降到 0.57 但不泛化。

## 问题诊断

对失败版本的根因分析发现 **四个问题**：

| 问题 | 严重程度 | 详情 |
|------|---------|------|
| MedQA CF 100% 截断 | **致命 bug** | `max_seq_len=64`，MedQA prompt 平均 216 tokens → answer token 全部丢失，5,346 样本 zero gradient |
| 梯度方向冲突 | **核心矛盾** | news（温和增强 gate）与 CF（强力覆写 gate）对 W_A/W_B 的梯度方向不同 |
| 36 层 gate 累积偏移 | **放大机制** | 每层 gate 被 CF 推向更激进 → 累积后 residual stream 偏离 frozen LM head 解码空间 |
| 验证集盲区 | **诊断障碍** | val set 只有 news，无法衡量 CF 学习进展 |

### 数据健康度（失败版本）

| 数据集 | 总样本 | 截断无效 | 有效 |
|--------|--------|----------|------|
| News train | 50,000 | ~14,000 (28%) | ~36,000 |
| CF arc_easy (×2) | 9,974 | ~2,990 (30%) | ~6,984 |
| **CF medqa (×2)** | **5,346** | **5,346 (100%)** | **0** |

## 实验设计

### 修复措施

1. **`max_seq_len` 64→512**: 消除 MedQA 100% 截断（news/CF 截断率从 ~29% 降到 ~0%）
2. **Curriculum 两阶段训练**: Phase 1 纯 news 3 epoch → Phase 2 news+CF 5 epoch
3. **Phase 2 重置 optimizer**: 数据分布变化，独立 warmup
4. **分开追踪 val_loss**: `val/news_loss` + `val/cf_loss`，best 按 news 选

### 训练配置

```yaml
# Phase 1 (纯 news)
data: News train 50K
epochs: 3
batch_size: 4 (per GPU) × 2 GPU = 8, grad_accum=8 → effective 64
optimizer: Lamb lr=1e-3 (reset)
scheduler: LinearLR(1/3→1, 10 steps)
max_seq_len: 512
knowledge_max_len: 256, strided→64
ckpt_dir: checkpoints/qwen3-4b_sft_p1

# Phase 2 (news + CF)
data: News 50K + CF arc_easy 4987×2 + CF medqa 2673×2 = 65,320
val: News val 8,663 + CF test (arc_easy 2745 + medqa 1146 = 3,891)
epochs: 5
load_gates: Phase 1 best
optimizer: Lamb lr=1e-3 (reset, 独立 warmup)
其余同 Phase 1
ckpt_dir: checkpoints/qwen3-4b_sft_p2
```

### 显存适配

`max_seq_len=512` 导致 `batch_size=32` OOM (47.5/49GB per 4090)。降为 `batch_size=4 + grad_accum_steps=8`，有效 batch 不变。实测 ~16GB/卡（单卡 smoke test 34GB）。

## 结果

### Phase 1: 纯 News（3 epoch）

| Epoch | Train Loss | News Val Loss | 备注 |
|-------|-----------|---------------|------|
| 1 | 1.039 | 0.661 | |
| 2 | 0.715 | 0.534 | |
| **3** | **0.584** | **0.468** | **best** |

**对比旧版**: 旧版（`max_seq_len=64`）纯 news 5 epoch val_loss ~0.5。本版 3 epoch 即达 0.468 → `max_seq_len=512` 修复有效，更多有效样本加速收敛。

### Phase 2: News + CF（5 epoch，加载 Phase 1 best gate）

| Epoch | Train Loss | News Val Loss | CF Val Loss | 备注 |
|-------|-----------|---------------|-------------|------|
| 1 | 0.780 | 0.497 | 0.458 | gate 加载后初始适应 |
| 2 | 0.571 | 0.462 | 0.321 | |
| 3 | 0.545 | 0.444 | 0.266 | |
| 4 | 0.483 | 0.415 | 0.239 | |
| **5** | **0.496** | **0.404** | **0.210** | **best** |

### CF Val Loss 逐 step 追踪

| Step | CF Val Loss | 趋势 |
|------|------------|------|
| 500 | 1.005 | 初始（gate 刚从纯 news 切换） |
| 1000 | 0.473 | 快速下降 |
| 1500 | 0.371 | |
| 2000 | 0.324 | |
| 2500 | 0.295 | |
| 3000 | 0.273 | |
| 3500 | 0.251 | |
| 4000 | 0.237 | |
| 4500 | 0.233 | |
| 5000 | **0.210** | 仍在下降 |

### 全版本对比

| 指标 | 旧版 (news only, seq=64) | 失败版 (混合, seq=64) | **Curriculum (seq=512)** |
|------|-------------------------|----------------------|--------------------------|
| News Val Loss | ~0.5 | **1.0** (停滞) | **0.404** ✅ |
| CF Val Loss | — | 不可观测 | **0.210** ✅ |
| MedQA 有效率 | — | 0% | ~100% ✅ |
| 过拟合 | 无 | 严重 | **无** ✅ |

## 分析

### 关键发现

1. **`max_seq_len=64` 是致命 bug，不是超参选择**：MedQA prompt 100% 被截断，5,346 样本完全不产生梯度。修复后 news val_loss 从 ~0.5 改善到 0.468（Phase 1）→ 0.404（Phase 2）。

2. **Curriculum 策略解决了梯度冲突**：直接混合训练时 news val_loss 停在 1.0；curriculum 训练后 news val_loss 降到 0.404，甚至优于纯 news 训练。原因：Phase 1 的稳定 gate 为 Phase 2 提供了好的初始化，CF 梯度在此基础上微调而非破坏性更新。

3. **两个 loss 同时下降**：Phase 2 中 news val_loss（0.497→0.404）和 CF val_loss（1.005→0.210）均持续下降，无此消彼长现象。这证明 curriculum 策略成功避免了梯度冲突。

4. **CF 学习速度很快**：从 step 500（1.005）到 step 1000（0.473），仅 500 步 CF val_loss 就减半。模型在 Phase 1 建立的稳定 gate 基础上快速适应 CF 数据。

### 对 C1 (Faithful Injection) 的影响

本实验验证了 SFT 训练侧可以同时学习一致性知识和反事实知识。但 C1 的验证需要下游评测实验（E2 Counterfactual Compliance），比较 TokenMem vs RAG 在反事实条件下的 Knowledge Compliance。当前的 low CF val_loss (0.210) 是积极信号——模型能从 CF passage 中提取正确答案。

### 运行环境

- **服务器**: 4090-serve, GPU 2+4 双卡
- **Phase 1 耗时**: ~55 分钟（3 epoch × ~18 min/epoch）
- **Phase 2 耗时**: ~3.5 小时（5 epoch × ~40 min/epoch，CF 数据量增大）
- **显存**: ~16-42 GB/卡（Phase 2 因 CF 数据较长占用更多）

## 代码改动

| 文件 | 改动 |
|------|------|
| `training/sft.py` | 新增 `--load-gates`, `--cf-val-jsonl` 参数 + gate 加载 + CF val eval |
| `scripts/qwen3-4b_sft_phase1.sh` | Phase 1: 纯 news, 3 epoch, seq_len=512 |
| `scripts/qwen3-4b_sft_phase2.sh` | Phase 2: news+CF, 5 epoch, load P1 gate |

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
