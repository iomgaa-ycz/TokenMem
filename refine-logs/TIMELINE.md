# TIMELINE: TokenMem 7天冲刺计划 (v3修订)

**开始日期**: 2026-04-27 (Day 1)
**当前日期**: 2026-04-28 (Day 2 晚)
**提交日期**: 2026-05-03 (Day 7)
**可用资源**: 8x RTX 4090 (4090-serve) + 本地 4070 Ti Super + 8x A100 (需手动启动)

**v3变更**: Day 3-4 从"E1全模型评测+消融"改为"C4/C1 faithfulness核心实验优先"

---

## 总览甘特图

```
Day 1 ██████████  代码实现 + News数据 + 首批SFT                    ✅ 完成
Day 2 ██████████  4B/8B SFT + Baseline + E4(4B/8B)评测             ✅ 完成
Day 3 ██████████  反事实数据生成 + ⚑E1(Sensitivity) + E2开始 + 剩余模型SFT
Day 4 ██████████  ⚑E2(Compliance+conflict分层) + E3(公平基线) + E7(效率) + 剩余评测 + ⚑决策
Day 5 ██████████  消融(E5) + Domain-SFT(E6) + 论文写作开始
Day 6 ██████████  论文写作（全天）
Day 7 ██████████  定稿 + 排版 + 提交
```

---

## Day 1 (4/27) — 代码实现 + 数据准备 ✅

| 时段 | 任务 | 状态 | 产出 |
|------|------|------|------|
| 上午 | TokenMemoryBank类实现 | ✅ | `memory_lora/token_bank.py` (56/56 tests) |
| 上午 | 知识融合模块实现 | ✅ | `linear_fusion.py`, `modified_models/`, `tokenmem_model.py` (81/81 tests) |
| 下午 | HuggingFace适配(6模型) | ✅ | 模型映射 + `modeling_ministral.py` + 15测试 |
| 下午 | SFT训练脚本 | ✅ | `training/sft.py` + 6个sh脚本 |
| 下午 | News数据分割 | ✅ | 50K train / 10K test |
| 晚上 | E1 Baseline启动(远程) | ✅ | 36 OOD JSON开始生成 |

---

## Day 2 (4/28) — SFT + Baseline + E4部分评测 ✅

| 时段 | 任务 | 状态 | 产出 |
|------|------|------|------|
| 上午 | E4 Baseline完成 | ✅ | 48 JSON (6模型×4ds×2方法) |
| 并行 | 4B SFT (News 50K) | ✅ | best_val_loss=0.5279 |
| 并行 | 8B SFT (News 50K) | ✅ | best_val_loss=0.4804 |
| 下午 | E4-eval: 4B (4 datasets) | ✅ | +37.3/+14.0/+4.3/+8.4 pp |
| 晚上 | E4-eval: 8B (3 datasets) | ✅ | +32.5/+12.9/+4.1 pp (MMLU未完成) |
| 晚上 | GPT-5.4 三轮审稿 + v3 revision | ✅ | 3→5→6/10, v3 faithful injection |

**Day 2 总结**: 4B/8B基础结果出齐。Recovery 29-74%，低于RAG。经GPT-5.4审稿后确定v3 faithfulness故事。

---

## Day 3 (4/29) — ⚑ 核心实验（E1 + E2开始） + 剩余模型SFT

| 时段 | 任务 | GPU | 时间 | 产出 | 依赖 |
|------|------|-----|------|------|------|
| 上午 | **反事实数据生成**: ARC+MedQA counterfactual passages (E1+E2共用) | API | ~2h | counterfactual_*.jsonl | DeepSeek V4 Flash |
| 上午 | **⚑ E1: Sensitivity (4B)**: Oracle/Topic-Matched-Wrong/Empty on ARC+MedQA | GPU2 | ~2h | E1结果 | 反事实数据(Topic-Matched复用) |
| 并行 | 0.6B SFT | GPU3 | ~30min | adapter weights | - |
| 并行 | 1.7B SFT | GPU6 | ~1h | adapter weights | - |
| **⚑ 中午** | **E1 Go/No-Go 检查** | - | - | Oracle>>Empty? | E1结果 |
| 下午 | **E2: Compliance (4B)**: 正确+反事实 × TM+RAG on MedQA | GPU2 | ~3h | E2核心结果 | E1通过 + counterfactual data |
| 并行 | Ministral-3B SFT | GPU3(0.6B完成后) | ~1.5h | adapter weights | - |
| 并行 | Gemma3-1B SFT | GPU6(1.7B完成后) | ~45min | adapter weights | - |
| 晚上 | E1: Sensitivity (8B) 复制 | GPU6 | ~2h | 8B E1结果 | - |
| 晚上 | 8B MMLU补测 | GPU2(E2完成后) | ~1.5h | 8B MMLU | - |

**Day 3 关键产出**:
- ⚑ E1 结果（决定是否继续E2）
- E2 4B 初步结果
- 4个新模型SFT完成

---

## Day 4 (4/30) — ⚑ E2完成 + E3公平基线 + 剩余评测 + 决策

| 时段 | 任务 | GPU | 时间 | 产出 | 依赖 |
|------|------|-----|------|------|------|
| 上午 | **E2: Compliance (4B)** ARC补完 + conflict分层分析 | GPU2 | ~2h | E2完整4B结果(含High/Low-Prior分层) | counterfactual data |
| 上午 | **E3: Strong-prompt RAG** + **E7: 效率数据** | GPU3/GPU6 | ~3h | 公平基线compliance + 延迟/显存数据 | - |
| 并行 | E4-eval: 0.6B (4ds) | GPU6 | ~1h | 0.6B结果 | Day3 adapter |
| 并行 | E4-eval: 1.7B (4ds) | GPU6(0.6B后) | ~1.5h | 1.7B结果 | Day3 adapter |
| 下午 | **E2: Compliance (8B)** 复制 | GPU2 | ~3h | 8B E2结果 | - |
| 下午 | E4-eval: Ministral-3B (4ds) | GPU3 | ~1.5h | Ministral结果 | Day3 adapter |
| 下午 | E4-eval: Gemma3-1B (4ds) | GPU6 | ~1h | Gemma结果 | Day3 adapter |
| **⚑ 晚上** | **综合决策点** | - | - | NeurIPS / EMNLP / 换方向 | E1+E2+E4全部 |

**Day 4 关键产出**:
- ⚑ E2 + E3 完整结果 → **决定论文能不能投**
- E4 全6模型结果表

---

## Day 5 (5/1) — 消融 + 效率 + 论文写作开始

| 时段 | 任务 | GPU | 产出 |
|------|------|-----|------|
| 上午 | E5: 注入层消融 (4B, 4配置) | GPU2,3 | 消融表 |
| 上午 | E7: 效率数据 (延迟/显存) | GPU6 | 效率表 |
| 下午 | E6: Domain-SFT消融 (MedQA SFT, 4B) | GPU2 | 跨域分析 |
| 下午 | 论文: Introduction + Method 初稿 | 本地 | §1-§3 |
| 晚上 | 论文: Related Work | 本地 | §2 |
| 晚上 | 图表生成 (核心compliance图 + E1表格) | 本地 | figures/ |

---

## Day 6 (5/2) — 论文写作（全天）

| 时段 | 任务 | 产出 |
|------|------|------|
| 上午 | §4 Experiments (E1表 + C4/C1表 + 消融表) | experiments.tex |
| 上午 | §5 Analysis (tradeoff表 + compliance图) | analysis.tex |
| 下午 | Abstract + Conclusion | abstract.tex |
| 下午 | Appendix (实验详情 + 超参 + 额外结果) | appendix.tex |
| 晚上 | 通读全文 + 内部审查 | 修改清单 |

---

## Day 7 (5/3) — 定稿 + 提交

| 时段 | 任务 | 产出 |
|------|------|------|
| 上午 | 根据审查修改 + 补充实验(如需) | 修改版 |
| 下午 | NeurIPS格式排版 + 数字一致性检查 | camera-ready |
| 晚上 | **提交** | ✅ |

---

## GPU并行分配表（v3修订）

```
         Day1(完成)  Day2(完成)      Day3              Day4             Day5
GPU 2  | smoke     | 4B SFT→eval | E1→E2(4B MedQA) | E2(ARC)→E2(8B) | E5(part)
GPU 3  | -         | 8B SFT      | 0.6B SFT→Min SFT | E3→Min eval    | E5(part)
GPU 6  | -         | 4B eval     | 1.7B SFT→Gem SFT | 0.6B→1.7B eval | E7→E6
GPU 2+3| -         | -           | -                | Gem eval       | -

本地    | 编码      | v3 revision | 反事实数据生成    | 结果分析        | 写作开始
```

---

## 里程碑检查表（v3修订）

| 日期 | 里程碑 | 检查标准 | 状态 |
|------|--------|---------|------|
| Day 1 晚 | 代码实现 + Baseline启动 | 137/137 tests ✅ + Baseline JSON ✅ | ✅ |
| Day 2 晚 | 4B/8B SFT+评测 + v3 revision | E4 4B/8B结果 ✅ + v3确定 ✅ | ✅ |
| **Day 3 中午** | **⚑ E1 Go/No-Go** | Oracle >> Empty by ≥10pp | ⏳ |
| **Day 4 上午** | **⚑ E2 核心结果** | TM-Counter KC > RAG-Counter KC by ≥15pp | ⏳ |
| Day 4 晚 | **⚑ 综合决策** | E1✅+E2✅→NeurIPS; E2❌→EMNLP | ⏳ |
| Day 5 晚 | 消融+效率+写作启动 | E5/E7/E6 + §1-§3初稿 | ⏳ |
| Day 6 晚 | 论文初稿完成 | 内部审查通过 | ⏳ |
| Day 7 晚 | **提交** | NeurIPS投稿系统 | ⏳ |

---

## 风险缓冲（v3修订）

| 风险场景 | 触发条件 | 缓冲方案 | 时间代价 |
|---------|---------|---------|---------|
| **E-C4失败** | Day3, Oracle≈Empty | **停止C1。** 重新评估训练方案（可能需要更多SFT数据或不同loss） | 致命 — 可能需要放弃NeurIPS |
| **E-C1效果弱** | Day4, KC差距<10pp | 尝试strong-prompt RAG是否更弱; 如果仍不够，降级EMNLP | +0h (已有数据，只是结论不同) |
| **Strong-prompt抹平差距** | Day4, StrongRAG-Counter KC ≈ TM-Counter KC | 需要trained prompt baseline (+4h); 如仍无差距，faithfulness故事不成立 | +4h |
| 反事实段落生成质量差 | Day3, passages明显不自然 | 换为minimal-edit方式（修改正确段落中的关键词） | +3h |
| 剩余模型SFT不收敛 | Day3-4 | 调lr; 最差情况从6模型砍到4模型 | +2h |
| Gemma3-1B评测仍失效 | Day4 | 从scope移除，改为5模型×2.5家族 | 0 |
| 论文写作时间不够 | Day6 | 砍E-Mech(机制分析)和E2(编辑); 聚焦E1+C4+C1 | 0 |
| 8B OOM on counterfactual eval | Day4 | gradient_checkpointing; 或仅在4B上做完整C1 | +1h |
