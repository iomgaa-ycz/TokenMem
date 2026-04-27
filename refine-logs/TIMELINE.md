# TIMELINE: TokenMem 7天冲刺计划

**开始日期**: 2026-04-27 (Day 1)
**提交日期**: 2026-05-03 (Day 7)
**可用资源**: 8x RTX 4090 (4090-serve) + 本地 4070 Ti Super + 8x A100 (需手动启动)

---

## 总览甘特图

```
Day 1 ██████████  代码实现 + News扩展 + 首批SFT
Day 2 ██████████  Bank构建 + 全部SFT + E1开始
Day 3 ██████████  E1评测 + E5(DecoupledRAG基线) + ⚑决策点
Day 4 ██████████  E2/E3/E4消融 + 论文写作开始
Day 5 ██████████  E6 + 补充实验 + 论文写作
Day 6 ██████████  论文写作（完成初稿）
Day 7 ██████████  修改 + 排版 + 提交
```

---

## 详细任务表

### Day 1 (4/27) — 代码实现 + 数据准备

| 时段 | 任务 | 负责 | GPU | 产出 | 依赖 |
|------|------|------|-----|------|------|
| 上午 | ✅ 实现TokenMemoryBank类(token_ids+emb存储) | 编码 | 本地 | `memory_lora/token_bank.py` (56/56 tests) | 无 |
| 上午 | ✅ 实现知识融合模块(LinearFusion+modified modeling) | 编码 | 本地 | `memory_lora/linear_fusion.py`, `modified_models/`, `tokenmem_model.py` (81/81 tests, smoke通过) | 无 |
| 上午 | News数据集扩展到60K | 脚本 | 本地 | `data/news/qa_full_60k.jsonl` | 无 |
| 下午 | ✅ 适配HuggingFace框架(6模型加载) | 编码 | 本地 | `tokenmem_model.py`映射+`modeling_ministral.py`+15测试通过 | TokenMemoryBank | ⚠️ 待下载权重后补跑完整GPU测试 |
| 下午 | ✅ 实现SFT训练脚本 | 编码 | 本地 | `training/sft.py` + `training/data.py` + 6个`scripts/*_sft.sh`（Lamb+LinearLR, smoke通过） | GateCrossAttention |
| 下午 | News时间分割(50K train/10K test) | 脚本 | 本地 | `data/news/qa_{train,test}.jsonl` | 60K数据 |
| 晚上 | Smoke test: Qwen3-0.6B SFT | 测试 | 4090 GPU2 | 验证训练流程通 | 全部代码 |

**Day 1 产出检查**: SFT训练流程在Qwen3-0.6B上跑通（loss下降）

---

### Day 2 (4/28) — Bank构建 + 全模型SFT

| 时段 | 任务 | 负责 | GPU | 产出 | 依赖 |
|------|------|------|-----|------|------|
| 上午 | 构建TokenMemoryBank(6模型×4数据集, token_ids+emb) | 脚本 | 4090 GPU2 | `data/tokenbank_*.pt` (~550MB/model) | Day1代码 |
| 上午 | 构建FAISS索引(6模型×4数据集) | 脚本 | CPU | `data/faiss_*.index` | TokenMemoryBank |
| 并行 | SFT: Qwen3-0.6B (News 50K) | 训练 | 4090 GPU2 | adapter weights | Bank+FAISS |
| 并行 | SFT: Qwen3-1.7B | 训练 | 4090 GPU3 | adapter weights | Bank+FAISS |
| 并行 | SFT: Qwen3-4B | 训练 | 4090 GPU6 | adapter weights | Bank+FAISS |
| 并行 | SFT: Gemma3-1B | 训练 | 4090 GPU2(完成后) | adapter weights | Bank+FAISS |
| 并行 | SFT: Ministral-3B | 训练 | 4090 GPU3(完成后) | adapter weights | Bank+FAISS |
| 晚上 | SFT: Qwen3-8B (需要更多显存) | 训练 | 4090 GPU2+3 | adapter weights | Bank+FAISS |

**Day 2 产出检查**: 6个模型的adapter weights全部训练完成

---

### Day 3 (4/29) — E1核心实验 + ⚑决策点

| 时段 | 任务 | 负责 | GPU | 产出 | 依赖 |
|------|------|------|-----|------|------|
| 上午 | E1评测: Qwen3-0.6B (4数据集) | 评测 | 4090 GPU2 | accuracy结果 | Day2 adapter |
| 上午 | E1评测: Qwen3-1.7B (4数据集) | 评测 | 4090 GPU3 | accuracy结果 | Day2 adapter |
| 上午 | E1评测: Qwen3-4B (4数据集) | 评测 | 4090 GPU6 | accuracy结果 | Day2 adapter |
| 下午 | E1评测: Gemma3-1B, Ministral-3B | 评测 | 4090 GPU2,3 | accuracy结果 | Day2 adapter |
| 下午 | E1评测: Qwen3-8B (4数据集) | 评测 | 4090 GPU6 | accuracy结果 | Day2 adapter |
| 下午 | No-Memory基线 (6模型×4数据集) | 评测 | 并行各卡 | baseline accuracy | 无 |
| 下午 | VanillaRAG基线 (6模型×4数据集) | 评测 | 并行各卡 | baseline accuracy | FAISS索引 |
| 晚上 | E5: DecoupledRAG基线 (Qwen3-4B) | 训练+评测 | 4090 GPU2 | 对比数据 | DecoupledRAG代码适配 |

**⚑ Day 3 决策点** (晚上):

| E1结果 | 行动 |
|--------|------|
| ≥4/6模型在News+≥1 OOD有效 | ✅ 按计划继续 |
| in-domain好但OOD差 | ⚠️ 分析原因；考虑混入少量多领域SFT数据 |
| 全面不work | ❌ 检查adapter设计；可能需要加入Phase 1预训练 |

**Day 3 产出检查**: E1完整结果表格（6模型×4数据集×3方法）

---

### Day 4 (4/30) — 消融实验 + 论文写作启动

| 时段 | 任务 | 负责 | GPU | 产出 | 依赖 |
|------|------|------|-----|------|------|
| 上午 | E2: 知识编辑实验 (Qwen3-4B) | 实验 | 4090 GPU2 | Edit Success Rate | Day2 adapter |
| 上午 | E3: 注入层消融 (Qwen3-4B, 5配置) | 训练+评测 | 4090 GPU3,6 | 消融表格 | Day1代码 |
| 下午 | E4: adapter设计消融 (rank+基座LoRA) | 训练+评测 | 4090 GPU2,3 | 消融表格 | Day1代码 |
| 下午 | 论文: Introduction初稿 | 写作 | 本地 | intro.tex | E1结果 |
| 晚上 | 论文: Method初稿 | 写作 | 本地 | method.tex | 确定最终架构 |

**Day 4 产出检查**: E2/E3/E4结果 + Introduction/Method初稿

---

### Day 5 (5/1) — 补充实验 + 论文主体

| 时段 | 任务 | 负责 | GPU | 产出 | 依赖 |
|------|------|------|-----|------|------|
| 上午 | E6: 检索效率实验 | 实验 | 4090 GPU2 | FAISS性能数据 | Bank数据 |
| 上午 | 补充实验（如需要） | 实验 | 4090其他卡 | 补充数据 | Day3/4结果分析 |
| 下午 | 论文: Related Work | 写作 | 本地 | related.tex | 无 |
| 下午 | 论文: Experiments | 写作 | 本地 | experiments.tex | E1-E6结果 |
| 晚上 | 生成所有图表和表格 | 制图 | 本地 | figures/ | 全部实验结果 |

**Day 5 产出检查**: 全部实验完成 + Related Work/Experiments初稿 + 图表

---

### Day 6 (5/2) — 论文完成初稿

| 时段 | 任务 | 负责 | GPU | 产出 | 依赖 |
|------|------|------|-----|------|------|
| 上午 | 论文: Analysis & Discussion | 写作 | 本地 | analysis.tex | 全部实验 |
| 上午 | 论文: Abstract + Conclusion | 写作 | 本地 | abstract.tex | 全文 |
| 下午 | 统一全文风格/格式 | 修改 | 本地 | main.tex | 全部section |
| 下午 | Appendix: 补充实验详情 | 写作 | 本地 | appendix.tex | 全部实验 |
| 晚上 | **内部审查**: 通读全文 | 审查 | - | 修改清单 | 初稿完成 |

**Day 6 产出检查**: 论文完整初稿（可提交状态）

---

### Day 7 (5/3) — 最终修改 + 提交

| 时段 | 任务 | 负责 | GPU | 产出 | 依赖 |
|------|------|------|-----|------|------|
| 上午 | 根据内部审查修改 | 修改 | 本地 | 修改版 | Day6审查 |
| 上午 | 最后一轮补充实验（如需） | 实验 | 4090 | 补充数据 | 审查反馈 |
| 下午 | 排版检查(NeurIPS格式) | 排版 | 本地 | camera-ready | 修改完成 |
| 下午 | 检查所有数字/表格一致性 | 校对 | 本地 | 最终版 | 排版完成 |
| 晚上 | **提交** | - | - | ✅ | 一切就绪 |

---

## GPU并行分配表

```
         Day1      Day2           Day3           Day4        Day5
GPU 2  | smoke   | 0.6B SFT→1B  | eval 0.6B    | E2        | E6
GPU 3  | -       | 1.7B SFT→3B  | eval 1.7B    | E3(part)  | 补充
GPU 6  | -       | 4B SFT       | eval 4B→8B   | E3(part)  | 补充
GPU 2+3| -       | 8B SFT(晚)   | E5           | E4        | -
其他卡  | -       | -            | NM+RAG基线   | -         | -

本地    | 编码     | -            | -            | 写作      | 写作+图表
```

---

## 里程碑检查表

| 日期 | 里程碑 | 检查标准 | 状态 |
|------|--------|---------|------|
| Day 1 晚 | 代码实现完成 | Qwen3-0.6B SFT loss下降 | ⏳ |
| Day 2 晚 | 全模型SFT完成 | 6个adapter weights文件 | ⏳ |
| Day 3 晚 | **⚑ E1核心结果** | 6模型×4数据集结果表格 | ⏳ |
| Day 4 晚 | 消融完成 | E2/E3/E4结果 + Intro/Method初稿 | ⏳ |
| Day 5 晚 | 实验全部完成 | 全部图表生成 | ⏳ |
| Day 6 晚 | 论文初稿完成 | 内部审查通过 | ⏳ |
| Day 7 晚 | **提交** | NeurIPS投稿系统 | ⏳ |

---

## 风险缓冲

| 风险场景 | 触发条件 | 缓冲方案 | 时间代价 |
|---------|---------|---------|---------|
| SFT训练不收敛 | Day 2, loss不下降 | 调lr/batch_size; 加warmup | +4h |
| 8B显存不够 | Day 2, OOM | 用gradient_checkpointing+smaller batch; 或移至A100 | +2h |
| OOD泛化差 | Day 3, OOD提升<5% | 混入少量多领域数据重新SFT | +6h (从Day4消融时间借) |
| DecoupledRAG适配困难 | Day 3, 代码不兼容 | 降低E5优先级; 从论文引用数据 | 0 |
| 论文写作时间不够 | Day 6, 初稿未完成 | 砍Analysis section; 聚焦核心实验 | 0 |
