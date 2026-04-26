# News 数据集扩展到 60K 设计文档

**日期**: 2026-04-26
**目标**: 将 News MCQ 数据集从 ~11K 去重条目扩展到 ≥60K 去重条目
**策略**: 完整迁移 Reference 代码/数据 → 扩展爬虫源 → 提高 QA 密度
**时间约束**: Day 1 (4/27) 完成，Phase 3 可过夜运行

---

## 1. 现状

| 阶段 | 数量 | 文件 |
|------|------|------|
| 原始文章 | 2,112 篇 | `raw_articles.jsonl` (9.4MB) |
| 提取段落 | 3,753 条 | `passages.jsonl` (2.5MB) |
| 原始 QA | 11,232 条 | `qa_raw.jsonl` (9.0MB) |
| MCQ (含干扰项) | 19,146 条 | `qa_full.jsonl` (18.4MB) |
| 去重后 MCQ | 11,338 条 | `qa_full_dedup.jsonl` (10.9MB) |

产出比: 5.37 去重 MCQ / 文章。13 个新闻源。

## 2. 目标产出

| 指标 | 目标值 |
|------|--------|
| 去重后 MCQ | ≥ 60,000 条 |
| 原始文章 | ~8,000 篇 |
| 新闻源 | ≥ 20 个（英文） |
| 类别覆盖 | ≥ 5 类 |
| `qa_per_passage` | 5（从当前 3 提升） |

数学验证: 8,000 × 1.8 段/篇 × 5 QA/段 × 0.75 去重率 = 54,000 + 现有 11,338 = ~65,000 ✓

## 3. 文件迁移清单

### 3.1 代码迁移

| 源 (Reference/Memory-LoRA-old/) | 目标 (Memory-LoRA/) | 改动 |
|------|------|------|
| `tools/build_news_qa.py` | `tools/build_news_qa.py` | 调整 import、日志对接 |
| `tools/news_crawlers.py` | `tools/news_crawlers.py` | 调整 import、扩展新源 |
| `tools/__init__.py` | `tools/__init__.py` | 新建 |

不迁移: `build_fusion_bank.py`、`build_oracle_map.py`、`build_knowledge_jsonl.py`（后续阶段按需迁移）

### 3.2 数据迁移

| 文件 | 大小 | 说明 |
|------|------|------|
| `data/news/raw_articles.jsonl` | 9.4MB | 2,112 篇原始文章 |
| `data/news/passages.jsonl` | 2.5MB | 3,753 条段落 |
| `data/news/qa_raw.jsonl` | 9.0MB | 11,232 条 QA |
| `data/news/qa_full.jsonl` | 18.4MB | 19,146 条 MCQ |
| `data/news/qa_full_dedup.jsonl` | 10.9MB | 11,338 条去重 MCQ |

不迁移: `.bak` 文件、`qa_full_final.jsonl`

## 4. 爬虫扩展方案

### 4.1 现有 13 源

BBC, AP News, TechCrunch, NPR, Guardian, Al Jazeera, Ars Technica, CBS News, ScienceDaily, Independent, Wired, France24, CNBC

### 4.2 新增第一梯队（结构简单，大概率跑通）

| 源 | 类别 | 预估产量 |
|------|------|------|
| Reuters | 综合 | 500+ |
| The Verge | 科技 | 400+ |
| Politico | 政治 | 400+ |
| ESPN | 体育 | 500+ |
| Nature News | 科学 | 300+ |
| Phys.org | 科学 | 400+ |

### 4.3 新增第二梯队（可能需调试反爬）

| 源 | 类别 | 预估产量 |
|------|------|------|
| CNN | 综合 | 500+ |
| Bloomberg | 商业 | 300+ |
| The Hill | 政治 | 400+ |
| Sky News | 综合 | 400+ |
| New Scientist | 科学 | 300+ |
| ABC News (AU) | 综合 | 400+ |

### 4.4 实现策略

- 沿用现有 `crawl_xxx()` 异步函数模式，统一 `Article` dataclass
- 优先 RSS/sitemap 获取 URL 列表，fallback 到页面解析
- 单源失败不阻塞整体，`try/except` + 日志跳过
- 标题前 80 字符 + URL 双重去重
- 每个源写完立即测试，能跑通就保留

### 4.5 类别多样性目标

- Science: ~30%
- Politics: ~20%
- Technology: ~15%
- Business: ~10%
- Sports: ~10%
- World/综合: ~15%

## 5. Pipeline 参数调整

### 5.1 QA 密度

`qa_per_passage`: 3 → **5**

### 5.2 质量保障

1. **Prompt 端**: Step 2 prompt 强调"5个QA必须覆盖段落不同方面，禁止仅改措辞的重复题"
2. **去重端**: 增加 question 文本相似度去重（余弦相似度 > 0.85 过滤）

## 6. 执行流程

```
Phase 1: 迁移代码+数据                    (~30min, Claude 执行)
Phase 2: 扩展爬虫 + Step 0 爬取            (~2-3h, Claude 执行)
Phase 3: Step 1-3 增量处理                 (~13h, 后台/过夜运行)
Phase 4: 去重 + 质量检查 + Step 4 OOD      (~1h)
Phase 5: Step 5 v2 schema 转换             (~1h)
```

### 6.1 LLM 调用估算

| Step | 调用次数 | 模型 | 预估耗时 |
|------|----------|------|----------|
| Step 1 (段落提取) | ~6,000 | qwen3.6-plus | ~2h |
| Step 2 (QA 生成) | ~11,000 | qwen3.6-plus | ~3h |
| Step 3 (干扰项) | ~55,000 | qwen3.6-plus | ~8h |
| 合计 | ~72,000 | — | ~13h |

并发: 8 semaphore，支持断点续跑。

## 7. 验收标准

| 指标 | 阈值 | 验证方式 |
|------|------|----------|
| 去重后 MCQ 总量 | ≥ 60,000 | `wc -l qa_full_dedup.jsonl` |
| 新闻源数量 | ≥ 20 | `jq '.source' raw_articles.jsonl \| sort -u \| wc -l` |
| 类别覆盖 | ≥ 5 类 | `jq '.category' \| sort -u` |
| OOD baseline 准确率 | ≤ 40% | Step 4 输出 |
| 每源文章数 | ≥ 50 篇 | 避免单源过少 |
| QA schema 完整 | 所有字段非空 | jsonl 校验 |

## 8. 风险与应对

| 风险 | 应对 |
|------|------|
| 新爬虫被反爬拦截 | 跳过该源，用其他源补量 |
| qa_per_passage=5 质量下降 | 抽样 100 条检查，必要时回退到 4 |
| LLM API 限流 | 降低并发到 4，延长运行时间 |
| 去重率高于预期 | 增加爬取量到 10,000 篇 |
| 总量仍不足 60K | 追加第三梯队源或提高段落提取数 |

## 9. 交付目录结构

```
Memory-LoRA/
├── tools/
│   ├── __init__.py
│   ├── build_news_qa.py      # 迁移+适配
│   └── news_crawlers.py      # 扩展到 25+ 源
├── data/news/
│   ├── raw_articles.jsonl    # ~8,000+ 篇
│   ├── passages.jsonl        # ~14,000+ 条
│   ├── qa_raw.jsonl          # ~70,000+ 条
│   ├── qa_full.jsonl         # ~70,000+ 条
│   └── qa_full_dedup.jsonl   # ≥60,000 条 ← 核心交付
```
