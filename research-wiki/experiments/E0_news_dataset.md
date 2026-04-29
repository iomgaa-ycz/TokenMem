---
type: experiment
node_id: exp:E0_news_dataset
title: "E0 News Dataset Construction: 58K MCQ from 25 English Sources"
date: 2026-04-27T13:00:00Z
status: completed
---

# E0 News Dataset Construction

**目的**: 构建 TokenMem 训练/验证数据集 — 基于 2025-2026 英文新闻的多选题 (MCQ)，确保内容在 LLM 训练截止日期之后（OOD）。

## 数据集规模

| 指标 | 数值 |
|------|------|
| 原始文章 | 5,308 篇 |
| 提取段落 | 12,309 条 |
| 原始 QA | 61,401 条 |
| MCQ (含干扰项) | 61,343 条 (失败率 0.09%) |
| **去重后 MCQ** | **58,663 条** |
| 训练集 | 50,000 条 (按时间排序，较早) |
| 验证集 | 8,663 条 (按时间排序，较晚) |

## 数据来源

**25 个英文新闻源**，分两阶段采集：

**Phase 1 — crawl4ai 爬虫 (3,170 篇)**:
BBC, AP News, TechCrunch, NPR, Guardian, Al Jazeera, Ars Technica, CBS News, ScienceDaily, Independent, Wired, France24, CNBC, CNN, ESPN, Nature News, Politico, The Hill, ABC News AU, Bloomberg

**Phase 2 — RSS 聚合补量 (2,138 篇)**:
Google News (多查询), BBC RSS, CNN RSS, TechCrunch RSS, Ars Technica RSS, Al Jazeera RSS, France24 RSS, ESPN RSS, CNBC RSS, Politico RSS, arXiv RSS, New Scientist RSS, Hacker News RSS, The Verge RSS

## 类别分布

| 类别 | 数量 | 占比 |
|------|------|------|
| science | 13,507 | 23.0% |
| sports | 11,704 | 20.0% |
| politics | 9,991 | 17.0% |
| business | 8,447 | 14.4% |
| world | 7,527 | 12.8% |
| technology | 7,487 | 12.8% |

## 构建 Pipeline

```
Step 0: 爬取 (crawl4ai + RSS/feedparser + googlenewsdecoder)
Step 1: 段落提取 (DeepSeek v4-flash, temperature=0.3, thinking=disabled)
Step 2: QA 生成 (DeepSeek v4-flash, qa_per_passage=5, thinking=disabled)
Step 3: 干扰项生成 (DeepSeek v4-flash, temperature=0.7, thinking=disabled)
去重: question 精确 + 近似去重 (前60字符+answer)
划分: 按新闻发布日期排序，前50K训练/后8663验证
```

## 数据质量

- **QA 密度**: 5.0 QA/段落（完美命中目标）
- **段落/文章**: 2.3 段落/文章
- **去重率**: 4.4% (2,680/61,343)
- **Schema 完整**: 所有字段 (question, correct_answer, options, passage, source, category) 非空
- **时间切分点**: 训练集 ≤ 2026-04-26 / 验证集 ≥ 2026-04-26

## 文件位置

```
data/news/
├── raw_articles.jsonl      # 5,308 篇原始文章
├── passages.jsonl           # 12,309 条段落
├── qa_raw.jsonl             # 61,401 条 QA
├── qa_full.jsonl            # 61,343 条 MCQ
├── qa_full_dedup.jsonl      # 58,663 条去重 MCQ
├── train.jsonl              # 50,000 条训练集
└── val.jsonl                # 8,663 条验证集
```

## 关键决策

1. **LLM 选择**: QWEN 额度耗尽后切换到 DeepSeek v4-flash，关闭 thinking 模式节省 token（速度提升 10x）
2. **RSS 补量**: crawl4ai 受反爬限制仅获 3,170 篇，通过 RSS + Google News URL 解码补充 2,138 篇
3. **qa_per_passage=5**: 从默认 3 提升到 5，配合多样性 prompt 约束确保质量
4. **时间划分**: 按新闻日期排序后切分，验证集为更晚的新闻，模拟真实 OOD 场景

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
