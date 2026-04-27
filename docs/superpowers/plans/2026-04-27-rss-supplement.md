# RSS 聚合补量实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 通过 RSS 聚合源补充 ~4,000 篇新闻文章，配合现有 pipeline 产出 ~36,000 MCQ，使总量达到 ~60K。

**Architecture:** 新增 `crawl_rss_feeds()` 函数，用 feedparser 解析 RSS/Atom feed 获取文章 URL，用 requests + trafilatura 提取正文（无需浏览器），去重后追加到 `raw_articles.jsonl`。复用现有 Step 1-3 pipeline 生成 MCQ。

**Tech Stack:** feedparser (新增), requests, trafilatura (已有), asyncio

**当前状态:**
- 已有: 3,170 篇文章 → 27,571 去重 MCQ
- 需要: ~4,000 篇新文章 → ~36,000 新 MCQ → 总计 ~60K
- LLM: DeepSeek v4-flash（thinking=disabled）
- Conda: ExplicitLLM

---

### Task 1: 安装 feedparser

**Files:** 无代码变更

- [ ] **Step 1: 安装 feedparser**

```bash
conda run -n ExplicitLLM pip install feedparser
```

- [ ] **Step 2: 验证安装**

```bash
conda run -n ExplicitLLM python -c "import feedparser; print('feedparser', feedparser.__version__)"
```

Expected: `feedparser 6.x.x`

- [ ] **Step 3: Commit**

不需要 commit（仅安装依赖）。

---

### Task 2: 新增 RSS 爬取函数

**Files:**
- Modify: `tools/news_crawlers.py` — 在文件末尾 CLI 入口之前，添加 RSS 常量 + `crawl_rss_feeds()` 函数

- [ ] **Step 1: 在 `tools/news_crawlers.py` 的 import 区域添加新依赖**

在 `import trafilatura` 之后添加：

```python
import feedparser  # noqa: E402
import requests as _requests  # noqa: E402
```

- [ ] **Step 2: 在 `crawl_all_sites()` 函数之后、CLI 入口之前，添加 RSS_FEEDS 常量**

```python
# ---------------------------------------------------------------------------
# RSS 聚合补量
# ---------------------------------------------------------------------------

RSS_FEEDS: Dict[str, List[Dict[str, str]]] = {
    "science": [
        {"url": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml", "source": "bbc"},
        {"url": "https://rss.arxiv.org/rss/cs.AI", "source": "arxiv"},
        {"url": "https://www.newscientist.com/section/news/feed/", "source": "newscientist"},
        {"url": "https://phys.org/rss-feed/science-news/", "source": "physorg"},
        {"url": "https://news.google.com/rss/search?q=science+2026&hl=en&gl=US&ceid=US:en", "source": "googlenews"},
    ],
    "technology": [
        {"url": "https://feeds.bbci.co.uk/news/technology/rss.xml", "source": "bbc"},
        {"url": "https://techcrunch.com/feed/", "source": "techcrunch"},
        {"url": "https://feeds.arstechnica.com/arstechnica/index", "source": "arstechnica"},
        {"url": "https://www.theverge.com/rss/index.xml", "source": "theverge"},
        {"url": "https://news.google.com/rss/search?q=technology+2026&hl=en&gl=US&ceid=US:en", "source": "googlenews"},
        {"url": "https://hnrss.org/frontpage?points=50", "source": "hackernews"},
    ],
    "business": [
        {"url": "https://feeds.bbci.co.uk/news/business/rss.xml", "source": "bbc"},
        {"url": "https://rss.cnn.com/rss/edition_business.rss", "source": "cnn"},
        {"url": "https://news.google.com/rss/search?q=business+economy+2026&hl=en&gl=US&ceid=US:en", "source": "googlenews"},
        {"url": "https://www.cnbc.com/id/100003114/device/rss/rss.html", "source": "cnbc"},
    ],
    "politics": [
        {"url": "https://feeds.bbci.co.uk/news/politics/rss.xml", "source": "bbc"},
        {"url": "https://rss.cnn.com/rss/edition_world.rss", "source": "cnn"},
        {"url": "https://news.google.com/rss/search?q=politics+2026&hl=en&gl=US&ceid=US:en", "source": "googlenews"},
        {"url": "https://rss.politico.com/politics-news.xml", "source": "politico"},
    ],
    "sports": [
        {"url": "https://feeds.bbci.co.uk/sport/rss.xml", "source": "bbc"},
        {"url": "https://rss.cnn.com/rss/edition_sport.rss", "source": "cnn"},
        {"url": "https://news.google.com/rss/search?q=sports+2026&hl=en&gl=US&ceid=US:en", "source": "googlenews"},
        {"url": "https://www.espn.com/espn/rss/news", "source": "espn"},
    ],
    "world": [
        {"url": "https://feeds.bbci.co.uk/news/world/rss.xml", "source": "bbc"},
        {"url": "https://rss.cnn.com/rss/edition.rss", "source": "cnn"},
        {"url": "https://news.google.com/rss?topic=WORLD&hl=en&gl=US&ceid=US:en", "source": "googlenews"},
        {"url": "https://www.aljazeera.com/xml/rss/all.xml", "source": "aljazeera"},
        {"url": "https://www.france24.com/en/rss", "source": "france24"},
    ],
}
```

- [ ] **Step 3: 添加 `_fetch_article_text()` 辅助函数**

在 `RSS_FEEDS` 之后添加：

```python
def _fetch_article_text(url: str, timeout: int = 15) -> Optional[str]:
    """用 requests + trafilatura 提取文章正文（无需浏览器）。

    参数：
        url:     文章 URL。
        timeout: HTTP 请求超时秒数。

    返回：
        提取到的正文文本，失败时返回 None。
    """
    try:
        resp = _requests.get(
            url,
            timeout=timeout,
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"},
        )
        if resp.status_code != 200:
            return None
        text = trafilatura.extract(resp.text)
        return text if text and len(text) >= 200 else None
    except Exception:
        return None


def _extract_title_from_html(html: str) -> str:
    """从 HTML 中提取标题。"""
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if m:
        raw = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        return re.split(r"\s*[-|]\s*", raw)[0].strip()
    return ""
```

- [ ] **Step 4: 添加 `crawl_rss_feeds()` 主函数**

```python
async def crawl_rss_feeds(
    output_path: str,
    max_per_feed: int = 100,
    min_date: str = "2025-01-01",
) -> int:
    """通过 RSS/Atom feed 聚合爬取新闻文章，追加到 JSONL。

    参数：
        output_path:  输出 JSONL 文件路径。
        max_per_feed: 每个 feed 最多处理的条目数。
        min_date:     最早发布日期过滤。

    返回：
        新写入的文章数。
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 预加载已有 URL 和标题用于去重
    seen_titles: set[str] = set()
    seen_urls: set[str] = set()
    if output.exists():
        with output.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    art = json.loads(line)
                    title_key = art.get("title", "").lower()[:80]
                    url_key = art.get("url", "").rstrip("/")
                    if title_key:
                        seen_titles.add(title_key)
                    if url_key:
                        seen_urls.add(url_key)
                except json.JSONDecodeError:
                    pass
        logger.info("RSS: 预加载 %d 标题, %d URL", len(seen_titles), len(seen_urls))

    total_written = 0
    for category, feeds in RSS_FEEDS.items():
        for feed_info in feeds:
            feed_url = feed_info["url"]
            source = feed_info["source"]
            logger.info("RSS: 解析 %s / %s → %s", source, category, feed_url)

            try:
                parsed = feedparser.parse(feed_url)
            except Exception as exc:
                logger.warning("RSS 解析失败 %s: %s", feed_url, exc)
                continue

            entries = parsed.entries[:max_per_feed]
            logger.info("RSS: %s / %s 获得 %d 条目", source, category, len(entries))

            for entry in entries:
                link = entry.get("link", "")
                if not link:
                    continue

                # Google News redirect: 跟随重定向获取实际 URL
                url_key = link.rstrip("/")
                title = entry.get("title", "")
                title_key = title.lower()[:80]

                if url_key in seen_urls or title_key in seen_titles:
                    continue

                # 提取正文
                body = _fetch_article_text(link)
                if not body:
                    continue

                # 日期提取
                date = ""
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    import time
                    date = time.strftime("%Y-%m-%d", entry.published_parsed)
                elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                    import time
                    date = time.strftime("%Y-%m-%d", entry.updated_parsed)

                if date and date < min_date:
                    continue

                article = Article(
                    id=hashlib.md5(link.encode()).hexdigest()[:8],
                    title=title,
                    body=body,
                    source=source,
                    date=date,
                    category=category,
                    url=link,
                )

                if not article.is_valid():
                    continue

                seen_titles.add(title_key)
                seen_urls.add(url_key)

                with output.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(article.to_dict(), ensure_ascii=False) + "\n")
                total_written += 1

            logger.info("RSS: %s / %s 写入后累计新增 %d 篇", source, category, total_written)

    logger.info("RSS 爬取完成，共新增 %d 篇文章 → %s", total_written, output_path)
    return total_written
```

- [ ] **Step 5: 验证 import 正常**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
conda run -n ExplicitLLM python -c "
from tools.news_crawlers import crawl_rss_feeds, RSS_FEEDS
total_feeds = sum(len(v) for v in RSS_FEEDS.values())
print(f'RSS_FEEDS: {len(RSS_FEEDS)} categories, {total_feeds} feeds')
print('crawl_rss_feeds imported OK')
"
```

Expected: `RSS_FEEDS: 6 categories, ~27 feeds`

- [ ] **Step 6: Commit**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
git add tools/news_crawlers.py
git commit -m "feat: add RSS feed aggregation (crawl_rss_feeds) for supplemental crawling"
```

---

### Task 3: 集成 RSS 爬取到 build_news_qa.py pipeline

**Files:**
- Modify: `tools/build_news_qa.py` — 在 Step 0 中增加 RSS 爬取作为补充

- [ ] **Step 1: 在 `run_pipeline()` 的 Step 0 块中，crawl_all_sites 之后追加 RSS 爬取**

找到 `run_pipeline()` 函数中的 Step 0 块（包含 `crawl_all_sites` 调用），在 `logger.info("Step 0 完成：写入 %d 篇文章", count)` 之后，Step 1 之前，追加：

```python
            # RSS 补充爬取
            from tools.news_crawlers import crawl_rss_feeds  # noqa: PLC0415

            logger.info("Step 0-RSS: RSS 聚合补充爬取…")
            rss_count = await crawl_rss_feeds(
                output_path=str(raw_path),
                max_per_feed=200,
            )
            logger.info("Step 0-RSS 完成：新增 %d 篇文章", rss_count)
```

- [ ] **Step 2: 验证 pipeline 可 import**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
conda run -n ExplicitLLM python -c "from tools.build_news_qa import run_pipeline; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
git add tools/build_news_qa.py
git commit -m "feat: integrate RSS crawl into Step 0 pipeline"
```

---

### Task 4: Smoke test — RSS 单 feed 测试

**Files:** 无代码变更

- [ ] **Step 1: 测试单个 RSS feed 解析**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
conda run -n ExplicitLLM python -c "
import feedparser
feed = feedparser.parse('https://feeds.bbci.co.uk/news/science_and_environment/rss.xml')
print(f'Feed title: {feed.feed.get(\"title\", \"N/A\")}')
print(f'Entries: {len(feed.entries)}')
for e in feed.entries[:3]:
    print(f'  [{e.get(\"published\", \"\")}] {e.get(\"title\", \"\")[:60]}')
    print(f'    URL: {e.get(\"link\", \"\")}')
"
```

Expected: feed 标题 + 3 条条目（含 title、link、published）。

- [ ] **Step 2: 测试文章正文提取**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
conda run -n ExplicitLLM python -c "
from tools.news_crawlers import _fetch_article_text
import feedparser
feed = feedparser.parse('https://feeds.bbci.co.uk/news/science_and_environment/rss.xml')
if feed.entries:
    url = feed.entries[0].get('link', '')
    print(f'Testing URL: {url}')
    text = _fetch_article_text(url)
    if text:
        print(f'Extracted {len(text)} chars')
        print(f'Preview: {text[:200]}...')
    else:
        print('FAIL: no text extracted')
"
```

Expected: 200+ 字符的正文文本。

- [ ] **Step 3: 测试小规模 RSS 爬取（1 个 feed）**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
conda run -n ExplicitLLM python -c "
import asyncio
from tools.news_crawlers import crawl_rss_feeds, RSS_FEEDS

# 临时只用 1 个 feed 测试
import tools.news_crawlers as nc
original = nc.RSS_FEEDS
nc.RSS_FEEDS = {'science': [{'url': 'https://feeds.bbci.co.uk/news/science_and_environment/rss.xml', 'source': 'bbc'}]}

count = asyncio.run(crawl_rss_feeds('data/news/raw_articles.jsonl', max_per_feed=5))
print(f'RSS test: {count} new articles written')

nc.RSS_FEEDS = original
"
```

Expected: 1-5 篇新文章。

---

### Task 5: 执行完整 RSS 爬取

**Files:** 输出追加到 `data/news/raw_articles.jsonl`

- [ ] **Step 1: 运行完整 RSS 爬取**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
conda run -n ExplicitLLM python -c "
import asyncio, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
from tools.news_crawlers import crawl_rss_feeds
count = asyncio.run(crawl_rss_feeds('data/news/raw_articles.jsonl', max_per_feed=200))
print(f'Total new articles: {count}')
" 2>&1 | tee logs/rss_crawl.log
```

预计耗时 10-30 分钟（纯 HTTP 请求，无需浏览器）。

- [ ] **Step 2: 验证结果**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
conda run -n ExplicitLLM python -c "
import json
from collections import Counter
with open('data/news/raw_articles.jsonl') as f:
    articles = [json.loads(l) for l in f if l.strip()]
sources = Counter(a['source'] for a in articles)
print(f'Total articles: {len(articles)}')
print(f'Sources: {len(sources)}')
for s, c in sources.most_common():
    print(f'  {s}: {c}')
"
```

Expected: 总文章数 > 4,000（现有 3,170 + RSS 新增）。

如果新增不足 1,000 篇，可以多跑几轮（函数有去重，不会重复写入），或调大 `max_per_feed`。

---

### Task 6: 运行增量 Step 1-3 pipeline

**Files:** 输出到 `data/news/passages.jsonl`, `qa_raw.jsonl`, `qa_full.jsonl`, `qa_full_dedup.jsonl`

- [ ] **Step 1: 运行增量 Step 1-3（断点续跑，仅处理新文章）**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
tmux new-session -d -s pipeline 'eval "$(conda shell.bash hook)" && conda activate ExplicitLLM && python -m tools.build_news_qa --steps 1-3 --qa-per-passage 5 2>&1 | tee logs/pipeline_rss_increment.log; echo "=== DONE ===" >> logs/pipeline_rss_increment.log; bash'
```

Pipeline 有断点续跑机制：
- Step 1 会跳过已处理的 article_id，只处理新增文章
- Step 2 会跳过已处理的 passage_id
- Step 3 会跳过已处理的 qa_id
- 最后自动执行去重

- [ ] **Step 2: 监控进度**

```bash
wc -l data/news/qa_full_dedup.jsonl
```

- [ ] **Step 3: 验收最终数量**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
conda run -n ExplicitLLM python -c "
import json
from collections import Counter
with open('data/news/qa_full_dedup.jsonl') as f:
    items = [json.loads(l) for l in f if l.strip()]
print(f'Total deduped MCQ: {len(items)}')
sources = Counter(i['source'] for i in items)
print(f'Sources: {len(sources)}')
categories = Counter(i['category'] for i in items)
print(f'Categories: {dict(categories)}')
"
```

Expected: 总量显著增加，目标接近或超过 60K。
