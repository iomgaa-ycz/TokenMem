# News 60K 数据集扩展实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 News MCQ 数据集从 ~11K 扩展到 ≥60K 去重条目，通过迁移现有代码/数据 + 扩展爬虫源 + 提高 QA 密度实现。

**Architecture:** 从 Reference/Memory-LoRA-old/ 迁移 `build_news_qa.py` 和 `news_crawlers.py` 到主项目 `tools/`，适配 .env 中的 QWEN LLM 配置；扩展爬虫到 25+ 英文新闻源；将 `qa_per_passage` 从 3 提升到 5；新增 question 级别去重。

**Tech Stack:** Python 3.11, crawl4ai, trafilatura, openai (async), tqdm, dotenv

**关键环境约束:**
- Conda 环境: `ExplicitLLM`
- LLM: `QWEN_LLM_MODEL=qwen3.6-plus` (主项目 .env)
- Reference 代码使用 `DEEPSEEK_LLM_*` / `LLM_*` 变量名，需适配

---

### Task 1: 创建目录结构 + 迁移数据文件

**Files:**
- Create: `tools/__init__.py`
- Create: `data/news/` (directory)
- Copy: `data/news/raw_articles.jsonl` (from Reference)
- Copy: `data/news/passages.jsonl` (from Reference)
- Copy: `data/news/qa_raw.jsonl` (from Reference)
- Copy: `data/news/qa_full.jsonl` (from Reference)
- Copy: `data/news/qa_full_dedup.jsonl` (from Reference)

- [ ] **Step 1: 创建目录**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
mkdir -p tools data/news data/v2
```

- [ ] **Step 2: 创建 tools/__init__.py**

```python
# tools/__init__.py — 空包标记
```

- [ ] **Step 3: 复制数据文件**

```bash
REF=/home/iomgaa/Projects/Memory-LoRA/Reference/Memory-LoRA-old/data/news
DEST=/home/iomgaa/Projects/Memory-LoRA/data/news

cp "$REF/raw_articles.jsonl" "$DEST/"
cp "$REF/passages.jsonl" "$DEST/"
cp "$REF/qa_raw.jsonl" "$DEST/"
cp "$REF/qa_full.jsonl" "$DEST/"
cp "$REF/qa_full_dedup.jsonl" "$DEST/"
```

- [ ] **Step 4: 验证数据完整性**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
wc -l data/news/*.jsonl
```

Expected:
```
  2112 data/news/raw_articles.jsonl
  3753 data/news/passages.jsonl
 11232 data/news/qa_raw.jsonl
 19146 data/news/qa_full.jsonl
 11338 data/news/qa_full_dedup.jsonl
```

- [ ] **Step 5: Commit**

```bash
git add tools/__init__.py
git commit -m "chore: init tools package + migrate news data from Reference"
```

注意: data/ 下的 jsonl 文件较大，根据 .gitignore 策略决定是否加入 git。如果没有 .gitignore，先创建一个排除 `data/news/*.jsonl` 的规则，数据不入库。

---

### Task 2: 迁移并适配 news_crawlers.py

**Files:**
- Create: `tools/news_crawlers.py` (从 Reference 复制后修改)

**改动点:** 仅需调整 `_PROJECT_ROOT` 路径解析逻辑，其余代码无外部依赖差异。

- [ ] **Step 1: 复制源文件**

```bash
cp /home/iomgaa/Projects/Memory-LoRA/Reference/Memory-LoRA-old/tools/news_crawlers.py \
   /home/iomgaa/Projects/Memory-LoRA/tools/news_crawlers.py
```

- [ ] **Step 2: 验证 import 无报错**

```bash
conda run -n ExplicitLLM python -c "from tools.news_crawlers import crawl_all_sites, SITE_CATEGORY_PLAN; print('OK, crawlers:', len(SITE_CATEGORY_PLAN), 'categories')"
```

Expected: `OK, crawlers: 6 categories`

如果报 `ModuleNotFoundError: No module named 'crawl4ai'`，需要先安装：
```bash
conda run -n ExplicitLLM pip install crawl4ai trafilatura
```

- [ ] **Step 3: Commit**

```bash
git add tools/news_crawlers.py
git commit -m "feat: migrate news_crawlers.py with 13 crawlers from Reference"
```

---

### Task 3: 迁移并适配 build_news_qa.py（LLM 配置）

**Files:**
- Create: `tools/build_news_qa.py` (从 Reference 复制后修改 LLM 配置)

**核心改动:** Reference 代码的 LLM 变量优先级是 `DEEPSEEK_LLM_* > LLM_*`，主项目 .env 使用 `QWEN_LLM_*`。需要在 fallback chain 中加入 QWEN 支持。

- [ ] **Step 1: 复制源文件**

```bash
cp /home/iomgaa/Projects/Memory-LoRA/Reference/Memory-LoRA-old/tools/build_news_qa.py \
   /home/iomgaa/Projects/Memory-LoRA/tools/build_news_qa.py
```

- [ ] **Step 2: 修改 LLM 配置 fallback chain**

在 `tools/build_news_qa.py` 第 54-56 行，将 LLM 变量优先级改为 `QWEN_LLM_* > DEEPSEEK_LLM_* > LLM_*`：

原始代码（第 54-56 行）：
```python
LLM_MODEL: str = os.getenv("DEEPSEEK_LLM_MODEL", os.getenv("LLM_MODEL", "deepseek-v4-flash"))
LLM_BASE_URL: str = os.getenv("DEEPSEEK_LLM_BASE_URL", os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"))
LLM_API_KEY: str = os.getenv("DEEPSEEK_LLM_API_KEY", os.getenv("LLM_API_KEY", ""))
```

替换为：
```python
LLM_MODEL: str = os.getenv(
    "QWEN_LLM_MODEL",
    os.getenv("DEEPSEEK_LLM_MODEL", os.getenv("LLM_MODEL", "qwen3.6-plus")),
)
LLM_BASE_URL: str = os.getenv(
    "QWEN_LLM_BASE_URL",
    os.getenv("DEEPSEEK_LLM_BASE_URL", os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")),
)
LLM_API_KEY: str = os.getenv(
    "QWEN_LLM_API_KEY",
    os.getenv("DEEPSEEK_LLM_API_KEY", os.getenv("LLM_API_KEY", "")),
)
```

- [ ] **Step 3: 验证 LLM 配置加载正确**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
conda run -n ExplicitLLM python -c "
from tools.build_news_qa import LLM_MODEL, LLM_BASE_URL, LLM_API_KEY
print('Model:', LLM_MODEL)
print('Base URL:', LLM_BASE_URL[:50] + '...')
print('API Key set:', bool(LLM_API_KEY))
"
```

Expected:
```
Model: qwen3.6-plus
Base URL: https://token-plan.cn-beijing.maas.aliyuncs.com...
API Key set: True
```

- [ ] **Step 4: 验证 pipeline 函数可 import**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
conda run -n ExplicitLLM python -c "
from tools.build_news_qa import run_pipeline, run_step1, run_step2, run_step3
print('All pipeline functions imported OK')
"
```

Expected: `All pipeline functions imported OK`

- [ ] **Step 5: Commit**

```bash
git add tools/build_news_qa.py
git commit -m "feat: migrate build_news_qa.py, adapt LLM config for QWEN"
```

---

### Task 4: 新增第一梯队爬虫（6 个源）

**Files:**
- Modify: `tools/news_crawlers.py` (追加 6 个 crawler 类 + 注册)

新增: ReutersCrawler, TheVergeCrawler, PoliticoCrawler, ESPNCrawler, NatureNewsCrawler, PhysOrgCrawler

- [ ] **Step 1: 在 `_CRAWLER_REGISTRY` 之前（约第 993 行），追加 6 个爬虫类**

在 `CNBCCrawler` 类之后、`_CRAWLER_REGISTRY` 之前插入：

```python
class ReutersCrawler(BaseCrawler):
    """Reuters 爬虫（reuters.com）。

    综合国际新闻，文章质量高。
    文章 URL 格式：/<section>/YYYY-MM-DD/<slug>-<id>/
    支持类别：world, business, technology, science。
    """

    base_url = "https://www.reuters.com"
    source_name = "reuters"
    category_urls: Dict[str, str] = {
        "world": "https://www.reuters.com/world/",
        "business": "https://www.reuters.com/business/",
        "technology": "https://www.reuters.com/technology/",
        "science": "https://www.reuters.com/science/",
        "politics": "https://www.reuters.com/world/us/",
    }

    def _is_article_url(self, path: str) -> bool:
        """Reuters 文章路径含日期段 YYYY-MM-DD。"""
        if re.match(r"^/(?:graphics|video|pictures|podcasts|investigates)/", path):
            return False
        return bool(re.search(r"/20\d{2}-\d{2}-\d{2}/", path))


class TheVergeCrawler(BaseCrawler):
    """The Verge 爬虫（theverge.com）。

    科技与消费电子深度报道，文章结构清晰。
    文章 URL 格式：/YYYY/M/DD/<id>/<slug>
    支持类别：technology, science。
    """

    base_url = "https://www.theverge.com"
    source_name = "theverge"
    category_urls: Dict[str, str] = {
        "technology": "https://www.theverge.com/tech",
        "science": "https://www.theverge.com/science",
        "ai": "https://www.theverge.com/ai-artificial-intelligence",
    }

    def _is_article_url(self, path: str) -> bool:
        """The Verge 文章路径形如 /YYYY/M/DD/<id>/<slug>。"""
        if re.match(r"^/(?:archives|authors|about|contact|pages)/", path):
            return False
        return bool(re.match(r"^/\d{4}/\d{1,2}/\d{1,2}/\d+/", path))


class PoliticoCrawler(BaseCrawler):
    """Politico 爬虫（politico.com）。

    美国政治新闻，公开 RSS，文章结构规范。
    文章 URL 格式：/news/<YYYY>/<MM>/<DD>/<slug>-<id>
    支持类别：politics。
    """

    base_url = "https://www.politico.com"
    source_name = "politico"
    category_urls: Dict[str, str] = {
        "politics": "https://www.politico.com/politics",
        "world": "https://www.politico.com/foreign-policy",
        "business": "https://www.politico.com/economy",
    }

    def _is_article_url(self, path: str) -> bool:
        """Politico 文章路径含 /news/ 或日期段。"""
        if re.match(r"^/(?:newsletters|staff|about|podcasts|video|live)/", path):
            return False
        if re.match(r"^/news/\d{4}/\d{2}/\d{2}/", path):
            return True
        segments = [s for s in path.split("/") if s]
        return len(segments) >= 3 and any(re.fullmatch(r"20\d{2}", s) for s in segments)


class ESPNCrawler(BaseCrawler):
    """ESPN 爬虫（espn.com）。

    体育新闻，量大结构简单。
    文章 URL 格式：/<sport>/story/_/id/<数字>/<slug>
    支持类别：sports。
    """

    base_url = "https://www.espn.com"
    source_name = "espn"
    category_urls: Dict[str, str] = {
        "sports": "https://www.espn.com/espn/latestnews",
    }

    def _is_article_url(self, path: str) -> bool:
        """ESPN 文章路径含 /story/_/id/<数字>。"""
        return bool(re.search(r"/story/_/id/\d+", path))


class NatureNewsCrawler(BaseCrawler):
    """Nature News 爬虫（nature.com/news）。

    顶级科学新闻，文章质量极高。
    文章 URL 格式：/articles/<doi-slug>
    支持类别：science。
    """

    base_url = "https://www.nature.com"
    source_name = "nature"
    category_urls: Dict[str, str] = {
        "science": "https://www.nature.com/news",
    }

    def _is_article_url(self, path: str) -> bool:
        """Nature 文章路径形如 /articles/xxxxx。"""
        if re.match(r"^/(?:subjects|authors|collections|about)/", path):
            return False
        return bool(re.match(r"^/articles/[a-z0-9-]+$", path))


class PhysOrgCrawler(BaseCrawler):
    """Phys.org 爬虫（phys.org）。

    科学新闻聚合，开放访问，trafilatura 提取效果好。
    文章 URL 格式：/news/YYYY-MM-<slug>.html
    支持类别：science, technology。
    """

    base_url = "https://phys.org"
    source_name = "physorg"
    category_urls: Dict[str, str] = {
        "science": "https://phys.org/science-news/",
        "technology": "https://phys.org/technology-news/",
    }

    def _is_article_url(self, path: str) -> bool:
        """Phys.org 文章路径形如 /news/YYYY-MM-<slug>.html。"""
        return bool(re.match(r"^/news/20\d{2}-\d{2}-.+\.html$", path))
```

- [ ] **Step 2: 在 `_CRAWLER_REGISTRY`（约第 999 行）中注册新爬虫**

在 `"cnbc": CNBCCrawler(),` 之后追加：

```python
    "reuters": ReutersCrawler(),
    "theverge": TheVergeCrawler(),
    "politico": PoliticoCrawler(),
    "espn": ESPNCrawler(),
    "nature": NatureNewsCrawler(),
    "physorg": PhysOrgCrawler(),
```

- [ ] **Step 3: 验证新爬虫可实例化**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
conda run -n ExplicitLLM python -c "
from tools.news_crawlers import _CRAWLER_REGISTRY
print('Total crawlers:', len(_CRAWLER_REGISTRY))
for name in sorted(_CRAWLER_REGISTRY):
    c = _CRAWLER_REGISTRY[name]
    cats = list(c.category_urls.keys())
    print(f'  {name}: {cats}')
"
```

Expected: `Total crawlers: 19`（13 原有 + 6 新增）

- [ ] **Step 4: Commit**

```bash
git add tools/news_crawlers.py
git commit -m "feat: add 6 tier-1 crawlers (Reuters, Verge, Politico, ESPN, Nature, PhysOrg)"
```

---

### Task 5: 新增第二梯队爬虫（6 个源）

**Files:**
- Modify: `tools/news_crawlers.py` (追加 6 个 crawler 类 + 注册)

新增: CNNCrawler, BloombergCrawler, TheHillCrawler, SkyNewsCrawler, NewScientistCrawler, ABCNewsAUCrawler

- [ ] **Step 1: 在 `_CRAWLER_REGISTRY` 之前追加 6 个爬虫类**

在 Task 4 新增的 PhysOrgCrawler 之后插入：

```python
class CNNCrawler(BaseCrawler):
    """CNN 爬虫（cnn.com）。

    美国综合新闻，文章量大。部分页面需 JS 渲染，优先尝试 trafilatura。
    文章 URL 格式：/YYYY/MM/DD/<section>/<slug>/index.html
    支持类别：world, politics, business, technology。
    """

    base_url = "https://www.cnn.com"
    source_name = "cnn"
    category_urls: Dict[str, str] = {
        "world": "https://www.cnn.com/world",
        "politics": "https://www.cnn.com/politics",
        "business": "https://www.cnn.com/business",
        "technology": "https://www.cnn.com/business/tech",
        "science": "https://www.cnn.com/science",
    }

    def _is_article_url(self, path: str) -> bool:
        """CNN 文章路径含日期 /YYYY/MM/DD/。"""
        if re.match(r"^/(?:videos|audio|live-news|gallery)/", path):
            return False
        return bool(re.match(r"^/\d{4}/\d{2}/\d{2}/", path))


class BloombergCrawler(BaseCrawler):
    """Bloomberg 爬虫（bloomberg.com）。

    商业财经新闻，部分有 paywall，提取公开摘要部分。
    文章 URL 格式：/news/articles/YYYY-MM-DD/<slug>
    支持类别：business, technology。
    """

    base_url = "https://www.bloomberg.com"
    source_name = "bloomberg"
    category_urls: Dict[str, str] = {
        "business": "https://www.bloomberg.com/markets",
        "technology": "https://www.bloomberg.com/technology",
        "politics": "https://www.bloomberg.com/politics",
    }

    def _is_article_url(self, path: str) -> bool:
        """Bloomberg 文章路径含 /news/articles/ 或 /news/features/。"""
        return bool(re.match(r"^/news/(?:articles|features)/20\d{2}-\d{2}-\d{2}/", path))


class TheHillCrawler(BaseCrawler):
    """The Hill 爬虫（thehill.com）。

    美国政治与政策新闻，结构清晰，无 paywall。
    文章 URL 格式：/<section>/<数字>-<slug>/
    支持类别：politics, business。
    """

    base_url = "https://thehill.com"
    source_name = "thehill"
    category_urls: Dict[str, str] = {
        "politics": "https://thehill.com/news/administration/",
        "business": "https://thehill.com/business/",
        "technology": "https://thehill.com/policy/technology/",
    }

    def _is_article_url(self, path: str) -> bool:
        """The Hill 文章路径含长数字 ID。"""
        if re.match(r"^/(?:people|newsletters|events|about)/", path):
            return False
        return bool(re.search(r"/\d{7,}-", path))


class SkyNewsCrawler(BaseCrawler):
    """Sky News 爬虫（news.sky.com）。

    英国综合新闻，HTML 结构干净，无 paywall。
    文章 URL 格式：/story/<slug>-<数字>
    支持类别：world, politics, business, technology, science。
    """

    base_url = "https://news.sky.com"
    source_name = "skynews"
    category_urls: Dict[str, str] = {
        "world": "https://news.sky.com/world",
        "politics": "https://news.sky.com/politics",
        "business": "https://news.sky.com/business",
        "technology": "https://news.sky.com/technology",
        "science": "https://news.sky.com/science",
    }

    def _is_article_url(self, path: str) -> bool:
        """Sky News 文章路径形如 /story/<slug>-<数字>。"""
        return bool(re.match(r"^/story/.+-\d{7,}$", path))


class NewScientistCrawler(BaseCrawler):
    """New Scientist 爬虫（newscientist.com）。

    科学新闻与深度报道，部分有 paywall，提取公开摘要。
    文章 URL 格式：/article/<id>/<slug>/
    支持类别：science, technology。
    """

    base_url = "https://www.newscientist.com"
    source_name = "newscientist"
    category_urls: Dict[str, str] = {
        "science": "https://www.newscientist.com/subject/physics/",
        "technology": "https://www.newscientist.com/subject/technology/",
    }

    def _is_article_url(self, path: str) -> bool:
        """New Scientist 文章路径形如 /article/<id>/<slug>/。"""
        return bool(re.match(r"^/article/\d+/", path))


class ABCNewsAUCrawler(BaseCrawler):
    """ABC News Australia 爬虫（abc.net.au）。

    澳洲公共媒体，完全开放访问，trafilatura 效果好。
    文章 URL 格式：/news/YYYY-MM-DD/<slug>/<id>
    支持类别：world, science, business, politics。
    """

    base_url = "https://www.abc.net.au"
    source_name = "abcau"
    category_urls: Dict[str, str] = {
        "world": "https://www.abc.net.au/news/world",
        "science": "https://www.abc.net.au/news/science",
        "business": "https://www.abc.net.au/news/business",
        "politics": "https://www.abc.net.au/news/politics",
    }

    def _is_article_url(self, path: str) -> bool:
        """ABC AU 文章路径含日期 /YYYY-MM-DD/ 或 /news/<slug>/<数字>。"""
        if re.match(r"^/news/\d{4}-\d{2}-\d{2}/", path):
            return True
        return bool(re.match(r"^/news/[^/]+/\d{7,}$", path))
```

- [ ] **Step 2: 在 `_CRAWLER_REGISTRY` 中注册新爬虫**

在 Task 4 注册的 `"physorg"` 之后追加：

```python
    "cnn": CNNCrawler(),
    "bloomberg": BloombergCrawler(),
    "thehill": TheHillCrawler(),
    "skynews": SkyNewsCrawler(),
    "newscientist": NewScientistCrawler(),
    "abcau": ABCNewsAUCrawler(),
```

- [ ] **Step 3: 验证所有爬虫可实例化**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
conda run -n ExplicitLLM python -c "
from tools.news_crawlers import _CRAWLER_REGISTRY
print('Total crawlers:', len(_CRAWLER_REGISTRY))
new_ones = ['reuters','theverge','politico','espn','nature','physorg','cnn','bloomberg','thehill','skynews','newscientist','abcau']
for name in new_ones:
    c = _CRAWLER_REGISTRY[name]
    print(f'  {name} ({c.source_name}): {list(c.category_urls.keys())}')
"
```

Expected: `Total crawlers: 25`（13 + 12 新增）

- [ ] **Step 4: Commit**

```bash
git add tools/news_crawlers.py
git commit -m "feat: add 6 tier-2 crawlers (CNN, Bloomberg, TheHill, SkyNews, NewScientist, ABC AU)"
```

---

### Task 6: 更新 SITE_CATEGORY_PLAN 支持 60K 产能

**Files:**
- Modify: `tools/news_crawlers.py` — 替换 `SITE_CATEGORY_PLAN` 字典

**目标:** 总原始请求容量 ~24,000，按 60-70% 成功率 → ~14,400-16,800 篇 → 去重后 ~8,000-10,000 篇。

- [ ] **Step 1: 替换 SITE_CATEGORY_PLAN**

将 `tools/news_crawlers.py` 中的 `SITE_CATEGORY_PLAN` 字典（约第 120-180 行）整体替换为：

```python
SITE_CATEGORY_PLAN: Dict[str, Dict[str, int]] = {
    # 总原始请求容量约 24000，考虑 ~60-70% 成功率后预期产出约 14400-16800 篇。
    # 叠加去重损耗后目标实际入库 ~8000-10000 篇。
    "science": {
        "bbc": 300, "apnews": 200, "npr": 200, "guardian": 400,
        "arstechnica": 350, "cbsnews": 200, "sciencedaily": 500,
        "independent": 300, "wired": 250, "france24": 200,
        # 新增
        "nature": 400, "physorg": 500, "newscientist": 300,
        "skynews": 150, "cnn": 200, "abcau": 200,
    },
    "technology": {
        "bbc": 250, "techcrunch": 300, "npr": 150, "guardian": 300,
        "arstechnica": 350, "cbsnews": 150, "wired": 300, "cnbc": 250,
        "independent": 200,
        # 新增
        "theverge": 400, "cnn": 200, "bloomberg": 200,
        "physorg": 200, "thehill": 150, "abcau": 150,
    },
    "business": {
        "bbc": 250, "apnews": 250, "npr": 150, "guardian": 250,
        "aljazeera": 200, "cbsnews": 150, "cnbc": 350, "independent": 200,
        "france24": 200,
        # 新增
        "reuters": 400, "bloomberg": 300, "thehill": 200,
        "cnn": 200, "skynews": 150, "abcau": 150,
    },
    "politics": {
        "bbc": 250, "apnews": 250, "npr": 150, "guardian": 250,
        "aljazeera": 250, "cbsnews": 150, "independent": 200,
        "france24": 200,
        # 新增
        "reuters": 300, "politico": 500, "thehill": 400,
        "cnn": 300, "skynews": 200, "abcau": 200,
    },
    "sports": {
        "bbc": 300, "apnews": 200, "guardian": 300, "aljazeera": 200,
        # 新增
        "espn": 600, "cnn": 150, "skynews": 200, "abcau": 150,
    },
    "world": {
        "aljazeera": 300, "guardian": 300, "apnews": 200,
        "france24": 250, "independent": 200,
        # 新增
        "reuters": 400, "cnn": 300, "skynews": 300,
        "abcau": 250, "bbc": 200,
    },
}
```

- [ ] **Step 2: 验证总容量**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
conda run -n ExplicitLLM python -c "
from tools.news_crawlers import SITE_CATEGORY_PLAN
total = sum(v for cat in SITE_CATEGORY_PLAN.values() for v in cat.values())
sites = set()
for cat in SITE_CATEGORY_PLAN.values():
    sites.update(cat.keys())
print(f'Total raw capacity: {total}')
print(f'Unique sites: {len(sites)}')
print(f'Categories: {list(SITE_CATEGORY_PLAN.keys())}')
print(f'Expected yield (65% success, 80% dedup): ~{int(total * 0.65 * 0.80)} articles')
"
```

Expected:
- Total raw capacity: ~24,000
- Unique sites: 25
- Expected yield: ~12,400+ articles（远超 8,000 目标，留有充足 buffer）

- [ ] **Step 3: Commit**

```bash
git add tools/news_crawlers.py
git commit -m "feat: expand SITE_CATEGORY_PLAN to ~24K capacity for 60K MCQ target"
```

---

### Task 7: build_news_qa.py — 提高 QA 密度 + 新增去重步骤

**Files:**
- Modify: `tools/build_news_qa.py`

**改动点:**
1. `qa_per_passage` CLI 默认值从 3 → 5
2. Step 2 多 QA prompt 增加多样性约束
3. 新增 `run_dedup()` 函数（question 文本级去重）
4. 在 `run_pipeline()` 中 Step 3 之后调用去重

- [ ] **Step 1: 修改 CLI 默认 qa_per_passage 为 5**

在 `tools/build_news_qa.py` 的 `_build_arg_parser()` 函数中（约第 1009 行），修改：

```python
    parser.add_argument(
        "--qa-per-passage",
        type=int,
        default=5,
        help="Step 2 每个段落生成的 QA 对数量（默认: 5）",
    )
```

- [ ] **Step 2: 增强 Step 2 多 QA prompt 的多样性约束**

在 `tools/build_news_qa.py` 中，将 `_GENERATE_MULTI_QA_PROMPT_TEMPLATE`（约第 235-251 行）替换为：

```python
_GENERATE_MULTI_QA_PROMPT_TEMPLATE = """\
You are a question-answer generation expert. Given a factual passage from a \
news article, generate {n} different factual questions. Each question MUST:
- Focus on a DIFFERENT fact, entity, or aspect of the passage — NO two \
questions may target the same piece of information
- Be answerable ONLY by reading the passage, not from general world knowledge
- Have a specific, concise answer (a name, number, date, or event)

Mandatory variety (pick {n} from these question types):
- A "who" question about a key person or organization
- A "what" question about a specific event or outcome
- A "when" or "how many" question about a date, quantity, or statistic
- A "why" or "how" question about a cause, effect, or mechanism
- A "where" question about a location or context

If two questions would have the same answer, DISCARD one and write a new one \
about a different fact.

Return your response as JSON with this exact format:
{{"qa_pairs": [{{"question": "...", "answer": "..."}}, ...]}}

Passage:
"""
```

- [ ] **Step 3: 新增 run_dedup() 函数**

在 `run_step3()` 函数之后（约第 706 行）、Step 4 之前，插入：

```python
def run_dedup(input_path: Path, output_path: Path) -> int:
    """对 qa_full.jsonl 进行 question 文本级去重，输出 qa_full_dedup.jsonl。

    去重策略：
    1. 精确去重：question 文本完全相同（忽略大小写和首尾空白）。
    2. 近似去重：question 前 60 字符 + correct_answer 组合重复则视为近似重复。

    参数：
        input_path:  qa_full.jsonl 路径。
        output_path: 去重后输出路径。

    返回：
        去重后保留的条目数。
    """
    items = _load_jsonl(input_path)
    seen_exact: set[str] = set()
    seen_fuzzy: set[str] = set()
    kept: List[Dict] = []

    for item in items:
        q = item.get("question", "").strip().lower()
        a = item.get("correct_answer", "").strip().lower()

        if q in seen_exact:
            continue
        seen_exact.add(q)

        fuzzy_key = q[:60] + "||" + a
        if fuzzy_key in seen_fuzzy:
            continue
        seen_fuzzy.add(fuzzy_key)

        kept.append(item)

    _save_jsonl(kept, output_path)
    logger.info("去重完成：%d → %d（移除 %d 重复）", len(items), len(kept), len(items) - len(kept))
    return len(kept)
```

- [ ] **Step 4: 在 run_pipeline() 中 Step 3 之后调用去重**

在 `tools/build_news_qa.py` 的 `run_pipeline()` 函数中，Step 3 的 if block（约第 966-968 行）之后，Step 4 之前，插入：

```python
    # 去重
    dedup_path = DATA_DIR / "qa_full_dedup.jsonl"
    if any(s in step_list for s in [1, 2, 3]) or not dedup_path.exists():
        logger.info("执行去重…")
        run_dedup(qa_full_path, dedup_path)
    else:
        logger.info("去重跳过（qa_full_dedup.jsonl 已存在且无上游更新）")
```

同时修改 Step 4 和 Step 5 的输入路径，让它们读取去重后的文件：

将 Step 4 中的 `run_step4_ood_check(qa_full_path)` 改为：
```python
        run_step4_ood_check(dedup_path)
```

将 Step 5 中的 `run_step5_convert(qa_full_path, train_path, test_path)` 改为：
```python
            run_step5_convert(dedup_path, train_path, test_path)
```

- [ ] **Step 5: 验证修改后 import 正常**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
conda run -n ExplicitLLM python -c "
from tools.build_news_qa import run_dedup, run_pipeline, _GENERATE_MULTI_QA_PROMPT_TEMPLATE
print('run_dedup imported OK')
print('Prompt mentions DIFFERENT:', 'DIFFERENT' in _GENERATE_MULTI_QA_PROMPT_TEMPLATE)
"
```

Expected:
```
run_dedup imported OK
Prompt mentions DIFFERENT: True
```

- [ ] **Step 6: 用现有数据测试 run_dedup()**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
conda run -n ExplicitLLM python -c "
from pathlib import Path
from tools.build_news_qa import run_dedup
count = run_dedup(Path('data/news/qa_full.jsonl'), Path('data/news/qa_full_dedup_test.jsonl'))
print(f'Dedup result: {count} items kept')
import os; os.remove('data/news/qa_full_dedup_test.jsonl')
"
```

Expected: 应输出约 11,000-12,000 条（与原始 qa_full_dedup.jsonl 的 11,338 接近）

- [ ] **Step 7: Commit**

```bash
git add tools/build_news_qa.py
git commit -m "feat: qa_per_passage=5 default, diversity prompt, question-level dedup"
```

---

### Task 8: Smoke Test — 单源爬取 + Step 1-2 验证

**Files:** 无新文件，纯测试

目的: 验证完整 pipeline 从爬取到 QA 生成可以端到端跑通（用最小量）

- [ ] **Step 1: 测试单源爬取（PhysOrg，最可能成功）**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
conda run -n ExplicitLLM python -c "
import asyncio, json
from tools.news_crawlers import PhysOrgCrawler, AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

async def test():
    browser_cfg = BrowserConfig(headless=True, verbose=False)
    run_cfg = CrawlerRunConfig(cache_mode='bypass')
    crawler_obj = PhysOrgCrawler()
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        articles = await crawler_obj.crawl_category('science', 3, crawler, run_cfg)
    print(f'Got {len(articles)} articles')
    for a in articles:
        print(f'  [{a.source}] {a.title[:60]}... ({len(a.body)} chars, date={a.date})')
    return articles

articles = asyncio.run(test())
"
```

Expected: 1-3 篇文章，每篇有 title、body (200+ chars)、date。

如果某个源失败（HTTP 403/401），记录并跳过，尝试另一个源（如 ESPN 或 ScienceDaily）。

- [ ] **Step 2: 测试 LLM 调用（Step 1 段落提取，用 1 篇文章）**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
conda run -n ExplicitLLM python -c "
import asyncio
from tools.build_news_qa import extract_passages_from_article, _make_llm_client, LLM_MODEL

async def test():
    client = _make_llm_client()
    test_body = '''In April 2026, NASA announced a new mission to explore Europa, Jupiter's icy moon. The mission, called Europa Clipper II, will launch in 2028 and aims to analyze the ocean beneath Europa's ice shell. Scientists at the Jet Propulsion Laboratory have developed new instruments capable of detecting organic compounds at parts-per-billion levels. The mission budget is estimated at 4.2 billion dollars.'''
    passages = await extract_passages_from_article(client, test_body, LLM_MODEL)
    print(f'Extracted {len(passages)} passages:')
    for i, p in enumerate(passages):
        print(f'  [{i+1}] {p[:80]}...')

asyncio.run(test())
"
```

Expected: 1-3 个段落，每个 50-2000 字符。

- [ ] **Step 3: 测试 Step 2 QA 生成（qa_per_passage=5）**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
conda run -n ExplicitLLM python -c "
import asyncio
from tools.build_news_qa import generate_qa_from_passage, _make_llm_client, LLM_MODEL

async def test():
    client = _make_llm_client()
    passage = 'In April 2026, NASA announced the Europa Clipper II mission to explore the ocean beneath Europa ice shell. The mission will launch in 2028 with a budget of 4.2 billion dollars. Scientists at the Jet Propulsion Laboratory developed instruments detecting organic compounds at parts-per-billion levels.'
    qa_list = await generate_qa_from_passage(client, passage, LLM_MODEL, qa_per_passage=5)
    print(f'Generated {len(qa_list)} QA pairs:')
    for qa in qa_list:
        print(f'  Q: {qa[\"question\"]}')
        print(f'  A: {qa[\"answer\"]}')
        print()

asyncio.run(test())
"
```

Expected: 3-5 个 QA 对，每个有不同的 question 和 answer。

- [ ] **Step 4: Commit smoke test 通过的记录**

如果 Step 1-3 全部通过，不需要额外 commit（无文件变更）。如果发现了需要修复的 bug，在此处修复并 commit。

---

### Task 9: 执行完整爬取（Step 0）

**Files:** 输出到 `data/news/raw_articles.jsonl`（增量追加）

- [ ] **Step 1: 启动完整爬取（目标 8000 篇，预计 2-3 小时）**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
conda run -n ExplicitLLM python -m tools.build_news_qa --steps 0 --n-articles 8000 --resume 2>&1 | tee logs/crawl_step0.log
```

`--resume` 参数会跳过已存在的 raw_articles.jsonl 中的 URL，增量追加新文章到现有 2,112 篇之上。

注意：此命令耗时较长（2-3h），建议在 screen/tmux 会话中运行：
```bash
screen -S crawl
cd /home/iomgaa/Projects/Memory-LoRA
conda run -n ExplicitLLM python -m tools.build_news_qa --steps 0 --n-articles 8000 --resume 2>&1 | tee logs/crawl_step0.log
# Ctrl+A, D 断开
```

- [ ] **Step 2: 验证爬取结果**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
wc -l data/news/raw_articles.jsonl
conda run -n ExplicitLLM python -c "
import json
from collections import Counter
with open('data/news/raw_articles.jsonl') as f:
    articles = [json.loads(l) for l in f if l.strip()]
print(f'Total articles: {len(articles)}')
sources = Counter(a['source'] for a in articles)
print(f'Sources ({len(sources)}):')
for s, c in sources.most_common():
    print(f'  {s}: {c}')
categories = Counter(a['category'] for a in articles)
print(f'Categories: {dict(categories)}')
"
```

Expected:
- Total articles: ≥ 6,000（目标 8,000，考虑反爬损耗）
- Sources: ≥ 20 个不同源
- 每个源 ≥ 50 篇

如果总量 < 6,000，可以二次运行 `--n-articles 12000` 补量。

- [ ] **Step 3: Commit 爬取完成状态**

不 commit 数据文件（已在 .gitignore），仅确认日志。

---

### Task 10: 执行 Step 1-3 + 去重（QA 生成全流程）

**Files:** 输出到 `data/news/passages.jsonl`, `qa_raw.jsonl`, `qa_full.jsonl`, `qa_full_dedup.jsonl`

- [ ] **Step 1: 运行 Step 1-3（预计 13 小时，建议过夜）**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
screen -S pipeline
conda run -n ExplicitLLM python -m tools.build_news_qa --steps 1-3 --qa-per-passage 5 --resume 2>&1 | tee logs/pipeline_step1-3.log
# Ctrl+A, D 断开
```

断点续跑机制确保：如果中途中断，重新运行同一命令会从断点继续。

- [ ] **Step 2: 检查中间产出（可在运行中执行）**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
echo "=== Progress ==="
wc -l data/news/passages.jsonl data/news/qa_raw.jsonl data/news/qa_full.jsonl 2>/dev/null
```

- [ ] **Step 3: 完成后验证最终数量**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
wc -l data/news/qa_full_dedup.jsonl
```

Expected: ≥ 60,000 行

如果 < 60,000：
- 检查 `wc -l data/news/qa_full.jsonl`（去重前数量）
- 如果去重前也不够，需要回到 Task 9 补充爬取
- 如果去重前够但去重后不够，放宽去重阈值（将 `q[:60]` 改为 `q[:40]`）

- [ ] **Step 4: 数据质量抽检**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
conda run -n ExplicitLLM python -c "
import json, random
with open('data/news/qa_full_dedup.jsonl') as f:
    items = [json.loads(l) for l in f if l.strip()]
sample = random.sample(items, min(5, len(items)))
for item in sample:
    print(f'Source: {item[\"source\"]}, Category: {item[\"category\"]}')
    print(f'Q: {item[\"question\"]}')
    print(f'A: {item[\"correct_answer\"]}')
    print(f'Options: {item[\"options\"]}')
    print('---')
"
```

Expected: 5 条随机样本，每条有完整的 question/correct_answer/options/passage。

---

### Task 11: 最终验收

**Files:** 无新文件

- [ ] **Step 1: 运行完整验收检查**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
conda run -n ExplicitLLM python -c "
import json
from collections import Counter

with open('data/news/qa_full_dedup.jsonl') as f:
    items = [json.loads(l) for l in f if l.strip()]

print(f'=== 验收报告 ===')
print(f'去重后 MCQ 总量: {len(items)}')
assert len(items) >= 60000, f'FAIL: {len(items)} < 60000'
print('  ✓ ≥ 60,000')

sources = Counter(i['source'] for i in items)
print(f'新闻源数量: {len(sources)}')
assert len(sources) >= 20, f'FAIL: {len(sources)} < 20'
print('  ✓ ≥ 20 sources')

categories = Counter(i['category'] for i in items)
print(f'类别数量: {len(categories)} — {list(categories.keys())}')
assert len(categories) >= 5, f'FAIL: {len(categories)} < 5'
print('  ✓ ≥ 5 categories')

min_source = min(sources.values())
min_source_name = min(sources, key=sources.get)
print(f'最少源: {min_source_name} = {min_source}')

# Schema 完整性
for field in ['question', 'correct_answer', 'options', 'passage', 'source', 'category']:
    empty = sum(1 for i in items if not i.get(field))
    assert empty == 0, f'FAIL: {empty} items missing {field}'
print('  ✓ Schema 完整（所有字段非空）')

print(f'\\n=== 源分布 ===')
for s, c in sources.most_common():
    print(f'  {s}: {c}')

print(f'\\n=== 类别分布 ===')
for cat, c in categories.most_common():
    print(f'  {cat}: {c}')

print(f'\\n✓ 全部验收通过')
"
```

Expected: 所有 assert 通过，输出"全部验收通过"。

- [ ] **Step 2: 最终 commit**

```bash
cd /home/iomgaa/Projects/Memory-LoRA
git add tools/build_news_qa.py tools/news_crawlers.py tools/__init__.py
git commit -m "feat: News 60K dataset expansion — 25 crawlers, qa_per_passage=5, question dedup"
```
