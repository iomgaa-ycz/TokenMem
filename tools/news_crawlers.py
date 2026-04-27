"""news_crawlers —— 基于 crawl4ai + trafilatura 的多站点新闻爬虫模块。

提供以下功能：
1. Article dataclass：统一新闻文章数据结构，含验证与序列化。
2. BaseCrawler：通用爬虫基类，封装 crawl4ai 集成与 trafilatura 正文提取。
3. 十二个具体爬虫子类：BBCCrawler / APNewsCrawler / TechCrunchCrawler /
   NPRCrawler / GuardianCrawler / AlJazeeraCrawler /
   ArsTechnicaCrawler / CBSNewsCrawler / ScienceDailyCrawler /
   IndependentCrawler / WiredCrawler / France24Crawler / CNBCCrawler。
4. SITE_CATEGORY_PLAN：站点-类别-目标数量映射表。
5. crawl_all_sites：异步主入口，输出按标题+URL双重去重后的 JSONL。

使用：
    python -m tools.news_crawlers --output data/news_raw.jsonl --n-articles 6000

变更记录 (2026-04-25)：
    - 移除 Reuters（HTTP 401）和 PhysOrg（HTTP 403），不可爬取。
    - 新增 NPR 作为替代（干净 HTML，无付费墙）。
    - 修复 BBC URL 模式：新格式为 /news/articles/c...。
    - 正文提取改用 trafilatura（crawl4ai markdown 含大量导航噪声）。
    - 放宽 max_chars 至 10000（trafilatura 提取结果有时超 5000）。
    - 新增 Guardian / Al Jazeera / Ars Technica / CBS News 四个爬虫。
    - AlJazeeraCrawler 使用日期 sitemap 驱动，规避 SPA 渲染问题。
    - SITE_CATEGORY_PLAN 总容量提升至 ~8000，支持 10K+ QA 对生成。
    - 新增 ScienceDaily / Independent / Wired / France24 / CNBC 五个爬虫。
    - ScienceDailyCrawler：科学新闻聚合，URL 格式 /releases/YYYY/MM/<id>.htm。
    - IndependentCrawler：英国综合新闻，URL 格式 /news/<section>/<slug>-b<id>.html。
    - WiredCrawler：科技/科学深度报道，URL 格式 /story/<slug>/（绝对链接）。
    - France24Crawler：国际新闻英文版，URL 格式 /en/<section>/YYYYMMDD-<slug>。
    - CNBCCrawler：商业/科技新闻，URL 格式 /YYYY/MM/DD/<slug>.html（绝对链接）。
    - crawl_all_sites 去重逻辑升级：同时按标题（前 80 字符）和 URL 去重。
    - SITE_CATEGORY_PLAN 总容量扩展至 ~14000，预期产出 ~6000+ 篇。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import trafilatura  # noqa: E402
import feedparser  # noqa: E402
import requests as _requests  # noqa: E402
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Article dataclass
# ---------------------------------------------------------------------------


@dataclass
class Article:
    """新闻文章数据模型。

    参数：
        id:       文章唯一标识符（UUID 或派生哈希）。
        title:    文章标题。
        body:     文章正文纯文本。
        source:   来源站点标识，如 "bbc"、"reuters"。
        date:     发布日期字符串，格式 "YYYY-MM-DD"。
        category: 所属类别，如 "science"、"technology"。
        url:      文章原始 URL。
    """

    id: str
    title: str
    body: str
    source: str
    date: str
    category: str
    url: str

    def to_dict(self) -> Dict[str, str]:
        """将文章序列化为字典。

        返回：
            包含所有字段的 dict，键顺序与字段定义一致。
        """
        return {
            "id": self.id,
            "title": self.title,
            "body": self.body,
            "source": self.source,
            "date": self.date,
            "category": self.category,
            "url": self.url,
        }

    def is_valid(self, min_chars: int = 200, max_chars: int = 10000) -> bool:
        """验证文章正文长度是否在合理范围内。

        参数：
            min_chars: 正文最少字符数，默认 200。
            max_chars: 正文最多字符数，默认 10000。

        返回：
            正文长度在 [min_chars, max_chars] 范围内时返回 True，否则 False。
        """
        body_len = len(self.body)
        return min_chars <= body_len <= max_chars


# ---------------------------------------------------------------------------
# 站点-类别-目标数量映射表
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# BaseCrawler
# ---------------------------------------------------------------------------


class BaseCrawler:
    """新闻爬虫基类，封装 crawl4ai 集成与通用 HTML 解析逻辑。

    子类需覆盖：
        - base_url:           站点根 URL。
        - source_name:        来源标识字符串。
        - category_urls:      类别名 → 列表页 URL 映射。
        - _is_article_url():  判断路径是否为文章 URL（可选覆盖）。
    """

    base_url: str = ""
    source_name: str = ""
    # 子类定义：category -> 列表页 URL
    category_urls: Dict[str, str] = {}

    def __init__(self) -> None:
        """初始化爬虫，设置日志记录器。"""
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    async def crawl_category(
        self,
        category: str,
        max_articles: int,
        crawler: AsyncWebCrawler,
        config: CrawlerRunConfig,
    ) -> List[Article]:
        """爬取指定类别的文章列表。

        参数：
            category:     目标类别名称。
            max_articles: 最多抓取文章数。
            crawler:      已初始化的 AsyncWebCrawler 实例。
            config:       爬取运行时配置。

        返回：
            Article 列表，每篇已验证通过 is_valid()。
        """
        list_url = self.category_urls.get(category)
        if not list_url:
            self._logger.warning("站点 %s 不支持类别 %s，跳过", self.source_name, category)
            return []

        self._logger.info("开始爬取 %s / %s，目标 %d 篇", self.source_name, category, max_articles)

        # 第一步：获取列表页，提取文章 URL
        try:
            list_result = await crawler.arun(url=list_url, config=config)
        except Exception as exc:
            self._logger.error("列表页请求失败 %s: %s", list_url, exc)
            return []

        if not list_result.success:
            self._logger.warning("列表页响应失败 %s (status=%s)", list_url, getattr(list_result, "status_code", "?"))
            return []

        article_urls = self._extract_article_urls(list_result.html or "", list_url)
        self._logger.info("从列表页提取到 %d 个候选 URL", len(article_urls))

        # 第二步：逐篇爬取文章
        articles: List[Article] = []
        for url in article_urls[:max_articles]:
            if len(articles) >= max_articles:
                break
            article = await self._fetch_article(url, category, crawler, config)
            if article is not None:
                articles.append(article)

        self._logger.info("成功采集 %s / %s: %d 篇", self.source_name, category, len(articles))
        return articles

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _extract_article_urls(self, html: str, base_url: str) -> List[str]:
        """从列表页 HTML 中提取文章链接。

        参数：
            html:     列表页 HTML 字符串。
            base_url: 用于补全相对 URL 的基础地址。

        返回：
            去重后的文章 URL 列表。
        """
        href_pattern = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
        seen: set[str] = set()
        urls: List[str] = []

        for match in href_pattern.finditer(html):
            href = match.group(1).strip()
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue

            # 补全相对 URL
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)

            # 只保留同域名下的路径
            base_parsed = urlparse(self.base_url)
            if parsed.netloc and base_parsed.netloc and parsed.netloc != base_parsed.netloc:
                continue

            path = parsed.path
            # 规范化 URL：去除 fragment（如 #comments），避免同文章重复采集
            normalized_url = parsed._replace(fragment="").geturl()
            if self._is_article_url(path) and normalized_url not in seen:
                seen.add(normalized_url)
                urls.append(normalized_url)

        return urls

    def _is_article_url(self, path: str) -> bool:
        """判断 URL 路径是否为文章页（子类可覆盖以实现站点特定逻辑）。

        参数：
            path: URL 路径部分（不含域名）。

        返回：
            认定为文章 URL 时返回 True。

        默认策略：路径包含数字或路径段 >= 3 则视为文章。
        """
        segments = [s for s in path.split("/") if s]
        if len(segments) < 2:
            return False
        # 包含数字段或路径较深视为文章
        has_number = any(re.search(r"\d{4,}", seg) for seg in segments)
        return has_number or len(segments) >= 3

    def _extract_title(self, html: str) -> str:
        """从文章页 HTML 提取标题。

        参数：
            html: 文章页 HTML 字符串。

        返回：
            标题字符串，提取失败时返回空字符串。
        """
        # 优先从 <h1> 提取
        m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.IGNORECASE | re.DOTALL)
        if m:
            return re.sub(r"<[^>]+>", "", m.group(1)).strip()
        # 回退到 <title>
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if m:
            raw = re.sub(r"<[^>]+>", "", m.group(1)).strip()
            # 去掉站点名后缀（" - BBC News" 等）
            return re.split(r"\s*[-|]\s*", raw)[0].strip()
        return ""

    def _extract_date(self, html: str) -> str:
        """从文章页 HTML 提取发布日期。

        参数：
            html: 文章页 HTML 字符串。

        返回：
            "YYYY-MM-DD" 格式日期字符串，提取失败时返回空字符串。
        """
        # 优先查找 datetime 属性
        m = re.search(
            r'(?:datetime|content|datePublished)["\s]*[:=]["\s]*["\']([0-9]{4}-[0-9]{2}-[0-9]{2})',
            html,
            re.IGNORECASE,
        )
        if m:
            return m.group(1)
        # 查找可见日期文本（YYYY-MM-DD）
        m = re.search(r"\b(20[2-9][0-9]-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01]))\b", html)
        if m:
            return m.group(1)
        return ""

    def _clean_body(self, text: str) -> str:
        """清洗正文：去除多余空白、广告关键词段落。

        参数：
            text: Markdown 或纯文本正文。

        返回：
            清洗后的纯文本字符串。
        """
        # 去除 Markdown 链接
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # 去除 HTML 标签残留
        text = re.sub(r"<[^>]+>", " ", text)
        # 去除连续空行
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 去除首尾空白
        text = text.strip()
        return text

    def _make_id(self, url: str) -> str:
        """基于 URL 生成稳定唯一 ID。

        参数：
            url: 文章 URL。

        返回：
            8 位十六进制哈希字符串。
        """
        return hashlib.md5(url.encode()).hexdigest()[:8]

    async def _fetch_article(
        self,
        url: str,
        category: str,
        crawler: AsyncWebCrawler,
        config: CrawlerRunConfig,
    ) -> Optional[Article]:
        """爬取单篇文章并解析为 Article 对象。

        使用 trafilatura 提取正文（crawl4ai markdown 含大量导航噪声），
        trafilatura 失败时回退到 crawl4ai markdown + _clean_body。

        参数：
            url:      文章 URL。
            category: 所属类别。
            crawler:  AsyncWebCrawler 实例。
            config:   爬取配置。

        返回：
            解析成功且 is_valid() 通过时返回 Article，否则返回 None。
        """
        try:
            result = await crawler.arun(url=url, config=config)
        except Exception as exc:
            self._logger.debug("文章请求异常 %s: %s", url, exc)
            return None

        if not result.success:
            self._logger.debug("文章请求失败 %s (status=%s)", url, getattr(result, "status_code", "?"))
            return None

        html = result.html or ""

        title = self._extract_title(html)
        date = self._extract_date(html)

        # 优先使用 trafilatura 提取干净正文
        body = ""
        try:
            extracted = trafilatura.extract(html)
            if extracted:
                body = self._clean_body(extracted)
        except Exception as exc:
            self._logger.debug("trafilatura 提取异常 %s: %s", url, exc)

        # trafilatura 失败时回退到 crawl4ai markdown
        if not body or len(body) < 100:
            body_raw = result.markdown or ""
            body = self._clean_body(body_raw)

        if not title or not body:
            self._logger.debug("标题或正文为空，跳过: %s (title=%r, body_len=%d)", url, title[:50] if title else "", len(body))
            return None

        article = Article(
            id=self._make_id(url),
            title=title,
            body=body,
            source=self.source_name,
            date=date,
            category=category,
            url=url,
        )

        if not article.is_valid():
            self._logger.debug("文章正文不合规，跳过: %s (len=%d)", url, len(body))
            return None

        return article


# ---------------------------------------------------------------------------
# 具体爬虫子类
# ---------------------------------------------------------------------------


class BBCCrawler(BaseCrawler):
    """BBC News 爬虫（bbc.com/news）。

    支持类别：science, technology, business, world。
    """

    base_url = "https://www.bbc.com"
    source_name = "bbc"
    category_urls: Dict[str, str] = {
        "science": "https://www.bbc.com/news/science_and_environment",
        "technology": "https://www.bbc.com/news/technology",
        "business": "https://www.bbc.com/news/business",
        "world": "https://www.bbc.com/news/world",
        "politics": "https://www.bbc.com/news/politics",
        "sports": "https://www.bbc.com/sport",
    }

    def _is_article_url(self, path: str) -> bool:
        """BBC 文章 URL 判断。

        新格式（2025+）：/news/articles/c... 或 /sport/.../articles/c...
        旧格式：/news/<slug>-<8+位数字> 或 /sport/<id>
        """
        # 新格式: /news/articles/c... 或 /sport/**/articles/c...
        if re.search(r"/articles/c[a-z0-9]+$", path):
            return True
        # 旧格式: /news/<category>-<8+位数字> 或 /sport/<数字ID>
        if re.match(r"^/(?:news|sport)/[^/]+-\d{8,}", path):
            return True
        if re.match(r"^/sport/\d{7,}$", path):
            return True
        return False


class APNewsCrawler(BaseCrawler):
    """AP News 爬虫（apnews.com）。

    支持类别：politics, science, technology, business。
    """

    base_url = "https://apnews.com"
    source_name = "apnews"
    category_urls: Dict[str, str] = {
        "politics": "https://apnews.com/politics",
        "science": "https://apnews.com/science",
        "technology": "https://apnews.com/technology",
        "business": "https://apnews.com/business",
        "world": "https://apnews.com/world-news",
        "sports": "https://apnews.com/sports",
    }

    def _is_article_url(self, path: str) -> bool:
        """AP News 文章路径形如 /article/<slug>-<hex>。"""
        return bool(re.match(r"^/article/", path))


class TechCrunchCrawler(BaseCrawler):
    """TechCrunch 爬虫（techcrunch.com）。

    支持类别：technology, ai。
    """

    base_url = "https://techcrunch.com"
    source_name = "techcrunch"
    category_urls: Dict[str, str] = {
        "technology": "https://techcrunch.com/latest/",
        "ai": "https://techcrunch.com/category/artificial-intelligence/",
    }

    def _is_article_url(self, path: str) -> bool:
        """TechCrunch 文章路径形如 /YYYY/MM/DD/<slug>/。"""
        return bool(re.match(r"^/\d{4}/\d{2}/\d{2}/", path))


class NPRCrawler(BaseCrawler):
    """NPR 爬虫（npr.org）。

    支持类别：science, technology, business, politics。
    干净 HTML 结构，无付费墙，trafilatura 提取效果好。
    """

    base_url = "https://www.npr.org"
    source_name = "npr"
    category_urls: Dict[str, str] = {
        "science": "https://www.npr.org/sections/science/",
        "technology": "https://www.npr.org/sections/technology/",
        "business": "https://www.npr.org/sections/business/",
        "politics": "https://www.npr.org/sections/politics/",
    }

    def _is_article_url(self, path: str) -> bool:
        """NPR 文章路径形如 /YYYY/MM/DD/<id>/<slug>。

        要求至少 4 段路径（日期 + ID + slug），排除纯 section 页。
        """
        # 匹配 /2026/04/24/nx-s1-5793988/slug-text
        if re.match(r"^/\d{4}/\d{2}/\d{2}/", path):
            segments = [s for s in path.split("/") if s]
            # 日期 3 段 + ID + slug = 至少 5 段
            return len(segments) >= 4
        return False


class GuardianCrawler(BaseCrawler):
    """The Guardian 爬虫（theguardian.com）。

    支持类别：science, technology, business, politics, sports, world。
    文章 URL 格式：/<section>/YYYY/<mon>/<DD>/<slug>（4 段以上，含年月日）。
    """

    base_url = "https://www.theguardian.com"
    source_name = "guardian"
    category_urls: Dict[str, str] = {
        "science": "https://www.theguardian.com/science",
        "technology": "https://www.theguardian.com/technology",
        "business": "https://www.theguardian.com/business",
        "politics": "https://www.theguardian.com/politics",
        "sports": "https://www.theguardian.com/sport",
        "world": "https://www.theguardian.com/world",
    }

    def _is_article_url(self, path: str) -> bool:
        """Guardian 文章路径含年份（4 位数字）和月份缩写，如：
        /science/2026/apr/23/<slug>
        /news/2026/apr/23/<slug>
        /commentisfree/2026/apr/25/<slug>
        排除纯 section 页、preference 页、audio/video 页。
        """
        # 排除非文章页
        if re.match(
            r"^/(?:preference|type|search|profile|help|about|membership|"
            r"guardian-live-events|crosswords|newsletters|apps|info)/",
            path,
        ):
            return False
        # 排除音频和视频列表页（但保留含日期的音频文章）
        segments = [s for s in path.split("/") if s]
        if len(segments) < 4:
            return False
        # 检查是否含年份段（格式 YYYY）
        has_year = any(re.fullmatch(r"20\d{2}", seg) for seg in segments)
        # 检查是否含月份缩写（jan-dec）
        has_month = any(re.fullmatch(r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", seg) for seg in segments)
        return has_year and has_month


class AlJazeeraCrawler(BaseCrawler):
    """Al Jazeera 爬虫（aljazeera.com）。

    Al Jazeera 使用 SPA 渲染，列表页的链接均在 JS 中，无法通过 HTML href 提取。
    本实现采用日期 sitemap 驱动：
        https://www.aljazeera.com/sitemap.xml?yyyy=YYYY&mm=MM&dd=DD
    从近期多天的 sitemap 中提取文章 URL，按类别过滤后抓取正文。

    支持类别：news, business (economy), sports, world (news)。
    文章 URL 格式：/<section>/YYYY/M/DD/<slug>
    """

    base_url = "https://www.aljazeera.com"
    source_name = "aljazeera"
    # 类别 → sitemap 过滤路径前缀（用于从 sitemap URL 中识别类别）
    # 列表 URL 仅作占位，crawl_category 将被覆盖，不实际使用
    category_urls: Dict[str, str] = {
        "news": "https://www.aljazeera.com/news/",
        "business": "https://www.aljazeera.com/economy/",
        "sports": "https://www.aljazeera.com/sports/",
        "world": "https://www.aljazeera.com/news/",
        "politics": "https://www.aljazeera.com/news/",
    }

    # 类别 → sitemap URL 路径前缀（用于过滤 sitemap 中的 URL）
    _CATEGORY_PREFIXES: Dict[str, list[str]] = {
        "news": ["/news/"],
        "business": ["/economy/", "/business/"],
        "sports": ["/sports/"],
        "world": ["/news/", "/features/"],
        "politics": ["/news/", "/opinion/"],
    }

    def _is_article_url(self, path: str) -> bool:
        """Al Jazeera 文章路径：/<section>/YYYY/M/DD/<slug>，含数字年份段。"""
        segments = [s for s in path.split("/") if s]
        if len(segments) < 4:
            return False
        # 需含年份段
        return any(re.fullmatch(r"20\d{2}", seg) for seg in segments)

    def _extract_sitemap_urls(self, html: str, prefixes: list[str]) -> List[str]:
        """从 sitemap XML HTML 中提取符合类别前缀的文章 URL。

        参数：
            html:     sitemap 页面 HTML。
            prefixes: 要保留的 URL 路径前缀列表。

        返回：
            文章 URL 列表（已去重）。
        """
        loc_pattern = re.compile(r"<loc>(https://www\.aljazeera\.com/([^<]+))</loc>")
        seen: set[str] = set()
        urls: List[str] = []
        for m in loc_pattern.finditer(html):
            full_url = m.group(1)
            path = "/" + m.group(2)
            # 过滤 video/newsfeed/liveblog/gallery 等非文章类型
            if any(seg in path for seg in ["/video/", "/liveblog/", "/gallery/", "/program/"]):
                continue
            if any(path.startswith(prefix) for prefix in prefixes) and full_url not in seen:
                seen.add(full_url)
                urls.append(full_url)
        return urls

    async def crawl_category(
        self,
        category: str,
        max_articles: int,
        crawler: AsyncWebCrawler,
        config: CrawlerRunConfig,
    ) -> List[Article]:
        """覆盖基类方法：通过近期日期 sitemap 收集文章 URL 再批量抓取。

        参数：
            category:     目标类别。
            max_articles: 最多采集文章数。
            crawler:      AsyncWebCrawler 实例。
            config:       爬取配置。

        返回：
            Article 列表。

        实现细节：
            - 从今天往前取最近 N 天的日期 sitemap，直到收集到足够候选 URL。
            - 每个日期 sitemap 约含 30-80 个 URL，默认检查最近 30 天。
        """
        import datetime

        prefixes = self._CATEGORY_PREFIXES.get(category, ["/news/"])
        if not prefixes:
            self._logger.warning("AlJazeera 不支持类别 %s，跳过", category)
            return []

        self._logger.info("AlJazeera / %s 通过 sitemap 收集 URL，目标 %d 篇", category, max_articles)

        # 收集候选 URL：检查近 45 天的 sitemap
        candidate_urls: List[str] = []
        today = datetime.date.today()
        for days_back in range(0, 45):
            if len(candidate_urls) >= max_articles * 3:
                break
            day = today - datetime.timedelta(days=days_back)
            sitemap_url = (
                f"https://www.aljazeera.com/sitemap.xml"
                f"?yyyy={day.year}&mm={day.month:02d}&dd={day.day:02d}"
            )
            try:
                result = await crawler.arun(url=sitemap_url, config=config)
            except Exception as exc:
                self._logger.debug("sitemap 请求异常 %s: %s", sitemap_url, exc)
                continue
            if not result.success:
                continue
            day_urls = self._extract_sitemap_urls(result.html or "", prefixes)
            candidate_urls.extend(day_urls)
            self._logger.debug("日期 %s sitemap 提取 %d 个 URL（%s）", day, len(day_urls), category)

        self._logger.info("AlJazeera / %s: 共 %d 候选 URL，开始抓取", category, len(candidate_urls))

        # 批量抓取文章
        articles: List[Article] = []
        for url in candidate_urls[:max_articles]:
            if len(articles) >= max_articles:
                break
            article = await self._fetch_article(url, category, crawler, config)
            if article is not None:
                articles.append(article)

        self._logger.info("AlJazeera / %s: 成功采集 %d 篇", category, len(articles))
        return articles


class ArsTechnicaCrawler(BaseCrawler):
    """Ars Technica 爬虫（arstechnica.com）。

    专注科技与科学深度报道，文章质量高，无付费墙。
    文章 URL 格式：/<section>/YYYY/MM/<slug>/
    支持类别：science, technology。
    """

    base_url = "https://arstechnica.com"
    source_name = "arstechnica"
    category_urls: Dict[str, str] = {
        "science": "https://arstechnica.com/science/",
        "technology": "https://arstechnica.com/gadgets/",
        "space": "https://arstechnica.com/space/",
        "health": "https://arstechnica.com/health/",
    }

    def _is_article_url(self, path: str) -> bool:
        """Ars Technica 文章路径形如 /<section>/YYYY/MM/<slug>/。

        格式：以 4 位年份段为特征，排除列表页、tag 页、search 页。
        """
        # 排除非文章页
        if re.match(r"^/(?:tag|author|civis|search|store|subscribe|about)/", path):
            return False
        # 匹配 /section/YYYY/MM/slug/ 或 /section/YYYY/MM/slug
        return bool(re.match(r"^/[^/]+/20\d{2}/\d{2}/.+", path))


class CBSNewsCrawler(BaseCrawler):
    """CBS News 爬虫（cbsnews.com）。

    美国综合新闻，HTML 结构干净，无付费墙。
    文章 URL 格式：/news/<slug>/
    支持类别：science, technology, business, politics, sports。
    """

    base_url = "https://www.cbsnews.com"
    source_name = "cbsnews"
    category_urls: Dict[str, str] = {
        "science": "https://www.cbsnews.com/science/",
        "technology": "https://www.cbsnews.com/technology/",
        "business": "https://www.cbsnews.com/moneywatch/",
        "politics": "https://www.cbsnews.com/politics/",
        "sports": "https://www.cbsnews.com/sports/",
    }

    def _is_article_url(self, path: str) -> bool:
        """CBS News 文章路径形如 /news/<slug>/。

        排除 /live/、/live-updates/、/video/、/fly/ 等非文章路径。
        """
        if re.match(
            r"^/(?:live|live-updates|video|fly|local|latest|embed|amp)/",
            path,
        ):
            return False
        # 主文章路径：/news/<slug>
        if re.match(r"^/news/[^/]{10,}/?$", path):
            return True
        # 本地站文章路径：/<city>/news/<slug>
        if re.match(r"^/[a-z]+/news/[^/]{10,}/?$", path):
            return True
        return False

    def _extract_title(self, html: str) -> str:
        """CBS News 标题提取：<h1> 中含浏览器兼容性注入，优先从 <title> 提取。

        参数：
            html: 文章页 HTML 字符串。

        返回：
            标题字符串，提取失败时返回空字符串。
        """
        # CBS News 的 <h1> 中常被注入浏览器警告，直接用 <title>
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if m:
            raw = re.sub(r"<[^>]+>", "", m.group(1)).strip()
            # 去掉 " - CBS News" 等后缀
            return re.split(r"\s*[-|]\s*CBS", raw)[0].strip()
        return ""


class ScienceDailyCrawler(BaseCrawler):
    """ScienceDaily 爬虫（sciencedaily.com）。

    科学新闻聚合站点，报道学术研究发现，内容干净且无付费墙。
    文章 URL 格式：/releases/YYYY/MM/<id>.htm
    支持类别：science（含 health / environment / technology 子页）。
    """

    base_url = "https://www.sciencedaily.com"
    source_name = "sciencedaily"
    category_urls: Dict[str, str] = {
        "science": "https://www.sciencedaily.com/news/top/science/",
        "health": "https://www.sciencedaily.com/news/top/health/",
        "technology": "https://www.sciencedaily.com/news/computers_math/",
        "environment": "https://www.sciencedaily.com/news/earth_climate/",
    }

    def _is_article_url(self, path: str) -> bool:
        """ScienceDaily 文章路径形如 /releases/YYYY/MM/<id>.htm。

        格式：/releases/<4位年份>/<2位月份>/<id>.htm
        """
        return bool(re.match(r"^/releases/20\d{2}/\d{2}/\d+\.htm$", path))


class IndependentCrawler(BaseCrawler):
    """The Independent 爬虫（independent.co.uk）。

    英国综合新闻，涵盖科学、政治、商业等，无付费墙，HTML 结构清晰。
    文章 URL 格式：/news/<section>/<slug>-b<7位数字>.html
    支持类别：science, technology, business, politics, world。
    """

    base_url = "https://www.independent.co.uk"
    source_name = "independent"
    category_urls: Dict[str, str] = {
        "science": "https://www.independent.co.uk/news/science",
        "technology": "https://www.independent.co.uk/tech",
        "business": "https://www.independent.co.uk/business",
        "politics": "https://www.independent.co.uk/news/uk/politics",
        "world": "https://www.independent.co.uk/news/world",
    }

    def _is_article_url(self, path: str) -> bool:
        """The Independent 文章路径形如 /news/<section>/<slug>-b<id>.html。

        特征：路径以 -b<6+位数字>.html 或 -a<6+位数字>.html 结尾。
        排除纯分类页（路径段 < 3）和非 HTML 资源。
        """
        # 必须是 .html 结尾
        if not path.endswith(".html"):
            return False
        # 排除非文章路径
        if re.match(r"^/(?:img|static|fonts|icons)/", path):
            return False
        # 匹配文章 ID 后缀：-b<6+位数字>.html 或 -a<6+位数字>.html
        if re.search(r"-[ab]\d{6,}\.html$", path):
            return True
        return False


class WiredCrawler(BaseCrawler):
    """Wired 爬虫（wired.com）。

    科技与科学深度报道，无付费墙基础文章，trafilatura 提取效果好。
    文章 URL 格式：/story/<slug>/（绝对链接，基类可正确提取同域名链接）
    支持类别：science, technology, business, ai。
    """

    base_url = "https://www.wired.com"
    source_name = "wired"
    category_urls: Dict[str, str] = {
        "science": "https://www.wired.com/category/science/",
        "technology": "https://www.wired.com/category/gear/",
        "business": "https://www.wired.com/category/business/",
        "ai": "https://www.wired.com/category/artificial-intelligence/",
    }

    def _is_article_url(self, path: str) -> bool:
        """Wired 文章路径形如 /story/<slug>/。

        排除非文章页：/category/、/tag/、/author/、/subscribe/ 等。
        """
        # 排除非文章页
        if re.match(
            r"^/(?:category|tag|author|subscribe|coupons|about|magazine|"
            r"video|podcast|gallery|quiz|contribute)/",
            path,
        ):
            return False
        # 文章路径：/story/<slug>（可带或不带尾部 /）
        return bool(re.match(r"^/story/[^/]{5,}/?$", path))


class France24Crawler(BaseCrawler):
    """France24 爬虫（france24.com/en）。

    法国国际广播英文版，覆盖欧洲、世界、商业、科技话题，无付费墙。
    文章 URL 格式：/en/<section>/YYYYMMDD-<slug>
    支持类别：world, business, science, politics。
    """

    base_url = "https://www.france24.com"
    source_name = "france24"
    category_urls: Dict[str, str] = {
        "world": "https://www.france24.com/en/europe/",
        "business": "https://www.france24.com/en/economy/",
        "science": "https://www.france24.com/en/science-technology/",
        "politics": "https://www.france24.com/en/americas/",
    }

    def _is_article_url(self, path: str) -> bool:
        """France24 文章路径形如 /en/<section>/YYYYMMDD-<slug>。

        特征：路径包含 8 位日期前缀（如 20260425-）。
        排除 TV show 页、live news、replay 等非文章内容。
        """
        # 排除非文章页
        if re.match(
            r"^/en/(?:tv-shows|live-news|replay|podcasts|[a-z]+-show)/",
            path,
        ):
            return False
        # 匹配含日期前缀的文章路径：/en/<section>/YYYYMMDD-<slug>
        return bool(re.search(r"/en/[^/]+/20\d{6}-[a-z]", path))


class CNBCCrawler(BaseCrawler):
    """CNBC 爬虫（cnbc.com）。

    美国商业与科技财经新闻，内容无付费墙，trafilatura 可提取正文。
    文章 URL 格式（绝对）：https://www.cnbc.com/YYYY/MM/DD/<slug>.html
    支持类别：technology, business, science。

    注意：CNBC 列表页 href 均为绝对 URL（https://www.cnbc.com/...），
    基类 _extract_article_urls 以 base_url 为域名过滤，可正确处理。
    """

    base_url = "https://www.cnbc.com"
    source_name = "cnbc"
    category_urls: Dict[str, str] = {
        "technology": "https://www.cnbc.com/technology/",
        "business": "https://www.cnbc.com/business/",
        "science": "https://www.cnbc.com/science/",
    }

    def _is_article_url(self, path: str) -> bool:
        """CNBC 文章路径形如 /YYYY/MM/DD/<slug>.html。

        排除 /video/、/live-tv/、/pro/、/select/ 等非文章路径。
        """
        # 排除非文章路径
        if re.match(
            r"^/(?:video|live-tv|pro|select|investingclub|piped|id|widget|"
            r"amp|json|site-map|application)/",
            path,
        ):
            return False
        # 匹配日期格式路径：/YYYY/MM/DD/<slug>.html
        return bool(re.match(r"^/20\d{2}/\d{2}/\d{2}/.+\.html$", path))


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


# ---------------------------------------------------------------------------
# 爬虫实例注册表
# ---------------------------------------------------------------------------

_CRAWLER_REGISTRY: Dict[str, BaseCrawler] = {
    "bbc": BBCCrawler(),
    "apnews": APNewsCrawler(),
    "techcrunch": TechCrunchCrawler(),
    "npr": NPRCrawler(),
    "guardian": GuardianCrawler(),
    "aljazeera": AlJazeeraCrawler(),
    "arstechnica": ArsTechnicaCrawler(),
    "cbsnews": CBSNewsCrawler(),
    "sciencedaily": ScienceDailyCrawler(),
    "independent": IndependentCrawler(),
    "wired": WiredCrawler(),
    "france24": France24Crawler(),
    "cnbc": CNBCCrawler(),
    "reuters": ReutersCrawler(),
    "theverge": TheVergeCrawler(),
    "politico": PoliticoCrawler(),
    "espn": ESPNCrawler(),
    "nature": NatureNewsCrawler(),
    "physorg": PhysOrgCrawler(),
    "cnn": CNNCrawler(),
    "bloomberg": BloombergCrawler(),
    "thehill": TheHillCrawler(),
    "skynews": SkyNewsCrawler(),
    "newscientist": NewScientistCrawler(),
    "abcau": ABCNewsAUCrawler(),
}


# ---------------------------------------------------------------------------
# 主入口：crawl_all_sites
# ---------------------------------------------------------------------------


async def crawl_all_sites(
    output_path: str,
    n_articles: int = 6000,
    min_date: str = "2026-01-01",
) -> int:
    """异步爬取所有站点，输出去重后的 JSONL 文件。

    参数：
        output_path: 输出 JSONL 文件路径。
        n_articles:  目标总文章数，用于等比缩放 SITE_CATEGORY_PLAN 中的计划量。
        min_date:    最早发布日期过滤阈值，格式 "YYYY-MM-DD"。

    返回：
        实际写入文章总数。

    实现细节：
        - SITE_CATEGORY_PLAN 总原始容量约 14000，以此为基准进行等比缩放。
        - scale = n_articles / plan_total，确保请求量与目标产出成比例。
        - 双重去重：同时按标题前 80 字符（小写）和规范化 URL 去重。
          URL 去重可防止多轮爬取时重复写入同一篇文章。
        - 过滤发布日期早于 min_date 的文章。
        - 输出每行一个 JSON 对象（JSONL 格式）。
    """
    # SITE_CATEGORY_PLAN 中所有 raw_target 之和（动态计算）
    _PLAN_TOTAL = sum(v for cat in SITE_CATEGORY_PLAN.values() for v in cat.values())
    scale = n_articles / _PLAN_TOTAL
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    browser_cfg = BrowserConfig(headless=True, verbose=False)
    run_cfg = CrawlerRunConfig(cache_mode="bypass")

    seen_titles: set[str] = set()
    seen_urls: set[str] = set()
    total_written = 0

    # 若输出文件已存在（多轮爬取），预加载已有 URL 集合
    if output.exists():
        try:
            with output.open("r", encoding="utf-8") as _f:
                for _line in _f:
                    try:
                        _art = json.loads(_line)
                        _title_key = _art.get("title", "").lower()[:80]
                        _url_key = _art.get("url", "").rstrip("/")
                        if _title_key:
                            seen_titles.add(_title_key)
                        if _url_key:
                            seen_urls.add(_url_key)
                    except json.JSONDecodeError:
                        pass
            logger.info("预加载已有文章：%d 标题，%d URL（来自 %s）", len(seen_titles), len(seen_urls), output_path)
        except OSError as exc:
            logger.warning("预加载已有文章失败: %s", exc)

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        for category, site_targets in SITE_CATEGORY_PLAN.items():
            for site_key, raw_target in site_targets.items():
                target = max(1, int(raw_target * scale))
                site_crawler = _CRAWLER_REGISTRY.get(site_key)
                if site_crawler is None:
                    logger.warning("未注册的站点 key: %s，跳过", site_key)
                    continue

                articles = await site_crawler.crawl_category(
                    category=category,
                    max_articles=target,
                    crawler=crawler,
                    config=run_cfg,
                )

                with output.open("a", encoding="utf-8") as f:
                    for art in articles:
                        # 日期过滤
                        if art.date and art.date < min_date:
                            continue
                        # 双重去重：标题（前 80 字符）+ URL
                        title_key = art.title.lower()[:80]
                        url_key = art.url.rstrip("/")
                        if title_key in seen_titles or url_key in seen_urls:
                            continue
                        seen_titles.add(title_key)
                        seen_urls.add(url_key)
                        f.write(json.dumps(art.to_dict(), ensure_ascii=False) + "\n")
                        total_written += 1

    logger.info("爬取完成，共写入 %d 篇文章 → %s", total_written, output_path)
    return total_written


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
    import time as _time

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

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

                url_key = link.rstrip("/")
                title = entry.get("title", "")
                title_key = title.lower()[:80]

                if url_key in seen_urls or title_key in seen_titles:
                    continue

                body = _fetch_article_text(link)
                if not body:
                    continue

                date = ""
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    date = _time.strftime("%Y-%m-%d", entry.published_parsed)
                elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                    date = _time.strftime("%Y-%m-%d", entry.updated_parsed)

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


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def _parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="多站点新闻爬虫，输出 JSONL")
    parser.add_argument("--output", default="data/news_raw.jsonl", help="输出文件路径")
    parser.add_argument("--n-articles", type=int, default=6000, help="目标总文章数")
    parser.add_argument("--min-date", default="2026-01-01", help="最早日期过滤 YYYY-MM-DD")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    total = asyncio.run(crawl_all_sites(args.output, args.n_articles, args.min_date))
    sys.exit(0 if total > 0 else 1)
