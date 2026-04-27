"""build_news_qa —— 新闻问答数据集构建流水线（Step 0-5）。

核心功能：
- Step 0: 爬取新闻文章（调用 tools.news_crawlers.crawl_all_sites）
- Step 1: 从文章中提取知识段落（LLM 生成）
- Step 2: 为每个段落生成问答对（LLM 生成）
- Step 3: 为每个问答对生成干扰项并组装选择题（LLM 生成）
- Step 4: OOD 零样本基线检查（确认模型无法轻松答对）
- Step 5: LLMLingua 压缩 + 转换为 v2 schema，拆分 train/test

数据路径：
    data/news/raw_articles.jsonl → passages.jsonl → qa_raw.jsonl → qa_full.jsonl
    data/v2/news_knowledge_k64.jsonl（train）
    data/v2/news_knowledge_k64_test.jsonl（test）

环境变量（通过 .env 加载）：
    DEEPSEEK_LLM_MODEL:    DeepSeek 模型名称（优先；默认 deepseek-v4-flash）
    DEEPSEEK_LLM_BASE_URL: DeepSeek API base URL（优先）
    DEEPSEEK_LLM_API_KEY:  DeepSeek API 密钥（优先）
    LLM_MODEL:             备用 LLM 模型名称
    LLM_BASE_URL:          备用 OpenAI-compatible API base URL
    LLM_API_KEY:           备用 API 密钥
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─── 三方库 ───────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

# ─── 项目内部 ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

# ─── 配置 ─────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

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

DATA_DIR = Path("data/news")
V2_DIR = Path("data/v2")

raw_path = DATA_DIR / "raw_articles.jsonl"
passages_path = DATA_DIR / "passages.jsonl"
qa_raw_path = DATA_DIR / "qa_raw.jsonl"
qa_full_path = DATA_DIR / "qa_full.jsonl"
train_path = V2_DIR / "news_knowledge_k64.jsonl"
test_path = V2_DIR / "news_knowledge_k64_test.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────


def parse_llm_json_response(raw: str, key: str) -> Optional[Any]:
    """从 LLM 输出中解析 JSON，支持 Markdown 代码块。

    参数：
        raw: LLM 原始输出字符串。
        key: 要提取的 JSON 顶层键名。

    返回：
        对应键的值；若解析失败或键不存在则返回 None。

    实现细节：
        - 优先从 ```json ... ``` 或 ``` ... ``` 代码块中提取 JSON。
        - 若无代码块则直接尝试解析整个字符串。
    """
    # Phase 1: 尝试从 Markdown 代码块提取
    code_block_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
    match = code_block_pattern.search(raw)
    candidate = match.group(1).strip() if match else raw.strip()

    # Phase 2: 解析 JSON
    try:
        parsed = json.loads(candidate)
        return parsed.get(key) if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, AttributeError):
        return None


def _load_jsonl(path: Path) -> List[Dict]:
    """加载 JSONL 文件，返回字典列表。

    参数：
        path: JSONL 文件路径。

    返回：
        解析后的字典列表，跳过空行。
    """
    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _save_jsonl(items: List[Dict], path: Path) -> None:
    """将字典列表保存为 JSONL 文件，自动创建父目录。

    参数：
        items: 要保存的字典列表。
        path:  输出文件路径。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _make_llm_client() -> AsyncOpenAI:
    """根据环境变量创建 OpenAI-compatible 异步客户端。

    返回：
        配置好 base_url 和 api_key 的 AsyncOpenAI 实例。
    """
    return AsyncOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: 提取段落
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACT_PASSAGES_PROMPT = """\
You are a knowledge extraction expert. Given a news article, extract 1 to 5 \
independent, self-contained knowledge passages about RECENT events.

Requirements for each passage:
- 2 to 4 sentences long, approximately 256 tokens
- MUST describe events or facts from 2025 or later — reject any content about \
events before 2025
- Must contain specific named entities (people, organizations, places), numbers, \
or dates — no unresolved pronouns (e.g., do NOT extract a passage starting with \
"she", "he", "they", "it" without naming the referent)
- Must be self-contained and fully understandable without reading the rest of \
the article — include enough context so the passage stands alone
- Focus on factual, verifiable information about real-world events

REJECT and do NOT include:
- Passages about historical events before 2025
- Metadata: podcast credits, "produced by", "edited by", correction notices, \
  erratum, editorial notes, subscription prompts, or author bios
- Passages that open with a pronoun whose referent is not stated within the passage
- Opinion paragraphs without verifiable facts

Return your response as JSON with this exact format:
{"passages": ["passage 1 text here", "passage 2 text here"]}

Article:
"""


async def extract_passages_from_article(
    client: AsyncOpenAI,
    body: str,
    model: str,
) -> List[str]:
    """从新闻文章正文中提取 1-5 个知识段落。

    参数：
        client: AsyncOpenAI 客户端。
        body:   文章正文。
        model:  LLM 模型名称。

    返回：
        长度在 (50, 2000) 字符之间的段落字符串列表；出错时返回空列表。

    实现细节：
        - 使用英文 prompt 指导 LLM 提取包含实体/数字/日期的段落。
        - 过滤：仅保留 50 < len < 2000 chars 的段落。
        - LLM 异常时静默返回空列表。
    """
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": _EXTRACT_PASSAGES_PROMPT + body},
            ],
            temperature=0.3,
            extra_body={"thinking": {"type": "disabled"}},
        )
        raw: str = resp.choices[0].message.content or ""
        passages: Optional[List] = parse_llm_json_response(raw, key="passages")
        if not isinstance(passages, list):
            return []
        return [p for p in passages if isinstance(p, str) and 50 < len(p) < 2000]
    except Exception as exc:  # noqa: BLE001
        logger.warning("extract_passages_from_article 失败: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: 生成问答对
# ─────────────────────────────────────────────────────────────────────────────

_GENERATE_QA_PROMPT = """\
You are a question-answer generation expert. Given a factual passage from a \
news article, generate one high-quality multiple-choice question.

Requirements:
- The question must require reading the passage to answer correctly
- It should NOT be answerable from general world knowledge alone
- The correct answer should be specific (a name, number, date, or event)
- Keep both question and answer concise

Return your response as JSON with this exact format:
{"question": "your question here", "answer": "the correct answer here"}

Passage:
"""

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


async def generate_qa_from_passage(
    client: AsyncOpenAI,
    passage: str,
    model: str,
    qa_per_passage: int = 3,
) -> List[Dict[str, str]]:
    """为单个知识段落生成多个问答对。

    参数：
        client:          AsyncOpenAI 客户端。
        passage:         知识段落文本。
        model:           LLM 模型名称。
        qa_per_passage:  期望生成的 QA 对数量（默认 3）。

    返回：
        含多个 {"question": str, "answer": str} 字典的列表；
        出错或解析失败时返回空列表。接受 1-N 个有效 QA（宽容解析）。

    实现细节：
        - qa_per_passage=1 时退回单问答 prompt（向后兼容）。
        - 多问答 prompt 返回 {"qa_pairs": [...]}；单问答 prompt 返回
          {"question": ..., "answer": ...}，统一转换为列表格式。
        - 宽容策略：LLM 返回少于 qa_per_passage 的 QA 对时仍接受。
    """
    try:
        if qa_per_passage == 1:
            # 向后兼容：沿用单问答 prompt
            prompt = _GENERATE_QA_PROMPT + passage
        else:
            prompt = (
                _GENERATE_MULTI_QA_PROMPT_TEMPLATE.format(n=qa_per_passage) + passage
            )

        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            extra_body={"thinking": {"type": "disabled"}},
        )
        raw: str = resp.choices[0].message.content or ""

        code_block_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
        match = code_block_pattern.search(raw)
        candidate = match.group(1).strip() if match else raw.strip()
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            return []

        # 多 QA 路径：{"qa_pairs": [...]}
        if "qa_pairs" in parsed:
            pairs = parsed["qa_pairs"]
            if not isinstance(pairs, list):
                return []
            result: List[Dict[str, str]] = []
            for item in pairs:
                if isinstance(item, dict) and "question" in item and "answer" in item:
                    result.append(
                        {
                            "question": str(item["question"]),
                            "answer": str(item["answer"]),
                        }
                    )
            return result

        # 单 QA 回退路径（LLM 有时会在多 QA prompt 下返回单 QA 格式）
        if "question" in parsed and "answer" in parsed:
            return [
                {
                    "question": str(parsed["question"]),
                    "answer": str(parsed["answer"]),
                }
            ]

        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("generate_qa_from_passage 失败: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: 生成干扰项
# ─────────────────────────────────────────────────────────────────────────────

_GENERATE_DISTRACTORS_PROMPT_TEMPLATE = """\
You are an expert at creating multiple-choice question distractors. Given a \
passage, a question, and the correct answer, generate exactly 3 plausible but \
incorrect answer options.

Requirements for distractors:
- Similar length and format to the correct answer
- DOMAIN-RELEVANT: distractors must come from the same domain as the correct answer:
    * If the answer is a person's name → use other real people from the same field \
(e.g., same profession, country, or event context) who are NOT the correct answer
    * If the answer is a number or quantity → use plausible near-miss numbers of \
the same unit and magnitude (e.g., if answer is "47 billion dollars", use values \
like "31 billion dollars", "62 billion dollars")
    * If the answer is a place → use other real locations in the same region or \
context
    * If the answer is an organization → use other real organizations in the same \
sector
- Plausible enough that they cannot be easily eliminated without reading the passage
- Factually incorrect based on the passage content
- Do NOT use obviously fabricated placeholder names (e.g., "Lisa Rodriguez", \
"John Smith") unless that name actually appears in the passage as a wrong answer
- Do not repeat the correct answer or each other

Return your response as JSON with this exact format:
{{"distractors": ["wrong option 1", "wrong option 2", "wrong option 3"]}}

Passage: {passage}
Question: {question}
Correct answer: {answer}
"""


async def generate_distractors(
    client: AsyncOpenAI,
    passage: str,
    question: str,
    answer: str,
    model: str,
) -> List[str]:
    """为选择题生成 3 个干扰项。

    参数：
        client:   AsyncOpenAI 客户端。
        passage:  知识段落文本。
        question: 问题文本。
        answer:   正确答案文本。
        model:    LLM 模型名称。

    返回：
        包含 3 个干扰项字符串的列表；出错或解析失败时返回空列表。

    实现细节：
        - prompt 要求干扰项与正确答案格式相似但内容错误。
        - 若解析结果不足 3 项则返回空列表。
    """
    prompt = _GENERATE_DISTRACTORS_PROMPT_TEMPLATE.format(
        passage=passage,
        question=question,
        answer=answer,
    )
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            extra_body={"thinking": {"type": "disabled"}},
        )
        raw: str = resp.choices[0].message.content or ""
        distractors: Optional[List] = parse_llm_json_response(raw, key="distractors")
        if not isinstance(distractors, list):
            return []
        result = [d for d in distractors if isinstance(d, str)]
        return result if len(result) >= 3 else []
    except Exception as exc:  # noqa: BLE001
        logger.warning("generate_distractors 失败: %s", exc)
        return []


def _assemble_mcq(answer: str, distractors: List[str]) -> Dict[str, Any]:
    """将正确答案与干扰项打乱后组装为 A/B/C/D 选择题格式。

    参数：
        answer:      正确答案文本。
        distractors: 3 个干扰项文本列表。

    返回：
        含 "options"（{A/B/C/D: str}）和 "correct_letter" 的字典。
    """
    all_options = [answer] + distractors[:3]
    random.shuffle(all_options)
    labels = ["A", "B", "C", "D"]
    options: Dict[str, str] = {labels[i]: all_options[i] for i in range(4)}
    correct_letter = next(k for k, v in options.items() if v == answer)
    return {"options": options, "correct_letter": correct_letter}


# ─────────────────────────────────────────────────────────────────────────────
# Step 编排（批量异步）
# ─────────────────────────────────────────────────────────────────────────────

_SEMAPHORE_SIZE = 5
_BATCH_SIZE = 50  # 每批并发处理的条目数


def _load_done_ids(output_path: Path) -> set:
    """从已有输出文件中加载已完成的 ID 集合（断点续跑用）。"""
    done = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    done.add(obj.get("id", ""))
                    if "article_id" in obj:
                        done.add(obj["article_id"])
                    if "passage_id" in obj:
                        done.add(obj["passage_id"])
                except json.JSONDecodeError:
                    pass
    return done


def _append_jsonl(item: Dict, path: Path) -> None:
    """追加一条记录到 JSONL 文件。"""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# 元数据关键词黑名单（小写匹配）
_METADATA_KEYWORDS: Tuple[str, ...] = (
    "produced by",
    "edited by",
    "correction:",
    "erratum",
    "subscribe to",
    "this podcast",
    "follow us",
    "send us",
    "support the show",
    "transcript by",
    "host:",
    "guest:",
)

# 2025+ 年份正则
_YEAR_2025_PLUS_RE = re.compile(r"\b(202[5-9]|20[3-9]\d)\b")


def _is_valid_passage(passage: str) -> bool:
    """对 LLM 提取的段落执行后置质量过滤。

    拒绝条件（满足任一即拒绝）：
    - 长度 < 100 字符
    - 包含元数据关键词（制作信息、勘误等）
    - 不含任何 2025 或更晚年份

    参数：
        passage: 待检查的段落字符串。

    返回：
        True 表示段落通过质量检查，False 表示应被丢弃。
    """
    if len(passage) < 100:
        return False

    lower = passage.lower()
    for kw in _METADATA_KEYWORDS:
        if kw in lower:
            return False

    if not _YEAR_2025_PLUS_RE.search(passage):
        return False

    return True


async def run_step1(articles_path: Path, output_path: Path) -> int:
    """批量处理文章，提取段落，逐条追加写入 passages.jsonl。

    支持断点续跑：已处理的 article_id 会被跳过。
    """
    articles = _load_jsonl(articles_path)
    done_ids = _load_done_ids(output_path)
    todo = [a for a in articles if a.get("id", "") not in done_ids]
    logger.info("Step 1: %d 篇文章，已完成 %d，待处理 %d", len(articles), len(articles) - len(todo), len(todo))

    if not todo:
        return len(done_ids)

    client = _make_llm_client()
    sem = asyncio.Semaphore(_SEMAPHORE_SIZE)
    count = len(done_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async def _process(article: Dict) -> List[Dict]:
        async with sem:
            passages = await extract_passages_from_article(
                client, article.get("body", ""), model=LLM_MODEL
            )
            results = []
            for p in passages:
                if not _is_valid_passage(p):
                    continue
                results.append({
                    "id": str(uuid.uuid4()),
                    "article_id": article.get("id", ""),
                    "source": article.get("source", ""),
                    "category": article.get("category", ""),
                    "date": article.get("date", ""),
                    "url": article.get("url", ""),
                    "passage": p,
                })
            return results

    pbar = tqdm(total=len(todo), desc="[Step1] 提取段落", unit="篇")
    for i in range(0, len(todo), _BATCH_SIZE):
        batch = todo[i : i + _BATCH_SIZE]
        batch_results = await asyncio.gather(*[_process(a) for a in batch], return_exceptions=True)
        for res in batch_results:
            if isinstance(res, list):
                for item in res:
                    _append_jsonl(item, output_path)
                    count += 1
            elif isinstance(res, Exception):
                logger.warning("Step 1 异常: %s", res)
            pbar.update(1)
    pbar.close()

    logger.info("Step 1 完成：%d 篇文章 → %d 段落", len(articles), count)
    return count


async def run_step2(
    passages_path: Path,
    output_path: Path,
    qa_per_passage: int = 3,
) -> int:
    """批量处理段落，生成问答对，逐条追加写入 qa_raw.jsonl。

    支持断点续跑：已处理的 passage_id 会被跳过。
    """
    passages = _load_jsonl(passages_path)
    done_ids = _load_done_ids(output_path)
    todo = [p for p in passages if p.get("id", "") not in done_ids]
    logger.info("Step 2: %d 段落，已完成 %d，待处理 %d（每段 %d QA）",
                len(passages), len(passages) - len(todo), len(todo), qa_per_passage)

    if not todo:
        existing = _load_jsonl(output_path)
        return len(existing)

    client = _make_llm_client()
    sem = asyncio.Semaphore(_SEMAPHORE_SIZE)
    count = len(_load_jsonl(output_path)) if output_path.exists() else 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async def _process(item: Dict) -> List[Dict]:
        async with sem:
            qa_list = await generate_qa_from_passage(
                client, item.get("passage", ""), model=LLM_MODEL,
                qa_per_passage=qa_per_passage,
            )
            base_id = str(uuid.uuid4())
            results = []
            for idx, qa in enumerate(qa_list):
                results.append({
                    "id": f"{base_id}_{idx}",
                    "passage_id": item.get("id", ""),
                    "source": item.get("source", ""),
                    "category": item.get("category", ""),
                    "date": item.get("date", ""),
                    "url": item.get("url", ""),
                    "passage": item.get("passage", ""),
                    "question": qa["question"],
                    "answer": qa["answer"],
                })
            return results

    pbar = tqdm(total=len(todo), desc="[Step2] 生成QA", unit="段")
    for i in range(0, len(todo), _BATCH_SIZE):
        batch = todo[i : i + _BATCH_SIZE]
        batch_results = await asyncio.gather(*[_process(p) for p in batch], return_exceptions=True)
        for res in batch_results:
            if isinstance(res, list):
                for item in res:
                    _append_jsonl(item, output_path)
                    count += 1
            elif isinstance(res, Exception):
                logger.warning("Step 2 异常: %s", res)
            pbar.update(1)
    pbar.close()

    logger.info("Step 2 完成：%d 段落 → %d 问答对", len(passages), count)
    return count


async def run_step3(qa_raw_path: Path, output_path: Path) -> int:
    """批量处理问答对，生成干扰项并组装选择题，逐条追加写入 qa_full.jsonl。

    支持断点续跑：已处理的 qa ID 会被跳过。每条最多重试 3 次。
    """
    qa_items = _load_jsonl(qa_raw_path)
    done_ids = _load_done_ids(output_path)
    todo = [q for q in qa_items if q.get("id", "") not in done_ids]
    logger.info("Step 3: %d 问答对，已完成 %d，待处理 %d",
                len(qa_items), len(qa_items) - len(todo), len(todo))

    if not todo:
        existing = _load_jsonl(output_path)
        return len(existing)

    client = _make_llm_client()
    sem = asyncio.Semaphore(_SEMAPHORE_SIZE)
    count = len(_load_jsonl(output_path)) if output_path.exists() else 0
    fail_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async def _process(item: Dict) -> Optional[Dict]:
        async with sem:
            distractors: List[str] = []
            for attempt in range(3):
                try:
                    distractors = await generate_distractors(
                        client,
                        passage=item.get("passage", ""),
                        question=item.get("question", ""),
                        answer=item.get("answer", ""),
                        model=LLM_MODEL,
                    )
                except Exception as exc:
                    logger.debug("distractor 生成异常 (attempt %d): %s", attempt, exc)
                    distractors = []
                if len(distractors) >= 3:
                    break
                await asyncio.sleep(1.0 * (attempt + 1))
            if len(distractors) < 3:
                return None
            mcq = _assemble_mcq(item["answer"], distractors)
            return {
                "id": item.get("id", str(uuid.uuid4())),
                "passage_id": item.get("passage_id", ""),
                "source": item.get("source", ""),
                "category": item.get("category", ""),
                "date": item.get("date", ""),
                "url": item.get("url", ""),
                "passage": item.get("passage", ""),
                "question": item.get("question", ""),
                "correct_answer": item.get("answer", ""),
                "options": mcq["options"],
                "correct_letter": mcq["correct_letter"],
            }

    pbar = tqdm(total=len(todo), desc="[Step3] 生成选择题", unit="题")
    for i in range(0, len(todo), _BATCH_SIZE):
        batch = todo[i : i + _BATCH_SIZE]
        batch_results = await asyncio.gather(*[_process(q) for q in batch], return_exceptions=True)
        for res in batch_results:
            if isinstance(res, dict):
                _append_jsonl(res, output_path)
                count += 1
            elif isinstance(res, Exception):
                logger.warning("Step 3 异常: %s", res)
                fail_count += 1
            elif res is None:
                fail_count += 1
            pbar.update(1)
    pbar.close()

    logger.info("Step 3 完成：%d 问答对 → %d 选择题（%d 失败）", len(qa_items), count, fail_count)
    return count


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


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: OOD 零样本基线检查
# ─────────────────────────────────────────────────────────────────────────────


def run_step4_ood_check(
    qa_full_path: Path,
    model_name: str = "Qwen/Qwen3-4B",
    sample_size: int = 500,
    device: str = "cuda:0",
) -> float:
    """对生成的选择题进行 OOD 零样本基线检查。

    参数：
        qa_full_path: 完整选择题 JSONL 路径。
        model_name:   使用的模型名称（默认 Qwen/Qwen3-4B）。
        sample_size:  随机采样数量（默认 500）。
        device:       PyTorch 设备（默认 cuda:0）。

    返回：
        零样本准确率（float，0-1）。期望约 0.25（随机基线）。
        若准确率 > 0.40 则发出警告（数据可能太简单）。

    实现细节：
        - 构建零样本 prompt，格式化选项 A/B/C/D。
        - 取 logits 中 A/B/C/D token 概率最高者作为预测。
        - 使用后清理 GPU 显存。
    """
    import gc  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    qa_items = _load_jsonl(qa_full_path)
    if len(qa_items) > sample_size:
        qa_items = random.sample(qa_items, sample_size)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    label_ids = {
        label: tokenizer.encode(label, add_special_tokens=False)[0]
        for label in ["A", "B", "C", "D"]
    }

    correct = 0
    for item in qa_items:
        opts = item.get("options", {})
        opts_text = "\n".join(f"{k}. {v}" for k, v in sorted(opts.items()))
        prompt = (
            f"Question: {item['question']}\n{opts_text}\n"
            "Answer with a single letter (A, B, C, or D):"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        probs = {k: logits[v].item() for k, v in label_ids.items()}
        predicted = max(probs, key=probs.__getitem__)
        if predicted == item.get("correct_letter"):
            correct += 1

    accuracy = correct / len(qa_items) if qa_items else 0.0

    # 清理 GPU 显存
    del model
    torch.cuda.empty_cache()
    gc.collect()

    if accuracy > 0.40:
        logger.warning(
            "OOD 准确率 %.1f%% > 40%%，数据可能太简单或存在数据泄露！",
            accuracy * 100,
        )
    else:
        logger.info("OOD 零样本准确率: %.1f%%（期望 ~25%%）", accuracy * 100)

    return accuracy


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: 转换为 v2 Schema
# ─────────────────────────────────────────────────────────────────────────────


def run_step5_convert(
    qa_full_path: Path,
    output_train_path: Path,
    output_test_path: Path,
    compression_ratio: float = 0.25,
    test_ratio: float = 0.2,
    llmlingua_model: str = "models/llmlingua-2-xlm-roberta-large",
) -> Tuple[int, int]:
    """将 qa_full.jsonl 压缩并转换为 v2 schema，拆分 train/test。

    参数：
        qa_full_path:      完整选择题 JSONL 路径。
        output_train_path: 训练集输出路径。
        output_test_path:  测试集输出路径。
        compression_ratio: LLMLingua 目标压缩比（0.25 = 压缩至原始长度的 25%）。
        test_ratio:        测试集比例（默认 0.2）。
        llmlingua_model:   LLMLingua 模型路径。

    返回：
        (train_count, test_count) 元组，分别为训练集和测试集样本数。

    实现细节：
        - 使用 LLMLingua-2 对 passage 进行 4x 压缩（ratio=0.25）。
        - 随机 80/20 拆分（固定随机种子 42）。
        - 输出 v2 schema：key/question/correct_answer/options/original_text/
          compressed_text/dataset/split/category。
    """
    from llmlingua import PromptCompressor  # noqa: PLC0415

    qa_items = _load_jsonl(qa_full_path)
    compressor = PromptCompressor(
        model_name=llmlingua_model,
        use_llmlingua2=True,
    )

    def _compress(text: str) -> str:
        """调用 LLMLingua 压缩单段文本。"""
        try:
            result = compressor.compress_prompt(
                text,
                rate=compression_ratio,
                force_tokens=["\n"],
            )
            return result.get("compressed_prompt", text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLMLingua 压缩失败: %s", exc)
            return text

    converted: List[Dict] = []
    for item in qa_items:
        original_text = item.get("passage", "")
        compressed_text = _compress(original_text)
        converted.append(
            {
                "key": item.get("id", str(uuid.uuid4())),
                "question": item.get("question", ""),
                "correct_answer": item.get("correct_answer", ""),
                "options": item.get("options", {}),
                "original_text": original_text,
                "compressed_text": compressed_text,
                "dataset": "news",
                "category": item.get("category", "general"),
                "url": item.get("url", ""),
                "split": None,  # 稍后填充
            }
        )

    # 80/20 拆分
    random.seed(42)
    random.shuffle(converted)
    split_idx = int(len(converted) * (1 - test_ratio))
    train_items = converted[:split_idx]
    test_items = converted[split_idx:]

    for item in train_items:
        item["split"] = "train"
    for item in test_items:
        item["split"] = "test"

    _save_jsonl(train_items, output_train_path)
    _save_jsonl(test_items, output_test_path)

    logger.info(
        "Step 5 完成：%d 训练样本，%d 测试样本",
        len(train_items),
        len(test_items),
    )
    return len(train_items), len(test_items)


# ─────────────────────────────────────────────────────────────────────────────
# 步骤解析与主流水线
# ─────────────────────────────────────────────────────────────────────────────


def _parse_steps(steps: str) -> List[int]:
    """将步骤字符串解析为整数列表。

    参数：
        steps: 步骤描述字符串，如 "0-5"、"1,3,5" 或 "2"。

    返回：
        去重排序后的步骤整数列表。

    示例：
        "0-5" → [0, 1, 2, 3, 4, 5]
        "1,3" → [1, 3]
        "2"   → [2]
    """
    result: List[int] = []
    for part in steps.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


async def run_pipeline(
    steps: str = "0-5",
    n_articles: int = 3000,
    resume: bool = False,
    qa_per_passage: int = 3,
) -> None:
    """编排完整的新闻问答数据集构建流水线。

    参数：
        steps:           要运行的步骤，如 "0-5"、"1-3"。
        n_articles:      Step 0 目标爬取文章数（默认 3000）。
        resume:          若为 True，则跳过输出文件已存在的步骤。
        qa_per_passage:  Step 2 每个段落期望生成的 QA 对数量（默认 3）。
    """
    step_list = _parse_steps(steps)
    logger.info("将运行步骤: %s", step_list)

    if 0 in step_list:
        if resume and raw_path.exists():
            logger.info("Step 0 跳过（输出文件已存在）")
        else:
            from tools.news_crawlers import crawl_all_sites  # noqa: PLC0415

            logger.info("Step 0: 爬取新闻文章（目标 %d 篇）…", n_articles)
            count = await crawl_all_sites(
                output_path=str(raw_path),
                n_articles=n_articles,
            )
            logger.info("Step 0 完成：写入 %d 篇文章", count)

    if 1 in step_list:
        if resume and passages_path.exists():
            logger.info("Step 1 跳过（输出文件已存在）")
        else:
            logger.info("Step 1: 提取知识段落…")
            await run_step1(raw_path, passages_path)

    if 2 in step_list:
        if resume and qa_raw_path.exists():
            logger.info("Step 2 跳过（输出文件已存在）")
        else:
            logger.info("Step 2: 生成问答对（每段落 %d 个）…", qa_per_passage)
            await run_step2(passages_path, qa_raw_path, qa_per_passage=qa_per_passage)

    if 3 in step_list:
        if resume and qa_full_path.exists():
            logger.info("Step 3 跳过（输出文件已存在）")
        else:
            logger.info("Step 3: 生成干扰项…")
            await run_step3(qa_raw_path, qa_full_path)

    # 去重
    dedup_path = DATA_DIR / "qa_full_dedup.jsonl"
    if any(s in step_list for s in [1, 2, 3]) or not dedup_path.exists():
        logger.info("执行去重…")
        run_dedup(qa_full_path, dedup_path)
    else:
        logger.info("去重跳过（qa_full_dedup.jsonl 已存在且无上游更新）")

    if 4 in step_list:
        logger.info("Step 4: OOD 零样本基线检查…")
        run_step4_ood_check(dedup_path)

    if 5 in step_list:
        if resume and train_path.exists() and test_path.exists():
            logger.info("Step 5 跳过（输出文件已存在）")
        else:
            logger.info("Step 5: 压缩并转换为 v2 schema…")
            run_step5_convert(dedup_path, train_path, test_path)

    logger.info("流水线执行完毕。")


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="新闻问答数据集构建流水线（Step 0-5）")
    parser.add_argument(
        "--steps",
        default="0-5",
        help="要运行的步骤，如 '0-5'、'1-3'、'2,4'（默认: 0-5）",
    )
    parser.add_argument(
        "--n-articles",
        type=int,
        default=3000,
        help="Step 0 目标爬取文章数（默认: 3000）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="跳过输出文件已存在的步骤",
    )
    parser.add_argument(
        "--qa-per-passage",
        type=int,
        default=5,
        help="Step 2 每个段落生成的 QA 对数量（默认: 5）",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = _build_arg_parser()
    args = parser.parse_args()
    asyncio.run(
        run_pipeline(
            steps=args.steps,
            n_articles=args.n_articles,
            resume=args.resume,
            qa_per_passage=args.qa_per_passage,
        )
    )
