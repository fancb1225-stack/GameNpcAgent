from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

T = TypeVar("T")

CHAPTER_TITLE_PATTERN = re.compile(
    r"^\s*(?:#{1,6}\s*)?"
    r"("
    r"(?:第[一二三四五六七八九十百千万零〇两\d]+[章节卷部篇回].*)"
    r"|(?:[一二三四五六七八九十百千万零〇两\d]+[、.．]\s*.+)"
    r"|(?:Chapter\s+\d+.*)"
    r")\s*$",
    re.IGNORECASE,
)

SENTENCE_PUNCTUATION = set("，。！？；：,.!?;:")


def split_text_by_chapter(
    text: str,
    max_chunk_size: int = 2000,
    overlap: int = 120,
) -> list[str]:
    """按章节标题切分文本，单章过长时再做带重叠的窗口切分。"""

    clean_text = (text or "").strip()
    if not clean_text:
        return []

    # 先按显式章节标题聚合，避免把同一章节打散成过碎片段。
    lines = [raw_line.strip() for raw_line in clean_text.splitlines()]
    sections: list[list[str]] = []
    current: list[str] = []
    for index, line in enumerate(lines):
        if not line:
            if current:
                current.append("")
            continue
        if _is_chapter_title(line, _next_content_line(lines, index)):
            if _section_has_body(current):
                sections.append(current)
            current = [line.lstrip("#").strip()]
        else:
            current.append(line.lstrip("#").strip() if not current else line)
    if _section_has_body(current):
        sections.append(current)

    # 没有识别出章节时，将整篇作为一个章节，再按长度兜底切分。
    chunks: list[str] = []
    for section in sections:
        section_text = "\n".join(section).strip()
        chunks.extend(_split_large_section(section_text, max_chunk_size, overlap))
    return [chunk for chunk in chunks if chunk]


def _next_content_line(lines: Sequence[str], current_index: int) -> str:
    """查找下一条非空行，辅助判断短标题是否真的是小节标题。"""

    for line in lines[current_index + 1:]:
        if line.strip():
            return line.strip()
    return ""


def _is_chapter_title(line: str, next_line: str = "") -> bool:
    """识别显式章节标题和 PDF 抽取后常见的短小节标题。"""

    clean_line = line.lstrip("#").strip()
    if CHAPTER_TITLE_PATTERN.match(line):
        return True
    if not clean_line or not next_line:
        return False
    if len(clean_line) > 24 or any(char in SENTENCE_PUNCTUATION for char in clean_line):
        return False
    if CHAPTER_TITLE_PATTERN.match(next_line):
        return False
    return any("\u4e00" <= char <= "\u9fff" for char in clean_line)


def _section_has_body(section: Sequence[str]) -> bool:
    """避免把只有父级标题、没有正文的小节写成无意义 chunk。"""

    if not section:
        return False
    first_line = section[0].strip()
    if CHAPTER_TITLE_PATTERN.match(first_line):
        return any(line.strip() for line in section[1:])
    return any(line.strip() for line in section)


def rank_hybrid_results(
    items: Sequence[T],
    query: str,
    content_getter: Callable[[T], str],
    distance_getter: Callable[[T], float | None],
    limit: int = 5,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
) -> list[dict[str, Any]]:
    """使用 BM25 和余弦相似度做加权融合排序。"""

    normalized_limit = max(1, int(limit))
    candidates = list(items)
    if not candidates:
        return []

    # 计算 BM25 原始分，并归一化到 0 到 1。
    contents = [content_getter(item) for item in candidates]
    bm25_scores = _bm25_scores(query, contents)
    max_bm25 = max(bm25_scores) if bm25_scores else 0.0

    ranked: list[dict[str, Any]] = []
    for item, bm25_score in zip(candidates, bm25_scores, strict=True):
        distance = distance_getter(item)
        vector_score = _distance_to_similarity(distance)
        normalized_bm25 = bm25_score / max_bm25 if max_bm25 > 0 else 0.0
        hybrid_score = (bm25_weight * normalized_bm25) + (vector_weight * vector_score)
        ranked.append(
            {
                "item": item,
                "hybrid_score": hybrid_score,
                "bm25_score": normalized_bm25,
                "vector_score": vector_score,
                "distance": distance,
            }
        )

    return sorted(ranked, key=lambda row: row["hybrid_score"], reverse=True)[:normalized_limit]


def _split_large_section(text: str, max_chunk_size: int, overlap: int) -> list[str]:
    """章节过长时使用固定窗口兜底，保持相邻片段有少量上下文。"""

    max_size = max(1, int(max_chunk_size))
    safe_overlap = max(0, min(int(overlap), max_size - 1))
    if len(text) <= max_size:
        return [text]

    chunks: list[str] = []
    step = max(1, max_size - safe_overlap)
    for start in range(0, len(text), step):
        chunk = text[start:start + max_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _bm25_scores(query: str, documents: Sequence[str]) -> list[float]:
    """计算轻量 BM25 分数，避免引入额外检索依赖。"""

    query_tokens = _tokenize(query)
    tokenized_docs = [_tokenize(document) for document in documents]
    if not query_tokens or not tokenized_docs:
        return [0.0 for _ in documents]

    doc_count = len(tokenized_docs)
    avg_doc_len = sum(len(doc) for doc in tokenized_docs) / max(doc_count, 1)
    doc_freq: Counter[str] = Counter()
    for doc in tokenized_docs:
        doc_freq.update(set(doc))

    k1 = 1.5
    b = 0.75
    scores: list[float] = []
    for doc in tokenized_docs:
        frequencies = Counter(doc)
        doc_len = len(doc) or 1
        score = 0.0
        for token in query_tokens:
            frequency = frequencies.get(token, 0)
            if frequency == 0:
                continue
            idf = math.log(1 + (doc_count - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5))
            denominator = frequency + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1))
            score += idf * frequency * (k1 + 1) / denominator
        scores.append(score)
    return scores


def _tokenize(text: str) -> list[str]:
    """同时支持英文词、数字和中文字符的轻量分词。"""

    tokens: list[str] = []
    for match in re.finditer(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", (text or "").lower()):
        tokens.append(match.group(0))
    return tokens


def _distance_to_similarity(distance: float | None) -> float:
    """将 pgvector cosine distance 转换为 0 到 1 的相似度。"""

    if distance is None:
        return 0.0
    try:
        value = 1.0 - float(distance)
    except Exception:
        return 0.0
    return max(0.0, min(value, 1.0))
