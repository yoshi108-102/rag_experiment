from __future__ import annotations

import re
from difflib import SequenceMatcher

from src.rag.models import KnowledgeRecord, RetrievalResult


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).strip().lower()


def _char_ngrams(text: str, n: int = 2) -> set[str]:
    normalized = _normalize(text)
    if not normalized:
        return set()
    if len(normalized) <= n:
        return {normalized}
    return {normalized[i : i + n] for i in range(len(normalized) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _tag_fragments(tags: list[str]) -> list[str]:
    fragments: list[str] = []
    for tag in tags:
        tag = tag.strip()
        if not tag:
            continue
        fragments.append(tag)
        if ":" in tag:
            _, right = tag.split(":", 1)
            right = right.strip()
            if right:
                fragments.append(right)
    return fragments


def score_record(query: str, record: KnowledgeRecord) -> tuple[float, list[str]]:
    q = _normalize(query)
    if not q:
        return 0.0, []

    reasons: list[str] = []
    score = 0.0

    q_grams = _char_ngrams(query)
    text_grams = _char_ngrams(record.text)
    topic_grams = _char_ngrams(record.topic)

    text_sim = _jaccard(q_grams, text_grams)
    topic_sim = _jaccard(q_grams, topic_grams)
    seq_sim = SequenceMatcher(None, q, _normalize(record.text)).ratio()

    score += text_sim * 0.55
    score += topic_sim * 0.25
    score += seq_sim * 0.20

    if text_sim >= 0.2:
        reasons.append("text-sim")
    if topic_sim >= 0.2:
        reasons.append("topic-sim")

    if q in _normalize(record.text) or _normalize(record.text) in q:
        score += 0.15
        reasons.append("substring")

    if q in _normalize(record.topic):
        score += 0.10
        reasons.append("topic-substring")

    tag_hits = 0
    for fragment in _tag_fragments(record.tags):
        if fragment and fragment.lower() in query.lower():
            tag_hits += 1

    if tag_hits:
        score += min(0.12, 0.04 * tag_hits)
        reasons.append(f"tag-hit:{tag_hits}")

    if record.record_type == "open_question":
        score += 0.03
        reasons.append("question-prioritized")

    return min(score, 1.0), reasons


def retrieve_similar(
    query: str,
    records: list[KnowledgeRecord],
    top_k: int = 3,
    min_score: float = 0.18,
) -> list[RetrievalResult]:
    results: list[RetrievalResult] = []
    for record in records:
        score, reasons = score_record(query, record)
        if score >= min_score:
            results.append(RetrievalResult(record=record, score=score, reasons=reasons))

    results.sort(key=lambda item: item.score, reverse=True)
    return results[:top_k]

