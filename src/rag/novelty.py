from __future__ import annotations

from src.rag.models import NoveltyDecision, RetrievalResult


def judge_novelty(
    query: str,
    retrieved: list[RetrievalResult],
    novelty_threshold: float = 0.38,
) -> NoveltyDecision:
    if not query.strip():
        return NoveltyDecision(
            is_novel=False,
            confidence=0.0,
            reason="empty query",
        )

    if not retrieved:
        return NoveltyDecision(
            is_novel=True,
            confidence=0.9,
            reason="no similar records found",
        )

    top = retrieved[0]
    if top.score >= novelty_threshold:
        confidence = min(0.95, 0.5 + top.score / 2)
        return NoveltyDecision(
            is_novel=False,
            confidence=confidence,
            reason=f"top match score {top.score:.2f} >= threshold",
        )

    confidence = min(0.9, 0.55 + (novelty_threshold - top.score))
    return NoveltyDecision(
        is_novel=True,
        confidence=confidence,
        reason=f"top match score {top.score:.2f} < threshold",
    )

