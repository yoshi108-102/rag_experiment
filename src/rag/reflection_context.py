from __future__ import annotations

from dataclasses import dataclass

from src.rag.embedding_search import (
    EmbeddingRetrieverError,
    search_similar_records_with_embeddings,
)
from src.rag.knowledge_reader import load_consolidated_knowledge_records
from src.rag.models import NoveltyDecision, RetrievalResult
from src.rag.novelty_rules import assess_novelty
from src.rag.pending_reflection_store import store_pending_reflection
from src.rag.record_search import search_similar_records


@dataclass(frozen=True)
class ReflectionContextAnalysis:
    enabled: bool
    retrieved: list[RetrievalResult]
    novelty: NoveltyDecision | None
    saved_pending: bool = False
    skipped_reason: str | None = None
    retrieval_method: str | None = None
    retrieval_note: str | None = None

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "retrieved": [item.to_dict() for item in self.retrieved],
            "novelty": self.novelty.to_dict() if self.novelty else None,
            "saved_pending": self.saved_pending,
            "skipped_reason": self.skipped_reason,
            "retrieval_method": self.retrieval_method,
            "retrieval_note": self.retrieval_note,
        }


def analyze_reflection_context(
    user_input: str,
    route: str,
    *,
    allowed_routes: tuple[str, ...] = ("DEEPEN", "CLARIFY"),
    top_k: int = 3,
) -> ReflectionContextAnalysis:
    if route not in allowed_routes:
        return ReflectionContextAnalysis(
            enabled=False,
            retrieved=[],
            novelty=None,
            skipped_reason=f"route {route} not eligible",
        )

    records = load_consolidated_knowledge_records()
    if not records:
        return ReflectionContextAnalysis(
            enabled=False,
            retrieved=[],
            novelty=None,
            skipped_reason="no knowledge records found",
        )

    retrieval_method = "openai-embedding"
    retrieval_note = None

    try:
        embedding_result = search_similar_records_with_embeddings(
            user_input,
            records,
            top_k=top_k,
        )
        retrieved = embedding_result.results
        retrieval_method = f"openai-embedding:{embedding_result.model}"
    except EmbeddingRetrieverError as exc:
        retrieved = search_similar_records(user_input, records, top_k=top_k)
        retrieval_method = "ngram-fallback"
        retrieval_note = str(exc)
    except Exception as exc:
        retrieved = search_similar_records(user_input, records, top_k=top_k)
        retrieval_method = "ngram-fallback"
        retrieval_note = f"embedding retrieval failed: {exc.__class__.__name__}"

    novelty = assess_novelty(user_input, retrieved)
    saved_pending = False

    if novelty.is_novel:
        saved_pending = store_pending_reflection(
            user_input=user_input,
            route=route,
            novelty=novelty,
            retrieved=retrieved,
        )

    return ReflectionContextAnalysis(
        enabled=True,
        retrieved=retrieved,
        novelty=novelty,
        saved_pending=saved_pending,
        retrieval_method=retrieval_method,
        retrieval_note=retrieval_note,
    )


# Backward-compatible aliases while callers migrate.
RagAnalysis = ReflectionContextAnalysis
analyze_with_rag = analyze_reflection_context
