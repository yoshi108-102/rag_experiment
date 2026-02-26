from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from math import sqrt
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from langchain_openai import OpenAIEmbeddings

from src.rag.models import KnowledgeRecord, RetrievalResult


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class EmbeddingRetrieverError(RuntimeError):
    """Raised when embedding retrieval cannot be executed."""


class TextEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class EmbeddingSearchResult:
    results: list[RetrievalResult]
    model: str


_RECORD_EMBEDDINGS_CACHE: dict[str, list[list[float]]] = {}


def search_similar_records_with_embeddings(
    query: str,
    records: list[KnowledgeRecord],
    *,
    top_k: int = 3,
    min_score: float = 0.15,
    model: str | None = None,
    embedder: TextEmbedder | None = None,
) -> EmbeddingSearchResult:
    normalized_query = query.strip()
    if not normalized_query:
        return EmbeddingSearchResult(results=[], model=model or _resolve_embedding_model())
    if not records:
        return EmbeddingSearchResult(results=[], model=model or _resolve_embedding_model())

    resolved_model = model or _resolve_embedding_model()
    resolved_embedder = embedder or _build_openai_embedder(resolved_model)

    query_vector = resolved_embedder.embed_query(normalized_query)
    record_vectors = _get_or_create_record_embeddings(records, resolved_model, resolved_embedder)

    results: list[RetrievalResult] = []
    for record, vector in zip(records, record_vectors, strict=False):
        score = _cosine_similarity(query_vector, vector)
        if score < min_score:
            continue
        results.append(
            RetrievalResult(
                record=record,
                score=score,
                reasons=[f"embedding-cosine:{resolved_model}"],
            )
        )

    results.sort(key=lambda item: item.score, reverse=True)
    return EmbeddingSearchResult(results=results[:top_k], model=resolved_model)


def _build_openai_embedder(model: str) -> TextEmbedder:
    if not os.getenv("OPENAI_API_KEY"):
        raise EmbeddingRetrieverError("OPENAI_API_KEY is not set")
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise EmbeddingRetrieverError("langchain_openai is not installed") from exc
    return OpenAIEmbeddings(model=model)


def _resolve_embedding_model() -> str:
    return os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL


def _get_or_create_record_embeddings(
    records: list[KnowledgeRecord],
    model: str,
    embedder: TextEmbedder,
) -> list[list[float]]:
    cache_key = _records_cache_key(records, model)
    cached = _RECORD_EMBEDDINGS_CACHE.get(cache_key)
    if cached is not None and len(cached) == len(records):
        return cached

    texts = [_record_to_embedding_text(record) for record in records]
    vectors = embedder.embed_documents(texts)
    _RECORD_EMBEDDINGS_CACHE[cache_key] = vectors
    return vectors


def _records_cache_key(records: list[KnowledgeRecord], model: str) -> str:
    payload = {
        "model": model,
        "records": [record.to_dict() for record in records],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _record_to_embedding_text(record: KnowledgeRecord) -> str:
    parts = [
        f"topic: {record.topic}",
        f"type: {record.record_type}",
        f"text: {record.text}",
    ]
    if record.tags:
        parts.append(f"tags: {', '.join(record.tags)}")
    if record.applicable_when:
        parts.append(f"applicable_when: {record.applicable_when}")
    return "\n".join(parts)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
