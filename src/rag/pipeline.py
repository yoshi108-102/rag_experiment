from __future__ import annotations

from dataclasses import dataclass

from src.rag.loader import load_knowledge_records
from src.rag.models import NoveltyDecision, RetrievalResult
from src.rag.novelty import judge_novelty
from src.rag.retriever import retrieve_similar
from src.rag.store import save_pending_reflection


@dataclass(frozen=True)
class RagAnalysis:
    enabled: bool
    retrieved: list[RetrievalResult]
    novelty: NoveltyDecision | None
    saved_pending: bool = False
    skipped_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "retrieved": [item.to_dict() for item in self.retrieved],
            "novelty": self.novelty.to_dict() if self.novelty else None,
            "saved_pending": self.saved_pending,
            "skipped_reason": self.skipped_reason,
        }


def analyze_with_rag(
    user_input: str,
    route: str,
    *,
    allowed_routes: tuple[str, ...] = ("DEEPEN", "CLARIFY"),
    top_k: int = 3,
) -> RagAnalysis:
    if route not in allowed_routes:
        return RagAnalysis(
            enabled=False,
            retrieved=[],
            novelty=None,
            skipped_reason=f"route {route} not eligible",
        )

    records = load_knowledge_records()
    if not records:
        return RagAnalysis(
            enabled=False,
            retrieved=[],
            novelty=None,
            skipped_reason="no knowledge records found",
        )

    retrieved = retrieve_similar(user_input, records, top_k=top_k)
    novelty = judge_novelty(user_input, retrieved)
    saved_pending = False

    if novelty.is_novel:
        saved_pending = save_pending_reflection(
            user_input=user_input,
            route=route,
            novelty=novelty,
            retrieved=retrieved,
        )

    return RagAnalysis(
        enabled=True,
        retrieved=retrieved,
        novelty=novelty,
        saved_pending=saved_pending,
    )
