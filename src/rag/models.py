from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal


RecordType = Literal["summary", "observation", "method", "open_question"]


@dataclass(frozen=True)
class KnowledgeRecord:
    record_id: str
    topic: str
    tags: list[str]
    record_type: RecordType
    text: str
    applicable_when: str | None = None
    source: str = "consolidated_knowledge"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalResult:
    record: KnowledgeRecord
    score: float
    reasons: list[str]

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "reasons": self.reasons,
            "record": self.record.to_dict(),
        }


@dataclass(frozen=True)
class NoveltyDecision:
    is_novel: bool
    confidence: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "is_novel": self.is_novel,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
        }

