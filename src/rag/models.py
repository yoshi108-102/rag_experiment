"""RAG処理で共通利用するデータ構造（知識レコード・検索結果・新規性判定）。"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal


RecordType = Literal["summary", "observation", "method", "open_question"]


@dataclass(frozen=True)
class KnowledgeRecord:
    """統合知識JSONから展開した1件分の知識レコード。"""

    record_id: str
    topic: str
    tags: list[str]
    record_type: RecordType
    text: str
    applicable_when: str | None = None
    source: str = "consolidated_knowledge"

    def to_dict(self) -> dict:
        """JSON出力向けに辞書へ変換する。"""
        return asdict(self)


@dataclass(frozen=True)
class RetrievalResult:
    """検索時のスコアと根拠を、対応レコードと一緒に保持する。"""

    record: KnowledgeRecord
    score: float
    reasons: list[str]

    def to_dict(self) -> dict:
        """表示・保存向けに丸め済みスコアを含む辞書へ変換する。"""
        return {
            "score": round(self.score, 4),
            "reasons": self.reasons,
            "record": self.record.to_dict(),
        }


@dataclass(frozen=True)
class NoveltyDecision:
    """入力内容の新規性判定結果。"""

    is_novel: bool
    confidence: float
    reason: str

    def to_dict(self) -> dict:
        """判定結果を辞書化し、confidenceを小数4桁へ丸める。"""
        return {
            "is_novel": self.is_novel,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
        }
