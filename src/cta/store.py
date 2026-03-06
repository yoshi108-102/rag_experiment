"""In-memory repositories for CTA Sprint 1."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import DefaultDict

from src.cta.models import (
    CTASessionState,
    CTATurnRecord,
    GenerationTrace,
    KnowledgeCandidate,
)


class InMemoryCTAStore:
    """In-memory store for sessions, turns and generation traces."""

    def __init__(self) -> None:
        self._sessions: dict[str, CTASessionState] = {}
        self._turns: DefaultDict[str, list[CTATurnRecord]] = defaultdict(list)
        self._generation_traces: DefaultDict[str, list[GenerationTrace]] = defaultdict(list)
        self._knowledge_candidates: DefaultDict[str, list[KnowledgeCandidate]] = defaultdict(list)

    def create_session(self, session: CTASessionState) -> None:
        if session.session_id in self._sessions:
            raise ValueError(f"session already exists: {session.session_id}")
        self._sessions[session.session_id] = session

    def get_session(self, session_id: str) -> CTASessionState:
        if session_id not in self._sessions:
            raise KeyError(f"session not found: {session_id}")
        return self._sessions[session_id]

    def list_session_ids(self) -> list[str]:
        return list(self._sessions.keys())

    def save_turn(self, session_id: str, turn: CTATurnRecord) -> None:
        _ = self.get_session(session_id)
        self._turns[session_id].append(turn)

    def list_turns(self, session_id: str) -> list[CTATurnRecord]:
        _ = self.get_session(session_id)
        return list(self._turns[session_id])

    def add_generation_trace(self, session_id: str, trace: GenerationTrace) -> None:
        _ = self.get_session(session_id)
        self._generation_traces[session_id].append(trace)

    def list_generation_traces(self, session_id: str) -> list[GenerationTrace]:
        _ = self.get_session(session_id)
        return list(self._generation_traces[session_id])

    def save_knowledge_candidates(
        self,
        session_id: str,
        candidates: list[KnowledgeCandidate],
    ) -> None:
        _ = self.get_session(session_id)
        if not candidates:
            return
        self._knowledge_candidates[session_id].extend(candidates)

    def list_knowledge_candidates(self, session_id: str) -> list[KnowledgeCandidate]:
        _ = self.get_session(session_id)
        return list(self._knowledge_candidates[session_id])

    def build_audit_snapshot(self, session_id: str) -> dict:
        session = self.get_session(session_id)
        return {
            "session": asdict(session),
            "turns": [asdict(turn) for turn in self.list_turns(session_id)],
            "generation_traces": [
                asdict(trace) for trace in self.list_generation_traces(session_id)
            ],
            "knowledge_candidates": [
                asdict(candidate)
                for candidate in self.list_knowledge_candidates(session_id)
            ],
        }
