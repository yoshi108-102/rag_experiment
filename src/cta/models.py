"""CTA interview domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


GenerationMode = Literal["TEMPLATE_RANDOM", "HYBRID_LLM"]
SessionStatus = Literal["ACTIVE", "FINISHED"]


@dataclass
class SubjectPlan:
    """A subject and its interview topics."""

    name: str
    topics: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        cleaned_topics = [topic.strip() for topic in self.topics if topic and topic.strip()]
        self.topics = cleaned_topics or ["全般"]


@dataclass(frozen=True)
class AnswerFeatures:
    """Features extracted from a user answer."""

    keywords: list[str]
    has_positive: bool
    has_negative: bool
    has_question: bool
    clause_count: int
    cognitive_action_label: str


@dataclass
class CTASessionState:
    """Mutable session state for in-memory interview execution."""

    session_id: str
    user_id: str | None
    generation_mode: GenerationMode
    subjects: list[SubjectPlan]
    status: SessionStatus = "ACTIVE"
    subject_index: int = 0
    topic_index: int = 0
    turn_count: int = 0
    topic_turn_count: int = 0
    last_question_type: str | None = None

    @property
    def current_subject(self) -> SubjectPlan:
        return self.subjects[self.subject_index]

    @property
    def current_subject_name(self) -> str:
        return self.current_subject.name

    @property
    def current_topic_name(self) -> str:
        return self.current_subject.topics[self.topic_index]

    def has_next_topic(self) -> bool:
        return self.topic_index < len(self.current_subject.topics) - 1

    def has_next_subject(self) -> bool:
        return self.subject_index < len(self.subjects) - 1

    def advance_topic(self) -> bool:
        if not self.has_next_topic():
            return False
        self.topic_index += 1
        self.topic_turn_count = 0
        return True

    def advance_subject(self) -> bool:
        if not self.has_next_subject():
            return False
        self.subject_index += 1
        self.topic_index = 0
        self.topic_turn_count = 0
        return True


@dataclass(frozen=True)
class CTATurnRecord:
    """One completed assistant turn."""

    turn_index: int
    user_text: str | None
    assistant_text: str
    question_type: str
    backchannel_type: str | None
    generation_mode: GenerationMode
    fallback_used: bool
    subject_name: str
    topic_name: str
    keywords: list[str] = field(default_factory=list)
    has_positive: bool = False
    has_negative: bool = False
    has_question: bool = False
    clause_count: int = 0
    cognitive_action_label: str = "other"


@dataclass(frozen=True)
class GenerationTrace:
    """Trace for LLM generation path."""

    turn_index: int
    model_name: str
    prompt_version: str
    latency_ms: int
    fallback_used: bool
    error: str | None = None


@dataclass(frozen=True)
class CTAResponse:
    """API response for session start / user turn handling."""

    session_id: str
    assistant_text: str
    question_type: str
    backchannel_type: str | None
    generation_mode: GenerationMode
    fallback_used: bool
    status: SessionStatus
    subject_name: str
    topic_name: str

