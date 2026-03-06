"""CTA Sprint 1 interview engine."""

from __future__ import annotations

from dataclasses import dataclass
import time
import uuid

from src.cta.features import AnswerFeatureExtractor
from src.cta.knowledge import extract_knowledge_candidates
from src.cta.llm import LLMNaturalizer
from src.cta.models import (
    AnswerFeatures,
    CTAResponse,
    CTASessionState,
    CTATurnRecord,
    GenerationMode,
    GenerationTrace,
    SubjectPlan,
)
from src.cta.store import InMemoryCTAStore
from src.cta.templates import TemplateRepository


@dataclass(frozen=True)
class _Decision:
    question_type: str
    reason: str
    advance: str | None = None  # "TOPIC" | "SUBJECT"
    finish: bool = False


class CTAInterviewEngine:
    """In-memory CTA interview engine covering Sprint 1 scope."""

    def __init__(
        self,
        store: InMemoryCTAStore | None = None,
        feature_extractor: AnswerFeatureExtractor | None = None,
        templates: TemplateRepository | None = None,
        llm_naturalizer: LLMNaturalizer | None = None,
        topic_turn_limit: int = 2,
        template_seed: int = 7,
    ) -> None:
        self.store = store or InMemoryCTAStore()
        self.feature_extractor = feature_extractor or AnswerFeatureExtractor()
        self.templates = templates or TemplateRepository(seed=template_seed)
        self.llm_naturalizer = llm_naturalizer or LLMNaturalizer()
        self.topic_turn_limit = topic_turn_limit
        self.template_seed = self.templates.seed
        self._session_templates: dict[str, TemplateRepository] = {}

    def start_session(
        self,
        subjects: list[SubjectPlan] | None = None,
        generation_mode: GenerationMode = "TEMPLATE_RANDOM",
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> CTAResponse:
        subject_plan = subjects or [
            SubjectPlan(
                name="業務での判断",
                topics=["状況把握", "判断基準", "実行した行動"],
            )
        ]
        normalized_subjects = [SubjectPlan(plan.name, list(plan.topics)) for plan in subject_plan]
        sid = session_id or uuid.uuid4().hex

        session_templates = TemplateRepository(seed=self.template_seed)
        self._session_templates[sid] = session_templates

        session = CTASessionState(
            session_id=sid,
            user_id=user_id,
            generation_mode=generation_mode,
            subjects=normalized_subjects,
            random_seed=self.template_seed,
        )
        self.store.create_session(session)

        response = self._generate_response(
            session=session,
            user_text=None,
            features=AnswerFeatures(
                keywords=[],
                has_positive=False,
                has_negative=False,
                has_question=False,
                clause_count=0,
                cognitive_action_label="other",
            ),
            question_type="STD1",
            backchannel_type=None,
            decision_reason="session-start",
            session_templates=session_templates,
        )
        session.turn_count += 1
        session.last_question_type = "STD1"
        return response

    def set_generation_mode(self, session_id: str, generation_mode: GenerationMode) -> None:
        session = self.store.get_session(session_id)
        if session.status == "FINISHED":
            raise ValueError("cannot change mode for finished session")
        session.generation_mode = generation_mode

    def handle_user_input(self, session_id: str, user_text: str) -> CTAResponse:
        session = self.store.get_session(session_id)
        if session.status == "FINISHED":
            raise ValueError("session is already finished")
        session_templates = self._session_templates.get(session_id)
        if session_templates is None:
            session_templates = TemplateRepository(seed=session.random_seed)
            self._session_templates[session_id] = session_templates

        turn_started_at = time.perf_counter()
        cleaned = (user_text or "").strip()
        features = self.feature_extractor.extract(cleaned)
        if cleaned:
            session.topic_turn_count += 1

        decision = self._decide_next(session, cleaned, features)
        if decision.advance == "TOPIC":
            session.advance_topic()
        elif decision.advance == "SUBJECT":
            session.advance_subject()
        if decision.finish:
            session.status = "FINISHED"
            session.topic_turn_count = 0

        backchannel_type = session_templates.select_backchannel(
            turn_count=session.turn_count,
            question_type=decision.question_type,
        )
        response = self._generate_response(
            session=session,
            user_text=cleaned or None,
            features=features,
            question_type=decision.question_type,
            backchannel_type=backchannel_type,
            decision_reason=decision.reason,
            session_templates=session_templates,
            turn_started_at=turn_started_at,
        )
        session.turn_count += 1
        session.last_question_type = decision.question_type
        if session.status == "FINISHED":
            self._finalize_session(session.session_id)
        return response

    def _decide_next(
        self,
        session: CTASessionState,
        user_text: str,
        features: AnswerFeatures,
    ) -> _Decision:
        if not user_text:
            return _Decision(question_type="STD4", reason="empty-answer")

        if self._is_finish_intent(user_text):
            return _Decision(
                question_type="STD11",
                reason="finish-intent",
                finish=True,
            )

        if features.has_negative:
            if session.has_next_topic():
                return _Decision(
                    question_type="STD7",
                    reason="negative-answer-advance-topic",
                    advance="TOPIC",
                )
            if session.has_next_subject():
                return _Decision(
                    question_type="STD3",
                    reason="negative-answer-advance-subject",
                    advance="SUBJECT",
                )
            return _Decision(
                question_type="STD11",
                reason="negative-answer-no-remaining-scope",
                finish=True,
            )

        if session.topic_turn_count >= self.topic_turn_limit:
            if session.has_next_topic():
                return _Decision(
                    question_type="STD7",
                    reason="topic-turn-limit-advance-topic",
                    advance="TOPIC",
                )
            if session.has_next_subject():
                return _Decision(
                    question_type="STD3",
                    reason="topic-turn-limit-advance-subject",
                    advance="SUBJECT",
                )
            return _Decision(
                question_type="STD11",
                reason="topic-turn-limit-finish",
                finish=True,
            )

        if features.has_question:
            return _Decision(question_type="STD5", reason="user-asked-question")

        if features.keywords and session.last_question_type in {"CDM1", "CDM2"}:
            return _Decision(question_type="CDM3", reason="keywords-after-cdm")
        if features.keywords and features.clause_count >= 2:
            return _Decision(question_type="CDM2", reason="keywords-and-rich-clauses")
        if features.keywords:
            return _Decision(question_type="CDM1", reason="keywords-detected")
        return _Decision(question_type="STD4", reason="fallback-clarify")

    def _generate_response(
        self,
        session: CTASessionState,
        user_text: str | None,
        features: AnswerFeatures,
        question_type: str,
        backchannel_type: str | None,
        decision_reason: str,
        session_templates: TemplateRepository,
        turn_started_at: float | None = None,
    ) -> CTAResponse:
        keyword = features.keywords[0] if features.keywords else session.current_topic_name
        probe = session_templates.get_cdm_probe(features.cognitive_action_label)
        question_text = session_templates.render_question(
            question_type=question_type,
            subject=session.current_subject_name,
            topic=session.current_topic_name,
            keyword=keyword,
            probe=probe,
        )
        backchannel_text = session_templates.render_backchannel(backchannel_type)
        template_response = session_templates.compose_response(question_text, backchannel_text)

        assistant_text = template_response
        fallback_used = False
        latency_ms = 0
        trace_error: str | None = None
        if session.generation_mode == "HYBRID_LLM":
            begin = time.perf_counter()
            try:
                assistant_text = self.llm_naturalizer.naturalize(
                    template_response=template_response,
                    session=session,
                    question_type=question_type,
                )
            except Exception as exc:
                fallback_used = True
                trace_error = f"{type(exc).__name__}: {exc}"
                assistant_text = template_response
            latency_ms = int((time.perf_counter() - begin) * 1000)
            self.store.add_generation_trace(
                session.session_id,
                GenerationTrace(
                    turn_index=session.turn_count + 1,
                    model_name=self.llm_naturalizer.model_name,
                    prompt_version=self.llm_naturalizer.prompt_version,
                    latency_ms=latency_ms,
                    fallback_used=fallback_used,
                    error=trace_error,
                ),
            )

        processing_latency_ms = 0
        if turn_started_at is not None:
            processing_latency_ms = int((time.perf_counter() - turn_started_at) * 1000)

        self.store.save_turn(
            session.session_id,
            CTATurnRecord(
                turn_index=session.turn_count + 1,
                user_text=user_text,
                assistant_text=assistant_text,
                question_type=question_type,
                backchannel_type=backchannel_type,
                generation_mode=session.generation_mode,
                fallback_used=fallback_used,
                decision_reason=decision_reason,
                processing_latency_ms=processing_latency_ms,
                subject_name=session.current_subject_name,
                topic_name=session.current_topic_name,
                keywords=features.keywords,
                has_positive=features.has_positive,
                has_negative=features.has_negative,
                has_question=features.has_question,
                clause_count=features.clause_count,
                cognitive_action_label=features.cognitive_action_label,
            ),
        )
        return CTAResponse(
            session_id=session.session_id,
            assistant_text=assistant_text,
            question_type=question_type,
            backchannel_type=backchannel_type,
            generation_mode=session.generation_mode,
            fallback_used=fallback_used,
            status=session.status,
            subject_name=session.current_subject_name,
            topic_name=session.current_topic_name,
        )

    def _is_finish_intent(self, user_text: str) -> bool:
        finish_markers = ("終了", "終わり", "以上", "これで大丈夫")
        return any(marker in user_text for marker in finish_markers)

    def _finalize_session(self, session_id: str) -> None:
        existing = self.store.list_knowledge_candidates(session_id)
        if existing:
            return
        turns = self.store.list_turns(session_id)
        candidates = extract_knowledge_candidates(session_id, turns)
        self.store.save_knowledge_candidates(session_id, candidates)
