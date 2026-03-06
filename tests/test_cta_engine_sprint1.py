from __future__ import annotations

from src.cta.engine import CTAInterviewEngine
from src.cta.models import SubjectPlan


class FakeNaturalizer:
    model_name = "fake-model"
    prompt_version = "test-v1"

    def naturalize(self, template_response, session, question_type):  # type: ignore[no-untyped-def]
        del session, question_type
        return f"[LLM]{template_response}"


class FailingNaturalizer(FakeNaturalizer):
    def naturalize(self, template_response, session, question_type):  # type: ignore[no-untyped-def]
        del template_response, session, question_type
        raise TimeoutError("forced timeout")


def build_subjects() -> list[SubjectPlan]:
    return [SubjectPlan(name="業務判断", topics=["状況把握", "判断基準"])]


def test_start_session_creates_intro_turn() -> None:
    engine = CTAInterviewEngine()

    response = engine.start_session(subjects=build_subjects(), generation_mode="TEMPLATE_RANDOM")
    turns = engine.store.list_turns(response.session_id)

    assert response.question_type == "STD1"
    assert response.status == "ACTIVE"
    assert response.subject_name == "業務判断"
    assert response.topic_name == "状況把握"
    assert len(turns) == 1
    assert turns[0].question_type == "STD1"


def test_negative_answer_advances_topic() -> None:
    engine = CTAInterviewEngine()
    session = engine.start_session(subjects=build_subjects(), generation_mode="TEMPLATE_RANDOM")

    response = engine.handle_user_input(session.session_id, "いいえ、特にありません。")

    assert response.question_type == "STD7"
    assert response.topic_name == "判断基準"
    assert response.status == "ACTIVE"


def test_rich_answer_selects_cdm_question() -> None:
    engine = CTAInterviewEngine()
    session = engine.start_session(subjects=build_subjects(), generation_mode="TEMPLATE_RANDOM")

    response = engine.handle_user_input(
        session.session_id,
        "状況を確認しました。関連情報を整理して、判断基準を明確にしました。",
    )

    assert response.question_type == "CDM2"
    assert response.status == "ACTIVE"


def test_generation_mode_switch_to_hybrid_llm() -> None:
    engine = CTAInterviewEngine(llm_naturalizer=FakeNaturalizer())
    session = engine.start_session(subjects=build_subjects(), generation_mode="TEMPLATE_RANDOM")
    engine.set_generation_mode(session.session_id, "HYBRID_LLM")

    response = engine.handle_user_input(
        session.session_id,
        "現場情報を確認しました。",
    )

    assert response.generation_mode == "HYBRID_LLM"
    assert response.fallback_used is False
    assert response.assistant_text.startswith("[LLM]")
    traces = engine.store.list_generation_traces(session.session_id)
    assert len(traces) == 1
    assert traces[0].fallback_used is False


def test_hybrid_llm_fallback_on_error() -> None:
    engine = CTAInterviewEngine(llm_naturalizer=FailingNaturalizer())

    start = engine.start_session(subjects=build_subjects(), generation_mode="HYBRID_LLM")
    response = engine.handle_user_input(start.session_id, "状況を確認しました。")

    assert start.fallback_used is True
    assert response.fallback_used is True
    traces = engine.store.list_generation_traces(start.session_id)
    assert len(traces) == 2
    assert traces[0].fallback_used is True
    assert traces[1].fallback_used is True
    assert traces[1].error is not None


def test_finish_intent_closes_session() -> None:
    engine = CTAInterviewEngine()
    session = engine.start_session(subjects=build_subjects(), generation_mode="TEMPLATE_RANDOM")

    response = engine.handle_user_input(session.session_id, "本日は以上です。終了します。")

    assert response.question_type == "STD11"
    assert response.status == "FINISHED"

