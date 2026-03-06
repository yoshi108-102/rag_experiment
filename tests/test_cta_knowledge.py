from __future__ import annotations

from src.cta.engine import CTAInterviewEngine
from src.cta.models import SubjectPlan


def test_knowledge_candidates_generated_on_finish() -> None:
    engine = CTAInterviewEngine(topic_turn_limit=3)
    response = engine.start_session(
        subjects=[SubjectPlan(name="業務判断", topics=["状況把握"])],
        generation_mode="TEMPLATE_RANDOM",
    )
    session_id = response.session_id

    _ = engine.handle_user_input(
        session_id,
        "状況を確認し、判断基準を整理しました。",
    )
    end = engine.handle_user_input(session_id, "終了します。")

    assert end.status == "FINISHED"
    candidates = engine.store.list_knowledge_candidates(session_id)
    assert len(candidates) >= 1
    assert candidates[0].session_id == session_id
    assert candidates[0].cue
    assert 0.0 <= candidates[0].confidence <= 1.0

