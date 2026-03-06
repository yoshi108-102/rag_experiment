from __future__ import annotations

import json
from pathlib import Path

from src.cta.engine import CTAInterviewEngine
from src.cta.export import export_session_artifacts
from src.cta.models import SubjectPlan


def test_export_session_artifacts_writes_files(tmp_path: Path) -> None:
    engine = CTAInterviewEngine()
    response = engine.start_session(
        subjects=[SubjectPlan(name="業務判断", topics=["状況把握"])],
        generation_mode="TEMPLATE_RANDOM",
    )
    session_id = response.session_id
    engine.handle_user_input(session_id, "状況を確認しました。")
    engine.handle_user_input(session_id, "終了します。")

    paths = export_session_artifacts(engine.store, session_id, tmp_path)

    turns_path = Path(paths["turns_jsonl"])
    traces_path = Path(paths["traces_jsonl"])
    knowledge_path = Path(paths["knowledge_jsonl"])
    summary_path = Path(paths["summary_json"])

    assert turns_path.exists()
    assert traces_path.exists()
    assert knowledge_path.exists()
    assert summary_path.exists()

    lines = [line for line in turns_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 3
    first_turn = json.loads(lines[0])
    assert first_turn["question_type"] == "STD1"
    assert "decision_reason" in first_turn
    assert "processing_latency_ms" in first_turn

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["session_id"] == session_id
    assert "performance" in summary
    assert "p95_ms" in summary["performance"]
    assert summary["knowledge_candidate_count"] >= 1
