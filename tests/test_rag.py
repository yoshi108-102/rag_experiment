import json

from src.rag.models import KnowledgeRecord, NoveltyDecision, RetrievalResult
from src.rag.novelty import judge_novelty
from src.rag.retriever import retrieve_similar
from src.rag.store import save_pending_reflection


def sample_records() -> list[KnowledgeRecord]:
    return [
        KnowledgeRecord(
            record_id="1",
            topic="太い棒の矯正方法",
            tags=["対象:太い", "動作:押す"],
            record_type="open_question",
            text="太い棒は細い棒よりも離れて見るのがよさそう？",
        ),
        KnowledgeRecord(
            record_id="2",
            topic="曲がりの確認と矯正手法",
            tags=["動作:確認"],
            record_type="method",
            text="押した後に90度戻して押し具合を確認する",
            applicable_when="押す強さを確認したい時",
        ),
        KnowledgeRecord(
            record_id="3",
            topic="視覚的確認の課題と工夫",
            tags=["視覚:確認"],
            record_type="summary",
            text="視点や姿勢の工夫が必要",
        ),
    ]


def test_retrieve_similar_returns_relevant_match():
    records = sample_records()
    results = retrieve_similar("太い棒の見方で迷ってる", records, top_k=2)

    assert results
    assert results[0].record.record_id == "1"
    assert results[0].score > 0


def test_judge_novelty_marks_non_novel_for_high_score():
    record = sample_records()[0]
    retrieved = [RetrievalResult(record=record, score=0.72, reasons=["text-sim"])]

    decision = judge_novelty("太い棒は離れて見る？", retrieved)

    assert decision.is_novel is False
    assert "threshold" in decision.reason


def test_judge_novelty_marks_novel_when_no_hits():
    decision = judge_novelty("全く別の新しい悩み", [])
    assert decision.is_novel is True


def test_save_pending_reflection_deduplicates(tmp_path):
    target = tmp_path / "pending_reflections.jsonl"
    novelty = NoveltyDecision(is_novel=True, confidence=0.8, reason="no similar")
    retrieved = []

    first = save_pending_reflection(
        user_input="新しい疑問です",
        route="DEEPEN",
        novelty=novelty,
        retrieved=retrieved,
        path=target,
    )
    second = save_pending_reflection(
        user_input="新しい疑問です",
        route="DEEPEN",
        novelty=novelty,
        retrieved=retrieved,
        path=target,
    )

    assert first is True
    assert second is False

    lines = target.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["status"] == "pending_review"
    assert payload["user_input"] == "新しい疑問です"
