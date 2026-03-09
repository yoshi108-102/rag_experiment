"""Knowledge candidate extraction for session-finalization."""

from __future__ import annotations

from src.cta.models import CTATurnRecord, KnowledgeCandidate


def extract_knowledge_candidates(
    session_id: str,
    turns: list[CTATurnRecord],
) -> list[KnowledgeCandidate]:
    """Build simple structured candidates from CDM-focused turns."""
    candidates: list[KnowledgeCandidate] = []
    for turn in turns:
        if not turn.question_type.startswith("CDM"):
            continue
        cue = turn.keywords[0] if turn.keywords else turn.topic_name
        decision = _clip(turn.assistant_text, 70)
        action = _action_from_label(turn.cognitive_action_label)
        difficulty = _difficulty_from_features(turn)
        exception = "未設定"
        confidence = 0.65 if turn.keywords else 0.5

        candidates.append(
            KnowledgeCandidate(
                session_id=session_id,
                turn_index=turn.turn_index,
                cue=cue,
                decision=decision,
                action=action,
                difficulty=difficulty,
                exception=exception,
                confidence=round(confidence, 2),
            )
        )
    return candidates


def _action_from_label(label: str) -> str:
    mapping = {
        "information": "情報を確認する",
        "situation_awareness": "状況を把握する",
        "decision": "判断基準を選ぶ",
        "action": "具体行動を実施する",
    }
    return mapping.get(label, "行動方針を確認する")


def _difficulty_from_features(turn: CTATurnRecord) -> str:
    if turn.has_negative:
        return "回答が限定的で追加確認が必要"
    if turn.clause_count >= 3:
        return "情報量が多く論点整理が必要"
    if turn.has_question:
        return "不確実性が残っている"
    return "大きな難所は未検出"


def _clip(text: str, limit: int) -> str:
    stripped = (text or "").strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 1] + "…"

