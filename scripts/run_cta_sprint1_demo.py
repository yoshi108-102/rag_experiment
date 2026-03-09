"""Minimal CLI demo for CTA Sprint 1 engine."""

from __future__ import annotations

from src.cta import SubjectPlan, create_default_cta_engine


def main() -> None:
    engine = create_default_cta_engine()

    subjects = [
        SubjectPlan(name="地震時の対応", topics=["状況把握", "判断基準", "実行した行動"]),
        SubjectPlan(name="帰宅判断", topics=["情報収集", "移動手段の選択"]),
    ]
    session = engine.start_session(subjects=subjects, generation_mode="TEMPLATE_RANDOM")
    session_id = session.session_id

    print(f"session_id: {session_id}")
    print(f"AI: {session.assistant_text}")
    print("Type `exit` to finish.")

    while True:
        user_text = input("User: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            user_text = "終了します"
        result = engine.handle_user_input(session_id, user_text)
        print(f"AI: {result.assistant_text}")
        print(
            f"  [question_type={result.question_type}, mode={result.generation_mode}, "
            f"fallback={result.fallback_used}, status={result.status}]"
        )
        if result.status == "FINISHED":
            break


if __name__ == "__main__":
    main()
