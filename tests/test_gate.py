import pytest
import json
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage
from src.core.models import GateDecision
from src.agents.gate import (
    analyze_input,
    build_clarify_completion_json,
    load_gate_prompt,
    _apply_decision_overrides,
)

def test_load_gate_prompt():
    """Test if the system prompt loads correctly."""
    prompt = load_gate_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "DEEPEN" in prompt
    assert "CLARIFY" in prompt
    assert "PARK" in prompt

@patch('src.agents.gate.ChatOpenAI')
@patch('src.agents.gate.translate_reasoning_to_japanese')
def test_analyze_input(mock_translate, mock_chatopenai):
    """
    Test the analyze_input function by mocking the ChatOpenAI response.
    Ensures that it returns a (GateDecision, reasoning) tuple.
    """
    # Create the mock decision JSON
    mock_decision_dict = {
        "route": "DEEPEN",
        "reason": "User shows uncertainty.",
        "first_question": "どの瞬間に引っかかりを感じましたか？"
    }
    mock_decision_json = json.dumps(mock_decision_dict)
    
    # Mock reasoning
    mock_reasoning_text = "The user is expressing doubt about the design."
    mock_translated_reasoning = "ユーザーは設計に疑問を持っています。"
    mock_translate.return_value = mock_translated_reasoning

    # Setup the mock AIMessage with content blocks
    mock_response = MagicMock(spec=AIMessage)
    mock_response.content = [
        {"type": "reasoning", "summary": [{"text": mock_reasoning_text}]},
        {"type": "text", "text": mock_decision_json}
    ]
    
    # Configure the mock LLM
    mock_llm_instance = MagicMock()
    mock_chatopenai.return_value = mock_llm_instance
    mock_llm_instance.invoke.return_value = mock_response

    # Call the function
    decision, reasoning = analyze_input("うーん、この設計でいいのか少し迷っています。")

    # Assertions
    assert isinstance(decision, GateDecision)
    assert decision.route == "DEEPEN"
    assert decision.reason == "User shows uncertainty."
    assert reasoning == mock_translated_reasoning
    
    # Verify the mock was called
    mock_llm_instance.invoke.assert_called_once()


@patch('src.agents.gate.ChatOpenAI')
@patch('src.agents.gate.translate_reasoning_to_japanese')
def test_analyze_input_does_not_duplicate_latest_user_message(mock_translate, mock_chatopenai):
    mock_translate.return_value = None

    mock_decision_json = json.dumps({
        "route": "CLARIFY",
        "reason": "Need clarification",
        "first_question": "いちばん気になっている点はどこですか？",
    })

    mock_response = MagicMock(spec=AIMessage)
    mock_response.content = [{"type": "text", "text": mock_decision_json}]

    captured_messages = {}

    def fake_invoke(messages, response_format=None):
        captured_messages["messages"] = messages
        return mock_response

    mock_llm_instance = MagicMock()
    mock_chatopenai.return_value = mock_llm_instance
    mock_llm_instance.invoke.side_effect = fake_invoke

    current_user_input = "仕様の切り方で少し迷っています。"
    analyze_input(
        current_user_input,
        chat_context=[
            {"role": "assistant", "content": "なるほど、どのあたりで迷っていますか？"},
            {"role": "user", "content": current_user_input},
        ],
    )

    human_messages = [
        m for m in captured_messages["messages"]
        if isinstance(m, HumanMessage) and m.content == current_user_input
    ]
    assert len(human_messages) == 1


def test_apply_decision_overrides_forces_finish_on_closure_intent():
    decision = GateDecision(
        route="DEEPEN",
        reason="User is reflecting",
        first_question="どの瞬間がいちばん引っかかりましたか？",
    )
    user_input = (
        "疑問の時は深掘りしても言葉に詰まっちゃうし、"
        "話すだけ話したら正直スッキリするからあとはいらないかなとなった"
    )

    overridden = _apply_decision_overrides(decision, user_input)

    assert overridden.route == "FINISH"
    assert "区切って" in overridden.first_question


def test_apply_decision_overrides_keeps_non_closure_input():
    decision = GateDecision(
        route="CLARIFY",
        reason="Need core issue",
        first_question="いちばん詰まった場面はどこでしたか？",
    )

    overridden = _apply_decision_overrides(
        decision,
        "真ん中の位置決めが難しくて、どこを見ればいいかまだ掴めないです",
    )

    assert overridden == decision


def test_apply_decision_overrides_turns_frustration_into_park():
    decision = GateDecision(
        route="CLARIFY",
        reason="Need detail",
        first_question="どの条件のときですか？",
    )

    overridden = _apply_decision_overrides(
        decision,
        "だから、支持台に乗らないんだから知ったって意味ない",
    )

    assert overridden.route == "PARK"
    assert "十分伝わってる" in overridden.first_question


def test_apply_decision_overrides_finishes_after_friction_and_confirmation():
    decision = GateDecision(
        route="CLARIFY",
        reason="Need wording",
        first_question="その言い方でいい？",
    )

    overridden = _apply_decision_overrides(
        decision,
        "そうです",
        chat_context=[
            {"role": "assistant", "content": "いま言語化したいのは、先端を除いた判定基準の言い方かな？"},
            {"role": "user", "content": "だからそうなの"},
            {"role": "assistant", "content": "じゃあ「支持台に乗らない先端部」で統一でいい？"},
            {"role": "user", "content": "そうです"},
        ],
    )

    assert overridden.route == "FINISH"
    assert "区切って" in overridden.first_question


def test_apply_decision_overrides_finishes_when_cause_and_result_are_both_stated():
    decision = GateDecision(
        route="CLARIFY",
        reason="Need to clarify",
        first_question="どういう条件でそうなりますか？",
    )

    overridden = _apply_decision_overrides(
        decision,
        "先端が強く曲がっているから、機械に合わせて見ると全体の曲がり判断が引っ張られる",
        chat_context=[
            {"role": "assistant", "content": "そのとき何が起きるの？"},
            {"role": "user", "content": "先端の影響で全体が見えにくい"},
        ],
    )

    assert overridden.route == "FINISH"
    assert "付け加えたいこと" in overridden.first_question


def test_build_clarify_completion_json_marks_complete_for_causal_statement():
    payload = build_clarify_completion_json(
        "先端が強く曲がっているから、機械に合わせて見ると全体の判断が引っ張られる"
    )

    assert payload["is_complete"] is True
    assert payload["kind"] == "idea"
    assert payload["item"]
    assert payload["reason"]


def test_build_clarify_completion_json_collects_reason_from_recent_context():
    payload = build_clarify_completion_json(
        "だから、先端を外して見るようにしてる",
        chat_context=[
            {"role": "assistant", "content": "なんでそうしてるの？"},
            {"role": "user", "content": "先端が曲がってるから全体の見え方が引っ張られる"},
            {"role": "assistant", "content": "で、どうしてる？"},
        ],
    )

    assert payload["is_complete"] is True
    assert payload["reason"] is not None
    assert payload["item"] is not None
