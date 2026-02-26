import pytest
import json
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage
from src.core.models import GateDecision
from src.agents.gate import analyze_input, load_gate_prompt, _apply_decision_overrides

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
