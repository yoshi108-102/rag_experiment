import pytest
import json
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage
from src.core.models import GateDecision
from src.agents.gate import analyze_input, load_gate_prompt

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
