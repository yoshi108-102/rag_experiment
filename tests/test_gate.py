import pytest
from unittest.mock import patch, MagicMock
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
def test_analyze_input(mock_chatopenai):
    """
    Test the analyze_input function by mocking the ChatOpenAI response.
    Ensures that it returns a GateDecision object.
    """
    # Create the mock decision we expect back
    mock_decision = GateDecision(
        route="DEEPEN",
        reason="User shows uncertainty.",
        first_question="どの瞬間に引っかかりを感じましたか？"
    )

    # Setup the mock chain
    mock_llm_instance = MagicMock()
    mock_structured_llm = MagicMock()
    
    # Configure the mock returns
    mock_chatopenai.return_value = mock_llm_instance
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm
    mock_structured_llm.invoke.return_value = mock_decision

    # Call the function
    result = analyze_input("うーん、この設計でいいのか少し迷っています。")

    # Assertions
    assert isinstance(result, GateDecision)
    assert result.route == "DEEPEN"
    assert result.reason == "User shows uncertainty."
    assert result.first_question == "どの瞬間に引っかかりを感じましたか？"
    
    # Verify the mock was called
    mock_structured_llm.invoke.assert_called_once()
