import pytest
from src.core.models import GateDecision
from src.routing.router import execute_route

def test_execute_route_deepen():
    decision = GateDecision(
        route="DEEPEN",
        reason="Testing deepen",
        first_question="どの瞬間に引っかかりを感じましたか？"
    )
    result = execute_route(decision)
    assert result == "どの瞬間に引っかかりを感じましたか？"

def test_execute_route_clarify():
    decision = GateDecision(
        route="CLARIFY",
        reason="Testing clarify",
        first_question="つまり、「〇〇」ということでしょうか？"
    )
    result = execute_route(decision)
    assert result == "つまり、「〇〇」ということでしょうか？"

def test_execute_route_park():
    decision = GateDecision(
        route="PARK",
        reason="Testing park",
        first_question="なるほど、そうなんですね！"
    )
    result = execute_route(decision)
    assert result == "なるほど、そうなんですね！"
