from src.core.models import GateDecision


def execute_route(decision: GateDecision) -> str:
    """
    Executes the appropriate action based on the GateDecision.

    Since this is a minimal implementation:
    - DEEPEN: Returns the selected deep-dive question.
    - CLARIFY: Returns the clarifying question.
    - PARK: Logs the decision and returns the response (e.g., confirmation).
    """
    route = decision.route
    
    print(f"\n[Route]: {route}")
    print(f"[Reason]: {decision.reason}")
    
    if route == "DEEPEN":
        return decision.first_question
    elif route == "CLARIFY":
        return decision.first_question
    elif route == "PARK":
        # For PARK, we might log it to a file or database in a real system.
        # Here we just print a log message and return the question/response.
        print(f"[Log] Parking this topic. Reason: {decision.reason}")
        return decision.first_question
    else:
        return f"Unknown route: {route}"
