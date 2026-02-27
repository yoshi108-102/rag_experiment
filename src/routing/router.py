"""Gate判定のroute値に応じて、返答文を選択・返却するルーティング処理。"""

from src.core.models import GateDecision


def execute_route(decision: GateDecision) -> str:
    """Gate判定に応じた応答文を返す。

    現状は最小実装として、全ルートで`first_question`を返しつつ、
    デバッグ用に標準出力へroute/reasonを出力する。
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
    elif route == "FINISH":
        print(f"[Log] Conversation finished. Reason: {decision.reason}")
        return decision.first_question
    else:
        return f"Unknown route: {route}"
