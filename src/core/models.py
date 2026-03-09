"""アプリケーション全体で利用するPydanticデータモデルを定義する。"""

from typing import Literal
from pydantic import BaseModel, Field


class GateDecision(BaseModel):
    """Gateモデルの判定結果を表す構造化データ。

    Attributes:
        route: 入力の遷移先。深掘り・明確化・保留・終了のいずれか。
        reason: その遷移を選んだ理由。
        first_question: 次にユーザーへ返す最初の短い応答文。必要なら1つだけ質問を含めてよい。
    """

    route: Literal["DEEPEN", "PARK", "CLARIFY", "FINISH"] = Field(
        description="The classification route for the user's input."
    )
    reason: str = Field(
        description="A concise reason (< 10 words) for the chosen route."
    )
    first_question: str = Field(
        description="The next short response to the user, optionally including one open follow-up question."
    )

    model_config = {
        "extra": "forbid"
    }
