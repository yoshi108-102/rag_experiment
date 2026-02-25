from typing import Literal
from pydantic import BaseModel, Field


class GateDecision(BaseModel):
    """Represents the classification and reasoning from the Reflective Gate Chat."""

    route: Literal["DEEPEN", "PARK", "CLARIFY"] = Field(
        description="The classification route for the user's input."
    )
    reason: str = Field(
        description="A concise reason (< 10 words) for the chosen route."
    )
    first_question: str = Field(
        description="The follow-up probing question to ask the user."
    )
