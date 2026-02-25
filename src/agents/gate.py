import os
from pathlib import Path
from langchain_openai import ChatOpenAI

from src.core.models import GateDecision


def load_gate_prompt() -> str:
    """
    Loads the system prompt used by the Gate Model.
    
    This prompt is defined in `prompts/gate_prompt.txt` and is responsible for
    instructing the LLM to classify user input into DEEPEN, CLARIFY, or PARK,
    and output a strictly formatted JSON conforming to the GateDecision model.
    It enforces rules such as limiting the reason length and selecting from
    specific question templates for the DEEPEN route.
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    prompt_path = base_dir / "prompts" / "gate_prompt.md"
    overall_path = base_dir / "prompts" / "overall.md"
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()
        
    if overall_path.exists():
        with open(overall_path, "r", encoding="utf-8") as f:
            overall_context = f.read()
        prompt += f"\n\n[ドメイン知識（前提）]\n以下の作業概要を前提知識として踏まえた上で、ユーザーの発言を解釈してください。\n{overall_context}"
        
    return prompt


def analyze_input(user_input: str) -> GateDecision:
    """
    Analyzes the user input and returns a classification decision.
    """
    # Use gpt-4o-mini as specified in the lightweight model requirement
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # Enforce structured output based on the GateDecision Pydantic model
    structured_llm = llm.with_structured_output(GateDecision)
    
    system_prompt = load_gate_prompt()
    
    # Combine system prompt and user input
    messages = [
        ("system", system_prompt),
        ("user", user_input)
    ]
    
    # Invoke the model and parse the output
    decision = structured_llm.invoke(messages)
    return decision
