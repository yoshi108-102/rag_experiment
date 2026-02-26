import os
from pathlib import Path
from langchain_openai import ChatOpenAI

from src.core.models import GateDecision
from src.agents.translator import translate_reasoning_to_japanese


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


def analyze_input(user_input: str, chat_context: list = None) -> tuple[GateDecision, str | None]:
    """
    Analyzes the user input and returns a classification decision along with the 
    reasoning content if available. Optionally takes recent chat context (pre-sized).
    """
    from openai import OpenAI
    import json
    
    client = OpenAI()
    system_prompt = load_gate_prompt()
    
    messages = [{"role": "system", "content": system_prompt}]
    
    if chat_context:
        # Pre-pend chat history context directly
        for msg in chat_context:
            # We only send role and content to the API
            if msg.get("role") in ["user", "assistant"]:
                messages.append({"role": msg["role"], "content": msg["content"]})
                
    messages.append({"role": "user", "content": user_input})
    
    response = client.responses.create(
        model="gpt-5.2",
        input=messages,
        reasoning={
            "effort": "high",
            "summary": "detailed"
        },
        text={
            "format": {
                "type": "json_schema",
                "name": "GateDecision",
                "schema": {
                    "type": "object",
                    "properties": {
                        "route": {"type": "string", "enum": ["DEEPEN", "PARK", "CLARIFY", "FINISH"]},
                        "reason": {"type": "string"},
                        "first_question": {"type": "string"}
                    },
                    "required": ["route", "reason", "first_question"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    )
    
    # Parse the new response format based on schema_ex.txt
    reasoning = None
    msg_content_json = None
    
    response_dict = json.loads(response.model_dump_json())
    for item in response_dict.get("output", []):
        if item.get("type") == "reasoning":
            summary = item.get("summary", [])
            if summary and len(summary) > 0:
                reasoning = summary[0].get("text")
        elif item.get("type") == "message":
            content_list = item.get("content", [])
            if content_list and len(content_list) > 0:
                msg_content_json = content_list[0].get("text")

    from src.core.models import GateDecision
    if msg_content_json:
        decision = GateDecision(**json.loads(msg_content_json))
    else:
        # Fallback if parsing fails for some reason
        decision = GateDecision(route="PARK", reason="Error parsing LLM output", first_question="エラーが発生しました。")
    
    # Translate reasoning to Japanese
    if reasoning:
        reasoning = translate_reasoning_to_japanese(reasoning)

    return decision, reasoning
