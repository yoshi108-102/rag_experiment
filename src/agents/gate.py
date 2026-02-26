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
    import json
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    
    system_prompt = load_gate_prompt()
    
    messages = [SystemMessage(content=system_prompt)]
    
    if chat_context:
        for msg in chat_context:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg["content"]))
                
    messages.append(HumanMessage(content=user_input))
    
    # Initialize ChatOpenAI with Reasoning support (Responses API)
    llm = ChatOpenAI(
        model="gpt-5.2",
        use_responses_api=True,
        output_version="responses/v1",
        reasoning={
            "effort": "high",
            "summary": "detailed"
        }
    )
    
    # Define the JSON schema for Structured Outputs
    schema = GateDecision.model_json_schema()
    # OpenAI Structured Outputs (Strict) requires additionalProperties: False
    schema["additionalProperties"] = False
    
    # Remove Pydantic-specific metadata that OpenAI might reject in strict mode
    schema.pop("title", None)
    schema.pop("description", None)
    
    gate_decision_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "GateDecision",
            "schema": schema,
            "strict": True
        }
    }
    
    # Invoke the LLM with direct response_format to enforce Structured Outputs
    response = llm.invoke(messages, response_format=gate_decision_schema)
    
    reasoning = None
    msg_content_json = None
    
    # Parse the content blocks from AIMessage
    if isinstance(response.content, list):
        for block in response.content:
            if isinstance(block, dict):
                if block.get("type") == "reasoning":
                    summary_list = block.get("summary", [])
                    if summary_list and len(summary_list) > 0:
                        reasoning = summary_list[0].get("text")
                elif block.get("type") == "text":
                    msg_content_json = block.get("text")
    elif isinstance(response.content, str):
        msg_content_json = response.content

    # Use Pydantic to validate and parse the JSON structure directly
    if msg_content_json:
        try:
            decision = GateDecision.model_validate_json(msg_content_json)
        except Exception as e:
            print(f"[Gate Error] Pydantic validation failed: {e}")
            decision = GateDecision(route="PARK", reason="Pydantic validation failed", first_question="解析エラーが発生しました。")
    else:
        decision = GateDecision(route="PARK", reason="No response content", first_question="応答が得られませんでした。")
    
    # Translate reasoning to Japanese
    if reasoning:
        reasoning = translate_reasoning_to_japanese(reasoning)

    return decision, reasoning
