from pathlib import Path
import re
from typing import Any

from langchain_openai import ChatOpenAI

from src.core.models import GateDecision
from src.core.token_usage import default_gate_model_name, extract_token_usage
from src.agents.translator import translate_reasoning_to_japanese


FINISH_INTENT_PATTERNS = (
    r"話すだけ話した",
    r"話せただけで.*(十分|よい|いい)",
    r"スッキリした",
    r"スッキリする",
    r"少しスッキリ",
    r"あとはいらない",
    r"これ以上(は)?(いらない|不要)",
    r"もう十分",
    r"ここまでで(いい|よい)",
    r"このへんで(いい|よい)",
    r"一旦(いい|よい)",
    r"区切り(にしたい|でいい|でよい)",
    r"終わり(でいい|でよい|にしたい)",
)

FRUSTRATION_PATTERNS = (
    r"だってば",
    r"に決まってる",
    r"だから(そう|それ)",
    r"意味ない",
    r"別に.*十分伝わる",
    r"わかるに決まってる",
    r"もう(その話|そこ)",
)

SHORT_AFFIRMATION_PATTERNS = (
    r"そうです[。！]?$",
    r"そうそう[。！]?$",
    r"その通り[。！]?$",
    r"そうなの[。！]?$",
    r"はい[。！]?$",
)

QUESTION_MARKER_PATTERNS = (
    r"[？?]$",
    r"疑問",
    r"気になる",
    r"わからない",
    r"知りたい",
    r"どうして",
    r"なぜ",
    r"なんで",
    r"どこ",
    r"どの",
    r"いつ",
    r"何が",
    r"何を",
)

CAUSAL_CONNECTOR_PATTERNS = (
    r"から",
    r"ので",
    r"ため",
    r"せいで",
    r"影響で",
    r"都合上",
)

IMAGE_FALLBACK_TEXT = "添付画像を見て会話を続けたいです。"


def _build_human_message_content(
    text: str,
    images: list[dict[str, str]] | None = None,
) -> str | list[dict[str, Any]]:
    normalized_text = (text or "").strip()
    if not images:
        return normalized_text

    content_blocks: list[dict[str, Any]] = [
        {"type": "text", "text": normalized_text or IMAGE_FALLBACK_TEXT}
    ]
    for image in images:
        mime_type = str(image.get("mime_type", "")).strip().lower()
        data_base64 = str(image.get("data_base64", "")).strip()
        if not mime_type.startswith("image/"):
            continue
        if not data_base64:
            continue
        content_blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{data_base64}"},
            }
        )

    if len(content_blocks) == 1:
        return content_blocks[0]["text"]
    return content_blocks


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _recent_user_texts(chat_context: list | None, limit: int = 4) -> list[str]:
    if not chat_context:
        return []
    texts = [msg.get("content", "") for msg in chat_context if msg.get("role") == "user"]
    return texts[-limit:]


def _last_assistant_text(chat_context: list | None) -> str:
    if not chat_context:
        return ""
    for msg in reversed(chat_context):
        if msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    return ""


def _looks_like_confirmation_question(text: str) -> bool:
    if not text:
        return False
    markers = ("つまり", "ってこと", "でいい", "かな", "ですか", "統一", "言い方")
    return "?" in text or "？" in text or any(marker in text for marker in markers)


def _normalize_text_for_slots(text: str) -> str:
    return re.sub(r"\s+", "", text or "").strip("。．!！")


def _extract_causal_pair(text: str) -> tuple[str | None, str | None]:
    compact = _normalize_text_for_slots(text)
    if not compact or len(compact) < 10:
        return None, None
    if compact.endswith(("?", "？")):
        return None, None

    # Prefer explicit "AだからB" style splits.
    for connector in CAUSAL_CONNECTOR_PATTERNS:
        if connector in compact:
            left, right = compact.split(connector, 1)
            left = left.strip("、, ")
            right = right.strip("、, ")
            if len(left) >= 3 and len(right) >= 3:
                return left, right

    # "そのため/その結果" style: treat prefix as idea and previous clause as reason-like context.
    for connector in ("そのため", "その結果"):
        if connector in compact:
            left, right = compact.split(connector, 1)
            left = left.strip("、, ")
            right = right.strip("、, ")
            if len(left) >= 3 and len(right) >= 3:
                return left, right

    return None, None


def _is_meaningful_user_text(text: str) -> bool:
    compact = _normalize_text_for_slots(text)
    if len(compact) < 4:
        return False
    if _matches_any(compact, SHORT_AFFIRMATION_PATTERNS):
        return False
    return True


def _infer_idea_or_question_kind(text: str) -> str | None:
    compact = _normalize_text_for_slots(text)
    if not compact:
        return None
    if _matches_any(compact, QUESTION_MARKER_PATTERNS):
        return "question"
    return "idea"


def build_clarify_completion_json(
    user_input: str,
    chat_context: list | None = None,
) -> dict[str, str | bool | None]:
    """
    Rule-based CLARIFY slot extraction.

    The CLARIFY phase is considered complete when the following JSON slots are filled:
    - kind: "idea" or "question"
    - item: the main idea/question the user is trying to convey
    - reason: why that idea/question matters / what causes it
    """
    user_texts = _recent_user_texts(chat_context, limit=6)
    if not user_texts or user_texts[-1] != user_input:
        user_texts.append(user_input)

    kind: str | None = None
    item: str | None = None
    reason: str | None = None

    for text in reversed(user_texts):
        if not _is_meaningful_user_text(text):
            continue

        compact = _normalize_text_for_slots(text)
        if kind is None:
            kind = _infer_idea_or_question_kind(compact)

        extracted_reason, extracted_item = _extract_causal_pair(compact)
        if reason is None and extracted_reason:
            reason = extracted_reason
        if item is None and extracted_item:
            item = extracted_item

        if item is None:
            item = compact

    return {
        "kind": kind,
        "item": item,
        "reason": reason,
        "is_complete": bool(kind and item and reason),
    }


def _apply_decision_overrides(
    decision: GateDecision,
    user_input: str,
    chat_context: list | None = None,
) -> GateDecision:
    """
    Post-processes the LLM decision with a small set of deterministic guardrails.

    This primarily catches explicit "I feel done after talking" signals and routes
    them to FINISH so the conversation can close without over-probing.
    """
    if decision.route == "FINISH":
        return decision

    text = user_input.strip()
    if text and _matches_any(text, FINISH_INTENT_PATTERNS):
        return GateDecision(
            route="FINISH",
            reason="Closure intent detected",
            first_question="話せて少しスッキリしたなら、今日はここで区切ってよさそうですね。",
        )

    if text and _matches_any(text, FRUSTRATION_PATTERNS):
        return GateDecision(
            route="PARK",
            reason="Frustration signal detected",
            first_question="なるほど、そこはもう十分伝わってるので、いったんその理解で置いておこう。",
        )

    clarify_json = build_clarify_completion_json(text, chat_context)
    if decision.route == "CLARIFY" and clarify_json.get("is_complete"):
        return GateDecision(
            route="FINISH",
            reason="Clarify JSON complete",
            first_question="いまので筋はかなり伝わったよ、ほかに付け加えたいことある？",
        )

    recent_user_texts = _recent_user_texts(chat_context)
    recent_has_frustration = any(_matches_any(t, FRUSTRATION_PATTERNS) for t in recent_user_texts[:-1])
    last_assistant = _last_assistant_text(chat_context)
    if (
        text
        and _matches_any(text, SHORT_AFFIRMATION_PATTERNS)
        and recent_has_frustration
        and _looks_like_confirmation_question(last_assistant)
    ):
        return GateDecision(
            route="FINISH",
            reason="Confirmed after friction",
            first_question="うん、意図は十分伝わったので、ここで区切って大丈夫そうです。",
        )

    return decision


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


def analyze_input(
    user_input: str,
    chat_context: list | None = None,
    user_images: list[dict[str, str]] | None = None,
) -> tuple[GateDecision, str | None, dict[str, int] | None]:
    """
    Analyzes the user input and returns a classification decision along with the 
    reasoning content if available. Optionally takes recent chat context (pre-sized).
    If the latest chat_context entry already contains the current user input, it is
    not appended again.
    """
    import json
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    
    system_prompt = load_gate_prompt()
    
    latest_context_is_same_user_input = bool(
        chat_context
        and chat_context[-1].get("role") == "user"
        and chat_context[-1].get("content") == user_input
    )
    latest_context_index = (len(chat_context) - 1) if chat_context else -1

    messages = [SystemMessage(content=system_prompt)]

    if chat_context:
        for idx, msg in enumerate(chat_context):
            if msg.get("role") == "user":
                msg_images: list[dict[str, str]] | None = None
                if idx == latest_context_index and latest_context_is_same_user_input and user_images:
                    msg_images = user_images

                messages.append(
                    HumanMessage(
                        content=_build_human_message_content(
                            str(msg.get("content", "")),
                            msg_images,
                        )
                    )
                )
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=str(msg.get("content", ""))))

    if not latest_context_is_same_user_input:
        messages.append(
            HumanMessage(content=_build_human_message_content(user_input, user_images))
        )
    
    # Initialize ChatOpenAI with Reasoning support (Responses API)
    llm = ChatOpenAI(
        model=default_gate_model_name(),
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
    token_usage = extract_token_usage(response)
    
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
            decision = _apply_decision_overrides(decision, user_input, chat_context)
        except Exception as e:
            print(f"[Gate Error] Pydantic validation failed: {e}")
            decision = GateDecision(route="PARK", reason="Pydantic validation failed", first_question="解析エラーが発生しました。")
    else:
        decision = GateDecision(route="PARK", reason="No response content", first_question="応答が得られませんでした。")
    
    # Translate reasoning to Japanese
    if reasoning:
        reasoning = translate_reasoning_to_japanese(reasoning)

    return decision, reasoning, token_usage
