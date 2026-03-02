"""ユーザー入力をGateルートへ分類し、補助ルールで最終判定を整える処理群。"""

from pathlib import Path
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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

LOW_SIGNAL_ENDING_PATTERNS = (
    r"^(特に)?ないです$",
    r"^(特に)?ありません$",
    r"^ないかな$",
    r"^大丈夫です$",
    r"^以上です$",
    r"^それだけです$",
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
PARK_PARSE_ERROR_MESSAGE = "解析エラーが発生しました。"
PARK_NO_CONTENT_MESSAGE = "応答が得られませんでした。"


def _build_human_message_content(
    text: str,
    images: list[dict[str, str]] | None = None,
) -> str | list[dict[str, Any]]:
    """テキストと画像群をResponses API向けのHumanMessage形式へ整形する。"""
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


def _build_chat_messages(
    system_prompt: str,
    user_input: str,
    chat_context: list | None = None,
    user_images: list[dict[str, str]] | None = None,
) -> list[SystemMessage | HumanMessage | AIMessage]:
    """system/context/current入力からLLM送信用メッセージ列を構築する。"""
    latest_context_is_same_user_input = bool(
        chat_context
        and chat_context[-1].get("role") == "user"
        and chat_context[-1].get("content") == user_input
    )
    latest_context_index = (len(chat_context) - 1) if chat_context else -1

    messages: list[SystemMessage | HumanMessage | AIMessage] = [
        SystemMessage(content=system_prompt)
    ]
    if chat_context:
        for idx, msg in enumerate(chat_context):
            if msg.get("role") == "user":
                msg_images: list[dict[str, str]] | None = None
                if (
                    idx == latest_context_index
                    and latest_context_is_same_user_input
                    and user_images
                ):
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
    return messages


def _build_gate_decision_schema() -> dict[str, Any]:
    """GateDecision用のStructured Outputsスキーマを生成する。"""
    schema = GateDecision.model_json_schema()
    schema["additionalProperties"] = False
    schema.pop("title", None)
    schema.pop("description", None)

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "GateDecision",
            "schema": schema,
            "strict": True,
        },
    }


def _extract_reasoning_and_decision_json(
    response_content: Any,
) -> tuple[str | None, str | None]:
    """LLMレスポンスからreasoning要約と判定JSON文字列を抽出する。"""
    reasoning = None
    msg_content_json = None

    if isinstance(response_content, list):
        for block in response_content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "reasoning":
                summary_list = block.get("summary", [])
                if summary_list and len(summary_list) > 0:
                    reasoning = summary_list[0].get("text")
            elif block.get("type") == "text":
                msg_content_json = block.get("text")
    elif isinstance(response_content, str):
        msg_content_json = response_content

    return reasoning, msg_content_json


def _park_decision(reason: str, first_question: str) -> GateDecision:
    """PARKルートのフォールバック判定を作る。"""
    return GateDecision(route="PARK", reason=reason, first_question=first_question)


def _parse_decision_with_override(
    msg_content_json: str | None,
    user_input: str,
    chat_context: list | None,
) -> GateDecision:
    """判定JSONを検証して読み込み、必要な後段overrideを適用する。"""
    if msg_content_json:
        try:
            decision = GateDecision.model_validate_json(msg_content_json)
            return _apply_decision_overrides(decision, user_input, chat_context)
        except Exception as e:
            print(f"[Gate Error] Pydantic validation failed: {e}")
            return _park_decision(
                "Pydantic validation failed",
                PARK_PARSE_ERROR_MESSAGE,
            )

    return _park_decision("No response content", PARK_NO_CONTENT_MESSAGE)


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    """文字列が正規表現パターン群のいずれかに一致するかを返す。"""
    return any(re.search(pattern, text) for pattern in patterns)


def _recent_user_texts(chat_context: list | None, limit: int = 4) -> list[str]:
    """履歴から直近ユーザー発話のみを取り出す。"""
    if not chat_context:
        return []
    texts = [msg.get("content", "") for msg in chat_context if msg.get("role") == "user"]
    return texts[-limit:]


def _last_assistant_text(chat_context: list | None) -> str:
    """履歴上で最後のassistant発話を返す。"""
    if not chat_context:
        return ""
    for msg in reversed(chat_context):
        if msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    return ""


def _looks_like_confirmation_question(text: str) -> bool:
    """確認質問らしい文かを簡易ヒューリスティックで判定する。"""
    if not text:
        return False
    markers = ("つまり", "ってこと", "でいい", "かな", "ですか", "統一", "言い方")
    return "?" in text or "？" in text or any(marker in text for marker in markers)


def _normalize_text_for_slots(text: str) -> str:
    """slot抽出向けに空白と文末記号を削って正規化する。"""
    return re.sub(r"\s+", "", text or "").strip("。．!！")


def _extract_causal_pair(text: str) -> tuple[str | None, str | None]:
    """因果接続語を手掛かりに、理由と項目のペアを抽出する。"""
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
    """短すぎる相槌を除外し、slot抽出対象として有効か判定する。"""
    compact = _normalize_text_for_slots(text)
    if len(compact) < 4:
        return False
    if _matches_any(compact, SHORT_AFFIRMATION_PATTERNS):
        return False
    if _matches_any(compact, LOW_SIGNAL_ENDING_PATTERNS):
        return False
    return True


def _infer_idea_or_question_kind(text: str) -> str | None:
    """発話を`idea`または`question`のどちらかへ推定する。"""
    compact = _normalize_text_for_slots(text)
    if not compact:
        return None
    if _matches_any(compact, QUESTION_MARKER_PATTERNS):
        return "question"
    return "idea"


def _text_information_score(text: str) -> int:
    """発話の情報量をざっくり採点する（高いほどitem候補として優先）。"""
    compact = _normalize_text_for_slots(text)
    if not compact:
        return -1

    score = len(compact)
    extracted_reason, extracted_item = _extract_causal_pair(compact)
    if extracted_reason and extracted_item:
        score += 50
    if _matches_any(compact, QUESTION_MARKER_PATTERNS):
        score += 10
    return score


def _select_best_text(candidates: list[tuple[int, str]]) -> str | None:
    """候補文から情報量と新しさを基準に最良の文を選ぶ。"""
    if not candidates:
        return None
    _, selected = max(candidates, key=lambda item: (_text_information_score(item[1]), item[0]))
    return selected


def build_clarify_completion_json(
    user_input: str,
    chat_context: list | None = None,
) -> dict[str, str | bool | None]:
    """CLARIFY完了判定のためのslotをルールベースで抽出する。

    戻り値は以下の4キーを持つ:
    - kind: `idea` または `question`
    - item: ユーザーが伝えたい主題
    - reason: その主題の背景理由
    - is_complete: 上記3slotが揃っているか
    """
    user_texts = _recent_user_texts(chat_context, limit=6)
    if not user_texts or user_texts[-1] != user_input:
        user_texts.append(user_input)

    meaningful_texts: list[tuple[int, str]] = []
    causal_pairs: list[tuple[int, str, str, str]] = []

    for idx, text in enumerate(user_texts):
        if not _is_meaningful_user_text(text):
            continue

        compact = _normalize_text_for_slots(text)
        meaningful_texts.append((idx, compact))

        extracted_reason, extracted_item = _extract_causal_pair(compact)
        if extracted_reason and extracted_item:
            causal_pairs.append((idx, extracted_reason, extracted_item, compact))

    kind: str | None = None
    item: str | None = None
    reason: str | None = None

    if causal_pairs:
        _, reason, item, pair_source_text = max(causal_pairs, key=lambda item: item[0])
        kind = _infer_idea_or_question_kind(pair_source_text)

    if kind is None:
        best_text = _select_best_text(meaningful_texts)
        if best_text is not None:
            kind = _infer_idea_or_question_kind(best_text)

    if item is None:
        item = _select_best_text(meaningful_texts)

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
    """LLM判定へ決定論的なガードレールを後適用する。

    明示的な終了意図、苛立ち表現、CLARIFY完了条件を優先して
    ルートを安全側へ補正する。
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
    """Gate判定に使用するsystem prompt本文を読み込む。

    `prompts/gate_prompt.md`を基本プロンプトとして読み、
    `prompts/overall.md`が存在する場合はドメイン前提文脈を追記する。
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
    """ユーザー入力を解析し、`GateDecision`と補助情報を返す。

    Returns:
        tuple[GateDecision, str | None, dict[str, int] | None]:
            1) ルーティング判定
            2) reasoning要約（翻訳後、無ければNone）
            3) token usage辞書（取得できない場合はNone）
    """
    system_prompt = load_gate_prompt()
    messages = _build_chat_messages(system_prompt, user_input, chat_context, user_images)

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
    
    gate_decision_schema = _build_gate_decision_schema()
    
    # Invoke the LLM with direct response_format to enforce Structured Outputs
    response = llm.invoke(messages, response_format=gate_decision_schema)
    token_usage = extract_token_usage(response)
    
    reasoning, msg_content_json = _extract_reasoning_and_decision_json(response.content)
    decision = _parse_decision_with_override(msg_content_json, user_input, chat_context)

    # Translate reasoning to Japanese
    if reasoning:
        reasoning = translate_reasoning_to_japanese(reasoning)

    return decision, reasoning, token_usage
