"""Gate判定後に適用する決定論ガードレール。"""

from __future__ import annotations

import re

from src.core.models import GateDecision


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

LOW_DETAIL_ABSTRACT_PATTERNS = (
    r"わかりづら",
    r"分かりづら",
    r"見えづら",
    r"見づら",
    r"難し",
    r"困っ",
    r"迷っ",
    r"きつ",
    r"しんど",
    r"疲れ",
)

DETAIL_REPORT_PATTERNS = (
    r"\d",
    r"(mm|cm|kg|度|秒|分|回)",
    r"どのタイミング",
    r"いつ",
    r"どこ",
)

VISIBILITY_PATTERNS = (
    r"見えづら",
    r"見づら",
    r"見にく",
    r"わかりづら",
    r"分かりづら",
)

FATIGUE_PATTERNS = (
    r"きつ",
    r"しんど",
    r"疲れ",
    r"重",
)

CONFUSION_PATTERNS = (
    r"迷っ",
    r"困っ",
    r"わから",
)

OVER_SPECIFIC_QUESTION_PATTERNS = (
    r"どのタイミング",
    r"何(mm|cm|kg|度|秒|分|回)",
)

CAUSAL_CONNECTOR_PATTERNS = (
    r"から",
    r"ので",
    r"ため",
    r"せいで",
    r"影響で",
    r"都合上",
)


def matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    """文字列が正規表現パターン群のいずれかに一致するかを返す。"""
    return any(re.search(pattern, text) for pattern in patterns)


def recent_user_texts(chat_context: list | None, limit: int = 4) -> list[str]:
    """履歴から直近ユーザー発話のみを取り出す。"""
    if not chat_context:
        return []
    texts = [msg.get("content", "") for msg in chat_context if msg.get("role") == "user"]
    return texts[-limit:]


def last_assistant_text(chat_context: list | None) -> str:
    """履歴上で最後のassistant発話を返す。"""
    if not chat_context:
        return ""
    for msg in reversed(chat_context):
        if msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    return ""


def looks_like_confirmation_question(text: str) -> bool:
    """確認質問らしい文かを簡易ヒューリスティックで判定する。"""
    if not text:
        return False
    markers = ("つまり", "ってこと", "でいい", "かな", "ですか", "統一", "言い方")
    return "?" in text or "？" in text or any(marker in text for marker in markers)


def normalize_text_for_slots(text: str) -> str:
    """slot抽出向けに空白と文末記号を削って正規化する。"""
    return re.sub(r"\s+", "", text or "").strip("。．!！")


def extract_causal_pair(text: str) -> tuple[str | None, str | None]:
    """因果接続語を手掛かりに、理由と項目のペアを抽出する。"""
    compact = normalize_text_for_slots(text)
    if not compact or len(compact) < 10:
        return None, None
    if compact.endswith(("?", "？")):
        return None, None

    for connector in CAUSAL_CONNECTOR_PATTERNS:
        if connector in compact:
            left, right = compact.split(connector, 1)
            left = left.strip("、, ")
            right = right.strip("、, ")
            if len(left) >= 3 and len(right) >= 3:
                return left, right

    for connector in ("そのため", "その結果"):
        if connector in compact:
            left, right = compact.split(connector, 1)
            left = left.strip("、, ")
            right = right.strip("、, ")
            if len(left) >= 3 and len(right) >= 3:
                return left, right

    return None, None


def is_meaningful_user_text(text: str) -> bool:
    """短すぎる相槌を除外し、slot抽出対象として有効か判定する。"""
    compact = normalize_text_for_slots(text)
    if len(compact) < 4:
        return False
    if matches_any(compact, SHORT_AFFIRMATION_PATTERNS):
        return False
    if matches_any(compact, LOW_SIGNAL_ENDING_PATTERNS):
        return False
    return True


def infer_idea_or_question_kind(text: str) -> str | None:
    """発話を`idea`または`question`のどちらかへ推定する。"""
    compact = normalize_text_for_slots(text)
    if not compact:
        return None
    if matches_any(compact, QUESTION_MARKER_PATTERNS):
        return "question"
    return "idea"


def text_information_score(text: str) -> int:
    """発話の情報量をざっくり採点する（高いほどitem候補として優先）。"""
    compact = normalize_text_for_slots(text)
    if not compact:
        return -1

    score = len(compact)
    extracted_reason, extracted_item = extract_causal_pair(compact)
    if extracted_reason and extracted_item:
        score += 50
    if matches_any(compact, QUESTION_MARKER_PATTERNS):
        score += 10
    return score


def select_best_text(candidates: list[tuple[int, str]]) -> str | None:
    """候補文から情報量と新しさを基準に最良の文を選ぶ。"""
    if not candidates:
        return None
    _, selected = max(candidates, key=lambda item: (text_information_score(item[1]), item[0]))
    return selected


def count_prior_user_turns(chat_context: list | None) -> int:
    """今回入力以前のユーザー発話数を返す。"""
    if not chat_context:
        return 0

    return sum(
        1
        for msg in chat_context
        if msg.get("role") == "user"
        and normalize_text_for_slots(str(msg.get("content", "")))
    )


def is_low_detail_abstract_report(text: str) -> bool:
    """抽象的で詳細が少ない報告かを判定する。"""
    compact = normalize_text_for_slots(text)
    if len(compact) < 5:
        return False
    if infer_idea_or_question_kind(compact) == "question":
        return False
    if not matches_any(compact, LOW_DETAIL_ABSTRACT_PATTERNS):
        return False
    if matches_any(compact, DETAIL_REPORT_PATTERNS):
        return False
    return True


def extract_binary_options(question: str) -> tuple[str, str] | None:
    """`AとBどっち` 形式から二択の候補を抽出する。"""
    match = re.search(
        r"(?P<a>[^、。！？?]{2,20})と(?P<b>[^、。！？?]{2,20})(どっち|どちら)",
        question,
    )
    if not match:
        return None
    return (
        normalize_text_for_slots(match.group("a")),
        normalize_text_for_slots(match.group("b")),
    )


def introduces_unseen_binary_options(question: str, user_input: str) -> bool:
    """質問内の二択候補がユーザー未提示かを判定する。"""
    options = extract_binary_options(question)
    if options is None:
        return False

    user_text = normalize_text_for_slots(user_input)
    return all(option and option not in user_text for option in options)


def looks_over_specific_follow_up(question: str, user_input: str) -> bool:
    """ユーザー入力に対して質問が絞り込み過剰かを判定する。"""
    compact_question = normalize_text_for_slots(question)
    if not compact_question:
        return False

    if introduces_unseen_binary_options(compact_question, user_input):
        return True

    return matches_any(compact_question, OVER_SPECIFIC_QUESTION_PATTERNS)


def build_broad_gather_question(user_input: str) -> str:
    """抽象入力に対して使う、広めの情報収集質問を返す。"""
    compact = normalize_text_for_slots(user_input)

    if matches_any(compact, VISIBILITY_PATTERNS):
        return "どんな感じで見づらかったのか、もう少し聞かせて。"
    if matches_any(compact, FATIGUE_PATTERNS):
        return "どんなやり方のときに一番きつかった？"
    if matches_any(compact, CONFUSION_PATTERNS):
        return "どこから迷い始めた感じだった？"
    return "どんな感じで困ったのか、もう少し聞かせて。"


def should_force_broad_gather(
    decision: GateDecision,
    user_input: str,
    chat_context: list | None = None,
) -> bool:
    """初期ターンでは絞り込みより広い聞き方を優先すべきかを判定する。"""
    if decision.route not in {"DEEPEN", "CLARIFY"}:
        return False
    if not is_low_detail_abstract_report(user_input):
        return False
    if count_prior_user_turns(chat_context) >= 2:
        return False
    return looks_over_specific_follow_up(decision.first_question, user_input)


def build_clarify_completion_json(
    user_input: str,
    chat_context: list | None = None,
) -> dict[str, str | bool | None]:
    """CLARIFY完了判定のためのslotをルールベースで抽出する。"""
    user_texts = recent_user_texts(chat_context, limit=6)
    if not user_texts or user_texts[-1] != user_input:
        user_texts.append(user_input)

    meaningful_texts: list[tuple[int, str]] = []
    causal_pairs: list[tuple[int, str, str, str]] = []

    for idx, text in enumerate(user_texts):
        if not is_meaningful_user_text(text):
            continue

        compact = normalize_text_for_slots(text)
        meaningful_texts.append((idx, compact))

        extracted_reason, extracted_item = extract_causal_pair(compact)
        if extracted_reason and extracted_item:
            causal_pairs.append((idx, extracted_reason, extracted_item, compact))

    kind: str | None = None
    item: str | None = None
    reason: str | None = None

    if causal_pairs:
        _, reason, item, pair_source_text = max(causal_pairs, key=lambda item: item[0])
        kind = infer_idea_or_question_kind(pair_source_text)

    if kind is None:
        best_text = select_best_text(meaningful_texts)
        if best_text is not None:
            kind = infer_idea_or_question_kind(best_text)

    if item is None:
        item = select_best_text(meaningful_texts)

    return {
        "kind": kind,
        "item": item,
        "reason": reason,
        "is_complete": bool(kind and item and reason),
    }


def apply_decision_overrides(
    decision: GateDecision,
    user_input: str,
    chat_context: list | None = None,
) -> GateDecision:
    """LLM判定へ決定論的なガードレールを後適用する。"""
    if decision.route == "FINISH":
        return decision

    text = user_input.strip()
    if text and matches_any(text, FINISH_INTENT_PATTERNS):
        return GateDecision(
            route="FINISH",
            reason="Closure intent detected",
            first_question="話せて少しスッキリしたなら、今日はここで区切ってよさそうですね。",
        )

    if text and matches_any(text, FRUSTRATION_PATTERNS):
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

    recent_texts = recent_user_texts(chat_context)
    recent_has_frustration = any(
        matches_any(candidate, FRUSTRATION_PATTERNS) for candidate in recent_texts[:-1]
    )
    last_assistant = last_assistant_text(chat_context)
    if (
        text
        and matches_any(text, SHORT_AFFIRMATION_PATTERNS)
        and recent_has_frustration
        and looks_like_confirmation_question(last_assistant)
    ):
        return GateDecision(
            route="FINISH",
            reason="Confirmed after friction",
            first_question="うん、意図は十分伝わったので、ここで区切って大丈夫そうです。",
        )

    if should_force_broad_gather(decision, text, chat_context):
        return GateDecision(
            route=decision.route,
            reason="Broad gather before narrowing",
            first_question=build_broad_gather_question(text),
        )

    return decision
