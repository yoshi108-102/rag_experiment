"""Gate分類を行うチェーン。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from src.core.models import GateDecision
from src.core.token_usage import default_gate_model_name, extract_token_usage
from src.middleware.decision_guard import apply_decision_overrides
from src.middleware.prompt_middleware import build_chat_messages, gate_system_prompt_middleware


PARK_PARSE_ERROR_MESSAGE = "解析エラーが発生しました。"
PARK_NO_CONTENT_MESSAGE = "応答が得られませんでした。"


def build_gate_decision_schema() -> dict[str, Any]:
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


def extract_reasoning_and_decision_json(
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


def park_decision(reason: str, first_question: str) -> GateDecision:
    """PARKルートのフォールバック判定を作る。"""
    return GateDecision(route="PARK", reason=reason, first_question=first_question)


def parse_decision_with_override(
    msg_content_json: str | None,
    user_input: str,
    chat_context: list | None,
) -> GateDecision:
    """判定JSONを検証して読み込み、必要な後段overrideを適用する。"""
    if msg_content_json:
        try:
            decision = GateDecision.model_validate_json(msg_content_json)
            return apply_decision_overrides(decision, user_input, chat_context)
        except Exception as e:
            print(f"[Gate Error] Pydantic validation failed: {e}")
            return park_decision(
                "Pydantic validation failed",
                PARK_PARSE_ERROR_MESSAGE,
            )

    return park_decision("No response content", PARK_NO_CONTENT_MESSAGE)


def build_gate_invoke_middleware(response_format: dict[str, Any]):
    """Gate判定向けのモデル実行 middleware を生成する。"""

    @wrap_model_call(name="GateInvokeMiddleware")
    def gate_invoke(request: ModelRequest, handler):  # type: ignore[no-untyped-def]
        del handler
        messages = list(request.messages)
        if request.system_message is not None:
            messages = [request.system_message, *messages]
        response = request.model.invoke(messages, response_format=response_format)

        # langgraph reducer expects message.id; some mocked AIMessage objects may miss it.
        try:
            _ = response.id
        except Exception:
            setattr(response, "id", "gate-response")

        return response

    return gate_invoke


def extract_last_ai_message(messages: list[Any]) -> AIMessage:
    """agent実行結果から最後のAIMessageを取り出す。"""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    raise ValueError("No AIMessage found in agent output")


@dataclass(frozen=True)
class GateClassifierChain:
    """Gate判定を1つの責務にまとめたチェーン。"""

    llm_factory: Callable[..., ChatOpenAI]
    reasoning_translator: Callable[[str], str | None]

    def classify(
        self,
        user_input: str,
        chat_context: list | None = None,
        user_images: list[dict[str, str]] | None = None,
    ) -> tuple[GateDecision, str | None, dict[str, int] | None]:
        """ユーザー入力を解析し、`GateDecision`と補助情報を返す。"""
        messages = build_chat_messages(user_input, chat_context, user_images)
        llm = self.llm_factory(
            model=default_gate_model_name(),
            use_responses_api=True,
            output_version="responses/v1",
            reasoning={
                "effort": "high",
                "summary": "detailed",
            },
        )
        gate_decision_schema = build_gate_decision_schema()
        agent = create_agent(
            model=llm,
            tools=[],
            middleware=[
                gate_system_prompt_middleware,
                build_gate_invoke_middleware(gate_decision_schema),
            ],
        )
        agent_output = agent.invoke({"messages": messages})
        response = extract_last_ai_message(
            list(agent_output.get("messages", [])),
        )

        token_usage = extract_token_usage(response)
        reasoning, msg_content_json = extract_reasoning_and_decision_json(response.content)
        decision = parse_decision_with_override(msg_content_json, user_input, chat_context)

        if reasoning:
            reasoning = self.reasoning_translator(reasoning)

        return decision, reasoning, token_usage
