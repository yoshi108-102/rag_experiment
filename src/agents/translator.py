"""Gateモデルのreasoningテキストを日本語へ翻訳する補助処理。"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


def translate_reasoning_to_japanese(reasoning_text: str) -> str | None:
    """英語中心のreasoning要約を、日本語として読みやすい文章へ変換する。

    翻訳に失敗した場合は会話継続を優先し、元テキストを返す。
    """
    if not reasoning_text:
        return None
        
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    system_prompt = (
        "You are an expert translator. "
        "Translate the following internal AI reasoning text into natural, easy-to-understand Japanese. "
        "Maintain the logical flow and meaning, but output only the translated Japanese text."
    )
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=reasoning_text)
        ])
        
        return str(response.content)
        
    except Exception as e:
        print(f"[Translator Error] Failed to translate reasoning: {e}")
        # 翻訳失敗時は機能停止を避けるため原文を返す。
        return reasoning_text
