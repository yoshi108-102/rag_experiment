import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


def translate_reasoning_to_japanese(reasoning_text: str) -> str | None:
    """
    Translates the AI's internal reasoning (which is usually in English) into natural Japanese.
    Utilizes a fast LLM model like gpt-4o-mini for efficient translation.
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
        # Fallback to the original text if translation fails
        return reasoning_text
