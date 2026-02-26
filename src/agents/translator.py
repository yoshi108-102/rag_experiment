import os
from openai import OpenAI

def translate_reasoning_to_japanese(reasoning_text: str) -> str | None:
    """
    Translates the AI's internal reasoning (which is usually in English) into natural Japanese.
    Utilizes a fast LLM model like gpt-4o-mini for efficient translation.
    """
    if not reasoning_text:
        return None
        
    client = OpenAI()
    
    system_prompt = (
        "You are an expert translator. "
        "Translate the following internal AI reasoning text into natural, easy-to-understand Japanese. "
        "Maintain the logical flow and meaning, but output only the translated Japanese text."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": reasoning_text}
            ],
            temperature=0.3  # Keep it relatively deterministic
        )
        
        translated_text = response.choices[0].message.content
        return translated_text
        
    except Exception as e:
        print(f"[Translator Error] Failed to translate reasoning: {e}")
        # Fallback to the original text if translation fails
        return reasoning_text
