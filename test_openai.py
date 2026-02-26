from openai import OpenAI
import os
import json
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()
try:
    response = client.responses.create(
        model="gpt-5.2",
        input=[{"role": "user", "content": "Critical Dicision Methodについて日本語で小学生向けに説明してください"}],
        reasoning={
            "effort": "high",
            "summary": "detailed"
        },
        response_format={
            "type": "json_schema",
            "json_schema": {
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
    print("gpt-5.2 response:", response.model_dump_json(indent=2))
except Exception as e:
    print("gpt-5.2 error:", e)
