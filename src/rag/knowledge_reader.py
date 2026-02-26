from __future__ import annotations

import json
from pathlib import Path

from src.rag.models import KnowledgeRecord


def default_consolidated_knowledge_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent.parent
        / "datasets"
        / "consolidated"
        / "consolidated_knowledge.json"
    )


def load_consolidated_knowledge_records(
    path: str | Path | None = None,
) -> list[KnowledgeRecord]:
    source_path = Path(path) if path else default_consolidated_knowledge_path()
    if not source_path.exists():
        return []

    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    topics = data.get("consolidated_knowledge", [])
    records: list[KnowledgeRecord] = []

    for topic_index, topic_item in enumerate(topics):
        topic = str(topic_item.get("topic", "")).strip()
        tags = [str(tag).strip() for tag in topic_item.get("tags", []) if str(tag).strip()]

        summary = str(topic_item.get("summary_statement", "")).strip()
        if summary:
            records.append(
                KnowledgeRecord(
                    record_id=f"topic-{topic_index}-summary",
                    topic=topic,
                    tags=tags,
                    record_type="summary",
                    text=summary,
                )
            )

        for obs_index, text in enumerate(topic_item.get("related_observations", [])):
            observation = str(text).strip()
            if not observation:
                continue
            records.append(
                KnowledgeRecord(
                    record_id=f"topic-{topic_index}-obs-{obs_index}",
                    topic=topic,
                    tags=tags,
                    record_type="observation",
                    text=observation,
                )
            )

        for method_index, method in enumerate(topic_item.get("methods", [])):
            description = str(method.get("description", "")).strip()
            if not description:
                continue
            applicable_when = str(method.get("applicable_when", "")).strip() or None
            records.append(
                KnowledgeRecord(
                    record_id=f"topic-{topic_index}-method-{method_index}",
                    topic=topic,
                    tags=tags,
                    record_type="method",
                    text=description,
                    applicable_when=applicable_when,
                )
            )

        for question_index, text in enumerate(topic_item.get("open_questions", [])):
            question = str(text).strip()
            if not question:
                continue
            records.append(
                KnowledgeRecord(
                    record_id=f"topic-{topic_index}-q-{question_index}",
                    topic=topic,
                    tags=tags,
                    record_type="open_question",
                    text=question,
                )
            )

    return records


# Backward-compatible aliases while callers migrate.
default_consolidated_path = default_consolidated_knowledge_path
load_knowledge_records = load_consolidated_knowledge_records
