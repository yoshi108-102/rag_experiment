"""Audit export utilities for in-memory CTA sessions."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from src.cta.metrics import summarize_turn_latency
from src.cta.store import InMemoryCTAStore


def export_session_artifacts(
    store: InMemoryCTAStore,
    session_id: str,
    output_dir: str | Path,
) -> dict[str, str]:
    """Export in-memory turns/traces/summary into JSONL + JSON files."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    turns = store.list_turns(session_id)
    traces = store.list_generation_traces(session_id)
    knowledge_candidates = store.list_knowledge_candidates(session_id)
    snapshot = store.build_audit_snapshot(session_id)
    metrics = summarize_turn_latency(turns)

    turns_path = out_dir / f"{session_id}_turns.jsonl"
    traces_path = out_dir / f"{session_id}_generation_traces.jsonl"
    knowledge_path = out_dir / f"{session_id}_knowledge_candidates.jsonl"
    summary_path = out_dir / f"{session_id}_summary.json"

    _write_jsonl(turns_path, [asdict(turn) for turn in turns])
    _write_jsonl(traces_path, [asdict(trace) for trace in traces])
    _write_jsonl(knowledge_path, [asdict(item) for item in knowledge_candidates])

    summary: dict[str, Any] = {
        "session_id": session_id,
        "session": snapshot["session"],
        "turn_count": len(turns),
        "trace_count": len(traces),
        "knowledge_candidate_count": len(knowledge_candidates),
        "performance": asdict(metrics),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "turns_jsonl": str(turns_path),
        "traces_jsonl": str(traces_path),
        "knowledge_jsonl": str(knowledge_path),
        "summary_json": str(summary_path),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    payload = "\n".join(lines)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")
