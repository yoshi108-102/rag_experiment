"""chat_sessions / gate_agent_traces のJSONLを集計して指標を出力する。"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass
class ChatMetrics:
    sessions: int
    message_events: int
    assistant_turns: int
    route_counts: Counter[str]
    rag_enabled_count: int
    rag_skipped_counts: Counter[str]
    input_tokens: list[int]
    output_tokens: list[int]
    total_tokens: list[int]


@dataclass
class GateTraceMetrics:
    trace_events: int
    model_counts: Counter[str]
    decision_route_counts: Counter[str]
    parse_failures: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                records.append(item)
    return records


def _scan_jsonl_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    return sorted(root.glob("*.jsonl"))


def collect_chat_metrics(chat_paths: list[Path]) -> ChatMetrics:
    route_counts: Counter[str] = Counter()
    rag_skipped_counts: Counter[str] = Counter()
    input_tokens: list[int] = []
    output_tokens: list[int] = []
    total_tokens: list[int] = []

    sessions = len(chat_paths)
    message_events = 0
    assistant_turns = 0
    rag_enabled_count = 0

    for path in chat_paths:
        for record in _read_jsonl(path):
            if record.get("event_type") != "message":
                continue
            message_events += 1
            payload = record.get("payload") or {}
            if payload.get("role") != "assistant":
                continue

            assistant_turns += 1
            debug = payload.get("debug_info") or {}
            route = debug.get("route")
            if isinstance(route, str) and route:
                route_counts[route] += 1

            rag = debug.get("rag") or {}
            if isinstance(rag, dict):
                if rag.get("enabled"):
                    rag_enabled_count += 1
                skipped_reason = rag.get("skipped_reason")
                if isinstance(skipped_reason, str) and skipped_reason:
                    rag_skipped_counts[skipped_reason] += 1

            usage = debug.get("token_usage") or {}
            if isinstance(usage, dict):
                _append_int(usage.get("input_tokens"), input_tokens)
                _append_int(usage.get("output_tokens"), output_tokens)
                _append_int(usage.get("total_tokens"), total_tokens)

    return ChatMetrics(
        sessions=sessions,
        message_events=message_events,
        assistant_turns=assistant_turns,
        route_counts=route_counts,
        rag_enabled_count=rag_enabled_count,
        rag_skipped_counts=rag_skipped_counts,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def collect_gate_trace_metrics(trace_paths: list[Path]) -> GateTraceMetrics:
    model_counts: Counter[str] = Counter()
    decision_route_counts: Counter[str] = Counter()
    parse_failures = 0
    trace_events = 0

    for path in trace_paths:
        for record in _read_jsonl(path):
            if record.get("event_type") != "gate_classifier":
                continue
            trace_events += 1
            payload = record.get("payload") or {}

            model_name = payload.get("model_name")
            if isinstance(model_name, str) and model_name:
                model_counts[model_name] += 1

            decision = payload.get("decision") or {}
            if isinstance(decision, dict):
                route = decision.get("route")
                if isinstance(route, str) and route:
                    decision_route_counts[route] += 1

            decision_error = payload.get("error")
            parse_json = payload.get("msg_content_json")
            if decision_error or not parse_json:
                parse_failures += 1

    return GateTraceMetrics(
        trace_events=trace_events,
        model_counts=model_counts,
        decision_route_counts=decision_route_counts,
        parse_failures=parse_failures,
    )


def _append_int(value: Any, dst: list[int]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        dst.append(value)


def _avg(values: list[int]) -> float | None:
    return round(mean(values), 2) if values else None


def build_summary(chat: ChatMetrics, gate: GateTraceMetrics) -> dict[str, Any]:
    return {
        "chat_sessions": {
            "session_files": chat.sessions,
            "message_events": chat.message_events,
            "assistant_turns": chat.assistant_turns,
            "route_counts": dict(chat.route_counts),
            "rag_enabled_count": chat.rag_enabled_count,
            "rag_skipped_counts": dict(chat.rag_skipped_counts),
            "avg_input_tokens": _avg(chat.input_tokens),
            "avg_output_tokens": _avg(chat.output_tokens),
            "avg_total_tokens": _avg(chat.total_tokens),
        },
        "gate_traces": {
            "trace_events": gate.trace_events,
            "model_counts": dict(gate.model_counts),
            "decision_route_counts": dict(gate.decision_route_counts),
            "parse_failures": gate.parse_failures,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="JSONLログの集計")
    parser.add_argument(
        "--chat-dir",
        default="logs/chat_sessions",
        help="chat session JSONL のディレクトリまたはファイル",
    )
    parser.add_argument(
        "--trace-dir",
        default="logs/gate_agent_traces",
        help="gate trace JSONL のディレクトリまたはファイル",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="集計JSONの保存先（省略時は標準出力のみ）",
    )
    args = parser.parse_args()

    chat_paths = _scan_jsonl_files(Path(args.chat_dir))
    trace_paths = _scan_jsonl_files(Path(args.trace_dir))

    chat_metrics = collect_chat_metrics(chat_paths)
    gate_metrics = collect_gate_trace_metrics(trace_paths)
    summary = build_summary(chat_metrics, gate_metrics)

    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
