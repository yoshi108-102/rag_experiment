import json

from src.core.chat_logging import ChatSessionLogger
from src.core.gate_trace_logging import log_gate_agent_trace


def test_chat_session_logger_writes_jsonl(tmp_path):
    logger = ChatSessionLogger.create(app_name="test-app", logs_root=tmp_path)
    logger.log_message("user", "こんにちは", message_index=1)
    logger.log_error(ValueError("bad input"), stage="unit_test")

    lines = logger.log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3

    records = [json.loads(line) for line in lines]
    assert records[0]["event_type"] == "session_started"
    assert records[1]["event_type"] == "message"
    assert records[1]["payload"]["role"] == "user"
    assert records[1]["payload"]["content"] == "こんにちは"
    assert records[2]["event_type"] == "error"
    assert records[2]["payload"]["error_type"] == "ValueError"
    assert records[2]["payload"]["error_message"] == "bad input"


def test_log_gate_agent_trace_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("GATE_TRACE_LOG_ENABLED", "1")

    log_gate_agent_trace(
        {
            "trace_id": "trace-test",
            "user_input": "曲がってるかわかりづらかった",
            "raw_response": {"type": "text", "text": '{"route":"CLARIFY"}'},
        },
        logs_root=tmp_path,
    )

    files = sorted(tmp_path.glob("*.jsonl"))
    assert len(files) == 1

    records = [
        json.loads(line) for line in files[0].read_text(encoding="utf-8").strip().splitlines()
    ]
    assert len(records) == 1
    assert records[0]["event_type"] == "gate_classifier"
    assert records[0]["payload"]["trace_id"] == "trace-test"
    assert records[0]["payload"]["user_input"] == "曲がってるかわかりづらかった"


def test_log_gate_agent_trace_skips_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("GATE_TRACE_LOG_ENABLED", "0")

    log_gate_agent_trace({"trace_id": "trace-off"}, logs_root=tmp_path)

    files = sorted(tmp_path.glob("*.jsonl"))
    assert files == []
