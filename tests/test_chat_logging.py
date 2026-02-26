import json

from src.core.chat_logging import ChatSessionLogger


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
