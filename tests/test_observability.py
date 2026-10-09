"""Structured local audit events and durable human review tickets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import medibank.observability as observability


@pytest.fixture
def local_records(monkeypatch, tmp_path):
    logs = tmp_path / "logs"
    handoffs = tmp_path / "handoffs"
    monkeypatch.setenv("LOG_DIRECTORY", str(logs))
    monkeypatch.setenv("HANDOFF_DIRECTORY", str(handoffs))
    monkeypatch.setenv("LOG_MAX_BYTES", "2097152")
    monkeypatch.setenv("LOG_BACKUP_COUNT", "3")
    yield logs, handoffs
    # Persistent service loggers must release fixture handles on Windows.
    for identity, logger in list(observability._LOGGERS.items()):
        if Path(identity).is_relative_to(tmp_path):
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)
            observability._LOGGERS.pop(identity, None)


def test_audit_redacts_nested_fields_and_inline_credentials_before_writing(local_records):
    logs, _ = local_records
    secrets = [
        "private-api-key",
        "private-auth-header",
        "private-url-token",
        "sk-abcdefgh12345678",
        "AIzaABCDEFGHIJKLMNOPQRSTUVWX",
        "private-bearer",
        "private-assignment",
    ]
    event_id = observability.audit(
        "provider_failed",
        level="WARNING",
        session_id="session-123",
        api_key=secrets[0],
        provider={"Authorization": secrets[1], "url": f"https://example.com/v1?token={secrets[2]}"},
        message=f"Key {secrets[3]} and {secrets[4]}; Bearer {secrets[5]}; password={secrets[6]}",
    )
    assert event_id
    events = observability.read_events()
    assert len(events) == 1
    assert events[0]["id"] == event_id
    assert events[0]["event"] == "provider_failed"
    assert events[0]["level"] == "WARNING"
    assert events[0]["session_id"] == "session-123"
    serialized = (logs / "events.jsonl").read_text(encoding="utf-8")
    assert all(secret not in serialized for secret in secrets)
    assert "[REDACTED]" in serialized


def test_read_events_skips_damaged_lines_and_redacts_legacy_records(local_records):
    logs, _ = local_records
    logs.mkdir()
    legacy = {
        "id": "legacy-id",
        "timestamp": "2026-10-09T01:00:00+00:00",
        "event": "legacy_event",
        "level": "INFO",
        "details": {"api_key": "legacy-private-key", "message": "Bearer legacy-private-token"},
    }
    path = logs / "events.jsonl"
    original = "{interrupted line\n" + json.dumps(legacy) + "\n[]\n"
    path.write_text(original, encoding="utf-8")
    events = observability.read_events()
    assert len(events) == 1
    assert events[0]["id"] == "legacy-id"
    assert "legacy-private-key" not in json.dumps(events)
    assert "legacy-private-token" not in json.dumps(events)
    assert path.read_text(encoding="utf-8") == original


def test_rotation_keeps_recent_events_readable_and_limit_selects_latest(local_records, monkeypatch):
    logs, _ = local_records
    monkeypatch.setenv("LOG_MAX_BYTES", "1024")
    event_ids = [
        observability.audit("rotating_event", sequence=index, message="x" * 650)
        for index in range(8)
    ]
    assert len(list(logs.glob("events.jsonl*"))) > 1
    events = observability.read_events()
    assert len(events) > 1
    assert [event["timestamp"] for event in events] == sorted(
        event["timestamp"] for event in events
    )
    assert events[-1]["id"] == event_ids[-1]
    assert [event["id"] for event in observability.read_events(limit=2)] == event_ids[-2:]


def test_review_ticket_updates_preserve_request_and_redact_saved_notes(local_records):
    _, handoffs = local_records
    ticket = observability.create_handoff(
        "Please review my claim.", "personal_request", "session-456"
    )
    assert ticket["status"] == "pending"
    reviewed = observability.update_handoff(
        ticket["id"], "reviewed", "Checked; token=private-review-token"
    )
    assert reviewed["id"] == ticket["id"]
    assert reviewed["question"] == ticket["question"]
    assert reviewed["reason"] == ticket["reason"]
    assert reviewed["timestamp"] == ticket["timestamp"]
    latest = observability.list_handoffs()
    assert len(latest) == 1
    assert latest[0]["status"] == "reviewed"
    assert "private-review-token" not in json.dumps(latest)
    assert "private-review-token" not in (handoffs / "requests.jsonl").read_text(encoding="utf-8")
    observability.update_handoff(ticket["id"], "resolved", "Review completed.")
    assert observability.list_handoffs()[0]["status"] == "resolved"


@pytest.mark.parametrize(
    "ticket_id,status", [("existing", "invalid-status"), ("not-found", "reviewed")]
)
def test_invalid_review_update_does_not_change_queue(local_records, ticket_id, status):
    _, handoffs = local_records
    ticket = observability.create_handoff("Please review my claim.", "personal_request", "session")
    path = handoffs / "requests.jsonl"
    before = path.read_bytes()
    selected_id = ticket["id"] if ticket_id == "existing" else ticket_id
    with pytest.raises(ValueError):
        observability.update_handoff(selected_id, status, "Invalid update")
    assert path.read_bytes() == before
    assert observability.list_handoffs()[0]["status"] == "pending"


def test_damaged_queue_lines_do_not_hide_valid_requests(local_records):
    _, handoffs = local_records
    ticket = observability.create_handoff("Please review my claim.", "personal_request")
    with (handoffs / "requests.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{interrupted write\n")
    assert [item["id"] for item in observability.list_handoffs()] == [ticket["id"]]


def test_reading_empty_logs_and_queue_creates_no_files(local_records):
    logs, handoffs = local_records
    assert observability.read_events() == []
    assert observability.list_handoffs() == []
    assert not logs.exists()
    assert not handoffs.exists()


def test_audit_storage_error_does_not_crash_the_application(local_records, monkeypatch, tmp_path):
    blocked = tmp_path / "blocked-directory"
    blocked.write_text("A file occupies the configured log directory.", encoding="utf-8")
    monkeypatch.setenv("LOG_DIRECTORY", str(blocked))
    assert observability.audit("test_event") is None
