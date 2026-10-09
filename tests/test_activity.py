"""Chat activity stays scoped and contains safe operation summaries."""

from medibank.activity import request_activity
from medibank.observability import audit


def event(name, request="request-1", session="session-1", **details):
    return {
        "timestamp": "2026-10-09T05:00:01.123+00:00",
        "event": name,
        "session_id": session,
        "details": {"request_id": request, **details},
    }


def test_activity_uses_only_the_matching_request_and_session():
    rows = request_activity(
        "request-1",
        "session-1",
        [
            event("question_received"),
            event("retrieval_completed", query="benefits", returned_pages=3),
            event("question_received", request="other-request"),
            event("question_received", session="other-session"),
            event("index_loaded"),
        ],
    )
    assert len(rows) == 2
    assert rows[0]["Time (UTC)"] == "05:00:01.123"
    assert rows[1]["Details"] == "3 sources found · benefits"


def test_activity_does_not_return_raw_reasoning_prompts_answers_or_credentials():
    rows = request_activity(
        "request-1",
        "session-1",
        [
            event(
                "answer_completed",
                answer="private answer",
                reasoning="private reasoning",
                system_prompt="private prompt",
                api_key="private key",
                elapsed_ms=123,
            ),
            event(
                "retrieval_completed", query="benefits api_key=sk-private123456", returned_pages=1
            ),
        ],
    )
    text = str(rows)
    assert "private answer" not in text and "private reasoning" not in text
    assert "private prompt" not in text and "private key" not in text
    assert "sk-private123456" not in text
    assert "[REDACTED]" in text


def test_activity_reads_persisted_events_and_includes_human_route_and_duration():
    audit(
        "guardrail_decision",
        session_id="session-1",
        request_id="request-1",
        reason="human_requested",
    )
    audit(
        "answer_completed",
        session_id="session-1",
        request_id="request-1",
        elapsed_ms=10,
        human_review=True,
    )
    rows = request_activity("request-1", "session-1")
    assert rows[0]["Details"] == "Human review requested"
    assert rows[1]["Details"] == "10 ms · Saved for human review"


def test_activity_ignores_missing_ids_and_damaged_records():
    assert request_activity(None, "session-1", [event("question_received")]) == []
    assert request_activity("request-1", "", [event("question_received")]) == []
    assert request_activity("request-1", "session-1", [None, {"details": None}]) == []
