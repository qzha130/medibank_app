"""Request-scoped summaries of observed agent operations for the chat UI."""

from __future__ import annotations

from medibank.observability import read_events, redact

_EVENTS = {
    "question_received": "Question received",
    "guardrail_decision": "Scope and evidence checked",
    "chat_model_started": "Preparing an answer",
    "retrieval_completed": "Knowledge searched",
    "retrieval_failed": "Knowledge search unavailable",
    "citation_revision_started": "Source references checked",
    "citation_validation_failed": "Source references could not be verified",
    "answer_completed": "Response completed",
    "chat_failed": "Response unavailable",
}
_REASONS = {
    "off_topic": "General Medibank questions only",
    "instruction_override": "Request outside the assistant's scope",
    "human_requested": "Human review requested",
    "personal_or_clinical_request": "Personal or professional review required",
    "insufficient_evidence": "More information or human review required",
    "retrieval_unavailable": "Knowledge unavailable; human review required",
    "relevant_evidence": "Relevant information found",
}


def request_activity(
    request_id: str | None,
    session_id: str,
    events: list[dict] | None = None,
) -> list[dict[str, str]]:
    """Show only operational events for this exact request and browser session."""
    if not request_id or not session_id:
        return []
    if events is None:
        try:
            events = read_events(limit=2000)
        except (OSError, ValueError, TypeError):
            return []
    rows = []
    for event in events:
        if not isinstance(event, dict) or event.get("session_id") != session_id:
            continue
        details = event.get("details")
        name = event.get("event")
        if (
            not isinstance(details, dict)
            or details.get("request_id") != request_id
            or not isinstance(name, str)
            or name not in _EVENTS
        ):
            continue
        detail = ""
        if name == "guardrail_decision":
            reason = details.get("reason")
            detail = (
                _REASONS.get(reason, "Request checked")
                if isinstance(reason, str)
                else "Request checked"
            )
        elif name == "chat_model_started":
            detail = str(details.get("model", ""))
        elif name == "retrieval_completed":
            detail = f"{details.get('returned_pages', 0)} sources found"
            if isinstance(details.get("query"), str):
                detail += " · " + details["query"][:500]
        elif name == "answer_completed":
            detail = f"{details.get('elapsed_ms', 0)} ms"
            if details.get("human_review"):
                detail += " · Saved for human review"
        elif name == "chat_failed":
            detail = "Please retry or use human review"
        timestamp = event.get("timestamp", "")
        rows.append(
            redact(
                {
                    "Time (UTC)": timestamp[11:23] if isinstance(timestamp, str) else "",
                    "Activity": _EVENTS[name],
                    "Details": detail,
                }
            )
        )
    return rows
