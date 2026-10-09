"""Local operation logs and human review requests."""

import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from medibank.deployment import public_mode
from medibank.observability import list_handoffs, read_events, redact, update_handoff

if public_mode():
    st.error("Logs and human review administration are available in the local app only.")
    st.stop()

PROJECT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_DIR / ".env")
st.set_page_config(page_title="Logs and human review", page_icon="📋", layout="wide")
st.title("Logs and human review")
st.caption("Review application events, guardrail decisions, and requests that need a person.")
events_tab, reviews_tab, runtime_tab = st.tabs(
    ["Application events", "Human review", "Server output"]
)
with st.sidebar:
    directory_text = st.text_input("Log directory", os.getenv("LOG_DIRECTORY", "data/logs"))
    directory = Path(directory_text).expanduser()
    if not directory.is_absolute():
        directory = PROJECT_DIR / directory
    st.button("Refresh logs")
    st.caption("Logs and review requests are stored locally. Credential values are redacted.")
with events_tab:
    try:
        events = read_events(directory, limit=2000)
    except OSError:
        events = []
        st.error("The log folder could not be read.")
    level = st.selectbox("Log level", ["All levels", "INFO", "WARNING", "ERROR"])
    event_type = st.selectbox(
        "Event type", ["All events", *sorted({item["event"] for item in events})]
    )
    contains = st.text_input(
        "Filter logs", placeholder="Question text, request ID, session, or event details"
    )
    filtered = [
        item
        for item in events
        if (level == "All levels" or item["level"] == level)
        and (event_type == "All events" or item["event"] == event_type)
        and (not contains or contains.casefold() in json.dumps(item).casefold())
    ]
    st.metric("Matching events", len(filtered))
    if filtered:
        st.dataframe(
            [
                {
                    "Time (UTC)": item["timestamp"],
                    "Level": item["level"],
                    "Event": item["event"],
                    "Session": item.get("session_id") or "CLI / backend",
                    "Details": json.dumps(item.get("details", {}), ensure_ascii=False),
                }
                for item in reversed(filtered)
            ],
            hide_index=True,
            width="stretch",
        )
        event_id = st.selectbox("Event to inspect", [item["id"] for item in reversed(filtered)])
        st.json(next(item for item in filtered if item["id"] == event_id))
        st.download_button(
            "Download filtered logs",
            json.dumps(filtered, indent=2, ensure_ascii=False),
            "application-logs.json",
            "application/json",
        )
    else:
        st.info("No log events match these filters yet.")
with reviews_tab:
    st.info("This is a local review queue. An external human agent is not connected automatically.")
    st.link_button("Contact Medibank", "https://www.medibank.com.au/contact-us/")
    try:
        tickets = list_handoffs()
    except OSError:
        tickets = []
        st.error("The human review queue could not be read.")
    status_filter = st.selectbox(
        "Review status", ["All requests", "pending", "reviewed", "resolved"]
    )
    tickets = [
        item
        for item in tickets
        if status_filter == "All requests" or item["status"] == status_filter
    ]
    st.metric("Review requests", len(tickets))
    if tickets:
        st.dataframe(
            [
                {
                    "Request": item["id"][:12],
                    "Created (UTC)": item["timestamp"],
                    "Status": item["status"],
                    "Reason": item["reason"],
                    "Question": item["question"],
                }
                for item in tickets
            ],
            hide_index=True,
            width="stretch",
        )
        ticket_id = st.selectbox("Request to review", [item["id"] for item in tickets])
        ticket = next(item for item in tickets if item["id"] == ticket_id)
        st.text(ticket["question"])
        status = st.selectbox(
            "Update status",
            ["pending", "reviewed", "resolved"],
            index=["pending", "reviewed", "resolved"].index(ticket["status"]),
            key=f"review_status_{ticket_id}",
        )
        note = st.text_area("Review note", ticket.get("note", ""), key=f"review_note_{ticket_id}")
        if st.button("Save review status"):
            try:
                update_handoff(ticket_id, status, note)
                st.rerun()
            except (OSError, ValueError):
                st.error("This review update could not be saved.")
        st.download_button(
            "Download review request",
            json.dumps(ticket, indent=2, ensure_ascii=False),
            f"human-review-{ticket_id[:12]}.json",
            "application/json",
        )
    else:
        st.info("No human review requests match this status.")
with runtime_tab:
    path = directory / "streamlit-server.log"
    if path.is_file():
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 8000))
            text = stream.read().decode("utf-8", errors="replace")
        st.code(redact(text), language="text")
    else:
        st.info("Start the application with run.ps1 to capture Streamlit server output here.")
