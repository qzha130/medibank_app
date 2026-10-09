"""Local, structured operation logs and a review queue; never sends messages."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import deque
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import RLock
from uuid import uuid4

PROJECT_DIR = Path(__file__).resolve().parents[1]
_LOCK = RLock()
_LOGGERS: dict[str, logging.Logger] = {}
_SECRET_KEY = re.compile(r"api.?key|token|secret|password|authorization", re.IGNORECASE)
_INLINE_SECRET = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{8,}|AIza[A-Za-z0-9_-]{20,})\b|"
    r"\bBearer\s+[^\s,;]+|"
    r"(?i:((?:api[_ -]?key|access[_ -]?token|token|password|authorization)\s*[:=]\s*))(?:Bearer\s+)?[^\s,;&]+"
)


def _directory(env: str, default: str) -> Path:
    path = Path(os.getenv(env, default)).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def redact(value):
    """Remove credential fields and common inline credentials before writing/exporting."""
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _SECRET_KEY.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value[:100]]
    if isinstance(value, str):
        value = re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@", r"\1[REDACTED]@", value)
        for name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
            configured_key = os.getenv(name, "")
            if len(configured_key) >= 8:
                value = value.replace(configured_key, "[REDACTED]")
        return _INLINE_SECRET.sub(lambda match: (match.group(1) or "") + "[REDACTED]", value)[:8000]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact(str(value))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _logger(directory: Path) -> logging.Logger:
    identity = str(directory.resolve())
    with _LOCK:
        if identity not in _LOGGERS:
            directory.mkdir(parents=True, exist_ok=True)
            logger = logging.getLogger(
                "medibank.audit." + hashlib.sha256(identity.encode()).hexdigest()
            )
            logger.setLevel(logging.INFO)
            logger.propagate = False
            try:
                max_bytes = max(1024, int(os.getenv("LOG_MAX_BYTES", "2097152")))
                backups = min(20, max(1, int(os.getenv("LOG_BACKUP_COUNT", "3"))))
            except ValueError:
                max_bytes, backups = 2097152, 3
            handler = RotatingFileHandler(
                directory / "events.jsonl",
                maxBytes=max_bytes,
                backupCount=backups,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
            _LOGGERS[identity] = logger
        return _LOGGERS[identity]


def audit(event: str, level: str = "INFO", session_id: str | None = None, **details) -> str | None:
    event_id = uuid4().hex
    record = redact(
        {
            "id": event_id,
            "timestamp": _now(),
            "level": level.upper(),
            "event": event,
            "session_id": session_id,
            "details": details,
        }
    )
    try:
        _logger(_directory("LOG_DIRECTORY", "data/logs")).log(
            getattr(logging, level.upper(), logging.INFO), json.dumps(record, ensure_ascii=False)
        )
        return event_id
    except (OSError, ValueError):
        return None


def read_events(directory: Path | None = None, limit: int = 1000) -> list[dict]:
    directory = directory if directory is not None else _directory("LOG_DIRECTORY", "data/logs")
    if not directory.is_dir():
        return []
    records: deque = deque(maxlen=min(10000, max(1, limit)))
    files = sorted(directory.glob("events.jsonl*"), key=lambda path: path.stat().st_mtime_ns)
    for path in files:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                    if isinstance(record, dict) and all(
                        key in record for key in ("timestamp", "event", "level")
                    ):
                        records.append(redact(record))
                except (ValueError, TypeError):
                    continue
    return sorted(records, key=lambda record: record["timestamp"])


def _queue_path() -> Path:
    return _directory("HANDOFF_DIRECTORY", "data/handoffs") / "requests.jsonl"


def list_handoffs() -> list[dict]:
    path = _queue_path()
    if not path.is_file():
        return []
    requests = {}
    with _LOCK, path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                item = json.loads(line)
                if isinstance(item, dict) and item.get("id"):
                    requests[item["id"]] = redact(item)
            except ValueError:
                continue
    return sorted(requests.values(), key=lambda item: item["timestamp"], reverse=True)


def _save_ticket(ticket: dict) -> None:
    path = _queue_path()
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(redact(ticket), ensure_ascii=False) + "\n")


def create_handoff(question: str, reason: str, session_id: str | None = None) -> dict:
    ticket = redact(
        {
            "id": uuid4().hex,
            "timestamp": _now(),
            "status": "pending",
            "question": question[:4000],
            "reason": reason,
            "session_id": session_id,
            "note": "",
        }
    )
    _save_ticket(ticket)
    audit("human_review_requested", session_id=session_id, ticket_id=ticket["id"], reason=reason)
    return ticket


def update_handoff(ticket_id: str, status: str, note: str = "") -> dict:
    if status not in {"pending", "reviewed", "resolved"}:
        raise ValueError("Choose pending, reviewed, or resolved.")
    with _LOCK:
        ticket = next((item for item in list_handoffs() if item["id"] == ticket_id), None)
        if ticket is None:
            raise ValueError("This human review request was not found.")
        ticket.update(status=status, note=note[:4000], updated_at=_now())
        _save_ticket(ticket)
    audit(
        "human_review_updated",
        session_id=ticket.get("session_id"),
        ticket_id=ticket_id,
        status=status,
    )
    return ticket
