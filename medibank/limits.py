"""A small, process-wide request budget for the anonymous public demo."""

from __future__ import annotations

import time
from collections import deque
from threading import Lock


class RequestLimiter:
    def __init__(
        self, limit: int = 60, window_seconds: float = 3600, cooldown_seconds: float = 5
    ) -> None:
        if limit < 1 or window_seconds <= 0 or cooldown_seconds < 0:
            raise ValueError("Request limits must be positive and cooldown cannot be negative.")
        self.limit = limit
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self._requests: deque[float] = deque()
        self._sessions: dict[str, float] = {}
        self._lock = Lock()

    def allow(self, session_id: str, now: float | None = None) -> bool:
        timestamp = time.monotonic() if now is None else now
        cutoff = timestamp - self.window_seconds
        with self._lock:
            while self._requests and self._requests[0] <= cutoff:
                self._requests.popleft()
            self._sessions = {
                session: last for session, last in self._sessions.items() if last > cutoff
            }
            previous = self._sessions.get(session_id)
            if previous is not None and timestamp - previous < self.cooldown_seconds:
                return False
            if len(self._requests) >= self.limit:
                return False
            self._requests.append(timestamp)
            self._sessions[session_id] = timestamp
            return True
