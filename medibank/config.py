"""Provider configuration shared by the UI and backend."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    base_url: str
    api_key: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if self.provider not in {"ollama", "openai", "gemini"}:
            raise ValueError("Select Ollama, OpenAI, or Gemini as the model provider.")
        if not self.model.strip():
            raise ValueError("Enter a model name.")
        url = urlsplit(self.base_url)
        if url.scheme not in {"http", "https"} or not url.hostname:
            raise ValueError("The model server URL must start with http:// or https://.")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError(
                "The model server URL must not contain credentials, a query, or a fragment."
            )


def embedding_identity(config: ProviderConfig) -> str:
    """Identify vector spaces without persisting API credentials."""
    identity = json.dumps(
        {
            "provider": config.provider,
            "model": config.model.strip(),
            "base_url": config.base_url.rstrip("/"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(identity.encode()).hexdigest()
