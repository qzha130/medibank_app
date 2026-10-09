"""Server-only configuration and repeatable index preparation for public hosting."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medibank.config import ProviderConfig, embedding_identity
from medibank.knowledge import IndexResult, build_index, discover_pdfs
from medibank.models import create_embeddings

PROJECT_DIR = Path(__file__).resolve().parents[1]
_PUBLIC_PROVIDERS = {
    "gemini": (
        "gemini-3.8-flash",
        "gemini-embedding-001",
        "https://generativelanguage.googleapis.com",
    ),
    "openai": ("gpt-4.1-mini", "text-embedding-3-small", "https://api.openai.com/v1"),
}


@dataclass(frozen=True)
class PublicSettings:
    chat: ProviderConfig
    embedding: ProviderConfig
    pdf_directory: Path
    chroma_directory: Path
    top_k: int = 5

    def __post_init__(self) -> None:
        for config in (self.chat, self.embedding):
            if config.provider not in _PUBLIC_PROVIDERS:
                raise ValueError("Public hosting requires Gemini or OpenAI; Ollama is local only.")
            expected_url = _PUBLIC_PROVIDERS[config.provider][2]
            if config.base_url.rstrip("/") != expected_url:
                raise ValueError("Public hosting requires the provider's official API URL.")
            if not config.api_key.strip():
                raise ValueError(f"Configure a {config.provider.title()} API key on the server.")
        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or not 1 <= self.top_k <= 20
        ):
            raise ValueError("PUBLIC_TOP_K must be an integer between 1 and 20.")


def public_mode() -> bool:
    """Check the server deployment flag, never a browser-controlled preference."""
    return os.getenv("APP_PUBLIC_MODE", "").strip().lower() in {"true", "1", "yes", "on"}


def read_public_settings(values: Mapping[str, Any] | None = None) -> PublicSettings:
    """Read deployment secrets over environment values without rendering them.

    Provider endpoints are fixed to their official URLs. Relative storage paths
    resolve from the repository, regardless of the host's current directory.
    """
    values = values if values is not None else {}

    def value(name: str, default: Any = "") -> Any:
        return values[name] if name in values else os.getenv(name, default)

    def text(name: str, default: str = "") -> str:
        result = value(name, default)
        if not isinstance(result, str):
            raise ValueError(f"{name} must be a string.")
        return result.strip()

    def config(role: str) -> ProviderConfig:
        provider = text(f"{role}_PROVIDER", "gemini").lower()
        if provider not in _PUBLIC_PROVIDERS:
            raise ValueError("Public hosting requires Gemini or OpenAI; Ollama is local only.")
        chat_model, embedding_model, official_url = _PUBLIC_PROVIDERS[provider]
        default_model = chat_model if role == "CHAT" else embedding_model
        model = text(f"{provider.upper()}_{role}_MODEL", default_model)
        base_url = text(f"{provider.upper()}_BASE_URL", official_url).rstrip("/")
        if base_url != official_url:
            raise ValueError("Public hosting requires the provider's official API URL.")
        key_name = f"{provider.upper()}_API_KEY"
        if provider == "gemini":
            if key_name in values:
                api_key = text(key_name)
            elif "GOOGLE_API_KEY" in values:
                api_key = text("GOOGLE_API_KEY")
            elif os.getenv(key_name):
                api_key = text(key_name)
            else:
                api_key = text("GOOGLE_API_KEY")
        else:
            api_key = text(key_name)
        return ProviderConfig(provider, model, official_url, api_key)

    def directory(name: str, default: str) -> Path:
        raw = text(name, default)
        if not raw:
            raise ValueError(f"{name} must identify a directory.")
        path = Path(raw).expanduser()
        if path.is_absolute():
            # Absolute paths can be supplied only through trusted server settings.
            return path.resolve()
        resolved = (PROJECT_DIR / path).resolve()
        if not resolved.is_relative_to(PROJECT_DIR.resolve()):
            raise ValueError(f"Relative {name} must stay inside the project directory.")
        return resolved

    raw_top_k = value("PUBLIC_TOP_K", 5)
    if isinstance(raw_top_k, bool) or not isinstance(raw_top_k, (str, int)):
        raise ValueError("PUBLIC_TOP_K must be an integer between 1 and 20.")
    try:
        top_k = int(raw_top_k)
    except ValueError:
        raise ValueError("PUBLIC_TOP_K must be an integer between 1 and 20.") from None
    return PublicSettings(
        chat=config("CHAT"),
        embedding=config("EMBEDDING"),
        pdf_directory=directory("PDF_DIRECTORY", "medibank_data"),
        chroma_directory=directory("CHROMA_DIRECTORY", "data/chroma"),
        top_k=top_k,
    )


def prepare_public_index(settings: PublicSettings) -> IndexResult:
    """Build the approved server corpus or reuse its complete compatible index."""
    pdfs = discover_pdfs(settings.pdf_directory)
    return build_index(
        pdfs=pdfs,
        embeddings=create_embeddings(settings.embedding),
        embedding_identity=embedding_identity(settings.embedding),
        persist_directory=settings.chroma_directory,
    )
