"""Construct provider clients without contacting the model service."""

from __future__ import annotations

import os

from google.genai.types import EmbedContentConfig, HttpOptions, HttpRetryOptions
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from medibank.config import ProviderConfig


class _BoundedGeminiEmbeddings(GoogleGenerativeAIEmbeddings):
    """Apply request limits through the SDK's public embedding config."""

    def _build_config(
        self,
        *,
        task_type: str | None = None,
        title: str | None = None,
        output_dimensionality: int | None = None,
    ) -> EmbedContentConfig:
        config = super()._build_config(
            task_type=task_type,
            title=title,
            output_dimensionality=output_dimensionality,
        )
        config.http_options = HttpOptions(
            timeout=60000,
            retry_options=HttpRetryOptions(attempts=2),
        )
        return config


def _gemini_key(config: ProviderConfig) -> str:
    key = config.api_key.strip()
    if not key:
        raise ValueError("Enter a Gemini API key from Google AI Studio.")
    # langchain-google-genai 4.4 does not forward vertexai=False to its SDK
    # client, which would otherwise re-read these flags and change backends.
    if any(
        os.getenv(name, "").lower() in {"true", "1"}
        for name in (
            "GOOGLE_GENAI_USE_VERTEXAI",
            "GOOGLE_GENAI_USE_ENTERPRISE",
        )
    ):
        raise ValueError(
            "This app uses the Gemini Developer API with a Google AI Studio key. "
            "Unset GOOGLE_GENAI_USE_VERTEXAI and GOOGLE_GENAI_USE_ENTERPRISE, "
            "or set them to false, before using Gemini."
        )
    return key


def _ollama_client_options(config: ProviderConfig) -> dict:
    options: dict = {"timeout": 120.0}
    if config.api_key:
        options["headers"] = {"Authorization": f"Bearer {config.api_key}"}
    return options


def create_embeddings(config: ProviderConfig) -> Embeddings:
    """Create the embedding client used for both indexing and retrieval."""
    if config.provider == "ollama":
        return OllamaEmbeddings(
            model=config.model,
            base_url=config.base_url,
            client_kwargs=_ollama_client_options(config),
            validate_model_on_init=False,
        )
    if config.provider == "openai":
        return OpenAIEmbeddings(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            request_timeout=60.0,
            max_retries=1,
            chunk_size=64,
        )
    if config.provider == "gemini":
        if config.model.removeprefix("models/").startswith("gemini-embedding-2"):
            raise ValueError(
                "The current LangChain adapter does not support Gemini Embedding 2 "
                "retrieval task types. Use gemini-embedding-001 for this PDF index."
            )
        return _BoundedGeminiEmbeddings(
            model=config.model,
            api_key=_gemini_key(config),
            vertexai=False,
            base_url=config.base_url,
            client_args={"timeout": 60.0},
            # Leave task_type unset: the adapter selects RETRIEVAL_DOCUMENT for
            # indexing and RETRIEVAL_QUERY for questions.
        )
    raise ValueError("Choose Ollama, OpenAI, or Gemini as the model provider.")


def create_chat_model(config: ProviderConfig) -> BaseChatModel:
    """Create a model client; the selected model must support tool calling."""
    if config.provider == "ollama":
        return ChatOllama(
            model=config.model,
            base_url=config.base_url,
            temperature=0.1,
            num_ctx=8192,
            num_predict=768,
            client_kwargs=_ollama_client_options(config),
            validate_model_on_init=False,
        )
    if config.provider == "openai":
        return ChatOpenAI(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=0.1,
            timeout=60.0,
            max_tokens=1024,
            max_retries=1,
        )
    if config.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model,
            api_key=_gemini_key(config),
            vertexai=False,
            base_url=config.base_url,
            timeout=60.0,
            max_output_tokens=4096,
            max_retries=2,
            # Preserve Google's sampling defaults for Gemini 3+.
        )
    raise ValueError("Choose Ollama, OpenAI, or Gemini as the model provider.")
