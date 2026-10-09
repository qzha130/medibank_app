"""Provider boundaries protect index compatibility and avoid startup requests."""

import os
from types import SimpleNamespace

import httpx
import pytest
from google import genai
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import embeddings as google_embeddings

from medibank.config import ProviderConfig, embedding_identity
from medibank.models import create_chat_model, create_embeddings


def test_index_identity_tracks_embedding_space_but_not_api_credentials():
    first = ProviderConfig(
        "openai", "text-embedding-3-small", "https://api.openai.com/v1", "private-key-one"
    )
    changed_key = ProviderConfig("openai", first.model, first.base_url + "/", "private-key-two")
    changed_model = ProviderConfig(
        "openai", "text-embedding-3-large", first.base_url, first.api_key
    )
    changed_server = ProviderConfig("openai", first.model, "https://example.com/v1", first.api_key)
    assert embedding_identity(first) == embedding_identity(changed_key)
    assert (
        len(
            {
                embedding_identity(first),
                embedding_identity(changed_model),
                embedding_identity(changed_server),
            }
        )
        == 3
    )
    assert "private-key-one" not in repr(first)
    assert "private-key-one" not in embedding_identity(first)


@pytest.mark.parametrize(
    "url",
    [
        "file:///private/model",
        "localhost:11434",
        "https://user:private-password@example.com/v1",
        "https://example.com/v1?token=private-token",
        "https://example.com/v1#private-fragment",
    ],
)
def test_model_server_urls_cannot_embed_credentials(url):
    with pytest.raises(ValueError) as failure:
        ProviderConfig("ollama", "model", url)
    assert "private-password" not in str(failure.value)
    assert "private-token" not in str(failure.value)


@pytest.mark.parametrize(
    "provider,chat_model,embedding_model",
    [
        ("ollama", "llama3.1:8b", "nomic-embed-text"),
        ("openai", "gpt-4.1-mini", "text-embedding-3-small"),
        ("gemini", "gemini-3.8-flash", "gemini-embedding-001"),
    ],
)
def test_provider_clients_initialize_without_any_network_requests(
    monkeypatch, provider, chat_model, embedding_model
):
    def unexpected_request(*args, **kwargs):
        raise AssertionError("Constructing a provider client must not contact a server")

    monkeypatch.setattr(httpx.Client, "send", unexpected_request)
    monkeypatch.setattr(httpx.AsyncClient, "send", unexpected_request)
    base_url = "http://127.0.0.1:9/v1" if provider == "openai" else "http://127.0.0.1:9"
    chat_config = ProviderConfig(provider, chat_model, base_url, "offline-test-key")
    embedding_config = ProviderConfig(provider, embedding_model, base_url, "offline-test-key")
    assert isinstance(create_chat_model(chat_config), BaseChatModel)
    assert isinstance(create_embeddings(embedding_config), Embeddings)


def test_gemini_identity_is_distinct_and_does_not_persist_credentials():
    first = ProviderConfig(
        "gemini",
        "gemini-embedding-001",
        "https://generativelanguage.googleapis.com",
        "private-gemini-one",
    )
    second = ProviderConfig("gemini", first.model, first.base_url + "/", "private-gemini-two")
    same_named_openai = ProviderConfig("openai", first.model, first.base_url, first.api_key)
    assert embedding_identity(first) == embedding_identity(second)
    assert embedding_identity(first) != embedding_identity(same_named_openai)
    assert "private-gemini" not in repr(first)
    assert "private-gemini" not in embedding_identity(first)


@pytest.mark.parametrize(
    "factory,model",
    [
        (create_chat_model, "gemini-3.8-flash"),
        (create_embeddings, "gemini-embedding-001"),
    ],
)
def test_gemini_clients_use_the_explicit_key_and_endpoint_without_changing_environment(
    monkeypatch,
    factory,
    model,
):
    for name, value in {
        "OPENAI_API_KEY": "other-openai-key",
        "GOOGLE_API_KEY": "other-google-key",
        "GEMINI_API_KEY": "other-gemini-key",
        "GOOGLE_GENAI_USE_VERTEXAI": "false",
        "GOOGLE_GENAI_USE_ENTERPRISE": "false",
        "GOOGLE_CLOUD_PROJECT": "other-vertex-project",
    }.items():
        monkeypatch.setenv(name, value)
    original = dict(os.environ)

    def unexpected_request(*args, **kwargs):
        raise AssertionError("Client initialization must be offline")

    monkeypatch.setattr(httpx.Client, "send", unexpected_request)
    monkeypatch.setattr(httpx.AsyncClient, "send", unexpected_request)
    first = factory(
        ProviderConfig("gemini", model, "http://127.0.0.1:9/gateway", "session-key-one")
    )
    second = factory(
        ProviderConfig("gemini", model, "http://127.0.0.1:9/gateway", "session-key-two")
    )
    assert first.vertexai is False and first.client.vertexai is False
    assert second.vertexai is False and second.client.vertexai is False
    assert first.client._api_client.api_key == "session-key-one"
    assert second.client._api_client.api_key == "session-key-two"
    assert first.google_api_key.get_secret_value() == "session-key-one"
    assert first.client._api_client._http_options.base_url == "http://127.0.0.1:9/gateway"
    assert "session-key-one" not in repr(first)
    assert dict(os.environ) == original


@pytest.mark.parametrize(
    "factory,model",
    [
        (create_chat_model, "gemini-3.8-flash"),
        (create_embeddings, "gemini-embedding-001"),
    ],
)
def test_gemini_empty_key_does_not_fall_back_to_another_sessions_environment_key(
    monkeypatch,
    factory,
    model,
):
    monkeypatch.setenv("GOOGLE_API_KEY", "other-session-google-key")
    monkeypatch.setenv("GEMINI_API_KEY", "other-session-gemini-key")
    with pytest.raises(ValueError, match="Enter a Gemini API key") as failure:
        factory(ProviderConfig("gemini", model, "https://generativelanguage.googleapis.com", ""))
    assert "other-session" not in str(failure.value)


@pytest.mark.parametrize("variable", ["GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE"])
@pytest.mark.parametrize(
    "factory,model",
    [
        (create_chat_model, "gemini-3.8-flash"),
        (create_embeddings, "gemini-embedding-001"),
    ],
)
def test_gemini_fails_closed_if_sdk_environment_would_switch_to_vertex(
    monkeypatch,
    variable,
    factory,
    model,
):
    monkeypatch.setenv(variable, "true")
    original = dict(os.environ)
    with pytest.raises(ValueError, match="Gemini Developer API") as failure:
        factory(
            ProviderConfig(
                "gemini", model, "https://generativelanguage.googleapis.com", "private-key"
            )
        )
    assert "private-key" not in str(failure.value)
    assert dict(os.environ) == original


def test_gemini_embeddings_preserve_document_and_query_retrieval_task_types(monkeypatch):
    embedding = create_embeddings(
        ProviderConfig(
            "gemini",
            "gemini-embedding-001",
            "https://generativelanguage.googleapis.com",
            "offline-key",
        )
    )
    calls = []

    def embed_content(**kwargs):
        calls.append(kwargs)
        count = len(kwargs["contents"]) if isinstance(kwargs["contents"], list) else 1
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[1.0, 0.0]) for _ in range(count)]
        )

    monkeypatch.setattr(embedding.client.models, "embed_content", embed_content)
    assert (
        embedding.embed_documents(["First PDF passage.", "Second PDF passage."]) == [[1.0, 0.0]] * 2
    )
    assert embedding.embed_query("What is covered?") == [1.0, 0.0]
    assert embedding.task_type is None
    assert calls[0]["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert calls[1]["config"].task_type == "RETRIEVAL_QUERY"
    for call in calls:
        assert call["config"].http_options.timeout == 60000
        assert call["config"].http_options.retry_options.attempts == 2


def test_gemini_embedding_requests_reach_the_real_sdk_with_bounded_timeouts(monkeypatch):
    requests = []
    outage = False

    def respond(request):
        requests.append(request)
        if outage:
            return httpx.Response(
                503,
                json={"error": {"code": 503, "message": "Temporary service outage."}},
                request=request,
            )
        return httpx.Response(
            200,
            json={"embeddings": [{"values": [1.0, 0.0]}]},
            request=request,
        )

    transport = httpx.MockTransport(respond)

    def offline_client(**kwargs):
        options = kwargs["http_options"]
        options.client_args = {**(options.client_args or {}), "transport": transport}
        options.async_client_args = {**(options.async_client_args or {}), "transport": transport}
        return genai.Client(**kwargs)

    monkeypatch.setattr(google_embeddings, "Client", offline_client)
    embedding = create_embeddings(
        ProviderConfig(
            "gemini",
            "gemini-embedding-001",
            "https://generativelanguage.googleapis.com",
            "offline-key",
        )
    )
    assert embedding.embed_documents(["A PDF passage."]) == [[1.0, 0.0]]
    assert embedding.embed_query("What is covered?") == [1.0, 0.0]
    assert len(requests) == 2
    for request in requests:
        assert request.extensions["timeout"] == {
            "connect": 60.0,
            "read": 60.0,
            "write": 60.0,
            "pool": 60.0,
        }
        assert request.headers["x-goog-api-key"] == "offline-key"
    outage = True
    with pytest.raises(google_embeddings.GoogleGenerativeAIError):
        embedding.embed_query("Retry during a service outage.")
    assert len(requests) == 4  # Two successful requests, then exactly two failed attempts.
    embedding.client.close()


@pytest.mark.parametrize("model", ["gemini-embedding-2", "models/gemini-embedding-2-preview"])
def test_gemini_embedding_2_has_a_clear_adapter_compatibility_error(model):
    with pytest.raises(ValueError, match="Use gemini-embedding-001"):
        create_embeddings(
            ProviderConfig(
                "gemini", model, "https://generativelanguage.googleapis.com", "offline-key"
            )
        )
