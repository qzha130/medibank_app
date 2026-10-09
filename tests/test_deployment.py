"""Public server settings and real Chroma bootstrap, without any model network."""

from __future__ import annotations

import json

import pytest
from conftest import searchable_pdf

from medibank import deployment
from medibank.config import ProviderConfig, embedding_identity
from medibank.deployment import (
    PublicSettings,
    prepare_public_index,
    public_mode,
    read_public_settings,
)


@pytest.fixture(autouse=True)
def isolated_server_environment(monkeypatch, tmp_path):
    for name in (
        "APP_PUBLIC_MODE",
        "CHAT_PROVIDER",
        "EMBEDDING_PROVIDER",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_BASE_URL",
        "OPENAI_BASE_URL",
        "GEMINI_CHAT_MODEL",
        "GEMINI_EMBEDDING_MODEL",
        "OPENAI_CHAT_MODEL",
        "OPENAI_EMBEDDING_MODEL",
        "PDF_DIRECTORY",
        "CHROMA_DIRECTORY",
        "PUBLIC_TOP_K",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(deployment, "PROJECT_DIR", tmp_path)


def test_public_defaults_are_fixed_gemini_settings_without_secret_repr(tmp_path):
    settings = read_public_settings({"GEMINI_API_KEY": "private-gemini-credential"})
    assert settings.chat.provider == settings.embedding.provider == "gemini"
    assert settings.chat.model == "gemini-3.8-flash"
    assert settings.embedding.model == "gemini-embedding-001"
    assert settings.chat.base_url == "https://generativelanguage.googleapis.com"
    assert settings.pdf_directory == tmp_path / "medibank_data"
    assert settings.chroma_directory == tmp_path / "data" / "chroma"
    assert settings.top_k == 5
    assert "private-gemini-credential" not in repr(settings)


def test_secrets_override_environment_and_google_key_is_supported(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "environment-private-key")
    monkeypatch.setenv("PUBLIC_TOP_K", "3")
    settings = read_public_settings({"GEMINI_API_KEY": "secrets-private-key", "PUBLIC_TOP_K": "7"})
    assert settings.chat.api_key == settings.embedding.api_key == "secrets-private-key"
    assert settings.top_k == 7
    alternate_secret = read_public_settings({"GOOGLE_API_KEY": "google-secrets-key"})
    assert alternate_secret.chat.api_key == "google-secrets-key"
    monkeypatch.delenv("GEMINI_API_KEY")
    fallback = read_public_settings({"GOOGLE_API_KEY": "google-private-key"})
    assert fallback.chat.api_key == "google-private-key"


def test_chat_and_embedding_providers_receive_only_their_own_server_key():
    settings = read_public_settings(
        {
            "CHAT_PROVIDER": "openai",
            "EMBEDDING_PROVIDER": "gemini",
            "OPENAI_API_KEY": "openai-private-key",
            "GEMINI_API_KEY": "gemini-private-key",
        }
    )
    assert settings.chat.provider == "openai"
    assert settings.chat.api_key == "openai-private-key"
    assert settings.chat.base_url == "https://api.openai.com/v1"
    assert settings.embedding.provider == "gemini"
    assert settings.embedding.api_key == "gemini-private-key"


@pytest.mark.parametrize("role", ["CHAT", "EMBEDDING"])
@pytest.mark.parametrize("provider", ["ollama", "unrecognized"])
def test_public_settings_reject_local_or_unknown_providers(role, provider):
    with pytest.raises(ValueError, match="Gemini or OpenAI"):
        read_public_settings({f"{role}_PROVIDER": provider, "GEMINI_API_KEY": "private-key"})


@pytest.mark.parametrize("provider", ["gemini", "openai"])
def test_untrusted_endpoint_is_rejected_before_credentials_can_be_used(provider):
    secret = "must-not-appear-in-error"
    values = {
        "CHAT_PROVIDER": provider,
        "EMBEDDING_PROVIDER": provider,
        f"{provider.upper()}_API_KEY": secret,
        f"{provider.upper()}_BASE_URL": "https://untrusted.example/api?token=" + secret,
    }
    with pytest.raises(ValueError, match="official API URL") as failure:
        read_public_settings(values)
    assert secret not in str(failure.value)


def test_missing_keys_and_explicit_empty_secret_do_not_use_ambient_credentials(monkeypatch):
    with pytest.raises(ValueError, match="API key"):
        read_public_settings()
    monkeypatch.setenv("GEMINI_API_KEY", "environment-private-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "fallback-private-key")
    with pytest.raises(ValueError, match="API key"):
        read_public_settings({"GEMINI_API_KEY": ""})


@pytest.mark.parametrize("top_k", [0, 21, True, 2.5, "2.5", "many"])
def test_invalid_search_limit_is_rejected(top_k):
    with pytest.raises(ValueError, match="PUBLIC_TOP_K"):
        read_public_settings({"GEMINI_API_KEY": "private-key", "PUBLIC_TOP_K": top_k})


def test_relative_storage_settings_cannot_escape_project():
    with pytest.raises(ValueError, match="stay inside the project"):
        read_public_settings({"GEMINI_API_KEY": "private-key", "CHROMA_DIRECTORY": "../other"})


def test_direct_settings_construction_still_rejects_unofficial_endpoints(tmp_path):
    wrong = ProviderConfig("gemini", "gemini-3.8-flash", "https://untrusted.example", "secret")
    correct = ProviderConfig(
        "gemini", "gemini-embedding-001", "https://generativelanguage.googleapis.com", "secret"
    )
    with pytest.raises(ValueError, match="official API URL"):
        PublicSettings(wrong, correct, tmp_path, tmp_path)


@pytest.mark.parametrize(
    "flag,enabled", [(None, False), ("false", False), ("true", True), ("1", True)]
)
def test_public_mode_is_controlled_by_the_server_environment(monkeypatch, flag, enabled):
    if flag is not None:
        monkeypatch.setenv("APP_PUBLIC_MODE", flag)
    assert public_mode() is enabled


def test_bootstrap_reuses_real_saved_vectors_without_reembedding(
    monkeypatch, pdf_bytes, embeddings, tmp_path
):
    folder = tmp_path / "medibank_data"
    folder.mkdir()
    (folder / "hospital.pdf").write_bytes(pdf_bytes[0])
    (folder / "extras.pdf").write_bytes(pdf_bytes[1])
    settings = read_public_settings({"GEMINI_API_KEY": "private-server-credential"})
    clients = []

    def fake_client(config):
        clients.append(config)
        return embeddings

    monkeypatch.setattr(deployment, "create_embeddings", fake_client)
    first = prepare_public_index(settings)
    batches = len(embeddings.document_batches)
    second = prepare_public_index(settings)
    assert first.reused is False
    assert second.reused is True
    assert first.dataset_id == second.dataset_id
    assert first.page_count == 3 and first.chunk_count == 3
    assert len(embeddings.document_batches) == batches
    assert clients == [settings.embedding, settings.embedding]
    assert all("private-server-credential" not in repr(client) for client in clients)
    manifest_path = settings.chroma_directory / first.dataset_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["specification"]["embedding_identity"] == embedding_identity(settings.embedding)
    assert "private-server-credential" not in manifest_path.read_text(encoding="utf-8")
    results = second.store.similarity_search("dental check-ups benefits", k=1)
    assert results[0].metadata["source"] == "extras.pdf"


def test_changed_public_corpus_gets_a_separate_index(monkeypatch, pdf_bytes, embeddings, tmp_path):
    folder = tmp_path / "medibank_data"
    folder.mkdir()
    (folder / "hospital.pdf").write_bytes(pdf_bytes[0])
    (folder / "extras.pdf").write_bytes(pdf_bytes[1])
    settings = read_public_settings({"GEMINI_API_KEY": "private-key"})
    monkeypatch.setattr(deployment, "create_embeddings", lambda config: embeddings)
    first = prepare_public_index(settings)
    (folder / "extras.pdf").write_bytes(searchable_pdf("Replacement maternity benefit."))
    second = prepare_public_index(settings)
    assert second.reused is False
    assert second.dataset_id != first.dataset_id
    assert "Replacement maternity" in " ".join(second.store.get()["documents"])
    assert "Dental benefits" not in " ".join(second.store.get()["documents"])


def test_missing_corpus_fails_before_creating_any_embedding_client(monkeypatch):
    settings = read_public_settings({"GEMINI_API_KEY": "private-key"})

    def unexpected_client(config):
        raise AssertionError("Missing corpus must not initialize a provider client")

    monkeypatch.setattr(deployment, "create_embeddings", unexpected_client)
    with pytest.raises(ValueError, match="PDF folder not found"):
        prepare_public_index(settings)
