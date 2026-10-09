"""Review and query saved Chroma data through the Streamlit page, offline."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from conftest import CountingEmbeddings, searchable_pdf
from streamlit.testing.v1 import AppTest

from medibank.config import ProviderConfig, embedding_identity
from medibank.knowledge import PdfInput, build_index

PAGE = Path(__file__).resolve().parents[1] / "pages" / "1_Vector_Database.py"


def widget(elements, label):
    return next(element for element in elements if element.label == label)


@pytest.fixture
def vector_ui(monkeypatch, tmp_path):
    chroma_root = tmp_path / "chroma"
    chroma_root.mkdir()
    pdf_directory = tmp_path / "pdfs"
    pdf_directory.mkdir()
    for key, value in {
        "CHROMA_DIRECTORY": str(chroma_root),
        "PDF_DIRECTORY": str(pdf_directory),
        "EMBEDDING_PROVIDER": "ollama",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_EMBEDDING_MODEL": "nomic-embed-text",
        "OPENAI_API_KEY": "",
        "GEMINI_API_KEY": "",
        "GOOGLE_API_KEY": "",
        "LOG_DIRECTORY": str(tmp_path / "logs"),
        "HANDOFF_DIRECTORY": str(tmp_path / "handoffs"),
    }.items():
        monkeypatch.setenv(key, value)
    embeddings = CountingEmbeddings()
    state = {
        "root": chroma_root,
        "pdf_directory": pdf_directory,
        "embeddings": embeddings,
        "configs": [],
    }
    page_module = importlib.import_module("medibank.vector_page")

    def fake_embeddings(config):
        state["configs"].append(config)
        return embeddings

    monkeypatch.setattr(page_module, "create_embeddings", fake_embeddings)
    return AppTest.from_file(PAGE, default_timeout=15), state


@pytest.fixture
def saved_ui(vector_ui, pdf_bytes):
    at, state = vector_ui
    config = ProviderConfig("ollama", "nomic-embed-text", "http://localhost:11434")
    result = build_index(
        [PdfInput("hospital.pdf", pdf_bytes[0]), PdfInput("extras.pdf", pdf_bytes[1])],
        state["embeddings"],
        embedding_identity(config),
        state["root"],
    )
    state["index"] = result
    return at, state


def test_vector_page_handles_empty_directory_without_creating_a_database(vector_ui):
    at, state = vector_ui
    at.run()
    assert not at.exception
    assert any("Chroma Vector Database" in title.value for title in at.title)
    assert state["configs"] == []
    assert list(state["root"].iterdir()) == []
    assert any("index" in info.value.lower() for info in at.info)
    assert widget(at.radio, "PDF source").value == "Local folder"
    assert widget(at.text_input, "PDF folder").value == str(state["pdf_directory"])
    assert widget(at.button, "Build / load knowledge base").disabled is True


def test_local_folder_build_creates_saved_index_and_selects_it_for_chat(vector_ui, pdf_bytes):
    at, state = vector_ui
    for name, data in zip(("hospital.pdf", "extras.pdf"), pdf_bytes):
        (state["pdf_directory"] / name).write_bytes(data)
    at.session_state["index_autoload_signature"] = "previous-attempt"
    at.session_state["index"] = "existing-chat-client"
    at.run()
    assert not at.exception
    assert state["configs"] == []
    assert state["embeddings"].document_batches == []
    widget(at.button, "Build / load knowledge base").click().run()
    assert not at.exception
    assert widget(at.metric, "Stored passages").value == "3"
    assert widget(at.metric, "PDF pages").value == "3"
    assert len(state["configs"]) == 1
    assert state["configs"][0].provider == "ollama"
    assert state["configs"][0].api_key == ""
    assert at.session_state["preferred_index_id"] == widget(at.selectbox, "Saved index").value
    assert at.session_state["preferred_chroma_directory"] == str(state["root"].resolve())
    assert "index_autoload_signature" not in at.session_state
    assert at.session_state["index"] == "existing-chat-client"
    batches = len(state["embeddings"].document_batches)
    widget(at.button, "Build / load knowledge base").click().run()
    assert not at.exception
    assert len(state["embeddings"].document_batches) == batches
    assert any("Saved index loaded" in success.value for success in at.success)


def test_upload_controls_are_available_without_saved_indexes(vector_ui):
    at, state = vector_ui
    at.run()
    widget(at.radio, "PDF source").set_value("Upload PDFs").run()
    assert not at.exception
    assert any(item.label == "Choose the two PDFs" for item in at.get("file_uploader"))
    assert widget(at.button, "Build / load knowledge base").disabled is True
    assert state["configs"] == []
    assert list(state["root"].iterdir()) == []


def test_online_build_requires_its_embedding_api_key(vector_ui, pdf_bytes):
    at, state = vector_ui
    for name, data in zip(("hospital.pdf", "extras.pdf"), pdf_bytes):
        (state["pdf_directory"] / name).write_bytes(data)
    at.run()
    widget(at.selectbox, "Index embedding provider").select("Gemini · online").run()
    assert not at.exception
    assert widget(at.text_input, "Index API key").value == ""
    assert widget(at.button, "Build / load knowledge base").disabled is True
    assert state["configs"] == []
    widget(at.text_input, "Index API key").set_value("offline-index-key").run()
    assert widget(at.button, "Build / load knowledge base").disabled is False
    widget(at.button, "Build / load knowledge base").click().run()
    assert not at.exception
    assert state["configs"][0].provider == "gemini"
    assert state["configs"][0].api_key == "offline-index-key"


def test_passive_review_does_not_change_chat_index_but_use_in_chat_does(saved_ui):
    at, state = saved_ui
    at.session_state["preferred_index_id"] = "another-chat-index"
    at.session_state["index_autoload_signature"] = "previous-attempt"
    at.run()
    assert not at.exception
    assert at.session_state["preferred_index_id"] == "another-chat-index"
    assert at.session_state["index_autoload_signature"] == "previous-attempt"
    widget(at.button, "Use in chat").click().run()
    assert not at.exception
    assert at.session_state["preferred_index_id"] == state["index"].dataset_id
    assert at.session_state["preferred_chroma_directory"] == str(state["root"].resolve())
    assert "index_autoload_signature" not in at.session_state
    assert state["configs"] == []


def test_saved_records_can_be_reviewed_without_api_keys_or_model_requests(saved_ui):
    at, state = saved_ui
    at.run()
    assert not at.exception
    assert widget(at.metric, "Stored passages").value == "3"
    assert widget(at.metric, "PDF pages").value == "3"
    assert len(widget(at.selectbox, "Record to inspect").options) == 3
    assert state["configs"] == []
    assert state["embeddings"].query_calls == []
    assert widget(at.button, "Prepare filtered export").disabled is False


def test_document_and_physical_page_filters_select_the_expected_records(saved_ui):
    at, state = saved_ui
    at.run()
    widget(at.selectbox, "Document filter").select("hospital.pdf").run()
    assert not at.exception
    assert len(widget(at.selectbox, "Record to inspect").options) == 2
    widget(at.number_input, "Page filter").set_value(2).run()
    assert not at.exception
    assert len(widget(at.selectbox, "Record to inspect").options) == 1
    selected_id = widget(at.selectbox, "Record to inspect").value
    selected = state["index"].store.get(ids=[selected_id])
    assert selected["metadatas"][0]["source"] == "hospital.pdf"
    assert selected["metadatas"][0]["page"] == 2
    assert any("Ambulance transport" in text.value for text in at.text)
    assert state["configs"] == []


def test_vector_values_can_be_loaded_without_an_embedding_request(saved_ui):
    at, state = saved_ui
    at.run()
    widget(at.button, "Load vector values").click().run()
    assert not at.exception
    assert state["configs"] == []
    assert state["embeddings"].query_calls == []
    assert widget(at.metric, "Vector dimensions").value == "64"


def test_exports_include_all_filtered_records_and_expire_when_filters_change(saved_ui):
    at, state = saved_ui
    at.run()
    widget(at.button, "Prepare filtered export").click().run()
    assert not at.exception
    exported = at.session_state["vector_export"][1]
    assert {record["id"] for record in exported} == set(state["index"].store.get()["ids"])
    labels = {item.label for item in at.get("download_button")}
    assert labels == {"Export filtered records (JSON)", "Export filtered records (CSV)"}
    widget(at.selectbox, "Document filter").select("hospital.pdf").run()
    assert not at.get("download_button")
    widget(at.button, "Prepare filtered export").click().run()
    exported = at.session_state["vector_export"][1]
    assert len(exported) == 2
    assert all(record["metadata"]["source"] == "hospital.pdf" for record in exported)


def test_pagination_displays_distinct_batches_of_real_saved_chunks(vector_ui, pdf_bytes):
    at, state = vector_ui
    long_text = " ".join(
        f"Coverage paragraph {index} explains hospital waiting periods and policy benefits."
        for index in range(40)
    )
    config = ProviderConfig("ollama", "nomic-embed-text", "http://localhost:11434")
    build_index(
        [PdfInput("hospital.pdf", searchable_pdf(long_text)), PdfInput("extras.pdf", pdf_bytes[1])],
        state["embeddings"],
        embedding_identity(config),
        state["root"],
        chunk_size=100,
        chunk_overlap=0,
    )
    at.run()
    widget(at.selectbox, "Records per page").select(10).run()
    first = set(widget(at.selectbox, "Record to inspect").options)
    assert len(first) == 10
    widget(at.number_input, "Record page").set_value(2).run()
    assert not at.exception
    second = set(widget(at.selectbox, "Record to inspect").options)
    assert len(second) == 10
    assert first.isdisjoint(second)
    assert state["configs"] == []


def test_semantic_search_uses_matching_embeddings_and_the_stored_text(saved_ui):
    at, state = saved_ui
    at.run()
    widget(at.text_input, "Search query").set_value("dental check-ups benefits").run()
    widget(at.number_input, "Result count").set_value(1).run()
    search = widget(at.button, "Search vectors")
    assert search.disabled is False
    search.click().run()
    assert not at.exception
    assert len(state["configs"]) == 1
    assert state["configs"][0].provider == "ollama"
    assert state["embeddings"].query_calls == ["dental check-ups benefits"]
    assert any("Dental benefits" in text.value for text in at.text)


def test_mismatched_embedding_model_disables_query_without_model_calls(saved_ui):
    at, state = saved_ui
    at.run()
    widget(at.text_input, "Search query").set_value("dental benefits").run()
    widget(at.text_input, "Search embedding model").set_value("a-different-vector-space").run()
    assert not at.exception
    assert widget(at.button, "Search vectors").disabled is True
    assert state["configs"] == []
    assert state["embeddings"].query_calls == []


def test_online_embedding_settings_allow_browsing_with_no_api_key(saved_ui):
    at, state = saved_ui
    at.run()
    widget(at.selectbox, "Search embedding provider").select("Gemini · online").run()
    assert not at.exception
    assert widget(at.text_input, "Search API key").value == ""
    assert widget(at.button, "Search vectors").disabled is True
    assert widget(at.metric, "Stored passages").value == "3"
    assert len(widget(at.selectbox, "Record to inspect").options) == 3
    assert state["configs"] == []


def test_matching_online_index_requires_api_key_only_for_search(vector_ui, pdf_bytes, monkeypatch):
    at, state = vector_ui
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")
    config = ProviderConfig(
        "gemini", "gemini-embedding-001", "https://generativelanguage.googleapis.com"
    )
    build_index(
        [PdfInput("hospital.pdf", pdf_bytes[0]), PdfInput("extras.pdf", pdf_bytes[1])],
        state["embeddings"],
        embedding_identity(config),
        state["root"],
    )
    at.run()
    widget(at.text_input, "Search query").set_value("dental benefits").run()
    assert not at.exception
    assert widget(at.metric, "Stored passages").value == "3"
    assert widget(at.button, "Search vectors").disabled is True
    assert state["configs"] == []
    widget(at.text_input, "Search API key").set_value("offline-gemini-key").run()
    assert widget(at.button, "Search vectors").disabled is False
    widget(at.button, "Search vectors").click().run()
    assert not at.exception
    assert state["configs"][0].api_key == "offline-gemini-key"
    assert state["embeddings"].query_calls == ["dental benefits"]
