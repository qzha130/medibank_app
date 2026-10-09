"""Streamlit state transitions stay offline and fail safely."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import CountingEmbeddings, ScriptedChatModel
from streamlit.testing.v1 import AppTest

import medibank.chat
import medibank.guardrails
import medibank.loading
import medibank.models
from medibank.chat import Answer, ChatError
from medibank.config import embedding_identity
from medibank.guardrails import GuardrailDecision
from medibank.guardrails import assess_question as real_assess_question
from medibank.knowledge import IndexResult

APP = Path(__file__).resolve().parents[1] / "app.py"


def widget(elements, label):
    return next(element for element in elements if element.label == label)


@pytest.fixture
def ui(monkeypatch, tmp_path):
    pdf_directory = tmp_path / "pdfs"
    pdf_directory.mkdir()
    for key, value in {
        "CHAT_PROVIDER": "ollama",
        "EMBEDDING_PROVIDER": "ollama",
        "OPENAI_API_KEY": "",
        "GEMINI_API_KEY": "",
        "GOOGLE_API_KEY": "",
        "PDF_DIRECTORY": str(pdf_directory),
        "CHROMA_DIRECTORY": str(tmp_path / "chroma"),
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_CHAT_MODEL": "llama3.1:8b",
        "OLLAMA_EMBEDDING_MODEL": "nomic-embed-text",
        "GEMINI_CHAT_MODEL": "gemini-3.8-flash",
        "GEMINI_EMBEDDING_MODEL": "gemini-embedding-001",
    }.items():
        monkeypatch.setenv(key, value)
    state = {
        "load_calls": [],
        "chat_calls": [],
        "chat_error": False,
        "pdf_directory": pdf_directory,
        "embedding_configs": [],
        "chat_configs": [],
    }

    def fake_find(directory, config, preferred_id=None):
        if len(list(pdf_directory.glob("*.pdf"))) != 2:
            return None
        return SimpleNamespace(id=embedding_identity(config))

    def fake_load(saved, embeddings):
        state["load_calls"].append({"embedding_identity": saved.id})
        return IndexResult(
            store=object(),
            dataset_id=saved.id,
            files=[{"name": path.name} for path in pdf_directory.glob("*.pdf")],
            page_count=3,
            chunk_count=3,
            skipped_pages=0,
            reused=True,
        )

    def fake_answer(**kwargs):
        state["chat_calls"].append(kwargs)
        if state["chat_error"]:
            raise ChatError("The chat model request failed. Check the connection.")
        return Answer(
            text="Dental check-ups are covered. [extras.pdf, p. 1]",
            sources=[
                {"source": "extras.pdf", "page": 1, "excerpt": "Dental check-ups are covered."}
            ],
        )

    def fake_embeddings(config):
        state["embedding_configs"].append(config)
        return CountingEmbeddings()

    def fake_chat_model(config):
        state["chat_configs"].append(config)
        return ScriptedChatModel()

    monkeypatch.setattr(medibank.loading, "find_saved_index", fake_find)
    monkeypatch.setattr(medibank.loading, "load_saved_index", fake_load)
    monkeypatch.setattr(medibank.models, "create_embeddings", fake_embeddings)
    monkeypatch.setattr(medibank.models, "create_chat_model", fake_chat_model)
    monkeypatch.setattr(medibank.chat, "answer_question", fake_answer)
    monkeypatch.setattr(
        medibank.guardrails,
        "assess_question",
        lambda *args, **kwargs: GuardrailDecision(True, "test_fixture", distance=0.1),
    )
    return AppTest.from_file(APP, default_timeout=15), state


def add_pdfs(state, pdf_bytes):
    for name, data in zip(("hospital.pdf", "extras.pdf"), pdf_bytes):
        (state["pdf_directory"] / name).write_bytes(data)


def load_index(at):
    at.run()
    assert not at.exception
    assert at.chat_input[0].disabled is False


def test_ui_starts_without_models_or_pdfs_and_disables_chat(ui):
    at, state = ui
    at.run()
    assert not at.exception
    assert at.chat_input[0].disabled is True
    assert state["load_calls"] == state["chat_calls"] == []
    assert any("Open Vector Database" in info.value for info in at.info)
    assert not any(b.label == "Build / load knowledge base" for b in at.button)
    assert not any(item.label == "PDF folder" for item in at.text_input)


def test_openai_saved_index_requires_query_api_key(ui, pdf_bytes):
    at, state = ui
    add_pdfs(state, pdf_bytes)
    at.run()
    widget(at.selectbox, "Embedding provider").select("OpenAI · online").run()
    assert not at.exception
    assert widget(at.text_input, "OpenAI API key").value == ""
    assert at.chat_input[0].disabled is True
    assert len(state["load_calls"]) == 1


def test_embedding_settings_change_automatically_loads_matching_saved_index(ui, pdf_bytes):
    at, state = ui
    add_pdfs(state, pdf_bytes)
    at.run()
    original_identity = state["load_calls"][0]["embedding_identity"]
    widget(at.text_input, "Embedding model").set_value("another-embedding-model").run()
    assert not at.exception
    assert at.chat_input[0].disabled is False
    assert len(state["load_calls"]) == 2
    assert state["load_calls"][-1]["embedding_identity"] != original_identity
    at.run()
    assert len(state["load_calls"]) == 2


def test_chat_model_change_reuses_index_and_online_chat_requires_key(ui, pdf_bytes):
    at, state = ui
    add_pdfs(state, pdf_bytes)
    at.run()
    load_index(at)
    widget(at.text_input, "Chat model").set_value("another-tool-capable-model").run()
    assert at.chat_input[0].disabled is False
    assert len(state["load_calls"]) == 1
    widget(at.selectbox, "Chat provider").select("OpenAI · online").run()
    assert not at.exception
    assert at.chat_input[0].disabled is True
    widget(at.text_input, "OpenAI API key").set_value("offline-test-key").run()
    assert at.chat_input[0].disabled is False
    assert len(state["load_calls"]) == 1


def test_chat_error_can_be_retried_without_poisoning_history(ui, pdf_bytes):
    at, state = ui
    add_pdfs(state, pdf_bytes)
    at.run()
    load_index(at)
    state["chat_error"] = True
    at.chat_input[0].set_value("Are dental check-ups covered?").run()
    assert not at.exception
    assert any("request failed" in error.value for error in at.error)
    assert at.session_state["messages"] == []
    state["chat_error"] = False
    at.chat_input[0].set_value("Are dental check-ups covered?").run()
    assert not at.exception
    assert [message["role"] for message in at.session_state["messages"]] == ["user", "assistant"]
    assert any("Sources" in expander.label for expander in at.expander)
    widget(at.button, "Clear conversation").click().run()
    assert at.session_state["messages"] == []
    assert at.chat_input[0].disabled is False


def test_embedding_key_rotation_reloads_client_without_changing_vector_space(ui, pdf_bytes):
    at, state = ui
    add_pdfs(state, pdf_bytes)
    at.run()
    widget(at.selectbox, "Embedding provider").select("OpenAI · online").run()
    widget(at.text_input, "OpenAI API key").set_value("offline-old-key").run()
    load_index(at)
    assert state["embedding_configs"][-1].api_key == "offline-old-key"
    widget(at.text_input, "OpenAI API key").set_value("offline-new-key").run()
    assert at.chat_input[0].disabled is False
    load_index(at)
    assert state["embedding_configs"][-1].api_key == "offline-new-key"
    assert (
        state["load_calls"][-2]["embedding_identity"]
        == state["load_calls"][-1]["embedding_identity"]
    )
    at.chat_input[0].set_value("Are dental check-ups covered?").run()
    assert state["chat_configs"][-1].provider == "ollama"
    assert state["chat_configs"][-1].api_key == ""


def test_online_chat_key_is_not_forwarded_to_local_embedding_service(ui, pdf_bytes):
    at, state = ui
    add_pdfs(state, pdf_bytes)
    at.run()
    widget(at.selectbox, "Chat provider").select("OpenAI · online").run()
    widget(at.text_input, "OpenAI API key").set_value("offline-online-chat-key").run()
    load_index(at)
    assert state["embedding_configs"][-1].provider == "ollama"
    assert state["embedding_configs"][-1].api_key == ""
    at.chat_input[0].set_value("Are dental check-ups covered?").run()
    assert state["chat_configs"][-1].api_key == "offline-online-chat-key"


def test_unrelated_question_is_blocked_before_model_creation(ui, pdf_bytes, monkeypatch):
    from medibank.observability import list_handoffs, read_events

    at, state = ui
    add_pdfs(state, pdf_bytes)
    monkeypatch.setattr(medibank.guardrails, "assess_question", real_assess_question)
    at.run()
    load_index(at)
    at.chat_input[0].set_value("What is the weather in Sydney?").run()
    assert not at.exception
    assert state["chat_configs"] == state["chat_calls"] == []
    assert "general questions about Medibank" in at.session_state["messages"][-1]["content"]
    assert list_handoffs() == []
    assert any(
        event["event"] == "guardrail_decision" and event["details"]["reason"] == "off_topic"
        for event in read_events()
    )


def test_explicit_human_request_is_saved_without_model_call(ui, pdf_bytes, monkeypatch):
    from medibank.observability import list_handoffs

    at, state = ui
    add_pdfs(state, pdf_bytes)
    monkeypatch.setattr(medibank.guardrails, "assess_question", real_assess_question)
    at.run()
    load_index(at)
    at.chat_input[0].set_value("I want to speak to a human about my Medibank membership.").run()
    assert not at.exception
    assert state["chat_configs"] == state["chat_calls"] == []
    tickets = list_handoffs()
    assert len(tickets) == 1 and tickets[0]["status"] == "pending"
    assert tickets[0]["reason"] == "human_requested"
    assert at.session_state["messages"][-1]["handoff"]["id"] == tickets[0]["id"]


def test_gemini_chat_uses_existing_local_index_and_requires_its_own_key(ui, pdf_bytes):
    at, state = ui
    add_pdfs(state, pdf_bytes)
    at.run()
    load_index(at)
    original_identity = state["load_calls"][0]["embedding_identity"]
    widget(at.selectbox, "Chat provider").select("Gemini · online").run()
    assert not at.exception
    assert widget(at.text_input, "Chat model").value == "gemini-3.8-flash"
    assert widget(at.text_input, "Gemini API key").value == ""
    assert at.chat_input[0].disabled is True
    widget(at.text_input, "Gemini API key").set_value("offline-gemini-chat-key").run()
    assert at.chat_input[0].disabled is False
    assert len(state["load_calls"]) == 1
    assert state["load_calls"][0]["embedding_identity"] == original_identity
    assert state["embedding_configs"][-1].provider == "ollama"
    assert state["embedding_configs"][-1].api_key == ""
    at.chat_input[0].set_value("Are dental check-ups covered?").run()
    assert not at.exception
    assert state["chat_configs"][-1].provider == "gemini"
    assert state["chat_configs"][-1].api_key == "offline-gemini-chat-key"


def test_gemini_embeddings_require_key_and_select_a_new_vector_space(ui, pdf_bytes):
    at, state = ui
    add_pdfs(state, pdf_bytes)
    at.run()
    load_index(at)
    original_identity = state["load_calls"][0]["embedding_identity"]
    widget(at.selectbox, "Embedding provider").select("Gemini · online").run()
    assert not at.exception
    assert widget(at.text_input, "Embedding model").value == "gemini-embedding-001"
    assert at.chat_input[0].disabled is True
    widget(at.text_input, "Gemini API key").set_value("offline-gemini-embedding-key").run()
    assert at.chat_input[0].disabled is False
    load_index(at)
    assert state["embedding_configs"][-1].provider == "gemini"
    assert state["embedding_configs"][-1].api_key == "offline-gemini-embedding-key"
    assert state["load_calls"][-1]["embedding_identity"] != original_identity
    at.chat_input[0].set_value("Are dental check-ups covered?").run()
    assert state["chat_configs"][-1].provider == "ollama"
    assert state["chat_configs"][-1].api_key == ""


def test_gemini_embedding_key_rotation_reloads_client_without_changing_index_identity(
    ui, pdf_bytes
):
    at, state = ui
    add_pdfs(state, pdf_bytes)
    at.run()
    widget(at.selectbox, "Embedding provider").select("Gemini · online").run()
    widget(at.text_input, "Gemini API key").set_value("offline-old-gemini-key").run()
    load_index(at)
    widget(at.text_input, "Gemini API key").set_value("offline-new-gemini-key").run()
    assert at.chat_input[0].disabled is False
    load_index(at)
    assert state["embedding_configs"][-1].api_key == "offline-new-gemini-key"
    assert (
        state["load_calls"][-2]["embedding_identity"]
        == state["load_calls"][-1]["embedding_identity"]
    )


@pytest.mark.parametrize(
    "chat_label,embedding_label,chat_provider,embedding_provider",
    [
        ("Gemini · online", "OpenAI · online", "gemini", "openai"),
        ("OpenAI · online", "Gemini · online", "openai", "gemini"),
    ],
)
def test_mixed_online_providers_receive_only_their_own_credentials(
    ui,
    pdf_bytes,
    chat_label,
    embedding_label,
    chat_provider,
    embedding_provider,
):
    at, state = ui
    add_pdfs(state, pdf_bytes)
    at.run()
    widget(at.selectbox, "Chat provider").select(chat_label).run()
    widget(at.selectbox, "Embedding provider").select(embedding_label).run()
    widget(at.text_input, "OpenAI API key").set_value("offline-openai-key").run()
    widget(at.text_input, "Gemini API key").set_value("offline-gemini-key").run()
    load_index(at)
    at.chat_input[0].set_value("Are dental check-ups covered?").run()
    assert not at.exception
    expected_keys = {"openai": "offline-openai-key", "gemini": "offline-gemini-key"}
    assert state["embedding_configs"][-1].provider == embedding_provider
    assert state["embedding_configs"][-1].api_key == expected_keys[embedding_provider]
    assert state["chat_configs"][-1].provider == chat_provider
    assert state["chat_configs"][-1].api_key == expected_keys[chat_provider]


def test_gemini_chat_accepts_google_api_key_fallback(ui, monkeypatch):
    at, _ = ui
    monkeypatch.setenv("GOOGLE_API_KEY", "offline-google-fallback-key")
    at.run()
    widget(at.selectbox, "Chat provider").select("Gemini · online").run()
    assert not at.exception
    assert widget(at.text_input, "Gemini API key").value == "offline-google-fallback-key"


def test_first_visit_autoloads_once_without_build_button(ui, pdf_bytes):
    at, state = ui
    add_pdfs(state, pdf_bytes)
    at.run()
    assert not at.exception
    assert at.chat_input[0].disabled is False
    assert len(state["load_calls"]) == 1
    assert not any(button.label == "Build / load knowledge base" for button in at.button)
    at.run()
    assert len(state["load_calls"]) == 1


def test_auto_load_failure_disables_chat_and_is_not_retried_on_every_rerun(
    ui, pdf_bytes, monkeypatch
):
    at, state = ui
    add_pdfs(state, pdf_bytes)
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(True)
        raise ValueError("The saved index is incomplete. Rebuild it on Vector Database.")

    monkeypatch.setattr(medibank.loading, "find_saved_index", unavailable)
    at.run()
    assert not at.exception
    assert at.chat_input[0].disabled
    assert any("incomplete" in error.value for error in at.error)
    at.run()
    assert len(calls) == 1
    assert state["embedding_configs"] == []


def test_vector_page_preference_reloads_without_pdf_ingestion(ui, pdf_bytes):
    at, state = ui
    add_pdfs(state, pdf_bytes)
    at.run()
    at.session_state["preferred_index_id"] = "selected-on-vector-page"
    at.session_state["preferred_chroma_directory"] = "other-chroma"
    at.session_state["index_autoload_signature"] = None
    at.run()
    assert not at.exception
    assert not at.chat_input[0].disabled
    assert len(state["load_calls"]) == 2


def test_chat_model_settings_survive_visiting_vector_database(ui, pdf_bytes):
    at, state = ui
    add_pdfs(state, pdf_bytes)
    at.run()
    widget(at.selectbox, "Chat provider").select("Gemini · online").run()
    widget(at.text_input, "Chat model").set_value("custom-gemini-model").run()
    widget(at.text_input, "Gemini API key").set_value("offline-kept-key").run()
    at.switch_page("pages/1_Vector_Database.py").run()
    assert not at.exception
    at.switch_page("app.py").run()
    assert not at.exception
    assert widget(at.selectbox, "Chat provider").value == "Gemini · online"
    assert widget(at.text_input, "Chat model").value == "custom-gemini-model"
    assert widget(at.text_input, "Gemini API key").value == "offline-kept-key"
    assert len(state["load_calls"]) == 1


def test_agent_activity_can_be_shown_and_hidden_without_repeating_model_calls(ui, pdf_bytes):
    at, state = ui
    add_pdfs(state, pdf_bytes)
    at.run()
    assert widget(at.toggle, "Show agent activity").value is False
    at.chat_input[0].set_value("How do I make a claim?").run()
    assert not at.exception
    assert not any(item.label == "Agent activity" for item in at.expander)
    saved_activity = at.session_state["messages"][-1]["activity"]
    assert any(row["Activity"] == "Response completed" for row in saved_activity)
    widget(at.toggle, "Show agent activity").set_value(True).run()
    assert any(item.label == "Agent activity" for item in at.expander)
    assert len(at.dataframe) == 1
    widget(at.toggle, "Show agent activity").set_value(False).run()
    assert not any(item.label == "Agent activity" for item in at.expander)
    assert len(state["chat_calls"]) == 1
    assert at.session_state["messages"][-1]["activity"] == saved_activity


def test_failed_attempt_activity_survives_toggle_and_successful_retry_clears_it(ui, pdf_bytes):
    at, state = ui
    add_pdfs(state, pdf_bytes)
    at.run()
    state["chat_error"] = True
    at.chat_input[0].set_value("How do I make a claim?").run()
    assert not at.exception
    assert at.session_state["messages"] == []
    failed = at.session_state["last_chat_error"]
    assert any(row["Activity"] == "Response unavailable" for row in failed["activity"])
    widget(at.toggle, "Show agent activity").set_value(True).run()
    assert not at.exception
    assert any(item.label == "Agent activity" for item in at.expander)
    assert any("request failed" in item.value for item in at.error)
    state["chat_error"] = False
    at.chat_input[0].set_value("How do I make a claim?").run()
    assert not at.exception
    assert "last_chat_error" not in at.session_state
    assert len(at.session_state["messages"]) == 2


def test_legacy_fallback_copy_updates_without_changing_user_messages(ui, pdf_bytes):
    from medibank.guardrails import OFF_TOPIC_MESSAGE

    at, state = ui
    add_pdfs(state, pdf_bytes)
    old = "I answer questions about the Medibank PDFs. Please ask about membership, benefits, claims, exclusions, or waiting periods."
    at.session_state["messages"] = [
        {"role": "user", "content": old},
        {"role": "assistant", "content": old, "sources": []},
    ]
    at.run()
    assert not at.exception
    assert at.session_state["messages"][0]["content"] == old
    assert at.session_state["messages"][1]["content"] == OFF_TOPIC_MESSAGE


def test_activity_option_survives_page_navigation_and_keeps_simple_default_screen(ui, pdf_bytes):
    at, state = ui
    add_pdfs(state, pdf_bytes)
    at.run()
    assert not at.metric
    assert any(item.label == "Model settings" for item in at.expander)
    assert any(title.value == "Medibank Assistant" for title in at.title)
    widget(at.toggle, "Show agent activity").set_value(True).run()
    at.switch_page("pages/1_Vector_Database.py").run()
    at.switch_page("app.py").run()
    assert not at.exception
    assert widget(at.toggle, "Show agent activity").value is True
    assert len(state["load_calls"]) == 1
