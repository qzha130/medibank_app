"""The public entry point keeps server settings and administrative data private."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from medibank.chat import Answer
from medibank.guardrails import GuardrailDecision
from medibank.knowledge import IndexResult

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def public_ui(monkeypatch, tmp_path):
    import app
    import medibank.deployment

    secret = "private-demo-key-" + tmp_path.name
    for key, value in {
        "APP_PUBLIC_MODE": "false",
        "CHAT_PROVIDER": "gemini",
        "EMBEDDING_PROVIDER": "gemini",
        "GEMINI_API_KEY": secret,
        "GEMINI_CHAT_MODEL": "gemini-3.8-flash",
        "GEMINI_EMBEDDING_MODEL": "gemini-embedding-001",
        "PDF_DIRECTORY": "medibank_data",
        "CHROMA_DIRECTORY": "data/chroma",
    }.items():
        monkeypatch.setenv(key, value)
    state = {"builds": 0, "answers": 0, "allow": True}

    def prepare(settings):
        state["builds"] += 1
        return IndexResult(object(), "public-test-index", [], 3, 3, 0, True)

    def answer(**kwargs):
        state["answers"] += 1
        return Answer("General membership information. [guide.pdf, p. 1]", [])

    monkeypatch.setattr(medibank.deployment, "prepare_public_index", prepare)
    monkeypatch.setattr(app, "answer_question", answer)
    monkeypatch.setattr(app, "create_chat_model", lambda config: object())
    monkeypatch.setattr(
        app,
        "assess_question",
        lambda *args: (
            GuardrailDecision(True, "test_fixture")
            if state["allow"]
            else GuardrailDecision(
                False, "human_requested", "A human should help.", needs_human=True
            )
        ),
    )

    def administrative_read(*args, **kwargs):
        raise AssertionError("Public chat must not load administrator-selected indexes or queues.")

    monkeypatch.setattr(app, "find_saved_index", administrative_read)
    monkeypatch.setattr(app, "list_handoffs", administrative_read)
    monkeypatch.setattr(app, "create_handoff", administrative_read)
    st.cache_resource.clear()
    yield AppTest.from_file(ROOT / "public_app.py", default_timeout=15), state, secret
    st.cache_resource.clear()


def test_public_chat_has_no_credentials_or_administration_and_reuses_bootstrap(public_ui):
    at, state, secret = public_ui
    at.run()
    assert not at.exception
    assert not at.text_input
    assert not at.selectbox
    assert not at.slider
    assert not any("Model settings" == item.label for item in at.expander)
    assert not any("Vector" in item.label or "Logs" in item.label for item in at.get("page_link"))
    assert at.chat_input[0].disabled is False
    assert secret not in str(at)
    at.run()
    assert not at.exception
    assert state["builds"] == 1


def test_public_human_fallback_only_contacts_medibank(public_ui):
    at, state, _ = public_ui
    state["allow"] = False
    at.run()
    at.chat_input[0].set_value("I want a human representative").run()
    assert not at.exception
    assert state["answers"] == 0
    assert at.session_state["messages"][-1]["handoff"]["status"] == "contact_required"
    assert any("Contact Medibank" in info.value for info in at.info)


def test_public_session_cooldown_blocks_repeat_model_calls(public_ui):
    at, state, _ = public_ui
    at.run()
    at.chat_input[0].set_value("What are membership benefits?").run()
    assert not at.exception
    assert state["answers"] == 1
    at.chat_input[0].set_value("What are membership benefits?").run()
    assert not at.exception
    assert state["answers"] == 1
    assert any("demo is busy" in warning.value for warning in at.warning)


def test_public_sessions_share_index_but_keep_conversations_and_activity_separate(public_ui):
    first, state, _ = public_ui
    first.run()
    first.chat_input[0].set_value("What are membership benefits?").run()
    second = AppTest.from_file(ROOT / "public_app.py", default_timeout=15).run()
    assert not first.exception and not second.exception
    assert first.session_state["messages"]
    assert second.session_state["messages"] == []
    assert first.session_state["session_id"] != second.session_state["session_id"]
    assert state["builds"] == 1
    second.toggle[0].set_value(True).run()
    assert not any(item.label == "Agent activity" for item in second.expander)


def test_secrets_cannot_disable_public_mode(public_ui):
    at, _, _ = public_ui
    at.secrets["APP_PUBLIC_MODE"] = "false"
    at.run()
    assert not at.exception
    from medibank.deployment import public_mode

    assert public_mode()
    assert not at.text_input


@pytest.mark.parametrize("page", ["1_Vector_Database.py", "2_Logs.py"])
def test_direct_administration_pages_stop_before_reading(monkeypatch, page):
    monkeypatch.setenv("APP_PUBLIC_MODE", "true")
    at = AppTest.from_file(ROOT / "pages" / page, default_timeout=15).run()
    assert not at.exception
    assert any("local app only" in error.value for error in at.error)
    assert not at.text_input
    assert not at.dataframe


def test_public_entry_point_fails_closed_when_key_missing(monkeypatch):
    monkeypatch.setenv("APP_PUBLIC_MODE", "false")
    monkeypatch.setenv("CHAT_PROVIDER", "gemini")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    at = AppTest.from_file(ROOT / "public_app.py", default_timeout=15).run()
    assert not at.exception
    assert any("not ready" in error.value for error in at.error)
    assert not at.chat_input
