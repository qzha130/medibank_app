"""Run with: .venv/Scripts/python.exe -m streamlit run app.py."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from medibank.activity import request_activity
from medibank.chat import Answer, ChatError, _abstention_only, answer_question
from medibank.config import ProviderConfig, embedding_identity
from medibank.deployment import PublicSettings, public_mode
from medibank.guardrails import LEGACY_FALLBACK_MESSAGES, assess_question
from medibank.knowledge import IndexResult
from medibank.limits import RequestLimiter
from medibank.loading import find_saved_index, load_saved_index
from medibank.models import create_chat_model, create_embeddings
from medibank.observability import audit, create_handoff, list_handoffs

PROJECT_DIR = Path(__file__).resolve().parent
if not public_mode():
    load_dotenv(PROJECT_DIR / ".env")
PROVIDERS = {
    "Ollama · local": "ollama",
    "OpenAI · online": "openai",
    "Gemini · online": "gemini",
}
MODEL_DEFAULTS = {
    "ollama": ("llama3.1:8b", "nomic-embed-text"),
    "openai": ("gpt-4.1-mini", "text-embedding-3-small"),
    "gemini": ("gemini-3.8-flash", "gemini-embedding-001"),
}


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def provider_index(value: str) -> int:
    return list(PROVIDERS.values()).index(value) if value in PROVIDERS.values() else 0


def answer_markdown(text: str, sources: list[dict] | None = None) -> str:
    """Keep answer formatting without auto-loading model-supplied remote images."""
    for source in sources or []:
        filename = source["source"]
        label = Path(filename).stem.replace("-", " ").replace("_", " ").title()
        text = text.replace(f"[{filename}, p. {source['page']}]", f"[{label}, p. {source['page']}]")
    return text.replace("![", r"\![")


def show_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources · {len(sources)}"):
        st.caption("Information used for this answer.")
        for source in sources:
            page = f" · page {source['page']}" if source.get("page") else ""
            st.caption(
                f"{Path(source['source']).stem.replace('-', ' ').replace('_', ' ').title()}{page}"
            )
            st.text(source["excerpt"])


def handoff_for(question: str, reason: str, session_id: str, public: bool) -> dict:
    if public:
        return {"id": uuid4().hex, "status": "contact_required", "reason": reason}
    return create_handoff(question, reason, session_id)


def show_handoff(ticket: dict) -> None:
    if ticket.get("status") == "contact_required":
        st.info("Contact Medibank for help from a human representative.")
        st.link_button("Contact Medibank", "https://www.medibank.com.au/contact-us/")
        return
    try:
        ticket = next((item for item in list_handoffs() if item["id"] == ticket["id"]), ticket)
    except OSError:
        pass
    st.info(f"Human review requested · {ticket['status']}")
    st.caption("Saved in this app’s review queue. Contact Medibank for direct help.")
    if ticket.get("note"):
        st.text("Reviewer note: " + ticket["note"])
    st.link_button("Contact Medibank", "https://www.medibank.com.au/contact-us/")
    st.download_button(
        "Download this review request",
        json.dumps(ticket, indent=2),
        f"human-review-{ticket['id'][:12]}.json",
        "application/json",
        key=f"handoff_{ticket['id']}",
    )


def show_activity(message: dict, enabled: bool, session_id: str) -> None:
    if not enabled:
        return
    activity = message.get("activity")
    if activity is None:
        activity = request_activity(message.get("request_id"), session_id)
    with st.expander("Agent activity", expanded=True):
        if activity:
            st.dataframe(activity, hide_index=True, width="stretch")
        else:
            st.caption("Activity is available for new replies.")


def remember_error(
    question: str, request_id: str, text: str, ticket: dict | None, enabled: bool
) -> None:
    message = {
        "question": question,
        "content": text,
        "handoff": ticket,
        "request_id": request_id,
        "activity": request_activity(request_id, st.session_state["session_id"]),
    }
    st.session_state["last_chat_error"] = message
    show_activity(message, enabled, st.session_state["session_id"])


def main(
    public_settings: PublicSettings | None = None,
    public_index: IndexResult | None = None,
    request_limiter: RequestLimiter | None = None,
) -> None:
    public = public_settings is not None
    st.set_page_config(
        page_title="Medibank Assistant",
        page_icon="📚",
        layout="wide",
    )
    st.session_state.setdefault("messages", [])
    for message in st.session_state["messages"]:
        if message.get("role") == "assistant":
            content = message.get("content", "")
            message["content"] = LEGACY_FALLBACK_MESSAGES.get(content, content)
    settings = st.session_state.setdefault("chat_settings", {})
    st.session_state.setdefault("session_id", uuid4().hex)
    session_id = st.session_state["session_id"]
    if not st.session_state.get("session_logged"):
        audit("session_started", session_id=session_id)
        st.session_state["session_logged"] = True

    with st.sidebar:
        st.title("Chat options")
        show_agent_activity = st.toggle(
            "Show agent activity",
            value=settings.get("show_agent_activity", False),
            key="show_agent_activity",
            help="Show checks, knowledge searches, and response timing under each answer.",
        )
        settings["show_agent_activity"] = show_agent_activity
        if public:
            chat_config = public_settings.chat
            embedding_config = public_settings.embedding
            chat_provider, chat_name = chat_config.provider, chat_config.model
            embedding_provider, embedding_name = embedding_config.provider, embedding_config.model
            api_keys = {
                chat_provider: chat_config.api_key,
                embedding_provider: embedding_config.api_key,
            }
            top_k = public_settings.top_k
            config_error = None
            providers = sorted({chat_provider, embedding_provider})
            st.caption(
                "Questions and relevant Medibank information are sent to "
                + " and ".join(name.title() for name in providers)
                + "."
            )
            st.link_button("Contact Medibank", "https://www.medibank.com.au/contact-us/")
        else:
            with st.expander("Model settings"):
                chat_label = st.selectbox(
                    "Chat provider",
                    list(PROVIDERS),
                    index=provider_index(
                        settings.get("chat_provider", os.getenv("CHAT_PROVIDER", "ollama"))
                    ),
                    key="chat_provider",
                )
                chat_provider = PROVIDERS[chat_label]
                chat_default = settings.get(
                    f"chat_model_{chat_provider}",
                    os.getenv(
                        f"{chat_provider.upper()}_CHAT_MODEL", MODEL_DEFAULTS[chat_provider][0]
                    ),
                )
                chat_name = st.text_input(
                    "Chat model", chat_default, key=f"chat_model_{chat_provider}"
                )
                embedding_label = st.selectbox(
                    "Embedding provider",
                    list(PROVIDERS),
                    index=provider_index(
                        settings.get(
                            "embedding_provider", os.getenv("EMBEDDING_PROVIDER", "ollama")
                        )
                    ),
                    key="embedding_provider",
                    help="The embedding model searches the Medibank knowledge base. It can differ from the chat model.",
                )
                embedding_provider = PROVIDERS[embedding_label]
                embedding_default = settings.get(
                    f"embedding_model_{embedding_provider}",
                    os.getenv(
                        f"{embedding_provider.upper()}_EMBEDDING_MODEL",
                        MODEL_DEFAULTS[embedding_provider][1],
                    ),
                )
                embedding_name = st.text_input(
                    "Embedding model",
                    embedding_default,
                    key=f"embedding_model_{embedding_provider}",
                )
                api_keys = {
                    "ollama": "",
                    "openai": settings.get("openai_api_key", os.getenv("OPENAI_API_KEY", "")),
                    "gemini": settings.get(
                        "gemini_api_key",
                        os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", ""),
                    ),
                }
                selected_providers = {chat_provider, embedding_provider}
                for provider, label in (("openai", "OpenAI"), ("gemini", "Gemini")):
                    if provider not in selected_providers:
                        continue
                    api_keys[provider] = st.text_input(
                        f"{label} API key",
                        api_keys[provider],
                        type="password",
                        key=f"{provider}_api_key",
                    )
                    if embedding_provider == provider:
                        st.caption(f"Search queries are sent to {label} for embeddings.")
                    if chat_provider == provider:
                        st.caption(
                            f"Chat sends your questions, history, and retrieved passages to {label}."
                        )
                if selected_providers == {"ollama"}:
                    st.caption("With a local Ollama server, chat stays on your computer.")

            with st.expander("Connection and search"):
                ollama_url = st.text_input(
                    "Ollama server URL",
                    settings.get(
                        "ollama_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                    ),
                )
                openai_url = st.text_input(
                    "OpenAI API URL",
                    settings.get(
                        "openai_url", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
                    ),
                )
                gemini_url = st.text_input(
                    "Gemini API URL",
                    settings.get(
                        "gemini_url",
                        os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com"),
                    ),
                )
                top_k = st.slider("Passages per search", 2, 12, settings.get("top_k", 5))

            settings.update(
                chat_provider=chat_provider,
                embedding_provider=embedding_provider,
                ollama_url=ollama_url,
                openai_url=openai_url,
                gemini_url=gemini_url,
                top_k=top_k,
            )
            settings[f"chat_model_{chat_provider}"] = chat_name
            settings[f"embedding_model_{embedding_provider}"] = embedding_name
            for provider in selected_providers - {"ollama"}:
                settings[f"{provider}_api_key"] = api_keys[provider]

            config_error = None
            chat_config = embedding_config = None
            try:
                urls = {
                    "ollama": ollama_url.strip(),
                    "openai": openai_url.strip(),
                    "gemini": gemini_url.strip(),
                }
                chat_config = ProviderConfig(
                    chat_provider,
                    chat_name.strip(),
                    urls[chat_provider],
                    api_keys[chat_provider].strip(),
                )
                embedding_config = ProviderConfig(
                    embedding_provider,
                    embedding_name.strip(),
                    urls[embedding_provider],
                    api_keys[embedding_provider].strip(),
                )
            except ValueError as error:
                config_error = str(error)

            st.page_link("pages/1_Vector_Database.py", label="Manage vector database", icon="🗄️")
        directory = (
            public_settings.chroma_directory
            if public
            else project_path(
                st.session_state.get("preferred_chroma_directory")
                or os.getenv("CHROMA_DIRECTORY", "data/chroma")
            ).resolve()
        )
        preferred_id = None if public else st.session_state.get("preferred_index_id")
        signature = (
            embedding_identity(embedding_config) if embedding_config else None,
            hashlib.sha256(embedding_config.api_key.encode()).hexdigest()
            if embedding_config
            else None,
            str(directory),
            preferred_id,
        )
        if public:
            st.session_state["index"] = public_index
            st.session_state["index_signature"] = signature
        if config_error:
            st.error(config_error)
        can_load = embedding_config is not None and (
            embedding_provider == "ollama" or bool(api_keys[embedding_provider].strip())
        )
        if not can_load and embedding_provider != "ollama":
            st.caption("Enter the embedding API key to search your saved index.")
        if (
            not public
            and can_load
            and st.session_state.get("index_autoload_signature") != signature
        ):
            old_index = st.session_state.pop("index", None)
            previous_dataset = st.session_state.get("loaded_dataset_id") or getattr(
                old_index, "dataset_id", None
            )
            st.session_state["index_autoload_signature"] = signature
            st.session_state.pop("index_autoload_error", None)
            audit(
                "index_load_started",
                session_id=session_id,
                provider=embedding_provider,
                model=embedding_name,
                automatic=True,
            )
            try:
                with st.spinner("Loading the saved Chroma index…"):
                    saved = find_saved_index(directory, embedding_config, preferred_id)
                    if saved is not None:
                        index = load_saved_index(saved, create_embeddings(embedding_config))
                        st.session_state["index"] = index
                        st.session_state["index_signature"] = signature
                        if previous_dataset is not None and previous_dataset != index.dataset_id:
                            st.session_state["messages"] = []
                        st.session_state["loaded_dataset_id"] = index.dataset_id
                        audit(
                            "index_loaded",
                            session_id=session_id,
                            dataset_id=index.dataset_id,
                            pages=index.page_count,
                            passages=index.chunk_count,
                            automatic=True,
                        )
                    else:
                        audit("index_not_found", session_id=session_id, automatic=True)
            except ValueError as error:
                st.session_state["index_autoload_error"] = str(error)
                audit(
                    "index_failed",
                    level="ERROR",
                    session_id=session_id,
                    error_type=type(error).__name__,
                    automatic=True,
                )
            except Exception:
                st.session_state["index_autoload_error"] = (
                    "The saved index could not be loaded. Check its database files on the Vector Database page."
                )
                audit(
                    "index_failed",
                    level="ERROR",
                    session_id=session_id,
                    error_type="unexpected_error",
                    automatic=True,
                )
        if st.session_state.get("index_autoload_error"):
            st.error(st.session_state["index_autoload_error"])
        active_index = st.session_state.get("index")
        if active_index is not None and st.session_state.get("index_signature") == signature:
            st.caption("Knowledge ready")

        st.divider()
        if st.button("Clear conversation", width="stretch"):
            st.session_state["messages"] = []
            st.session_state.pop("last_chat_error", None)
            audit("conversation_cleared", session_id=session_id)
        if st.session_state["messages"]:
            transcript = "\n\n".join(
                f"{m['role'].upper()}\n{m['content']}" for m in st.session_state["messages"]
            )
            st.download_button(
                "Download conversation",
                transcript,
                "conversation.txt",
                "text/plain",
                width="stretch",
            )
        if not public:
            with st.expander("Request human review"):
                review_question = st.text_area("Question for human review", max_chars=4000)
                if st.button("Save human review request", disabled=not review_question.strip()):
                    try:
                        ticket = create_handoff(review_question, "manual_request", session_id)
                        st.session_state["manual_handoff"] = ticket
                    except OSError:
                        st.error(
                            "The review request could not be saved. Contact Medibank directly."
                        )
                if st.session_state.get("manual_handoff"):
                    show_handoff(st.session_state["manual_handoff"])

    st.title("Medibank Assistant")
    st.caption(
        "General information about membership, benefits, claims, exclusions, and waiting periods."
    )
    if public:
        st.caption(
            "Independent demo. For personal or account help, contact Medibank directly. "
            "Please avoid sharing membership numbers or medical details."
        )
    index = st.session_state.get("index")
    index_matches = index is not None and st.session_state.get("index_signature") == signature
    chat_ready = chat_config is not None and (
        chat_provider == "ollama" or bool(api_keys[chat_provider].strip())
    )
    ready = index_matches and chat_ready
    if index is not None:
        if not index_matches:
            st.info("Open Model settings and select the search model used by your knowledge base.")
        elif not chat_ready:
            st.info("Open Model settings to complete your chat model and API key.")
    else:
        st.info(
            "Knowledge is unavailable for the selected search model. Open Vector Database to prepare it, "
            "or update Model settings."
        )

    suggested = None
    if not st.session_state["messages"]:
        with st.container(border=True):
            st.subheader("Start with a question")
            st.write("Ask about your membership, benefits, claims, exclusions, or waiting periods.")
            suggestions = [
                "What waiting periods apply?",
                "How do I make a claim?",
                "What exclusions apply?",
            ]
            for column, suggestion in zip(st.columns(3), suggestions):
                if column.button(suggestion, disabled=not ready, width="stretch"):
                    suggested = suggestion

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            content = message["content"]
            st.markdown(
                answer_markdown(content, message.get("sources", []))
                if message["role"] == "assistant"
                else answer_markdown(content)
            )
            show_sources(message.get("sources", []))
            if message["role"] == "assistant":
                show_activity(message, show_agent_activity, session_id)
            if message.get("handoff"):
                show_handoff(message["handoff"])

    if st.session_state.get("last_chat_error"):
        failed = st.session_state["last_chat_error"]
        with st.chat_message("user"):
            st.markdown(answer_markdown(failed["question"]))
        with st.chat_message("assistant"):
            st.error(failed["content"])
            show_activity(failed, show_agent_activity, session_id)
            if failed.get("handoff"):
                show_handoff(failed["handoff"])

    question = st.chat_input(
        "Ask a general question about Medibank…", disabled=not ready, max_chars=4000
    )
    question = question or suggested
    if question and ready:
        if request_limiter is not None and not request_limiter.allow(session_id):
            st.warning("The demo is busy. Please wait before trying again, or contact Medibank.")
            return
        request_id = uuid4().hex
        started = time.perf_counter()
        audit(
            "question_received",
            session_id=session_id,
            request_id=request_id,
            question=question,
            chat_provider=chat_provider,
            chat_model=chat_name,
        )
        with st.chat_message("user"):
            st.markdown(answer_markdown(question))
        with st.chat_message("assistant"):
            try:
                with st.spinner("Checking Medibank information…"):
                    decision = assess_question(question, st.session_state["messages"], index.store)
                    audit(
                        "guardrail_decision",
                        level="INFO" if decision.allowed else "WARNING",
                        session_id=session_id,
                        request_id=request_id,
                        allowed=decision.allowed,
                        reason=decision.reason,
                        distance=decision.distance,
                    )
                    ticket = None
                    if not decision.allowed:
                        answer = Answer(decision.message, [])
                        if decision.needs_human:
                            ticket = handoff_for(question, decision.reason, session_id, public)
                    else:
                        audit(
                            "chat_model_started",
                            session_id=session_id,
                            request_id=request_id,
                            provider=chat_provider,
                            model=chat_name,
                        )
                        answer = answer_question(
                            question=question,
                            history=st.session_state["messages"],
                            store=index.store,
                            model=create_chat_model(chat_config),
                            top_k=top_k,
                            request_id=request_id,
                            session_id=session_id,
                        )
                        if _abstention_only(answer.text):
                            ticket = handoff_for(
                                question, "answer_not_supported", session_id, public
                            )
                    audit(
                        "answer_completed",
                        session_id=session_id,
                        request_id=request_id,
                        answer=answer.text,
                        human_review=bool(ticket),
                        ticket_id=ticket["id"] if ticket else None,
                        sources=[
                            {"source": source["source"], "page": source["page"]}
                            for source in answer.sources
                        ],
                        elapsed_ms=round((time.perf_counter() - started) * 1000),
                    )
                st.session_state["messages"].extend(
                    [
                        {"role": "user", "content": question},
                        {
                            "role": "assistant",
                            "content": answer.text,
                            "sources": answer.sources,
                            "handoff": ticket,
                            "request_id": request_id,
                            "activity": request_activity(request_id, session_id),
                        },
                    ]
                )
                st.session_state.pop("last_chat_error", None)
                st.rerun()
            except (ChatError, ValueError) as error:
                st.error(str(error))
                audit(
                    "chat_failed",
                    level="ERROR",
                    session_id=session_id,
                    request_id=request_id,
                    error_type=type(error).__name__,
                    elapsed_ms=round((time.perf_counter() - started) * 1000),
                )
                ticket = None
                try:
                    ticket = handoff_for(
                        question, "automatic_answer_unavailable", session_id, public
                    )
                    show_handoff(ticket)
                except OSError:
                    st.info("Contact Medibank directly if you need a human representative.")
                remember_error(question, request_id, str(error), ticket, show_agent_activity)
            except Exception:
                audit(
                    "chat_failed",
                    level="ERROR",
                    session_id=session_id,
                    request_id=request_id,
                    error_type="unexpected_error",
                )
                st.error(
                    "Couldn't answer this question. Check the selected chat model and connection settings."
                )
                ticket = None
                try:
                    ticket = handoff_for(
                        question, "automatic_answer_unavailable", session_id, public
                    )
                    show_handoff(ticket)
                except OSError:
                    st.info("Contact Medibank directly if you need a human representative.")
                remember_error(
                    question,
                    request_id,
                    "This question could not be answered. Please retry or request human review.",
                    ticket,
                    show_agent_activity,
                )


if __name__ == "__main__":
    main()
