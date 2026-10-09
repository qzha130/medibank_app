"""Explicit PDF ingestion controls for the Vector Database page."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import streamlit as st
from langchain_core.embeddings import Embeddings

from medibank.config import ProviderConfig, embedding_identity
from medibank.knowledge import IndexResult, PdfInput, build_index, discover_pdfs
from medibank.observability import audit

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROVIDERS = {"Ollama · local": "ollama", "OpenAI · online": "openai", "Gemini · online": "gemini"}
DEFAULTS = {
    "ollama": ("nomic-embed-text", "http://localhost:11434"),
    "openai": ("text-embedding-3-small", "https://api.openai.com/v1"),
    "gemini": ("gemini-embedding-001", "https://generativelanguage.googleapis.com"),
}


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def prefer_index(dataset_id: str, directory: Path) -> None:
    """Communicate an explicit selection without replacing the chatbot's clients."""
    st.session_state["preferred_index_id"] = dataset_id
    st.session_state["preferred_chroma_directory"] = str(directory.resolve())
    st.session_state.pop("index_autoload_signature", None)


def render_ingestion_controls(
    directory: Path,
    create_embedding_client: Callable[[ProviderConfig], Embeddings],
) -> IndexResult | None:
    """Render local-folder/upload and indexing controls; build only after a click."""
    source_mode = st.radio(
        "PDF source", ["Local folder", "Upload PDFs"], horizontal=True, key="vector_pdf_source"
    )
    uploads = []
    folder = _project_path(os.getenv("PDF_DIRECTORY", "medibank_data"))
    source_error = None
    if source_mode == "Local folder":
        folder = _project_path(
            st.text_input(
                "PDF folder", os.getenv("PDF_DIRECTORY", "medibank_data"), key="vector_pdf_folder"
            )
        )
        try:
            if not folder.is_dir():
                paths = []
                source_error = "The PDF folder does not exist yet."
            else:
                paths = sorted(
                    (
                        path
                        for path in folder.iterdir()
                        if path.is_file() and path.suffix.lower() == ".pdf"
                    ),
                    key=lambda path: path.name.casefold(),
                )
        except OSError:
            paths = []
            source_error = "The PDF folder could not be read. Check its path and permissions."
        for path in paths:
            st.caption(f"📄 {path.name}")
        source_count = len(paths)
    else:
        uploads = st.file_uploader(
            "Choose the two PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Choose two searchable PDFs with distinct filenames, up to 50 MB each.",
            key="vector_pdf_uploads",
        )
        source_count = len(uploads)
    if source_error:
        st.info(source_error)
    if source_count != 2:
        st.caption(f"Add exactly two searchable PDFs to continue. Currently: {source_count}.")

    values = list(PROVIDERS.values())
    default_provider = os.getenv("EMBEDDING_PROVIDER", "ollama")
    label = st.selectbox(
        "Index embedding provider",
        list(PROVIDERS),
        index=values.index(default_provider) if default_provider in values else 0,
        key="vector_build_provider",
    )
    provider = PROVIDERS[label]
    model_default, url_default = DEFAULTS[provider]
    model = st.text_input(
        "Index embedding model",
        os.getenv(f"{provider.upper()}_EMBEDDING_MODEL", model_default),
        key=f"vector_build_model_{provider}",
    )
    base_url = st.text_input(
        "Index embedding server URL",
        os.getenv(f"{provider.upper()}_BASE_URL", url_default),
        key=f"vector_build_url_{provider}",
    )
    api_key = ""
    if provider != "ollama":
        default_key = (
            os.getenv("OPENAI_API_KEY", "")
            if provider == "openai"
            else (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", ""))
        )
        api_key = st.text_input(
            "Index API key", default_key, type="password", key=f"vector_build_key_{provider}"
        )
        st.caption(f"PDF text is sent to {provider.title()} to create embeddings.")
    else:
        st.caption("PDF text is embedded by the selected Ollama server.")
    chunk_size = st.slider(
        "Chunk size (characters)", 500, 2000, 1000, step=100, key="vector_chunk_size"
    )
    chunk_overlap = st.slider(
        "Chunk overlap (characters)", 0, 400, 200, step=50, key="vector_chunk_overlap"
    )
    config = None
    try:
        config = ProviderConfig(provider, model.strip(), base_url.strip(), api_key.strip())
    except ValueError as error:
        st.error(str(error))
    has_key = provider == "ollama" or bool(api_key.strip())
    if not has_key:
        st.caption("Enter the embedding provider API key to build this index.")
    can_build = source_count == 2 and not source_error and config is not None and has_key
    if not st.button(
        "Build / load knowledge base",
        type="primary",
        disabled=not can_build,
        width="stretch",
        key="vector_build_index",
    ):
        return None

    session_id = st.session_state.get("session_id")
    audit(
        "index_load_started",
        session_id=session_id,
        provider=provider,
        model=model,
        pdf_count=source_count,
    )
    try:
        with st.spinner("Reading PDFs and building or loading their index…"):
            pdfs = (
                discover_pdfs(folder)
                if source_mode == "Local folder"
                else [PdfInput(upload.name, upload.getvalue()) for upload in uploads]
            )
            result = build_index(
                pdfs=pdfs,
                embeddings=create_embedding_client(config),
                embedding_identity=embedding_identity(config),
                persist_directory=directory,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        prefer_index(result.dataset_id, directory)
        audit(
            "index_loaded" if result.reused else "index_built",
            session_id=session_id,
            dataset_id=result.dataset_id,
            pages=result.page_count,
            passages=result.chunk_count,
        )
        st.success("Saved index loaded." if result.reused else "Knowledge base ready.")
        st.caption(
            "This index is selected for chat. Use matching embedding settings on the chatbot page."
        )
        if result.skipped_pages:
            st.warning(
                f"Skipped {result.skipped_pages} pages without extractable text. Run OCR if they contain information."
            )
        return result
    except ValueError as error:
        st.error(str(error))
        audit("index_failed", level="ERROR", session_id=session_id, error_type=type(error).__name__)
    except Exception:
        audit("index_failed", level="ERROR", session_id=session_id, provider=provider)
        message = "Couldn't build the index. Check the embedding model, service, API key, and Chroma folder permissions."
        if provider == "ollama":
            message += f" For Ollama, run: ollama pull {model.strip()}"
        st.error(message)
    return None
