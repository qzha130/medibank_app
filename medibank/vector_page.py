"""Streamlit tools for reviewing and querying existing local Chroma indexes."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from uuid import uuid4

import streamlit as st

from medibank.config import ProviderConfig, embedding_identity
from medibank.ingestion_page import prefer_index, render_ingestion_controls
from medibank.inspection import (
    InspectionError,
    browse_records,
    count_records,
    get_record,
    list_indexes,
    open_collection,
    semantic_search,
)
from medibank.models import create_embeddings
from medibank.observability import audit

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROVIDERS = {"Ollama · local": "ollama", "OpenAI · online": "openai", "Gemini · online": "gemini"}
DEFAULTS = {
    "ollama": ("nomic-embed-text", "http://localhost:11434"),
    "openai": ("text-embedding-3-small", "https://api.openai.com/v1"),
    "gemini": ("gemini-embedding-001", "https://generativelanguage.googleapis.com"),
}


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def _record_label(record: dict) -> str:
    metadata = record["metadata"]
    return (
        f"{metadata.get('source', 'Unknown document')} · "
        f"page {metadata.get('page_number', metadata.get('page', '?'))} · "
        f"{record['id'][:12]}"
    )


def _table_record(record: dict) -> dict:
    metadata = record["metadata"]
    return {
        "Record ID": record["id"],
        "Document": metadata.get("source", ""),
        "Page": metadata.get("page_number", metadata.get("page", "")),
        "Chunk": metadata.get("chunk_index", ""),
        "Characters": len(record["document"]),
        "Text preview": record["document"][:180],
    }


def _csv_export(records: list[dict]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=["id", "source", "page", "document", "metadata"])
    writer.writeheader()
    for record in records:
        metadata = record["metadata"]
        writer.writerow(
            {
                "id": record["id"],
                "source": metadata.get("source", ""),
                "page": metadata.get("page_number", metadata.get("page", "")),
                "document": record["document"],
                "metadata": json.dumps(metadata, ensure_ascii=False),
            }
        )
    return output.getvalue()


def _search_settings() -> ProviderConfig | None:
    values = list(PROVIDERS.values())
    default_provider = os.getenv("EMBEDDING_PROVIDER", "ollama")
    label = st.selectbox(
        "Search embedding provider",
        list(PROVIDERS),
        index=values.index(default_provider) if default_provider in values else 0,
        key="vector_search_provider",
    )
    provider = PROVIDERS[label]
    model_default, url_default = DEFAULTS[provider]
    model = st.text_input(
        "Search embedding model",
        os.getenv(f"{provider.upper()}_EMBEDDING_MODEL", model_default),
        key=f"vector_search_model_{provider}",
    )
    base_url = st.text_input(
        "Search embedding server URL",
        os.getenv(f"{provider.upper()}_BASE_URL", url_default),
        key=f"vector_search_url_{provider}",
    )
    api_key = ""
    if provider != "ollama":
        default_key = (
            os.getenv("OPENAI_API_KEY", "")
            if provider == "openai"
            else (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", ""))
        )
        api_key = st.text_input(
            "Search API key", default_key, type="password", key=f"vector_search_key_{provider}"
        )
        st.caption(
            f"Your search query is sent to {provider.title()} for its embedding. "
            "The search uses the documents already saved in Chroma."
        )
    try:
        return ProviderConfig(provider, model.strip(), base_url.strip(), api_key.strip())
    except ValueError as error:
        st.error(str(error))
        return None


def _chat_navigation_hint() -> None:
    st.caption("Open the chatbot from the navigation sidebar.")


def render_vector_page() -> None:
    st.set_page_config(page_title="Chroma Vector Database", page_icon="🗄️", layout="wide")
    st.title("Chroma Vector Database")
    st.session_state.setdefault("session_id", uuid4().hex)
    st.caption("Review saved PDF passages and test what a question retrieves before asking an LLM.")

    with st.sidebar:
        st.subheader("Local database")
        directory_text = st.text_input(
            "Chroma directory",
            st.session_state.get(
                "preferred_chroma_directory", os.getenv("CHROMA_DIRECTORY", "data/chroma")
            ),
            key="vector_directory",
        )
        directory = _path(directory_text)
        try:
            indexes = list_indexes(directory)
        except (InspectionError, OSError):
            st.error("The Chroma directory could not be read. Check its path and permissions.")
            indexes = []
        with st.expander("Add or update PDFs", expanded=not indexes):
            built_index = render_ingestion_controls(directory, create_embeddings)
        if built_index is not None:
            indexes = list_indexes(directory)
        if not indexes:
            st.info(
                "No completed Chroma indexes found. Add your two PDFs and build the knowledge base here."
            )
            return
        by_id = {index.id: index for index in indexes}
        active = st.session_state.get("index")
        active_id = st.session_state.get("preferred_index_id") or getattr(
            active, "dataset_id", None
        )
        choices = list(by_id)
        selector_key = f"vector_index_{hashlib.sha256(str(directory).encode()).hexdigest()[:12]}"
        if built_index is not None:
            st.session_state[selector_key] = built_index.dataset_id
        selected_id = st.selectbox(
            "Saved index",
            choices,
            index=choices.index(active_id) if active_id in choices else 0,
            format_func=lambda value: (
                f"{value[:12]} · {by_id[value].manifest['chunk_count']} passages"
            ),
            key=selector_key,
        )
        index = by_id[selected_id]
        st.caption("Storage location")
        st.text(str(index.path))
        if st.button("Use in chat", width="stretch"):
            prefer_index(index.id, directory)
            audit(
                "chat_index_selected",
                session_id=st.session_state.get("session_id"),
                dataset_id=index.id,
            )
            st.success("This saved index is selected for chat.")
            st.caption(
                "Select its matching embedding provider, model, and server on the chatbot page."
            )
        _chat_navigation_hint()

    try:
        collection = open_collection(index)
        total = count_records(collection)
    except (InspectionError, OSError):
        st.error("This saved index could not be opened. Check its database files or rebuild it.")
        return
    manifest = index.manifest
    if st.session_state.get("last_reviewed_index") != str(index.path):
        audit(
            "database_review_opened",
            session_id=st.session_state.get("session_id"),
            dataset_id=index.id,
            passages=total,
        )
        st.session_state["last_reviewed_index"] = str(index.path)
    columns = st.columns(3)
    columns[0].metric("Stored passages", total)
    columns[1].metric("Knowledge PDFs", len(manifest["files"]))
    columns[2].metric("PDF pages", manifest["page_count"])
    if total != manifest["chunk_count"]:
        st.warning(
            f"The saved manifest expects {manifest['chunk_count']} passages, but Chroma contains {total}. "
            "Use the PDF indexing controls on this page to build or load the knowledge base and repair it."
        )
    with st.expander("Index details"):
        st.json(manifest)

    documents = sorted(file["name"] for file in manifest["files"])
    filters = st.columns(2)
    document_choice = filters[0].selectbox(
        "Document filter", ["All documents", *documents], key=f"vector_document_{selected_id}"
    )
    source = None if document_choice == "All documents" else document_choice
    maximum_page = max(file["page_count"] for file in manifest["files"])
    page_number = filters[1].number_input(
        "Page filter",
        min_value=0,
        max_value=maximum_page,
        value=0,
        step=1,
        help="0 includes every page. Page numbers are the physical PDF pages, starting at 1.",
        key=f"vector_page_{selected_id}",
    )
    page = int(page_number) or None
    filter_context = (str(index.path), source, page)
    filter_key = hashlib.sha256(repr(filter_context).encode()).hexdigest()[:16]
    try:
        filtered_count = count_records(collection, source=source, page=page)
    except InspectionError as error:
        st.error(str(error))
        return
    st.caption(f"{filtered_count} passages match the document and page filters.")
    browse_tab, search_tab = st.tabs(["Browse passages", "Semantic search"])

    with browse_tab:
        controls = st.columns(2)
        page_size = controls[0].selectbox("Records per page", [10, 25, 50, 100], index=1)
        page_count = max(1, math.ceil(filtered_count / page_size))
        record_page = controls[1].number_input(
            "Record page",
            min_value=1,
            max_value=page_count,
            value=1,
            step=1,
            key=f"vector_record_page_{filter_key}_{page_size}",
        )
        try:
            records = browse_records(
                collection,
                limit=page_size,
                offset=(int(record_page) - 1) * page_size,
                source=source,
                page=page,
            )
        except InspectionError as error:
            st.error(str(error))
            return
        if records:
            st.dataframe(
                [_table_record(record) for record in records], hide_index=True, width="stretch"
            )
            by_record_id = {record["id"]: record for record in records}
            record_id = st.selectbox(
                "Record to inspect",
                list(by_record_id),
                format_func=lambda value: _record_label(by_record_id[value]),
                key=f"vector_record_{filter_key}_{page_size}_{record_page}",
            )
            selected_record = by_record_id[record_id]
            st.subheader("Passage text")
            st.text(selected_record["document"])
            with st.expander("Record metadata", expanded=True):
                st.json(selected_record["metadata"])
            vector_context = (str(index.path), record_id)
            if st.button("Load vector values", key=f"vector_values_{filter_key}"):
                vector_record = get_record(collection, record_id, include_vector=True)
                st.session_state["vector_values"] = (vector_context, vector_record)
            loaded = st.session_state.get("vector_values")
            if loaded and loaded[0] == vector_context and loaded[1]:
                vector = loaded[1].get("embedding") or []
                if vector:
                    statistics = st.columns(2)
                    statistics[0].metric("Vector dimensions", len(vector))
                    statistics[1].metric(
                        "Vector norm", f"{math.sqrt(sum(v * v for v in vector)):.4f}"
                    )
                    st.line_chart(vector, x_label="Dimension", y_label="Value", height=220)
                    with st.expander("Vector values"):
                        st.json(vector)
                else:
                    st.info("No embedding is stored for this record.")
        else:
            st.info("No passages match these filters.")

        st.divider()
        st.subheader("Export passages")
        st.caption(
            "Exports include passage text and metadata. Vector values can be reviewed above."
        )
        if filtered_count > 10000:
            st.info("Use document or page filters to export up to 10,000 passages at once.")
        elif st.button("Prepare filtered export", disabled=filtered_count == 0):
            exported = []
            for offset in range(0, filtered_count, 100):
                exported.extend(
                    browse_records(collection, limit=100, offset=offset, source=source, page=page)
                )
            st.session_state["vector_export"] = (filter_context, exported)
            audit(
                "database_export_prepared",
                session_id=st.session_state.get("session_id"),
                dataset_id=index.id,
                passages=len(exported),
                source=source,
                page=page,
            )
        export = st.session_state.get("vector_export")
        if export and export[0] == filter_context:
            export_columns = st.columns(2)
            export_columns[0].download_button(
                "Export filtered records (JSON)",
                json.dumps(export[1], indent=2, ensure_ascii=False),
                "chroma-passages.json",
                "application/json",
                width="stretch",
            )
            export_columns[1].download_button(
                "Export filtered records (CSV)",
                _csv_export(export[1]),
                "chroma-passages.csv",
                "text/csv",
                width="stretch",
            )

    with search_tab:
        st.subheader("Search the stored vectors")
        st.caption("Search uses the filters above and an embedding model. No chat model is called.")
        with st.expander("Query embedding settings"):
            config = _search_settings()
        if config:
            st.caption(f"Query embedding: {config.provider.title()} · {config.model}")
        expected_identity = manifest["specification"]["embedding_identity"]
        compatible = config is not None and embedding_identity(config) == expected_identity
        has_key = config is not None and (config.provider == "ollama" or bool(config.api_key))
        if config and not compatible:
            st.info(
                "The embedding settings do not match this index. Select the settings used to build it."
            )
        elif config and not has_key:
            st.info(
                "Enter the embedding provider API key to search. Browsing does not require a key."
            )
        query = st.text_input(
            "Search query", placeholder="e.g. What waiting periods apply?", max_chars=4000
        )
        limit = st.number_input("Result count", min_value=1, max_value=20, value=5, step=1)
        search_context = (
            filter_context,
            embedding_identity(config) if config else None,
            hashlib.sha256(config.api_key.encode()).hexdigest() if config else None,
            query.strip(),
            int(limit),
        )
        if st.button(
            "Search vectors",
            type="primary",
            disabled=not (compatible and has_key and query.strip()),
        ):
            audit(
                "vector_search_started",
                session_id=st.session_state.get("session_id"),
                dataset_id=index.id,
                query=query.strip(),
                provider=config.provider,
                model=config.model,
            )
            st.session_state.pop("vector_results", None)
            try:
                with st.spinner("Embedding your query and searching Chroma…"):
                    results = semantic_search(
                        index,
                        collection,
                        query.strip(),
                        config,
                        create_embeddings(config),
                        limit=int(limit),
                        source=source,
                        page=page,
                    )
                st.session_state["vector_results"] = (search_context, results)
                audit(
                    "vector_search_completed",
                    session_id=st.session_state.get("session_id"),
                    dataset_id=index.id,
                    results=len(results),
                    distances=[row["distance"] for row in results],
                )
            except ValueError as error:
                st.error(str(error))
                audit("vector_search_failed", level="ERROR", error_type=type(error).__name__)
            except Exception:
                audit("vector_search_failed", level="ERROR", error_type="unexpected_error")
                st.error(
                    "Search failed. Check the embedding service, selected model, URL, and API key."
                )
        result_state = st.session_state.get("vector_results")
        if result_state and result_state[0] == search_context:
            results = result_state[1]
            st.caption(
                "Cosine distance: lower values mean closer vectors; this is not an answer confidence score."
            )
            if not results:
                st.info("No search results match these filters.")
            for rank, record in enumerate(results, start=1):
                with st.expander(
                    f"{rank} · {_record_label(record)} · distance {record['distance']:.4f}",
                    expanded=rank == 1,
                ):
                    st.text(record["document"])
                    st.json(record["metadata"])
