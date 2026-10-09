"""Existing-index inspection preserves storage and checks vector compatibility."""

from __future__ import annotations

import json
import sqlite3

import pytest
from conftest import CountingEmbeddings

from medibank.config import ProviderConfig, embedding_identity
from medibank.inspection import (
    InspectionError,
    SavedIndex,
    browse_records,
    get_record,
    list_indexes,
    open_collection,
    semantic_search,
)
from medibank.knowledge import PdfInput, build_index


@pytest.fixture
def stored_index(tmp_path, pdf_bytes):
    config = ProviderConfig("ollama", "nomic-embed-text", "http://localhost:11434")
    embeddings = CountingEmbeddings()
    result = build_index(
        [PdfInput("hospital.pdf", pdf_bytes[0]), PdfInput("extras.pdf", pdf_bytes[1])],
        embeddings,
        embedding_identity(config),
        tmp_path / "chroma",
    )
    saved = list_indexes(tmp_path / "chroma")[0]
    return saved, open_collection(saved), config, embeddings, result


@pytest.mark.parametrize("database_kind", ["missing", "damaged", "unrelated_sqlite"])
def test_missing_or_invalid_database_is_never_initialized(stored_index, tmp_path, database_kind):
    original, _, _, _, _ = stored_index
    root = tmp_path / database_kind
    path = root / original.id
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps(original.manifest), encoding="utf-8")
    database = path / "chroma.sqlite3"
    if database_kind == "damaged":
        database.write_bytes(b"Damaged database bytes")
    elif database_kind == "unrelated_sqlite":
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE unrelated (value TEXT)")
    paths_before = {str(item.relative_to(root)) for item in root.rglob("*")}
    bytes_before = database.read_bytes() if database.exists() else None
    assert list_indexes(root) == []
    with pytest.raises(InspectionError):
        open_collection(SavedIndex(original.id, path, original.manifest))
    assert {str(item.relative_to(root)) for item in root.rglob("*")} == paths_before
    assert (database.read_bytes() if database.exists() else None) == bytes_before


def test_incomplete_manifest_is_skipped_without_opening_storage(stored_index, tmp_path):
    original, _, _, _, _ = stored_index
    root = tmp_path / "incomplete"
    path = root / original.id
    path.mkdir(parents=True)
    manifest = dict(original.manifest, complete=False)
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert list_indexes(root) == []
    assert not (path / "chroma.sqlite3").exists()


def test_manifest_changed_after_selection_requires_refresh(stored_index):
    saved, _, _, _, _ = stored_index
    manifest_path = saved.path / "manifest.json"
    manifest = dict(saved.manifest)
    manifest["chunk_count"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(InspectionError):
        open_collection(saved)


def test_embedding_identity_mismatch_fails_before_any_query_request(stored_index):
    saved, collection, config, embeddings, _ = stored_index
    mismatched = ProviderConfig(config.provider, "another-vector-model", config.base_url)
    with pytest.raises(InspectionError, match="do not match"):
        semantic_search(saved, collection, "Dental cover", mismatched, embeddings)
    assert embeddings.query_calls == []


def test_incompatible_query_dimension_has_actionable_error(stored_index):
    saved, collection, config, _, _ = stored_index

    class WrongDimension(CountingEmbeddings):
        def embed_query(self, text):
            self.query_calls.append(text)
            return [0.1, 0.2, 0.3]

    embeddings = WrongDimension()
    with pytest.raises(InspectionError, match="vector dimensions"):
        semantic_search(saved, collection, "Dental cover", config, embeddings)
    assert embeddings.query_calls == ["Dental cover"]


def test_invalid_stored_page_metadata_is_rejected(stored_index):
    _, collection, _, _, _ = stored_index
    identifier = collection.get(include=[])["ids"][0]
    collection.update(ids=[identifier], metadatas=[{"page_number": 99}])
    with pytest.raises(InspectionError, match="source or page metadata"):
        get_record(collection, identifier)


def test_same_page_source_filters_and_vector_loading_do_not_call_model(stored_index):
    _, collection, _, embeddings, _ = stored_index
    records = browse_records(collection, source="hospital.pdf", page=2)
    assert len(records) == 1
    assert "Ambulance transport" in records[0]["document"]
    vector_record = get_record(collection, records[0]["id"], include_vector=True)
    assert vector_record["dimension"] == len(vector_record["embedding"]) == 64
    assert embeddings.query_calls == []
