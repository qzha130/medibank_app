"""Saved chat indexes load without PDF access, rebuilding, or embedding calls."""

import pytest
from conftest import CountingEmbeddings

from medibank.config import ProviderConfig, embedding_identity
from medibank.inspection import InspectionError, list_indexes
from medibank.knowledge import PdfInput, build_index
from medibank.loading import find_saved_index, load_saved_index


@pytest.fixture
def saved(tmp_path, pdf_bytes):
    config = ProviderConfig("ollama", "nomic-embed-text", "http://localhost:11434")
    embeddings = CountingEmbeddings()
    result = build_index(
        [PdfInput("hospital.pdf", pdf_bytes[0]), PdfInput("extras.pdf", pdf_bytes[1])],
        embeddings,
        embedding_identity(config),
        tmp_path,
    )
    return tmp_path, config, result, embeddings


def test_startup_load_reuses_vectors_without_embedding_documents(saved):
    root, config, result, embeddings = saved
    batches = len(embeddings.document_batches)
    loaded = load_saved_index(find_saved_index(root, config), embeddings)
    assert loaded.dataset_id == result.dataset_id and loaded.reused
    assert loaded.page_count == 3 and loaded.chunk_count == 3
    assert len(embeddings.document_batches) == batches
    assert embeddings.query_calls == []
    loaded.store.similarity_search("dental", k=1)
    assert embeddings.query_calls == ["dental"]


def test_missing_folder_is_not_created(tmp_path):
    folder = tmp_path / "missing"
    config = ProviderConfig("ollama", "nomic-embed-text", "http://localhost:11434")
    assert find_saved_index(folder, config) is None
    assert not folder.exists()


def test_embedding_model_mismatch_is_not_loaded(saved):
    root, _, _, embeddings = saved
    mismatch = ProviderConfig("ollama", "other-model", "http://localhost:11434")
    assert find_saved_index(root, mismatch) is None
    assert embeddings.query_calls == []


def test_explicit_missing_selection_does_not_fall_back_to_different_documents(saved):
    root, config, _, _ = saved
    assert find_saved_index(root, config, "f" * 64) is None


def test_explicit_incompatible_selection_does_not_fall_back_to_another_index(saved, pdf_bytes):
    root, config, first, embeddings = saved
    other_config = ProviderConfig("ollama", "other-model", "http://localhost:11434")
    second = build_index(
        [PdfInput("hospital.pdf", pdf_bytes[0]), PdfInput("extras.pdf", pdf_bytes[1])],
        embeddings,
        embedding_identity(other_config),
        root,
    )
    assert find_saved_index(root, config).id == first.dataset_id
    assert find_saved_index(root, config, second.dataset_id) is None


def test_preferred_index_wins_over_newer_compatible_index(saved, pdf_bytes):
    root, config, first, embeddings = saved
    second = build_index(
        [PdfInput("hospital.pdf", pdf_bytes[0]), PdfInput("extras.pdf", pdf_bytes[1])],
        embeddings,
        embedding_identity(config),
        root,
        chunk_size=500,
        chunk_overlap=100,
    )
    first_path = root / first.dataset_id / "manifest.json"
    second_path = root / second.dataset_id / "manifest.json"
    import os

    os.utime(first_path, ns=(1_000_000, 1_000_000))
    os.utime(second_path, ns=(2_000_000, 2_000_000))
    assert find_saved_index(root, config).id == second.dataset_id
    assert find_saved_index(root, config, first.dataset_id).id == first.dataset_id


def test_missing_passage_is_rejected_without_repair(saved):
    root, _, result, embeddings = saved
    index = list_indexes(root)[0]
    result.store.delete(ids=[result.store.get(include=[])["ids"][0]])
    batches = len(embeddings.document_batches)
    with pytest.raises(InspectionError, match="incomplete"):
        load_saved_index(index, embeddings)
    assert len(embeddings.document_batches) == batches
    assert len(result.store.get(include=[])["ids"]) == 2


def test_changed_chunk_inventory_is_rejected(saved):
    root, _, result, embeddings = saved
    index = list_indexes(root)[0]
    identifier = result.store.get(include=[])["ids"][0]
    result.store._collection.update(ids=[identifier], metadatas=[{"chunk_index": 999}])
    with pytest.raises(InspectionError, match="inventory"):
        load_saved_index(index, embeddings)


def test_invalid_database_does_not_autoload(saved):
    root, config, _, _ = saved
    # An unrelated/incomplete directory cannot become a ready chat index.
    missing = root / ("f" * 64)
    missing.mkdir()
    (missing / "manifest.json").write_text("{}", encoding="utf-8")
    indexes = list_indexes(root)
    assert len(indexes) == 1
    assert find_saved_index(root, config).id == indexes[0].id
