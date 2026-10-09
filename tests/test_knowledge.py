"""Real PDF extraction and Chroma persistence, with no model service."""

from __future__ import annotations

import io

import pytest
from conftest import CountingEmbeddings, searchable_pdf
from pypdf import PdfWriter

from medibank.knowledge import PdfInput, build_index, discover_pdfs


def inputs(pdf_bytes: tuple[bytes, bytes]) -> list[PdfInput]:
    return [PdfInput("hospital.pdf", pdf_bytes[0]), PdfInput("extras.pdf", pdf_bytes[1])]


def build(documents, embeddings, root, **kwargs):
    return build_index(
        documents,
        embeddings=embeddings,
        embedding_identity="offline:test-v1",
        persist_directory=root,
        **kwargs,
    )


@pytest.mark.parametrize("count", [0, 1, 3])
def test_requires_exactly_two_pdfs(count, pdf_bytes, embeddings, tmp_path):
    documents = [PdfInput(f"document-{index}.pdf", pdf_bytes[index % 2]) for index in range(count)]
    with pytest.raises(ValueError, match="two|2"):
        build(documents, embeddings, tmp_path)
    assert embeddings.document_batches == []


def test_extracts_real_text_and_one_based_page_metadata(pdf_bytes, embeddings, tmp_path):
    result = build(inputs(pdf_bytes), embeddings, tmp_path)
    assert result.page_count == 3
    assert result.chunk_count == 3
    assert result.skipped_pages == 0
    assert result.reused is False
    stored = result.store.get()
    assert len(stored["ids"]) == 3
    pages = {(item["source"], item["page"]) for item in stored["metadatas"]}
    assert pages == {("hospital.pdf", 1), ("hospital.pdf", 2), ("extras.pdf", 1)}
    assert all(item["file_hash"] for item in stored["metadatas"])
    assert any("Dental benefits" in text for text in stored["documents"])
    retrieved = result.store.similarity_search("dental check-ups benefits", k=1)
    assert retrieved[0].metadata["source"] == "extras.pdf"


def test_reuses_existing_index_even_when_input_order_changes(pdf_bytes, embeddings, tmp_path):
    documents = inputs(pdf_bytes)
    first = build(documents, embeddings, tmp_path)
    batch_count = len(embeddings.document_batches)
    second = build(list(reversed(documents)), embeddings, tmp_path)
    assert second.reused is True
    assert second.dataset_id == first.dataset_id
    assert len(embeddings.document_batches) == batch_count
    assert set(second.store.get()["ids"]) == set(first.store.get()["ids"])


def test_changed_documents_do_not_mix_with_previous_index(pdf_bytes, embeddings, tmp_path):
    first = build(inputs(pdf_bytes), embeddings, tmp_path)
    updated = [
        PdfInput("hospital.pdf", searchable_pdf("Replacement maternity benefit.")),
        inputs(pdf_bytes)[1],
    ]
    second = build(updated, embeddings, tmp_path)
    assert second.dataset_id != first.dataset_id
    assert second.reused is False
    new_text = " ".join(second.store.get()["documents"])
    assert "Replacement maternity" in new_text
    assert "Hospital waiting periods" not in new_text
    assert "Hospital waiting periods" in " ".join(first.store.get()["documents"])


def test_embedding_identity_and_chunk_settings_change_index(pdf_bytes, embeddings, tmp_path):
    documents = inputs(pdf_bytes)
    first = build(documents, embeddings, tmp_path)
    different_model = build_index(
        documents,
        embeddings=embeddings,
        embedding_identity="offline:different-model",
        persist_directory=tmp_path,
    )
    different_chunks = build(documents, embeddings, tmp_path, chunk_size=50, chunk_overlap=10)
    assert len({first.dataset_id, different_model.dataset_id, different_chunks.dataset_id}) == 3
    assert different_model.reused is False
    assert different_chunks.reused is False
    assert different_chunks.chunk_count > first.chunk_count


def test_missing_chunk_is_repaired_instead_of_reusing_incomplete_index(
    pdf_bytes, embeddings, tmp_path
):
    first = build(inputs(pdf_bytes), embeddings, tmp_path)
    expected = set(first.store.get()["ids"])
    first.store.delete(ids=[next(iter(expected))])
    repaired = build(inputs(pdf_bytes), embeddings, tmp_path)
    assert repaired.reused is False
    assert set(repaired.store.get()["ids"]) == expected
    assert repaired.chunk_count == len(expected)


def test_extraneous_chunk_is_removed_during_repair(pdf_bytes, embeddings, tmp_path):
    first = build(inputs(pdf_bytes), embeddings, tmp_path)
    expected = set(first.store.get()["ids"])
    first.store.add_texts(
        ["Stale unrelated document text"],
        metadatas=[{"source": "stale.pdf", "page": 1}],
        ids=["stale-id"],
    )
    repaired = build(inputs(pdf_bytes), embeddings, tmp_path)
    assert repaired.reused is False
    assert set(repaired.store.get()["ids"]) == expected


def test_corrupted_completion_manifest_forces_rebuild(pdf_bytes, embeddings, tmp_path):
    first = build(inputs(pdf_bytes), embeddings, tmp_path)
    manifest = tmp_path / first.dataset_id / "manifest.json"
    manifest.write_text("{interrupted write", encoding="utf-8")
    batches_before_repair = len(embeddings.document_batches)
    repaired = build(inputs(pdf_bytes), embeddings, tmp_path)
    assert repaired.reused is False
    assert len(embeddings.document_batches) > batches_before_repair
    assert len(repaired.store.get()["ids"]) == repaired.chunk_count
    assert build(inputs(pdf_bytes), embeddings, tmp_path).reused is True


def test_failed_embedding_build_can_be_retried(pdf_bytes, tmp_path):
    embeddings = CountingEmbeddings()
    embeddings.fail_next_batch = True
    with pytest.raises(RuntimeError, match="index could not be completed") as failure:
        build(inputs(pdf_bytes), embeddings, tmp_path)
    assert "Simulated embedding" in str(failure.value.__cause__)
    assert not list(tmp_path.glob("*/manifest.json"))
    recovered = build(inputs(pdf_bytes), embeddings, tmp_path)
    assert recovered.reused is False
    assert len(recovered.store.get()["ids"]) == recovered.chunk_count == 3
    assert build(inputs(pdf_bytes), embeddings, tmp_path).reused is True


def test_interrupted_multibatch_build_recovers_without_duplicate_chunks(
    pdf_bytes, tmp_path, monkeypatch
):
    embeddings = CountingEmbeddings()
    original_embed = embeddings.embed_documents
    calls = 0

    def fail_second_batch(texts):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Interrupted after the first persisted batch")
        return original_embed(texts)

    monkeypatch.setattr(embeddings, "embed_documents", fail_second_batch)
    long_text = " ".join(
        f"Coverage paragraph {index} describes hospital waiting periods and ambulance benefits under the policy."
        for index in range(200)
    )
    documents = [PdfInput("hospital.pdf", searchable_pdf(long_text)), inputs(pdf_bytes)[1]]
    with pytest.raises(RuntimeError, match="index could not be completed"):
        build(documents, embeddings, tmp_path, chunk_size=160, chunk_overlap=20)
    assert calls == 2
    assert len(embeddings.document_batches[0]) == 64
    assert not list(tmp_path.glob("*/manifest.json"))

    monkeypatch.setattr(embeddings, "embed_documents", original_embed)
    recovered = build(documents, embeddings, tmp_path, chunk_size=160, chunk_overlap=20)
    assert recovered.reused is False
    stored_ids = recovered.store.get()["ids"]
    assert len(set(stored_ids)) == len(stored_ids) == recovered.chunk_count
    assert recovered.chunk_count > 64
    assert build(documents, embeddings, tmp_path, chunk_size=160, chunk_overlap=20).reused is True


@pytest.mark.parametrize("duplicate_name", ["hospital.pdf", "HOSPITAL.PDF"])
def test_duplicate_names_are_rejected(duplicate_name, pdf_bytes, embeddings, tmp_path):
    with pytest.raises(ValueError, match="name|duplicate|unique"):
        build(
            [PdfInput("hospital.pdf", pdf_bytes[0]), PdfInput(duplicate_name, pdf_bytes[1])],
            embeddings,
            tmp_path,
        )
    assert embeddings.document_batches == []


def test_duplicate_content_is_rejected(pdf_bytes, embeddings, tmp_path):
    with pytest.raises(ValueError, match="same|duplicate|distinct|different"):
        build(
            [PdfInput("one.pdf", pdf_bytes[0]), PdfInput("two.pdf", pdf_bytes[0])],
            embeddings,
            tmp_path,
        )
    assert embeddings.document_batches == []


@pytest.mark.parametrize(
    "name,data",
    [("fake.pdf", b"not a PDF"), ("bad.pdf", b"%PDF-1.4\n"), ("wrong.txt", b"%PDF-1.4\n")],
)
def test_invalid_pdfs_fail_before_embedding(name, data, pdf_bytes, embeddings, tmp_path):
    with pytest.raises(ValueError):
        build([PdfInput(name, data), inputs(pdf_bytes)[1]], embeddings, tmp_path)
    assert embeddings.document_batches == []


def test_pdf_without_text_requires_ocr(pdf_bytes, embeddings, tmp_path):
    with pytest.raises(ValueError, match="text|OCR|scanned"):
        build(
            [PdfInput("scanned.pdf", searchable_pdf("")), inputs(pdf_bytes)[1]],
            embeddings,
            tmp_path,
        )
    assert embeddings.document_batches == []


def test_encrypted_pdf_is_rejected(pdf_bytes, embeddings, tmp_path):
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("test-password")
    output = io.BytesIO()
    writer.write(output)
    with pytest.raises(ValueError, match="encrypt|password"):
        build(
            [PdfInput("encrypted.pdf", output.getvalue()), inputs(pdf_bytes)[1]],
            embeddings,
            tmp_path,
        )
    assert embeddings.document_batches == []


def test_blank_pages_are_counted_but_not_indexed(pdf_bytes, embeddings, tmp_path):
    documents = [
        PdfInput("mixed.pdf", searchable_pdf("", "Text available on second page.")),
        inputs(pdf_bytes)[1],
    ]
    result = build(documents, embeddings, tmp_path)
    assert result.page_count == 3
    assert result.skipped_pages == 1
    assert result.chunk_count == 2
    metadata = result.store.get()["metadatas"]
    assert ("mixed.pdf", 2) in {(item["source"], item["page"]) for item in metadata}


def test_discovers_only_two_pdfs_directly_in_folder(pdf_bytes, tmp_path):
    (tmp_path / "Hospital.PDF").write_bytes(pdf_bytes[0])
    (tmp_path / "extras.pdf").write_bytes(pdf_bytes[1])
    (tmp_path / "notes.txt").write_text("Ignored non-PDF file", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "ignored.pdf").write_bytes(pdf_bytes[0])
    found = discover_pdfs(tmp_path)
    assert {document.name for document in found} == {"Hospital.PDF", "extras.pdf"}
    assert {document.data for document in found} == set(pdf_bytes)
    (tmp_path / "extra.pdf").write_bytes(searchable_pdf("Third document"))
    with pytest.raises(ValueError, match="exactly two"):
        discover_pdfs(tmp_path)


@pytest.mark.parametrize("chunk_size,chunk_overlap", [(0, 0), (100, -1), (100, 100), (True, 0)])
def test_invalid_chunk_settings_fail_before_model_calls(
    chunk_size, chunk_overlap, pdf_bytes, embeddings, tmp_path
):
    with pytest.raises(ValueError, match="Chunk"):
        build(
            inputs(pdf_bytes),
            embeddings,
            tmp_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    assert embeddings.document_batches == []
