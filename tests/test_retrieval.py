"""Page evidence retains nearby exception paragraphs beyond a vector hit."""

from __future__ import annotations

from conftest import searchable_pdf
from langchain_core.documents import Document

from medibank.knowledge import PdfInput, build_index
from medibank.retrieval import expand_page_context


def chunk(text: str, source="guide.pdf", page=20, start=0, index=0) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "page": page,
            "page_number": page,
            "start_index": start,
            "chunk_index": index,
        },
    )


class PageStore:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.calls = []

    def get(self, **kwargs):
        self.calls.append(kwargs)
        # Deliberately return other sources/pages too; the helper must check them.
        return {
            "documents": [item.page_content for item in self.documents],
            "metadatas": [item.metadata for item in self.documents],
        }


def test_reconstructs_overlap_and_tail_without_mixing_source_or_page():
    page = (
        "A twelve-month waiting period applies for a pre-existing condition. "
        "An emergency does not itself remove this waiting period."
    )
    first = chunk(page[:80])
    tail = chunk(page[60:], start=60, index=1)
    store = PageStore(
        [
            tail,
            chunk("Another PDF's unrelated waiver.", source="rules.pdf"),
            chunk("Another page's unrelated waiver.", page=21),
            first,
        ]
    )
    expanded = expand_page_context(store, [first, tail])
    assert len(expanded) == 1
    assert expanded[0].page_content == page
    assert expanded[0].page_content.count("waiting period applies") == 1
    assert "emergency does not" not in first.page_content
    assert "emergency does not" in expanded[0].page_content
    assert expanded[0].metadata["source"] == "guide.pdf"
    assert expanded[0].metadata["page_number"] == 20
    assert store.calls == [
        {
            "where": {"$and": [{"source": "guide.pdf"}, {"page_number": 20}]},
            "include": ["documents", "metadatas"],
        }
    ]


def test_missing_get_and_failed_page_lookup_preserve_all_original_chunks():
    hits = [chunk("First chunk."), chunk("Second chunk.", start=20, index=1)]
    assert expand_page_context(object(), hits) is hits

    class FailedStore:
        def get(self, **kwargs):
            raise RuntimeError("Unavailable index")

    assert expand_page_context(FailedStore(), hits) == hits
    assert expand_page_context(PageStore([]), hits) == hits


def test_page_context_is_limited_to_6000_characters():
    hits = [chunk("a" * 5000)]
    store = PageStore([hits[0], chunk("a" * 200 + "b" * 2000, start=4800, index=1)])
    text = expand_page_context(store, hits)[0].page_content
    assert text == "a" * 5000 + "b" * 1000


def test_real_chroma_page_lookup_recovers_an_unretrieved_emergency_paragraph(embeddings, tmp_path):
    page_text = (
        "A pre-existing condition has a twelve-month waiting period. "
        + "Membership administration information is provided here. " * 20
        + "An emergency does not itself remove this waiting period."
    )
    index = build_index(
        pdfs=[
            PdfInput("guide.pdf", searchable_pdf(page_text)),
            PdfInput("rules.pdf", searchable_pdf("An unrelated definition in another document.")),
        ],
        embeddings=embeddings,
        embedding_identity="offline-page-expansion",
        persist_directory=tmp_path / "chroma",
        chunk_size=200,
        chunk_overlap=60,
    )
    rows = index.store.get(
        where={"$and": [{"source": "guide.pdf"}, {"page_number": 1}]},
        include=["documents", "metadatas"],
    )
    hits = sorted(
        [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(rows["documents"], rows["metadatas"])
        ],
        key=lambda document: document.metadata["start_index"],
    )
    assert "An emergency" not in hits[0].page_content
    expanded = expand_page_context(index.store, [hits[0]])
    assert len(expanded) == 1
    assert "An emergency does not itself remove this waiting period." in expanded[0].page_content
    assert "unrelated definition" not in expanded[0].page_content
    assert expanded[0].metadata["source"] == "guide.pdf"
    assert expanded[0].metadata["page_number"] == 1
