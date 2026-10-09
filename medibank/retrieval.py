"""Expand vector hits into complete physical PDF pages for nearby exceptions."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document


def _number(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _page_key(metadata: dict) -> tuple[str, int] | None:
    source = metadata.get("source")
    page = _number(metadata.get("page_number", metadata.get("page")))
    if not isinstance(source, str) or not source or page is None or page < 1:
        return None
    return source, page


def _join_chunks(chunks: list[Document]) -> str:
    """Use source offsets to remove splitter overlap while retaining paragraph gaps."""
    chunks.sort(
        key=lambda chunk: (
            _number(chunk.metadata.get("start_index"))
            if _number(chunk.metadata.get("start_index")) is not None
            else float("inf"),
            _number(chunk.metadata.get("chunk_index")) or 0,
        )
    )
    combined = ""
    covered_end: int | None = None
    for chunk in chunks:
        text = chunk.page_content
        if not text:
            continue
        start = _number(chunk.metadata.get("start_index"))
        if start is not None and start < 0:
            start = None
        if not combined:
            combined = text
        elif start is not None and covered_end is not None:
            overlap = max(0, covered_end - start)
            combined += text[overlap:] if overlap else "\n\n" + text
        else:
            # Older indexes may lack offsets. Only remove a substantial exact
            # suffix/prefix match; a coincidental shared letter is not overlap.
            overlap = 0
            for length in range(min(len(combined), len(text)), 19, -1):
                if combined.endswith(text[:length]):
                    overlap = length
                    break
            combined += text[overlap:] if overlap else "\n\n" + text
        if start is not None:
            covered_end = max(covered_end or 0, start + len(text))
        else:
            covered_end = None
        if len(combined) >= 6000:
            return combined[:6000]
    return combined


def expand_page_context(store: Any, documents: list[Document]) -> list[Document]:
    """Fetch all chunks on each hit's physical page from the same Chroma store.

    A missing get API or failed page lookup preserves the original vector hits.
    Returned rows are checked again against both source and page before merging.
    """
    if not callable(getattr(store, "get", None)):
        return documents
    pages: dict[tuple[str, int], Document | None] = {}
    expanded: list[Document] = []
    emitted: set[tuple[str, int]] = set()
    for document in documents:
        key = _page_key(document.metadata)
        if key is None:
            expanded.append(document)
            continue
        if key not in pages:
            source, page = key
            try:
                result = store.get(
                    where={"$and": [{"source": source}, {"page_number": page}]},
                    include=["documents", "metadatas"],
                )
                chunks = [
                    Document(page_content=text, metadata=metadata)
                    for text, metadata in zip(
                        result.get("documents") or [],
                        result.get("metadatas") or [],
                    )
                    if isinstance(text, str)
                    and isinstance(metadata, dict)
                    and _page_key(metadata) == key
                ]
                text = _join_chunks(chunks)
                pages[key] = (
                    Document(
                        page_content=text,
                        metadata={
                            "source": source,
                            "page": page,
                            "page_number": page,
                            "start_index": 0,
                            "page_context": True,
                        },
                    )
                    if text
                    else None
                )
            except Exception:
                pages[key] = None
        context = pages[key]
        if context is None:
            expanded.append(document)
        elif key not in emitted:
            expanded.append(context)
            emitted.add(key)
    return expanded
