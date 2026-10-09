"""PDF ingestion and isolated, restart-safe Chroma knowledge indexes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator, Sequence

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

MAX_PDF_BYTES = 50 * 1024 * 1024
INDEX_VERSION = 1
INDEX_BATCH_SIZE = 64
COLLECTION_NAME = "medibank_knowledge"


@dataclass(frozen=True)
class PdfInput:
    name: str
    data: bytes


@dataclass
class IndexResult:
    store: Chroma
    dataset_id: str
    files: list[dict[str, Any]]
    page_count: int
    chunk_count: int
    skipped_pages: int
    reused: bool


def _safe_name(name: str) -> str:
    """Keep filenames useful for citations without retaining uploaded paths."""
    if not isinstance(name, str):
        raise ValueError("Every PDF must have a filename.")
    basename = unicodedata.normalize("NFKC", name.replace("\\", "/").rsplit("/", 1)[-1])
    basename = re.sub(r'[<>:"/\\|?*\[\]\x00-\x1f\x7f]', "_", basename).strip(" .")
    if not basename or Path(basename).suffix.lower() != ".pdf":
        raise ValueError("Choose files with a .pdf filename extension.")
    return basename


def _validated_inputs(pdfs: Sequence[PdfInput]) -> list[tuple[PdfInput, str]]:
    if len(pdfs) != 2:
        raise ValueError(f"The knowledge base needs exactly two PDFs; received {len(pdfs)}.")
    validated: list[tuple[PdfInput, str]] = []
    names: set[str] = set()
    hashes: set[str] = set()
    for pdf in pdfs:
        name = _safe_name(pdf.name)
        if not isinstance(pdf.data, bytes) or not pdf.data:
            raise ValueError(f"{name} is empty or is not a readable PDF upload.")
        if len(pdf.data) > MAX_PDF_BYTES:
            raise ValueError(f"{name} exceeds the 50 MB per-PDF limit.")
        if b"%PDF-" not in pdf.data[:1024]:
            raise ValueError(f"{name} does not have a valid PDF header.")
        digest = hashlib.sha256(pdf.data).hexdigest()
        if name.casefold() in names:
            raise ValueError("The two PDFs must have distinct filenames for source citations.")
        if digest in hashes:
            raise ValueError("The two PDFs contain the same file. Choose two distinct PDFs.")
        names.add(name.casefold())
        hashes.add(digest)
        validated.append((PdfInput(name, pdf.data), digest))
    return sorted(validated, key=lambda item: (item[0].name.casefold(), item[0].name, item[1]))


def discover_pdfs(directory: Path) -> list[PdfInput]:
    """Read exactly two PDFs directly inside a local knowledge-base folder."""
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise ValueError(
            f"PDF folder not found: {directory}. Create it and add the two PDFs, "
            "or use the PDF uploader."
        )
    paths = sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if len(paths) != 2:
        raise ValueError(
            f"Expected exactly two PDFs in {directory}; found {len(paths)}. "
            "Keep the two knowledge-base PDFs in that folder."
        )
    result: list[PdfInput] = []
    for path in paths:
        try:
            if path.stat().st_size > MAX_PDF_BYTES:
                raise ValueError(f"{path.name} exceeds the 50 MB per-PDF limit.")
            # Read one byte beyond the limit to guard a file changing after stat().
            with path.open("rb") as handle:
                data = handle.read(MAX_PDF_BYTES + 1)
        except OSError as exc:
            raise ValueError(
                f"Cannot read {path.name}. Check that the PDF is available locally."
            ) from exc
        result.append(PdfInput(path.name, data))
    return [pdf for pdf, _ in _validated_inputs(result)]


def _extract_pages(
    inputs: list[tuple[PdfInput, str]],
) -> tuple[list[Document], list[dict[str, Any]]]:
    documents: list[Document] = []
    files: list[dict[str, Any]] = []
    for pdf, digest in inputs:
        try:
            reader = PdfReader(BytesIO(pdf.data))
        except Exception as exc:
            raise ValueError(
                f"{pdf.name} could not be opened as a PDF. Check that it is not damaged."
            ) from exc
        if reader.is_encrypted:
            raise ValueError(f"{pdf.name} is encrypted. Upload an unencrypted copy of the PDF.")
        try:
            page_count = len(reader.pages)
        except Exception as exc:
            raise ValueError(
                f"{pdf.name} could not be opened as a PDF. Check that it is not damaged."
            ) from exc

        text_pages = 0
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = (page.extract_text() or "").replace("\x00", "").strip()
            except Exception as exc:
                raise ValueError(
                    f"Text could not be read from {pdf.name}, page {page_number}. "
                    "Try a fresh PDF export or run OCR before uploading."
                ) from exc
            if not text:
                continue
            text_pages += 1
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf.name,
                        "page": page_number,
                        "page_number": page_number,
                        "file_hash": digest,
                    },
                )
            )
        if not text_pages:
            raise ValueError(
                f"{pdf.name} has no extractable text. If it is a scanned PDF, "
                "run OCR and upload the searchable PDF."
            )
        files.append(
            {
                "name": pdf.name,
                "sha256": digest,
                "size_bytes": len(pdf.data),
                "page_count": page_count,
                "text_pages": text_pages,
                "skipped_pages": page_count - text_pages,
                "chunk_count": 0,
            }
        )
    return documents, files


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Publish a completed manifest atomically, after all Chroma writes finish."""
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix="manifest-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


@contextmanager
def _index_lock(path: Path, timeout: float = 60.0) -> Iterator[None]:
    """Use an OS lock, which also releases automatically if a process crashes."""
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "This PDF index is being built in another session. Try again when it finishes."
                    ) from exc
                time.sleep(0.1)
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def build_index(
    pdfs: Sequence[PdfInput],
    embeddings: Embeddings,
    embedding_identity: str,
    persist_directory: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> IndexResult:
    """Create or reuse a complete index for exactly these PDFs and embeddings.

    Changing PDFs, their filenames, embeddings, or splitting settings selects a
    new directory. Failed builds have no completed manifest and are fully
    upserted on retry, so a partial index is never returned as ready.
    """
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("Chunk size must be a positive integer.")
    if (
        not isinstance(chunk_overlap, int)
        or isinstance(chunk_overlap, bool)
        or not 0 <= chunk_overlap < chunk_size
    ):
        raise ValueError("Chunk overlap must be zero or greater and smaller than chunk size.")
    if not isinstance(embedding_identity, str) or not embedding_identity.strip():
        raise ValueError(
            "An embedding model identity is required to keep vector indexes compatible."
        )
    inputs = _validated_inputs(pdfs)
    specification = {
        "version": INDEX_VERSION,
        "files": [{"name": pdf.name, "sha256": digest} for pdf, digest in inputs],
        "embedding_identity": embedding_identity,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    dataset_id = hashlib.sha256(_canonical_json(specification).encode("utf-8")).hexdigest()
    pages, files = _extract_pages(inputs)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(pages)
    ids: list[str] = []
    counts = {file["name"]: 0 for file in files}
    for index, chunk in enumerate(chunks):
        identifier = hashlib.sha256(
            _canonical_json(
                {
                    "dataset_id": dataset_id,
                    "index": index,
                    "source": chunk.metadata["source"],
                    "page": chunk.metadata["page"],
                    "text": chunk.page_content,
                }
            ).encode("utf-8")
        ).hexdigest()
        chunk.metadata.update({"chunk_id": identifier, "chunk_index": index})
        ids.append(identifier)
        counts[chunk.metadata["source"]] += 1
    for file in files:
        file["chunk_count"] = counts[file["name"]]
    if not chunks:
        raise ValueError(
            "The PDFs did not produce any searchable text chunks. Run OCR if they are scanned."
        )

    root = Path(persist_directory).expanduser().resolve()
    dataset_directory = (root / dataset_id).resolve()
    if not dataset_directory.is_relative_to(root):
        raise ValueError(
            "The dataset index directory must remain inside the configured Chroma folder."
        )
    dataset_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_directory / "manifest.json"
    manifest = {
        "complete": True,
        "dataset_id": dataset_id,
        "specification": specification,
        "files": files,
        "page_count": sum(file["page_count"] for file in files),
        "chunk_count": len(chunks),
        "skipped_pages": sum(file["skipped_pages"] for file in files),
        "chunk_ids_sha256": hashlib.sha256(_canonical_json(ids).encode("utf-8")).hexdigest(),
    }
    expected_ids = set(ids)
    with _index_lock(dataset_directory / "index.lock"):
        try:
            store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=str(dataset_directory),
                client_settings=Settings(anonymized_telemetry=False),
                collection_metadata={"hnsw:space": "cosine"},
            )
            existing_ids = set(store.get(include=[])["ids"])
            reused = _read_manifest(manifest_path) == manifest and existing_ids == expected_ids
            if not reused:
                # Removing only the completion marker ensures failures remain retryable.
                manifest_path.unlink(missing_ok=True)
                unexpected_ids = sorted(existing_ids - expected_ids)
                for start in range(0, len(unexpected_ids), INDEX_BATCH_SIZE):
                    store.delete(ids=unexpected_ids[start : start + INDEX_BATCH_SIZE])
                # Re-embed every expected chunk when the manifest is incomplete:
                # this also repairs entries written by an interrupted upsert.
                for start in range(0, len(chunks), INDEX_BATCH_SIZE):
                    store.add_documents(
                        documents=chunks[start : start + INDEX_BATCH_SIZE],
                        ids=ids[start : start + INDEX_BATCH_SIZE],
                    )
                if set(store.get(include=[])["ids"]) != expected_ids:
                    raise RuntimeError(
                        "Chroma did not persist the expected complete chunk inventory."
                    )
                _write_manifest(manifest_path, manifest)
        except Exception as exc:
            raise RuntimeError(
                "The PDF index could not be completed. Check the embedding service/model "
                "and that the Chroma folder is writable, then retry."
            ) from exc
    return IndexResult(
        store=store,
        dataset_id=dataset_id,
        files=files,
        page_count=manifest["page_count"],
        chunk_count=len(chunks),
        skipped_pages=manifest["skipped_pages"],
        reused=reused,
    )
