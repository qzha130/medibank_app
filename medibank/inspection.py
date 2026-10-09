"""Browse existing Chroma indexes and run compatible searches without mutations."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from langchain_core.embeddings import Embeddings

from medibank.config import ProviderConfig, embedding_identity
from medibank.knowledge import COLLECTION_NAME, INDEX_VERSION

MAX_PAGE_SIZE = 200
_HASH = re.compile(r"[0-9a-f]{64}\Z")


class InspectionError(ValueError):
    """A saved index or inspection request is unavailable or invalid."""


@dataclass(frozen=True)
class SavedIndex:
    id: str
    path: Path
    manifest: dict[str, Any]


def _integer(value: Any, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _manifest_at(root: Path, path: Path) -> SavedIndex:
    """Validate the path and completion marker without creating any files."""
    root = root.expanduser().resolve()
    path = path.absolute()
    resolved = path.resolve()
    if (
        not _HASH.fullmatch(path.name)
        or path.parent.resolve() != root
        or resolved.parent != root
        or resolved.name != path.name
        or not resolved.is_dir()
    ):
        raise InspectionError(
            "The saved index must be a dataset directory inside the Chroma folder."
        )
    manifest_path = resolved / "manifest.json"
    if manifest_path.resolve().parent != resolved:
        raise InspectionError("The index manifest points outside its dataset directory.")
    try:
        if manifest_path.stat().st_size > 1024 * 1024:
            raise InspectionError("The saved index manifest is too large to inspect.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InspectionError("The saved index has no readable completion manifest.") from exc
    if not isinstance(manifest, dict) or manifest.get("complete") is not True:
        raise InspectionError("This index is incomplete. Build or load it in the chatbot first.")
    specification = manifest.get("specification")
    if (
        not isinstance(specification, dict)
        or not _integer(specification.get("version"), 1)
        or specification["version"] != INDEX_VERSION
    ):
        raise InspectionError("The saved index uses an unsupported manifest version.")
    canonical = json.dumps(specification, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    expected_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if manifest.get("dataset_id") != path.name or expected_id != path.name:
        raise InspectionError("The manifest identity does not match its dataset directory.")
    identity = specification.get("embedding_identity")
    size, overlap = specification.get("chunk_size"), specification.get("chunk_overlap")
    if (
        not isinstance(identity, str)
        or not identity.strip()
        or not _integer(size, 1)
        or not _integer(overlap)
        or overlap >= size
    ):
        raise InspectionError("The saved embedding or chunk settings are invalid.")
    spec_files, files = specification.get("files"), manifest.get("files")
    if (
        not isinstance(spec_files, list)
        or len(spec_files) != 2
        or not isinstance(files, list)
        or len(files) != 2
    ):
        raise InspectionError("The saved index must identify the two source PDFs.")
    sources: dict[str, str] = {}
    for file in spec_files:
        if not isinstance(file, dict):
            raise InspectionError("The source PDF manifest is invalid.")
        name, digest = file.get("name"), file.get("sha256")
        if (
            not isinstance(name, str)
            or not name
            or "/" in name
            or "\\" in name
            or not name.lower().endswith(".pdf")
            or not isinstance(digest, str)
            or not _HASH.fullmatch(digest)
            or name.casefold() in {source.casefold() for source in sources}
        ):
            raise InspectionError("The source PDF identity is invalid.")
        sources[name] = digest
    seen: set[str] = set()
    for file in files:
        if not isinstance(file, dict):
            raise InspectionError("The source PDF statistics are invalid.")
        name = file.get("name")
        if name not in sources or name in seen or file.get("sha256") != sources[name]:
            raise InspectionError("The source PDF statistics do not match the index specification.")
        seen.add(name)
        if not all(
            _integer(file.get(key), minimum)
            for key, minimum in (
                ("page_count", 1),
                ("text_pages", 1),
                ("skipped_pages", 0),
                ("chunk_count", 1),
                ("size_bytes", 1),
            )
        ):
            raise InspectionError("The source PDF statistics contain invalid counts.")
        if file["text_pages"] + file["skipped_pages"] != file["page_count"]:
            raise InspectionError("The source PDF page counts are inconsistent.")
    for field in ("page_count", "chunk_count", "skipped_pages"):
        if not _integer(manifest.get(field)) or manifest[field] != sum(
            file[field] for file in files
        ):
            raise InspectionError("The saved index totals are inconsistent.")
    if not isinstance(manifest.get("chunk_ids_sha256"), str) or not _HASH.fullmatch(
        manifest["chunk_ids_sha256"]
    ):
        raise InspectionError("The saved index has no valid chunk inventory identity.")
    return SavedIndex(path.name, resolved, manifest)


def _database_at(index: SavedIndex) -> Path:
    """Check SQLite read-only before invoking a client that could initialize a DB."""
    database = index.path / "chroma.sqlite3"
    if database.resolve().parent != index.path or not database.is_file():
        raise InspectionError("The saved Chroma database is missing or points outside its dataset.")
    try:
        with database.open("rb") as handle:
            if handle.read(16) != b"SQLite format 3\x00":
                raise InspectionError("The saved Chroma database is damaged or is not SQLite.")
        connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
        try:
            if connection.execute("PRAGMA quick_check(1)").fetchone() != ("ok",):
                raise InspectionError("The saved Chroma database failed its integrity check.")
            required = {
                "collections",
                "databases",
                "tenants",
                "segments",
                "embeddings",
                "migrations",
            }
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if not required.issubset(tables):
                raise InspectionError("The saved database does not contain the Chroma schema.")
            row = connection.execute(
                "SELECT id FROM collections WHERE name = ?", (COLLECTION_NAME,)
            ).fetchone()
            if row is None:
                raise InspectionError("The knowledge collection is missing from this database.")
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        raise InspectionError(
            "The saved Chroma database could not be read. Rebuild this index first."
        ) from exc
    return database


def list_indexes(root: Path) -> list[SavedIndex]:
    """List complete supported indexes; missing or broken folders are skipped."""
    root = Path(root).expanduser().resolve()
    if not root.exists():
        return []
    if not root.is_dir():
        raise InspectionError("The Chroma folder must be a directory.")
    result: list[SavedIndex] = []
    try:
        paths = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise InspectionError("The Chroma folder could not be read.") from exc
    for path in paths:
        if not _HASH.fullmatch(path.name):
            continue
        try:
            index = _manifest_at(root, path)
            _database_at(index)
        except (InspectionError, OSError):
            continue
        result.append(index)
    return result


def open_collection(index: SavedIndex) -> Collection:
    """Open only an existing validated collection, with embedding disabled."""
    validated = _manifest_at(index.path.parent, index.path)
    if index.id != validated.id or index.manifest != validated.manifest:
        raise InspectionError("The saved index changed. Refresh the index list before continuing.")
    _database_at(validated)
    try:
        client = chromadb.PersistentClient(
            path=str(validated.path), settings=Settings(anonymized_telemetry=False)
        )
        return client.get_collection(name=COLLECTION_NAME, embedding_function=None)
    except Exception as exc:
        raise InspectionError(
            "The saved knowledge collection could not be opened. Rebuild the index first."
        ) from exc


def make_filter(source: str | None = None, page: int | None = None) -> dict[str, Any] | None:
    if source is not None and (not isinstance(source, str) or not source):
        raise InspectionError("Choose a valid PDF source name.")
    if page is not None and not _integer(page, 1):
        raise InspectionError("Page numbers must be positive integers.")
    filters = []
    if source is not None:
        filters.append({"source": {"$eq": source}})
    if page is not None:
        filters.append({"page": {"$eq": page}})
    if not filters:
        return None
    return filters[0] if len(filters) == 1 else {"$and": filters}


def _limit(value: int, maximum: int = MAX_PAGE_SIZE) -> None:
    if not _integer(value, 1) or value > maximum:
        raise InspectionError(f"Choose a result limit between 1 and {maximum}.")


def count_records(
    collection: Collection, source: str | None = None, page: int | None = None
) -> int:
    """Count matching records without loading documents or vectors."""
    where = make_filter(source, page)
    try:
        if where is None:
            return int(collection.count())
        total = 0
        while True:
            records = collection.get(where=where, limit=MAX_PAGE_SIZE, offset=total, include=[])
            size = len(records["ids"])
            total += size
            if size < MAX_PAGE_SIZE:
                return total
    except Exception as exc:
        raise InspectionError("The collection count could not be read.") from exc


def _record(identifier: str, document: Any, metadata: Any) -> dict[str, Any]:
    if not isinstance(identifier, str) or not identifier or not isinstance(document, str):
        raise InspectionError("The collection contains a record without valid text or an ID.")
    if (
        not isinstance(metadata, dict)
        or not isinstance(metadata.get("source"), str)
        or not metadata["source"]
        or not _integer(metadata.get("page"), 1)
        or ("page_number" in metadata and metadata["page_number"] != metadata["page"])
        or ("chunk_id" in metadata and metadata["chunk_id"] != identifier)
    ):
        raise InspectionError(
            "A stored passage has invalid source or page metadata. Rebuild the index."
        )
    return {"id": identifier, "document": document, "metadata": dict(metadata)}


def browse_records(
    collection: Collection,
    limit: int = 25,
    offset: int = 0,
    source: str | None = None,
    page: int | None = None,
) -> list[dict[str, Any]]:
    """Read one bounded page of passages. Embeddings are deliberately excluded."""
    _limit(limit)
    if not _integer(offset):
        raise InspectionError("The page offset must be zero or greater.")
    where = make_filter(source, page)
    try:
        result = collection.get(
            limit=limit, offset=offset, where=where, include=["documents", "metadatas"]
        )
        return [
            _record(identifier, document, metadata)
            for identifier, document, metadata in zip(
                result["ids"],
                result["documents"],
                result["metadatas"],
                strict=True,
            )
        ]
    except InspectionError:
        raise
    except Exception as exc:
        raise InspectionError("The stored passages could not be read.") from exc


def get_record(
    collection: Collection, identifier: str, include_vector: bool = False
) -> dict[str, Any] | None:
    """Read a selected passage; optionally fetch only that passage's vector."""
    if not isinstance(identifier, str) or not identifier:
        raise InspectionError("Choose a stored passage ID.")
    include = ["documents", "metadatas"]
    if include_vector:
        include.append("embeddings")
    try:
        result = collection.get(ids=[identifier], include=include)
        if not result["ids"]:
            return None
        record = _record(result["ids"][0], result["documents"][0], result["metadatas"][0])
        if include_vector:
            vectors = result.get("embeddings")
            if vectors is None or len(vectors) != 1:
                raise InspectionError("The selected passage has no readable embedding vector.")
            vector = [float(value) for value in vectors[0]]
            if not vector or not all(math.isfinite(value) for value in vector):
                raise InspectionError("The selected embedding vector contains invalid values.")
            record.update({"embedding": vector, "dimension": len(vector)})
        return record
    except InspectionError:
        raise
    except Exception as exc:
        raise InspectionError("The selected passage could not be read.") from exc


def semantic_search(
    index: SavedIndex,
    collection: Collection,
    query: str,
    config: ProviderConfig,
    embeddings: Embeddings,
    limit: int = 5,
    source: str | None = None,
    page: int | None = None,
) -> list[dict[str, Any]]:
    """Search using the exact vector-space identity recorded during indexing."""
    if embedding_identity(config) != index.manifest["specification"]["embedding_identity"]:
        raise InspectionError(
            "These embedding settings do not match the saved index. Select the original provider, model, and API URL."
        )
    if not isinstance(query, str) or not query.strip():
        raise InspectionError("Enter a search question or phrase.")
    _limit(limit, maximum=100)
    where = make_filter(source, page)
    matched_count = count_records(collection, source, page)
    if matched_count == 0:
        return []
    try:
        vector = embeddings.embed_query(query.strip())
        if not vector or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            for value in vector
        ):
            raise InspectionError("The embedding model returned an invalid query vector.")
        result = collection.query(
            query_embeddings=[vector],
            n_results=min(limit, matched_count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        records = []
        files = {file["name"]: file for file in index.manifest["files"]}
        for identifier, document, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        ):
            record = _record(identifier, document, metadata)
            file = files.get(metadata["source"])
            if (
                file is None
                or metadata["page"] > file["page_count"]
                or metadata.get("file_hash") != file["sha256"]
            ):
                raise InspectionError("A search result does not match this index's source PDFs.")
            if not isinstance(distance, (int, float)) or not math.isfinite(distance):
                raise InspectionError("A search result has an invalid vector distance.")
            record["distance"] = float(distance)
            records.append(record)
        return records
    except InspectionError:
        raise
    except Exception as exc:
        raise InspectionError(
            "Semantic search failed. Check the original embedding model, service or API key, and vector dimensions."
        ) from exc
