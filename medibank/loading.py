"""Load complete saved indexes for chat without reading PDFs or embedding them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from medibank.config import ProviderConfig, embedding_identity
from medibank.inspection import InspectionError, SavedIndex, list_indexes, open_collection
from medibank.knowledge import COLLECTION_NAME, IndexResult


def find_saved_index(
    directory: Path, config: ProviderConfig, preferred_id: str | None = None
) -> SavedIndex | None:
    """Load an explicit compatible selection, or the newest match by default."""
    identity = embedding_identity(config)
    matches = [
        index
        for index in list_indexes(directory)
        if index.manifest["specification"]["embedding_identity"] == identity
    ]
    if preferred_id:
        return next((index for index in matches if index.id == preferred_id), None)
    return max(
        matches,
        key=lambda index: ((index.path / "manifest.json").stat().st_mtime_ns, index.id),
        default=None,
    )


def load_saved_index(index: SavedIndex, embeddings: Embeddings) -> IndexResult:
    """Attach a query embedding client only after checking the persisted inventory."""
    collection = open_collection(index)
    records = collection.get(include=["metadatas"])
    identifiers = records["ids"]
    metadata = records["metadatas"] or []
    expected_count = index.manifest["chunk_count"]
    if len(identifiers) != expected_count or len(metadata) != expected_count:
        raise InspectionError(
            "The saved index is incomplete. Rebuild it on the Vector Database page."
        )
    ordered = {}
    for identifier, item in zip(identifiers, metadata):
        position = item.get("chunk_index") if isinstance(item, dict) else None
        if (
            not isinstance(position, int)
            or isinstance(position, bool)
            or not 0 <= position < expected_count
            or position in ordered
        ):
            raise InspectionError("The saved passage inventory is invalid. Rebuild this index.")
        ordered[position] = identifier
    canonical = json.dumps(
        [ordered[position] for position in range(expected_count)],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != index.manifest["chunk_ids_sha256"]:
        raise InspectionError("The saved passage inventory changed. Rebuild this index.")
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(index.path),
        client_settings=Settings(anonymized_telemetry=False),
        create_collection_if_not_exists=False,
    )
    manifest = index.manifest
    return IndexResult(
        store=store,
        dataset_id=index.id,
        files=manifest["files"],
        page_count=manifest["page_count"],
        chunk_count=manifest["chunk_count"],
        skipped_pages=manifest["skipped_pages"],
        reused=True,
    )
