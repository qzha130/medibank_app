"""Build or reuse this project's persistent PDF knowledge index."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

from medibank.config import ProviderConfig, embedding_identity  # noqa: E402
from medibank.knowledge import build_index, discover_pdfs  # noqa: E402
from medibank.models import create_embeddings  # noqa: E402
from medibank.observability import audit  # noqa: E402


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv: list[str] | None = None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-directory", default=os.getenv("PDF_DIRECTORY", "medibank_data"))
    parser.add_argument("--chroma-directory", default=os.getenv("CHROMA_DIRECTORY", "data/chroma"))
    parser.add_argument(
        "--embedding-provider",
        choices=["ollama", "openai", "gemini"],
        default=os.getenv("EMBEDDING_PROVIDER", "ollama"),
    )
    parser.add_argument(
        "--embedding-model", help="Override the selected provider's embedding model."
    )
    parser.add_argument("--embedding-url", help="Override the selected provider's API base URL.")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args(argv)
    defaults = {
        "ollama": (
            "OLLAMA_EMBEDDING_MODEL",
            "nomic-embed-text",
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        ),
        "openai": (
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
            "OPENAI_BASE_URL",
            "https://api.openai.com/v1",
        ),
        "gemini": (
            "GEMINI_EMBEDDING_MODEL",
            "gemini-embedding-001",
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com",
        ),
    }
    try:
        model_env, model_default, url_env, url_default = defaults[args.embedding_provider]
        api_key = {
            "ollama": "",
            "openai": os.getenv("OPENAI_API_KEY", ""),
            "gemini": os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""),
        }[args.embedding_provider]
        if args.embedding_provider != "ollama" and not api_key.strip():
            raise ValueError("Set the selected provider's API key in .env or your environment.")
        config = ProviderConfig(
            args.embedding_provider,
            (args.embedding_model or os.getenv(model_env, model_default)).strip(),
            (args.embedding_url or os.getenv(url_env, url_default)).strip(),
            api_key.strip(),
        )
        audit("index_load_started", provider=config.provider, model=config.model, mode="CLI")
        directory = project_path(args.chroma_directory).resolve()
        index = build_index(
            pdfs=discover_pdfs(project_path(args.pdf_directory)),
            embeddings=create_embeddings(config),
            embedding_identity=embedding_identity(config),
            persist_directory=directory,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except (ValueError, KeyError) as error:
        audit("index_failed", level="ERROR", mode="CLI", error_type=type(error).__name__)
        message = (
            str(error) if isinstance(error, ValueError) else "Choose a valid embedding provider."
        )
        print(message, file=sys.stderr)
        return 2
    except Exception:
        audit("index_failed", level="ERROR", mode="CLI", error_type="service_or_storage_error")
        print(
            "The index could not be built or loaded. Check the PDFs, embedding service/model, "
            "API credentials, and Chroma directory permissions.",
            file=sys.stderr,
        )
        return 2
    print("Index reused." if index.reused else "Index built.")
    audit(
        "index_loaded" if index.reused else "index_built",
        mode="CLI",
        dataset_id=index.dataset_id,
        pages=index.page_count,
        passages=index.chunk_count,
    )
    for file in index.files:
        print(f"  {file['name']}: {file['page_count']} pages, {file['chunk_count']} passages")
    print(f"Total: {len(index.files)} PDFs, {index.page_count} pages, {index.chunk_count} passages")
    print(f"Skipped pages without text: {index.skipped_pages}")
    print(f"Persistent index: {directory / index.dataset_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
