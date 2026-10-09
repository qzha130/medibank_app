"""Offline fixtures: real searchable PDFs and deterministic LangChain doubles."""

from __future__ import annotations

import hashlib
import io
import math
import re
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


@pytest.fixture(autouse=True)
def isolate_operation_logs(monkeypatch, tmp_path):
    monkeypatch.setenv("LOG_DIRECTORY", str(tmp_path / "operation-logs"))
    monkeypatch.setenv("HANDOFF_DIRECTORY", str(tmp_path / "review-requests"))
    yield
    from medibank.observability import _LOGGERS

    for logger in _LOGGERS.values():
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
    _LOGGERS.clear()


def searchable_pdf(*pages: str) -> bytes:
    """Create actual PDF text streams without reportlab or downloaded fixtures."""
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


class CountingEmbeddings(Embeddings):
    """A normalized word-hash embedding; never calls an external model."""

    def __init__(self) -> None:
        self.document_batches: list[list[str]] = []
        self.query_calls: list[str] = []
        self.fail_next_batch = False

    @staticmethod
    def vector(text: str) -> list[float]:
        vector = [0.0] * 64
        for token in re.findall(r"[a-z]+", text.lower()):
            slot = hashlib.sha256(token.encode()).digest()[0] % len(vector)
            vector[slot] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_batches.append(list(texts))
        if self.fail_next_batch:
            self.fail_next_batch = False
            raise RuntimeError("Simulated embedding service outage")
        return [self.vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self.vector(text)


class ScriptedChatModel(BaseChatModel):
    """Tool-capable model double that exercises the real LangChain agent loop."""

    responses: list[AIMessage] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "offline-scripted-chat"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        index = len(self.calls) - 1
        if index >= len(self.responses):
            raise AssertionError("Unexpected additional model call")
        return ChatResult(generations=[ChatGeneration(message=self.responses[index])])


@pytest.fixture
def embeddings() -> CountingEmbeddings:
    return CountingEmbeddings()


@pytest.fixture
def pdf_bytes() -> tuple[bytes, bytes]:
    return (
        searchable_pdf(
            "Hospital waiting periods: twelve months for pre-existing conditions.",
            "Ambulance transport is included when medically necessary.",
        ),
        searchable_pdf("Dental benefits: two check-ups each year under this cover."),
    )
