"""Exercise the real LangChain agent with a scripted, offline chat model."""

from __future__ import annotations

import json
import time
from threading import get_ident

import pytest
from conftest import ScriptedChatModel
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGenerationChunk
from pydantic import Field

from medibank.chat import Answer, ChatError, answer_question, split_reasoning_text
from medibank.guardrails import NO_EVIDENCE_MESSAGE, RETRIEVAL_UNAVAILABLE_MESSAGE


class FakeStore:
    def __init__(self, *responses: list[Document]):
        self.responses = list(responses) or [[]]
        self.calls: list[tuple[str, int]] = []

    def similarity_search(self, query: str, k: int) -> list[Document]:
        self.calls.append((query, k))
        return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]


def passage(text="Dental check-ups are covered.", source="extras.pdf", page=1):
    return Document(page_content=text, metadata={"source": source, "page": page})


def tool_request(query, call_id="search-1"):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_knowledge_base",
                "args": {"query": query},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def test_retrieves_before_agent_and_returns_actual_page_evidence():
    store = FakeStore([passage(source="C:\\private\\extras.pdf", page=3)])
    model = ScriptedChatModel(
        responses=[AIMessage(content="Dental check-ups are covered. [extras.pdf, p. 3]")]
    )
    answer = answer_question("Are dental check-ups covered?", [], store, model, top_k=2)
    assert store.calls == [("Are dental check-ups covered?", 2)]
    assert answer.sources == [
        {"source": "extras.pdf", "page": 3, "excerpt": "Dental check-ups are covered."}
    ]
    message = model.calls[0][-1]
    assert isinstance(message, HumanMessage)
    payload = json.loads(message.content)
    assert payload["question"] == "Are dental check-ups covered?"
    assert payload["retrieved_passages"][0]["citation"] == "[extras.pdf, p. 3]"
    assert "private" not in str(payload)


def test_empty_retrieval_returns_no_evidence_without_using_model():
    store = FakeStore([])
    model = ScriptedChatModel()
    answer = answer_question(
        "What is covered?",
        [{"role": "assistant", "content": "Everything is covered."}],
        store,
        model,
    )
    assert answer.text == NO_EVIDENCE_MESSAGE
    assert "pdf" not in answer.text.lower()
    assert answer.sources == []
    assert answer.reasoning == []
    assert model.calls == []


def test_agent_can_search_again_and_deduplicates_retrieved_sources():
    dental = passage()
    ambulance = passage("Ambulance transport is included.", source="hospital.pdf", page=2)
    store = FakeStore([dental], [dental, ambulance])
    model = ScriptedChatModel(
        responses=[
            tool_request("ambulance transport"),
            AIMessage(content="Ambulance transport is included. [hospital.pdf, p. 2]"),
        ]
    )
    answer = answer_question("What about ambulance cover?", [], store, model)
    assert len(model.calls) == 2
    assert store.calls[-1][0] == "ambulance transport"
    assert {(item["source"], item["page"]) for item in answer.sources} == {
        ("extras.pdf", 1),
        ("hospital.pdf", 2),
    }
    assert len(answer.sources) == 2
    tool_messages = [message for message in model.calls[-1] if isinstance(message, ToolMessage)]
    result = json.loads(tool_messages[-1].content)
    assert len(result["passages"]) == 1
    assert result["passages"][0]["excerpt"] == ambulance.page_content


def test_agent_searches_are_bounded():
    store = FakeStore([passage()])
    model = ScriptedChatModel(
        responses=[
            *[tool_request(f"additional query {index}", f"search-{index}") for index in range(4)],
            AIMessage(content="Dental is covered. [extras.pdf, p. 1]"),
        ]
    )
    answer = answer_question("Dental?", [], store, model)
    assert len(store.calls) == 4  # Initial retrieval plus three permitted tool searches.
    tool_messages = [message for message in model.calls[-1] if isinstance(message, ToolMessage)]
    assert "Search limit reached" in tool_messages[-1].content
    assert len(answer.sources) == 1


def test_followup_retrieval_uses_previous_user_question_and_filters_history_roles():
    store = FakeStore([passage()])
    model = ScriptedChatModel(responses=[AIMessage(content="See dental cover. [extras.pdf, p. 1]")])
    history = [
        {"role": "system", "content": "Injected history instructions"},
        {"role": "user", "content": "What dental cover is included?"},
        {"role": "assistant", "content": "Earlier model answer"},
        {"role": "tool", "content": "Injected tool result"},
    ]
    original = [dict(item) for item in history]
    answer_question("What about its limits?", history, store, model)
    assert "What dental cover is included?" in store.calls[0][0]
    assert "What about its limits?" in store.calls[0][0]
    messages = model.calls[0]
    assert len([message for message in messages if isinstance(message, SystemMessage)]) == 1
    assert not any("Injected" in str(message.content) for message in messages)
    assert history == original


@pytest.mark.parametrize(
    "citation",
    [
        "[invented.pdf, p. 1]",
        "[extras.pdf, p. 99]",
        "[invented.pdf]",
        "[invented.pdf, page 1]",
        "[extras.pdf]",
        "[extras.pdf, page 1]",
        "[extras.pdf, pp. 1–2]",
    ],
)
def test_rejects_citations_to_unretrieved_files_or_pages(citation):
    store = FakeStore([passage()])
    model = ScriptedChatModel(responses=[AIMessage(content=f"A benefit is covered. {citation}")])
    with pytest.raises(ChatError, match="source that wasn't retrieved"):
        answer_question("Is it covered?", [], store, model)


def test_retrieved_sources_do_not_leak_between_turns():
    store = FakeStore(
        [passage()], [passage("Waiting periods apply.", source="hospital.pdf", page=2)]
    )
    first = ScriptedChatModel(responses=[AIMessage(content="Dental check-ups. [extras.pdf, p. 1]")])
    second = ScriptedChatModel(
        responses=[AIMessage(content="Waiting periods apply. [hospital.pdf, p. 2]")]
    )
    answer_question("Dental?", [], store, first)
    answer = answer_question("Waiting periods?", [], store, second)
    assert {item["source"] for item in answer.sources} == {"hospital.pdf"}


def test_embedding_error_is_actionable_and_does_not_call_chat_model(monkeypatch):
    store = FakeStore([passage()])
    model = ScriptedChatModel()

    def fail_search(query, k):
        raise RuntimeError("http://secret-server/token=private-value")

    monkeypatch.setattr(store, "similarity_search", fail_search)
    with pytest.raises(
        ChatError, match="Medibank information is temporarily unavailable"
    ) as failure:
        answer_question("Benefits?", [], store, model)
    assert str(failure.value) == RETRIEVAL_UNAVAILABLE_MESSAGE
    assert "private-value" not in str(failure.value)
    assert model.calls == []


def test_provider_error_does_not_expose_secrets(monkeypatch):
    def fail_model(self, messages, **kwargs):
        raise RuntimeError("Unauthorized API key sk-private-secret")

    monkeypatch.setattr(ScriptedChatModel, "_generate", fail_model)
    with pytest.raises(ChatError, match="model request failed") as failure:
        answer_question("Dental?", [], FakeStore([passage()]), ScriptedChatModel())
    assert "sk-private-secret" not in str(failure.value)


def test_non_tool_capable_model_has_clear_error():
    class UnsupportedModel(ScriptedChatModel):
        def bind_tools(self, tools, **kwargs):
            raise NotImplementedError("Tools are not supported")

    with pytest.raises(ChatError, match="tool calling"):
        answer_question("Dental?", [], FakeStore([passage()]), UnsupportedModel())


def test_empty_model_answer_has_clear_error():
    model = ScriptedChatModel(responses=[AIMessage(content="")])
    with pytest.raises(ChatError, match="empty answer"):
        answer_question("Dental?", [], FakeStore([passage()]), model)


def test_content_blocks_return_answer_without_reasoning():
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content=[
                    {"type": "reasoning", "reasoning": "Private model reasoning"},
                    {"type": "text", "text": "Dental is covered. [extras.pdf, p. 1]"},
                ]
            )
        ]
    )
    answer = answer_question("Dental?", [], FakeStore([passage()]), model)
    assert answer.text == "Dental is covered. [extras.pdf, p. 1]"
    assert answer.reasoning == ["Private model reasoning"]


def test_uncited_informative_answer_gets_one_citation_revision_without_search():
    store = FakeStore([passage()])
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="Dental check-ups are covered."),
            AIMessage(content="Dental check-ups are covered. [extras.pdf, p. 1]"),
        ]
    )
    answer = answer_question("Dental?", [], store, model)
    assert answer.text.endswith("[extras.pdf, p. 1]")
    assert len(model.calls) == 2
    assert len(store.calls) == 1
    assert any(
        isinstance(message, AIMessage) and message.content == "Dental check-ups are covered."
        for message in model.calls[-1]
    )
    assert "Do not search or call tools" in model.calls[-1][-1].content
    assert "[extras.pdf, p. 1]" in model.calls[-1][-1].content


def test_uncited_revision_is_rejected_without_appending_an_arbitrary_citation():
    store = FakeStore([passage()])
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="Dental is covered."),
            AIMessage(content="Dental is covered."),
        ]
    )
    with pytest.raises(ChatError, match="after one revision"):
        answer_question("Dental?", [], store, model)
    assert len(model.calls) == 2
    assert len(store.calls) == 1


def test_revision_with_an_unretrieved_citation_is_rejected():
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="Dental is covered."),
            AIMessage(content="Dental is covered. [invented.pdf, p. 10]"),
        ]
    )
    with pytest.raises(ChatError, match="source that wasn't retrieved"):
        answer_question("Dental?", [], FakeStore([passage()]), model)
    assert len(model.calls) == 2


def test_revision_cannot_perform_another_search():
    store = FakeStore([passage()])
    model = ScriptedChatModel(
        responses=[AIMessage(content="Dental is covered."), tool_request("new search")]
    )
    with pytest.raises(ChatError, match="did not finish"):
        answer_question("Dental?", [], store, model)
    assert len(store.calls) == 1
    assert len(model.calls) == 2


@pytest.mark.parametrize(
    "abstention",
    [
        NO_EVIDENCE_MESSAGE,
        "I couldn't find information about that in the provided PDFs.",
        "I cannot answer this question from the retrieved passages.",
        "The retrieved passages do not provide enough information to answer this question.",
    ],
)
def test_a_pure_inability_statement_does_not_need_citation_revision(abstention):
    model = ScriptedChatModel(responses=[AIMessage(content=abstention)])
    answer = answer_question("Premium price?", [], FakeStore([passage()]), model)
    assert answer.text == abstention
    assert len(model.calls) == 1


def test_inability_statement_with_an_uncited_claim_is_not_exempt_from_revision():
    text = "I couldn't find information in the PDFs. Dental is covered."
    model = ScriptedChatModel(responses=[AIMessage(content=text), AIMessage(content=text)])
    with pytest.raises(ChatError, match="after one revision"):
        answer_question("Premium price?", [], FakeStore([passage()]), model)
    assert len(model.calls) == 2


def test_medibank_fallback_with_an_uncited_claim_still_requires_citation_revision():
    text = NO_EVIDENCE_MESSAGE + " Dental is covered."
    model = ScriptedChatModel(responses=[AIMessage(content=text), AIMessage(content=text)])
    with pytest.raises(ChatError, match="after one revision"):
        answer_question("Premium price?", [], FakeStore([passage()]), model)
    assert len(model.calls) == 2


def test_recent_history_has_a_total_budget_and_retains_latest_user_context():
    history = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"message-{index} " + "x" * 3000,
        }
        for index in range(20)
    ]
    original = [dict(message) for message in history]
    store = FakeStore([passage()])
    model = ScriptedChatModel(
        responses=[AIMessage(content="Dental is covered. [extras.pdf, p. 1]")]
    )
    answer_question("What about that?", history, store, model)
    visible_history = model.calls[0][1:-1]
    assert sum(len(message.content) for message in visible_history) <= 4000
    assert visible_history[-1].content.startswith("message-19 ")
    assert any(message.content.startswith("message-18 ") for message in visible_history)
    assert "message-18 " in store.calls[0][0]
    assert history == original


def test_cumulative_passage_json_has_a_budget_and_only_visible_sources_are_returned():
    store = FakeStore(*[[passage(chr(97 + index) * 3000, page=index + 1)] for index in range(4)])
    model = ScriptedChatModel(
        responses=[
            tool_request("second passage", "search-2"),
            tool_request("third passage", "search-3"),
            tool_request("fourth passage", "search-4"),
            AIMessage(content="See these details. [extras.pdf, p. 3]"),
        ]
    )
    answer = answer_question("Benefits?", [], store, model)
    initial = json.loads(model.calls[0][-1].content)["retrieved_passages"]
    lists = [initial] + [
        json.loads(message.content)["passages"]
        for message in model.calls[-1]
        if isinstance(message, ToolMessage)
    ]
    assert sum(len(json.dumps(items, ensure_ascii=False)) for items in lists) <= 10000
    assert {source["page"] for source in answer.sources} == {1, 2, 3}
    assert lists[-1] == []
    assert len(store.calls) == 4


def test_a_citation_to_a_retrieved_but_budget_excluded_page_is_rejected():
    store = FakeStore(
        [
            passage("a" * 6000, page=1),
            passage("b" * 6000, page=2),
        ]
    )
    model = ScriptedChatModel(responses=[AIMessage(content="Details. [extras.pdf, p. 2]")])
    with pytest.raises(ChatError, match="source that wasn't retrieved"):
        answer_question("Benefits?", [], store, model)
    assert len(model.calls) == 1


@pytest.mark.parametrize(
    "question,top_k", [("", 5), ("x" * 4001, 5), ("Dental?", 0), ("Dental?", 21)]
)
def test_invalid_requests_fail_before_model_or_retrieval(question, top_k):
    store = FakeStore([passage()])
    model = ScriptedChatModel()
    with pytest.raises(ValueError):
        answer_question(question, [], store, model, top_k=top_k)
    assert store.calls == []
    assert model.calls == []


def test_answer_supports_old_callers_and_independent_reasoning_defaults():
    first = Answer(text="Answer", sources=[])
    second = Answer(text="Answer", sources=[])
    first.reasoning.append("Reasoning")
    assert second.reasoning == []


@pytest.mark.parametrize(
    "blocks,extra,expected",
    [
        (
            [{"type": "thinking", "thinking": "Check the dental benefit.", "signature": "opaque"}],
            {},
            ["Check the dental benefit."],
        ),
        (
            [
                {
                    "type": "text",
                    "text": "Check the dental benefit.",
                    "thought": True,
                    "thought_signature": "opaque",
                }
            ],
            {},
            ["Check the dental benefit."],
        ),
        (
            [
                {
                    "type": "reasoning",
                    "summary": [
                        {"type": "summary_text", "text": "Check the dental benefit."},
                        {"type": "summary_text", "text": "The page supports it."},
                    ],
                    "encrypted_content": "opaque",
                }
            ],
            {},
            ["Check the dental benefit.", "The page supports it."],
        ),
        ([], {"reasoning_content": "Check the dental benefit."}, ["Check the dental benefit."]),
        ([], {"thinking": "Check the dental benefit."}, ["Check the dental benefit."]),
        (
            [],
            {
                "reasoning": {
                    "summary": [{"type": "summary_text", "text": "Check the dental benefit."}],
                    "encrypted_content": "opaque",
                }
            },
            ["Check the dental benefit."],
        ),
        (
            [{"type": "reasoning", "reasoning": "Check the dental benefit."}],
            {"reasoning_content": "Check the dental benefit."},
            ["Check the dental benefit."],
        ),
        (
            [
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "signature", "text": "opaque", "thought": True},
                {"type": "reasoning", "encrypted_content": "opaque", "reasoning": 42},
            ],
            {
                "reasoning_content": {"type": "encrypted_reasoning", "text": "opaque"},
                "thinking": {"error": "provider error payload"},
                "signature": "opaque",
            },
            [],
        ),
    ],
)
def test_extracts_only_provider_plaintext_reasoning(blocks, extra, expected):
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content=[
                    *blocks,
                    {"type": "text", "text": "Dental is covered. [extras.pdf, p. 1]"},
                ],
                additional_kwargs=extra,
            )
        ]
    )
    answer = answer_question("Dental?", [], FakeStore([passage()]), model)
    assert answer.reasoning == expected
    assert answer.text == "Dental is covered. [extras.pdf, p. 1]"
    assert len(model.calls) == 1


@pytest.mark.parametrize(
    "content,expected",
    [
        (
            "<think>Consider [invented.pdf, p. 99].</think>\nDental is covered. [extras.pdf, p. 1]",
            ["Consider [invented.pdf, p. 99]."],
        ),
        (
            "<THINK>First step.</THINK><think>Second step.</think>"
            "Dental is covered. [extras.pdf, p. 1]",
            ["First step.", "Second step."],
        ),
        (
            [
                {"type": "text", "text": "<think>Check the"},
                {
                    "type": "text",
                    "text": "dental benefit.</think>Dental is covered. [extras.pdf, p. 1]",
                },
            ],
            ["Check the\ndental benefit."],
        ),
        (
            "Dental is covered. [extras.pdf, p. 1]<think>Unfinished hidden thought.",
            ["Unfinished hidden thought."],
        ),
    ],
)
def test_think_tags_are_separate_from_final_answer_and_citation_validation(content, expected):
    model = ScriptedChatModel(responses=[AIMessage(content=content)])
    answer = answer_question("Dental?", [], FakeStore([passage()]), model)
    assert answer.text == "Dental is covered. [extras.pdf, p. 1]"
    assert answer.reasoning == expected
    assert len(model.calls) == 1


def test_public_reasoning_splitter_normalizes_legacy_messages():
    assert split_reasoning_text("<think>Old reasoning.</think>Old answer.") == (
        "Old answer.",
        ["Old reasoning."],
    )
    assert split_reasoning_text("<think>Unfinished reasoning.") == ("", ["Unfinished reasoning."])


def test_reasoning_without_a_final_answer_has_clear_error():
    model = ScriptedChatModel(responses=[AIMessage(content="<think>Still thinking.")])
    with pytest.raises(ChatError, match="empty answer"):
        answer_question("Dental?", [], FakeStore([passage()]), model)


def test_reasoning_covers_tool_iterations_and_citation_repair_once(monkeypatch):
    events = []
    monkeypatch.setattr(
        "medibank.chat.audit", lambda event, **fields: events.append((event, fields))
    )
    store = FakeStore([passage()], [passage(), passage("Dental limits.", page=2)])
    search = tool_request("dental limits")
    search.content = [{"type": "reasoning", "reasoning": "Find the limits."}]
    model = ScriptedChatModel(
        responses=[
            search,
            AIMessage(content="<think>Check the evidence.</think>Dental is covered."),
            AIMessage(
                content=[
                    {"type": "text", "text": "Add the exact citation.", "thought": True},
                    {"type": "text", "text": "Dental is covered. [extras.pdf, p. 1]"},
                ]
            ),
        ]
    )
    answer = answer_question(
        "Dental?", [], store, model, request_id="request", session_id="session"
    )
    assert answer.reasoning == [
        "Find the limits.",
        "Check the evidence.",
        "Add the exact citation.",
    ]
    assert answer.text == "Dental is covered. [extras.pdf, p. 1]"
    assert [event for event, fields in events] == [
        "retrieval_completed",
        "retrieval_completed",
        "citation_revision_started",
    ]
    assert all(fields["request_id"] == "request" for event, fields in events)
    assert all(fields["session_id"] == "session" for event, fields in events)
    assert all(thought not in str(events) for thought in answer.reasoning)


def test_reasoning_is_excluded_from_old_history_and_later_turns():
    history = [
        {"role": "user", "content": "What is dental cover?"},
        {
            "role": "assistant",
            "content": "<think>Old tagged reasoning.</think>Old answer.",
            "reasoning": ["Old separately stored reasoning."],
        },
    ]
    original = [dict(item) for item in history]
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="Dental is covered. [extras.pdf, p. 1]",
                additional_kwargs={"reasoning_content": "Current reasoning."},
            )
        ]
    )
    answer = answer_question("What about its limits?", history, FakeStore([passage()]), model)
    assert answer.reasoning == ["Current reasoning."]
    assert model.calls[0][2].content == "Old answer."
    assert "Old tagged reasoning." not in str(model.calls[0])
    assert "Old separately stored reasoning." not in str(model.calls[0])
    assert history == original
    history.append({"role": "assistant", "content": answer.text, "reasoning": answer.reasoning})
    next_model = ScriptedChatModel(
        responses=[AIMessage(content="Dental is covered. [extras.pdf, p. 1]")]
    )
    following = answer_question("Dental?", history, FakeStore([passage()]), next_model)
    assert following.reasoning == []
    assert "Current reasoning." not in str(next_model.calls[0])


def test_citation_repair_preserves_provider_signatures_and_native_content_format():
    original = AIMessage(
        content=[
            {
                "type": "reasoning",
                "reasoning": "Check the benefit.",
                "extras": {"signature": "opaque-thought-signature"},
            },
            {"type": "text", "text": "Dental is covered."},
        ],
        additional_kwargs={
            "reasoning_content": "Check the benefit.",
            "__gemini_function_call_thought_signatures__": {"call-1": "opaque-call-signature"},
        },
        response_metadata={"output_version": "v1"},
    )
    model = ScriptedChatModel(
        responses=[original, AIMessage(content="Dental is covered. [extras.pdf, p. 1]")]
    )
    answer = answer_question("Dental?", [], FakeStore([passage()]), model)
    replay = next(message for message in model.calls[-1] if isinstance(message, AIMessage))
    assert replay.content == original.content
    assert replay.additional_kwargs == original.additional_kwargs
    assert replay.response_metadata == original.response_metadata
    assert answer.reasoning == ["Check the benefit."]
    assert "opaque" not in str(answer)


class StreamingChatModel(ScriptedChatModel):
    """Real chat-model streaming that pauses while the main thread renders progress."""

    stream_responses: list[list[AIMessageChunk]] = Field(default_factory=list)
    finished_calls: int = 0

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        index = len(self.calls) - 1
        for chunk in self.stream_responses[index]:
            yield ChatGenerationChunk(message=chunk)
            time.sleep(0.01)
        self.finished_calls += 1


def test_live_reasoning_arrives_before_model_finishes_and_handles_split_think_tags():
    model = StreamingChatModel(
        stream_responses=[
            [
                AIMessageChunk(content="<th"),
                AIMessageChunk(content="ink>Check "),
                AIMessageChunk(content="the benefit.</thi"),
                AIMessageChunk(content="nk>"),
                AIMessageChunk(content="Dental is covered. [extras.pdf, p. 1]"),
            ]
        ]
    )
    events = []
    caller = get_ident()

    def progress(event):
        events.append((dict(event), model.finished_calls, get_ident()))

    answer = answer_question("Dental?", [], FakeStore([passage()]), model, on_progress=progress)
    live_reasoning = [item for item in events if item[0]["type"] == "reasoning"]
    assert live_reasoning
    assert all(finished == 0 for event, finished, thread in live_reasoning)
    assert all(thread == caller for event, finished, thread in events)
    assert (
        "".join(event["text"] for event, finished, thread in live_reasoning) == "Check the benefit."
    )
    assert all("<" not in event["text"] for event, finished, thread in live_reasoning)
    assert answer.reasoning == ["Check the benefit."]
    assert answer.text == "Dental is covered. [extras.pdf, p. 1]"
    assert any(
        event["text"] == "Receiving the model response." for event, finished, thread in events
    )
    assert events[-1][0] == {"type": "step", "text": "The answer is ready."}


@pytest.mark.parametrize(
    "first,second",
    [
        (
            AIMessageChunk(
                content=[
                    {"type": "thinking", "thinking": "Check ", "index": 0, "signature": "opaque"}
                ]
            ),
            AIMessageChunk(content=[{"type": "thinking", "thinking": "the benefit.", "index": 0}]),
        ),
        (
            AIMessageChunk(
                content=[{"type": "text", "text": "Check ", "thought": True, "index": 0}]
            ),
            AIMessageChunk(
                content=[{"type": "text", "text": "the benefit.", "thought": True, "index": 0}]
            ),
        ),
        (
            AIMessageChunk(
                content=[
                    {
                        "type": "reasoning",
                        "index": 0,
                        "summary": [{"type": "summary_text", "text": "Check ", "index": 0}],
                    }
                ]
            ),
            AIMessageChunk(
                content=[
                    {
                        "type": "reasoning",
                        "index": 0,
                        "summary": [{"type": "summary_text", "text": "the benefit.", "index": 0}],
                    }
                ]
            ),
        ),
        (
            AIMessageChunk(content="", additional_kwargs={"reasoning_content": "Check "}),
            AIMessageChunk(
                content="",
                additional_kwargs={
                    "reasoning_content": "the benefit.",
                    "provider_debug": "sk-private-error",
                },
            ),
        ),
    ],
)
def test_live_provider_reasoning_preserves_word_spacing_and_ignores_opaque_fields(first, second):
    model = StreamingChatModel(
        stream_responses=[
            [
                first,
                second,
                AIMessageChunk(
                    content=[
                        {
                            "type": "text",
                            "text": "Dental is covered. [extras.pdf, p. 1]",
                            "index": 1,
                        }
                    ]
                ),
            ]
        ]
    )
    events = []
    answer = answer_question(
        "Dental?", [], FakeStore([passage()]), model, on_progress=events.append
    )
    reasoning = "".join(event["text"] for event in events if event["type"] == "reasoning")
    assert reasoning == "Check the benefit."
    assert answer.reasoning == ["Check the benefit."]
    assert "opaque" not in str(events)
    assert "sk-private" not in str(events)
    assert not any("Dental is covered" in event["text"] for event in events)


def test_live_progress_supports_parallel_tools_and_repair_without_duplicate_reasoning():
    caller = get_ident()
    search_threads = []

    class ThreadRecordingStore(FakeStore):
        def similarity_search(self, query, k):
            search_threads.append(get_ident())
            return super().similarity_search(query, k)

    store = ThreadRecordingStore([passage()], [passage(), passage("Dental limits.", page=2)])
    model = StreamingChatModel(
        stream_responses=[
            [
                AIMessageChunk(
                    content=[
                        {
                            "type": "reasoning",
                            "reasoning": "Find the limits.",
                            "index": 0,
                            "extras": {"signature": "opaque"},
                        }
                    ]
                ),
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": "search_knowledge_base",
                            "args": '{"query":"dental limits"}',
                            "id": "search-1",
                            "index": 0,
                        },
                        {
                            "name": "search_knowledge_base",
                            "args": '{"query":"dental details"}',
                            "id": "search-2",
                            "index": 1,
                        },
                    ],
                ),
            ],
            [AIMessageChunk(content="<think>Check the evidence.</think>Dental is covered.")],
            [
                AIMessageChunk(
                    content="", additional_kwargs={"reasoning_content": "Add the exact citation."}
                ),
                AIMessageChunk(content="Dental is covered. [extras.pdf, p. 1]"),
            ],
        ]
    )
    events = []
    callback_threads = []

    def progress(event):
        callback_threads.append(get_ident())
        events.append(dict(event))

    answer = answer_question("Dental?", [], store, model, on_progress=progress)
    assert answer.reasoning == [
        "Find the limits.",
        "Check the evidence.",
        "Add the exact citation.",
    ]
    assert "".join(event["text"] for event in events if event["type"] == "reasoning") == (
        "Find the limits.\n\nCheck the evidence.\n\nAdd the exact citation."
    )
    assert len(store.calls) == 3
    assert model.finished_calls == 3
    assert callback_threads and set(callback_threads) == {caller}
    assert search_threads[0] == caller
    assert all(thread != caller for thread in search_threads[1:])
    assert any("Revising the answer" in event["text"] for event in events)
    assert any("Searching for additional" in event["text"] for event in events)
    # Native signatures and complete tool messages survive in the streamed state.
    repair_ai = next(message for message in model.calls[-1] if isinstance(message, AIMessage))
    assert repair_ai.content[0]["extras"]["signature"] == "opaque"
    assert len([message for message in model.calls[-1] if isinstance(message, ToolMessage)]) == 2


def test_live_progress_never_shows_an_answer_that_fails_citation_validation():
    model = StreamingChatModel(
        stream_responses=[
            [
                AIMessageChunk(content="<think>Check the evidence.</think>"),
                AIMessageChunk(content="Invented cover. [invented.pdf, p. 9]"),
            ]
        ]
    )
    events = []
    with pytest.raises(ChatError, match="source that wasn't retrieved"):
        answer_question("Dental?", [], FakeStore([passage()]), model, on_progress=events.append)
    assert "Invented cover" not in str(events)
    assert "invented.pdf" not in str(events)
    assert not any(event["text"] == "The answer is ready." for event in events)


def test_live_progress_callback_failure_does_not_discard_a_valid_answer():
    model = ScriptedChatModel(
        responses=[AIMessage(content="Dental is covered. [extras.pdf, p. 1]")]
    )
    calls = []

    def progress(event):
        calls.append(event)
        raise RuntimeError("private UI failure sk-secret")

    answer = answer_question("Dental?", [], FakeStore([passage()]), model, on_progress=progress)
    assert answer.text == "Dental is covered. [extras.pdf, p. 1]"
    assert len(calls) == 1
    assert "sk-secret" not in str(answer)


def test_live_progress_without_evidence_does_not_call_model():
    model = ScriptedChatModel()
    events = []
    answer = answer_question("Dental?", [], FakeStore([]), model, on_progress=events.append)
    assert model.calls == []
    assert answer.reasoning == []
    assert [event["type"] for event in events] == ["step", "step", "step"]
    assert events[-1]["text"] == "No relevant information was found."
