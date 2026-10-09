"""Exercise the real LangChain agent with a scripted, offline chat model."""

from __future__ import annotations

import json

import pytest
from conftest import ScriptedChatModel
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from medibank.chat import ChatError, answer_question
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
