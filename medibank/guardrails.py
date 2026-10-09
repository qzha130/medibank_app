"""Conservative question routing before any chat-model request."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

OFF_TOPIC_MESSAGE = (
    "I can only answer general questions about Medibank. For example, ask about "
    "membership, benefits, claims, exclusions, or waiting periods."
)
HUMAN_REVIEW_MESSAGE = "A human representative should help with this request."
PERSONAL_REVIEW_MESSAGE = (
    "This request needs personal account access or professional review. "
    "A human representative should help."
)
RETRIEVAL_UNAVAILABLE_MESSAGE = (
    "Medibank information is temporarily unavailable. Please try again or request human review."
)
INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I couldn't find enough relevant Medibank information to answer reliably. "
    "Please request human review."
)
NO_EVIDENCE_MESSAGE = (
    "I couldn't find enough Medibank information to answer this question. "
    "Please try a more specific question."
)

# Exact historical fallbacks only; never replace arbitrary conversation text.
LEGACY_FALLBACK_MESSAGES = {
    "I can only use the PDFs to answer document questions.": OFF_TOPIC_MESSAGE,
    (
        "I answer questions about the Medibank PDFs. Please ask about membership, "
        "benefits, claims, exclusions, or waiting periods."
    ): OFF_TOPIC_MESSAGE,
    "A human representative should help with this request.": HUMAN_REVIEW_MESSAGE,
    (
        "This needs personal account access or professional review, "
        "which these PDFs cannot provide."
    ): PERSONAL_REVIEW_MESSAGE,
    (
        "The document evidence could not be checked. Please retry or request human review."
    ): RETRIEVAL_UNAVAILABLE_MESSAGE,
    (
        "I could not find sufficiently relevant evidence in the PDFs to answer reliably. "
        "A human should review this question."
    ): INSUFFICIENT_EVIDENCE_MESSAGE,
    (
        "I couldn't find information for this question in the indexed PDFs. "
        "Try a more specific question or check that the relevant document is indexed."
    ): NO_EVIDENCE_MESSAGE,
}

_DOMAIN = re.compile(
    r"\b(?:medibank|insurance|polic(?:y|ies)|member\w*|cover\w*|benefit\w*|wait\w*|"
    r"claim\w*|premium\w*|hospital\w*|dental|ambulance|extra\w*|exclu\w*|"
    r"refund\w*|cancel\w*|eligib\w*|accident\w*|emergency|admission\w*|"
    r"pre.?existing|treatment\w*|depend[ae]nt\w*|famil\w*|child\w*|kids|"
    r"fund|rebate\w*|medicare|health|mental|rehab\w*|maternity|pregnan\w*|"
    r"physio\w*|optical|surger\w*|diagnostic|imaging|overseas|resident\w*)\b",
    re.IGNORECASE,
)
_OVERRIDE = re.compile(
    r"ignore.{0,30}(?:instruction|rules)|reveal.{0,30}(?:secret|api.key|system.prompt)", re.I
)
_HUMAN = re.compile(
    r"\bhuman\b|\b(?:live|real) (?:agent|person|support)|(?:speak|talk|connect|contact).{0,30}(?:agent|someone|representative|support|staff)",
    re.I,
)
_PERSONAL = re.compile(
    r"claim.{0,30}status|status.{0,30}claim|access.{0,30}account|approve.{0,20}claim|my (?:policy|member\w*) (?:number|details)|"
    r"(?:change|update).{0,20}(?:bank|payment|address)|am i eligible|am i covered|"
    r"(?:diagnose|diagnosis for me|what dose|which medication|medical advice)",
    re.I,
)
_UNRELATED = re.compile(
    r"\b(?:python|javascript|bitcoin|crypto|weather|prime minister|president)\b|"
    r"(?:write|tell).{0,12}(?:poem|joke|song|code)",
    re.I,
)
_FOLLOWUP = re.compile(
    r"\b(?:it|its|that|this|they|them|those)\b|^(?:and|what about|how about)\b", re.I
)


@dataclass(frozen=True)
class GuardrailDecision:
    allowed: bool
    reason: str
    message: str = ""
    distance: float | None = None
    needs_human: bool = False


def assess_question(
    question: str, history: list[dict], store, max_distance: float | None = None
) -> GuardrailDecision:
    question = question.strip()
    if _OVERRIDE.search(question):
        return GuardrailDecision(False, "instruction_override", OFF_TOPIC_MESSAGE)
    if _HUMAN.search(question):
        return GuardrailDecision(
            False,
            "human_requested",
            HUMAN_REVIEW_MESSAGE,
            needs_human=True,
        )
    if _PERSONAL.search(question):
        return GuardrailDecision(
            False,
            "personal_or_clinical_request",
            PERSONAL_REVIEW_MESSAGE,
            needs_human=True,
        )
    query = question
    if history and _FOLLOWUP.search(question) and not history[-1].get("handoff"):
        previous = next(
            (item.get("content", "") for item in reversed(history) if item.get("role") == "user"),
            "",
        )
        query = previous[:600] + "\nFollow-up: " + question
    if _UNRELATED.search(question) or not _DOMAIN.search(query):
        return GuardrailDecision(
            False,
            "off_topic",
            OFF_TOPIC_MESSAGE,
        )
    if max_distance is None:
        try:
            max_distance = float(os.getenv("GUARDRAIL_MAX_DISTANCE", "0.35"))
        except ValueError:
            max_distance = 0.35
    if not math.isfinite(max_distance) or not 0 <= max_distance <= 2:
        max_distance = 0.35
    try:
        hits = store.similarity_search_with_score(query, k=1)
        distance = float(hits[0][1]) if hits else None
    except Exception:
        return GuardrailDecision(
            False,
            "retrieval_unavailable",
            RETRIEVAL_UNAVAILABLE_MESSAGE,
            needs_human=True,
        )
    if distance is None or not math.isfinite(distance) or distance > max_distance:
        return GuardrailDecision(
            False,
            "insufficient_evidence",
            INSUFFICIENT_EVIDENCE_MESSAGE,
            distance=distance if distance is not None and math.isfinite(distance) else None,
            needs_human=True,
        )
    return GuardrailDecision(True, "relevant_evidence", distance=distance)
