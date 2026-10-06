"""Typed state and structured decisions for the adaptive RAG graph."""

from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class RetrievalGrade(BaseModel):
    """Whether retrieved evidence, rather than its topic, answers the query."""

    verdict: Literal["correct", "ambiguous", "incorrect"]
    confidence: float = Field(ge=0.0, le=1.0)
    relevant_document_ids: list[str] = Field(default_factory=list)
    irrelevant_document_ids: list[str] = Field(default_factory=list)
    reasoning: str
    needs_web: bool = False


class StripSelection(BaseModel):
    """Batch decision used by knowledge refinement; IDs are supplied by us."""

    relevant_strip_ids: list[str] = Field(default_factory=list)
    reasoning: str = ""


class AnswerGrade(BaseModel):
    """Self-verification result. It is a guardrail, not a trained Self-RAG token."""

    grounded: bool
    complete: bool
    answerable: bool
    citation_valid: bool
    confidence: float = Field(ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    reasoning: str
    action: Literal["accept", "regenerate", "web_search", "abstain"]


class GraphState(TypedDict, total=False):
    question: str
    standalone_question: str
    chat_history: list[BaseMessage]
    retrieval_query: str
    retrieved_documents: list[Document]
    relevant_documents: list[Document]
    web_documents: list[Document]
    final_context: list[Document]
    retrieval_verdict: Literal["correct", "ambiguous", "incorrect"]
    retrieval_confidence: float
    retrieval_reasoning: str
    relevant_document_ids: list[str]
    needs_web: bool
    rewrite_query: str
    rewrite_purpose: Literal["recovery", "web"]
    draft_answer: str
    final_answer: str
    answer_grade: AnswerGrade | None
    unsupported_claims: list[str]
    missing_information: list[str]
    routing_decision: str
    iteration_count: int
    retrieval_retry_count: int
    generation_retry_count: int
    used_web_search: bool
    source_policy: Literal["adaptive", "corpus_only"]
    source_types: list[str]
    sources: list[dict]
    trace: list[str]
    llm_calls: int
    errors: list[str]
