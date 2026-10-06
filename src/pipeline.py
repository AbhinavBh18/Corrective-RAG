"""Backwards-compatible facade over the bounded LangGraph adaptive RAG flow.

``answer_question`` remains the public entry point used by the Streamlit app
and evaluation harness. Its orchestration is now explicit StateGraph nodes,
while parsing, Qdrant, BM25, RRF, and session memory remain unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

from src.config import DEFAULT_SOURCE_POLICY, RETRIEVAL_MODE
from src.graph.builder import build_adaptive_rag_graph, graph_mermaid, initial_state
from src.graph.nodes import final_sources
from src.memory import get_session
from src.pipeline_support import get_judge_llm, get_llm

logger = logging.getLogger(__name__)


def get_query_engine(mode: str = RETRIEVAL_MODE, top_k: int | None = None):
    """Return the compiled LangGraph runnable.

    ``top_k`` is retained for call compatibility. Candidate depth and final
    context are now separately configured by CRAG_CANDIDATE_K/refinement.
    """
    if top_k is not None:
        logger.debug("top_k=%s retained for compatibility; graph candidate depth is configured centrally", top_k)
    return build_adaptive_rag_graph(mode=mode)


def format_sources(docs: list[Document]) -> list[dict[str, Any]]:
    """Legacy source formatter extended with explicit local/web provenance."""
    return final_sources({"final_context": docs})


def answer_question(
    question: str,
    session_id: str | None = None,
    engine: Any | None = None,
    source_policy: str = DEFAULT_SOURCE_POLICY,
) -> dict[str, Any]:
    """Run one graph turn and update existing session-scoped memory on success."""
    engine = engine or get_query_engine()
    session = get_session(session_id) if session_id else None
    state = initial_state(question, session.messages if session else [], source_policy=source_policy)
    result = dict(engine.invoke(state))
    docs: list[Document] = list(result.get("final_context", []))
    answer = result.get("final_answer") or result.get("draft_answer") or (
        "I cannot answer this reliably from the available evidence."
    )
    if session:
        session.add_turn(question, answer)
    sources = final_sources(result)
    result["sources"] = sources
    source_types = sorted({s.get("source_type", "local") for s in sources})
    return {
        "question": question,
        "answer": answer,
        "documents": docs,
        "sources": sources,
        "trace": result.get("trace", []),
        "metadata": {
            "standalone_question": result.get("standalone_question", question),
            "retrieval_query": result.get("retrieval_query", question),
            "retrieval_verdict": result.get("retrieval_verdict"),
            "retrieval_confidence": result.get("retrieval_confidence"),
            "used_web_search": result.get("used_web_search", False),
            "source_policy": result.get("source_policy", source_policy),
            "source_types": source_types,
            "answer_grade": result.get("answer_grade").model_dump() if result.get("answer_grade") else None,
            "iteration_count": result.get("iteration_count", 0),
            "retrieval_retry_count": result.get("retrieval_retry_count", 0),
            "generation_retry_count": result.get("generation_retry_count", 0),
            "llm_calls": result.get("llm_calls", 0),
            "errors": result.get("errors", []),
        },
    }


def get_graph_mermaid() -> str:
    """Expose a diagram for README/demo tooling without leaking graph internals."""
    return graph_mermaid()


__all__ = ["answer_question", "format_sources", "get_graph_mermaid", "get_judge_llm", "get_llm", "get_query_engine"]
