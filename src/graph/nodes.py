"""Independently testable LangGraph node implementations.

Nodes return partial state only.  Network/LLM failures are recorded in state so
the router can choose a bounded, transparent fallback instead of crashing a
conversation or silently fabricating an answer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from src.config import (
    CRAG_CANDIDATE_K,
    CRAG_CORRECT_THRESHOLD,
    CRAG_INCORRECT_THRESHOLD,
    CRAG_MAX_STRIPS,
    CRAG_SKIP_REFINEMENT_CONFIDENCE,
    CRAG_SKIP_REFINEMENT_MAX_DOCS,
    TAVILY_API_KEY,
    TAVILY_MAX_RESULTS,
)
from src.graph.state import AnswerGrade, GraphState, RetrievalGrade, StripSelection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphServices:
    """Injected dependencies make every node easy to exercise with fakes."""

    llm_factory: Callable[[], Any]
    judge_llm_factory: Callable[[], Any]
    retriever_factory: Callable[..., Any]


def document_id(doc: Document, position: int | None = None) -> str:
    return str(doc.metadata.get("chunk_id") or doc.metadata.get("url") or f"doc-{position or 0}")


def _event(state: GraphState, message: str, **updates: Any) -> dict[str, Any]:
    """Append concise operational trace metadata; never store chain-of-thought."""
    updates["trace"] = [*state.get("trace", []), message]
    return updates


def _error(state: GraphState, message: str, **updates: Any) -> dict[str, Any]:
    logger.warning(message)
    updates["errors"] = [*state.get("errors", []), message]
    return _event(state, message, **updates)


def _text(value: Any) -> str:
    content = getattr(value, "content", value)
    return content if isinstance(content, str) else str(content)


def _history_text(history: list[BaseMessage]) -> str:
    return "\n".join(f"{getattr(m, 'type', 'message')}: {_text(m)}" for m in history)


def _structured(llm: Any, schema: type[Any], prompt: str) -> Any:
    result = llm.with_structured_output(schema).invoke(prompt)
    return result if isinstance(result, schema) else schema.model_validate(result)


def contextualize_query(state: GraphState, services: GraphServices) -> dict[str, Any]:
    question, history = state["question"], state.get("chat_history", [])
    if not history:
        return _event(state, "Query ready for retrieval", standalone_question=question, retrieval_query=question)
    prompt = (
        "Rewrite the latest question as a standalone retrieval question. Preserve technical terms, "
        "model names, acronyms, and requested constraints. Do not answer it.\n\n"
        f"CHAT HISTORY:\n{_history_text(history)}\n\nLATEST QUESTION:\n{question}"
    )
    try:
        rewritten = _text(services.llm_factory().invoke(prompt)).strip() or question
        return _event(state, "Query contextualized from conversation history", standalone_question=rewritten,
                      retrieval_query=rewritten, llm_calls=state.get("llm_calls", 0) + 1)
    except Exception as exc:
        return _error(state, f"Query contextualization unavailable: {exc}", standalone_question=question,
                      retrieval_query=question)


def retrieve(state: GraphState, services: GraphServices) -> dict[str, Any]:
    query = state.get("retrieval_query") or state["question"]
    try:
        retriever = services.retriever_factory(top_k=CRAG_CANDIDATE_K, dense_k=CRAG_CANDIDATE_K,
                                               sparse_k=CRAG_CANDIDATE_K)
        docs = list(retriever.invoke(query))
        return _event(state, f"Hybrid retrieval returned {len(docs)} candidates", retrieved_documents=docs)
    except Exception as exc:
        return _error(state, f"Hybrid retrieval failed: {exc}", retrieved_documents=[])


def _retrieval_prompt(question: str, docs: list[Document]) -> str:
    rendered = "\n\n".join(
        f"ID: {document_id(d, i)}\nSOURCE: {d.metadata.get('paper_title', d.metadata.get('title', 'unknown'))}"
        f"\nCONTENT:\n{d.page_content[:1800]}"
        for i, d in enumerate(docs, 1)
    ) or "(No documents were retrieved.)"
    return (
        "Evaluate whether the local retrieved passages contain evidence that actually answers the question, "
        "not merely a related topic. confidence is EVIDENCE SUFFICIENCY from 0 (none) to 1 (enough to answer). "
        "Use correct/ambiguous/incorrect consistently with that evidence. Identify IDs that contribute evidence. "
        "Do not answer the question.\n\nQUESTION:\n" + question + "\n\nPASSAGES:\n" + rendered
    )


def grade_retrieval(state: GraphState, services: GraphServices) -> dict[str, Any]:
    docs = state.get("retrieved_documents", [])
    try:
        grade: RetrievalGrade = _structured(services.judge_llm_factory(), RetrievalGrade,
                                            _retrieval_prompt(state["standalone_question"], docs))
        # Thresholds deliberately operate on the evaluator's evidence score,
        # never the non-calibrated RRF rank fusion score.
        if grade.confidence >= CRAG_CORRECT_THRESHOLD:
            verdict = "correct"
        elif grade.confidence <= CRAG_INCORRECT_THRESHOLD:
            verdict = "incorrect"
        else:
            verdict = "ambiguous"
        ids = set(grade.relevant_document_ids)
        relevant = [d for i, d in enumerate(docs, 1) if document_id(d, i) in ids]
        if verdict == "correct" and not relevant:
            relevant = docs  # preserve evidence if the judge omitted IDs
        return _event(state, f"Retrieval verdict: {verdict.upper()}", retrieval_verdict=verdict,
                      retrieval_confidence=grade.confidence, retrieval_reasoning=grade.reasoning,
                      relevant_document_ids=list(ids), relevant_documents=relevant,
                      needs_web=grade.needs_web or verdict != "correct",
                      llm_calls=state.get("llm_calls", 0) + 1)
    except Exception as exc:
        # Conservative failure: never claim evidence was verified.
        return _error(state, f"Retrieval evaluator unavailable: {exc}", retrieval_verdict="incorrect",
                      retrieval_confidence=0.0, retrieval_reasoning="Evaluator unavailable.",
                      relevant_documents=[], needs_web=True)


def rewrite_for_recovery(state: GraphState, services: GraphServices) -> dict[str, Any]:
    prompt = (
        "Rewrite this for a local PEFT-paper retriever. Preserve model names and acronyms, expand vague "
        "language, and target the missing evidence. Return only the search query.\n"
        f"QUESTION: {state['standalone_question']}\nEVALUATOR GAP: {state.get('retrieval_reasoning', '')}"
    )
    try:
        query = _text(services.llm_factory().invoke(prompt)).strip() or state["standalone_question"]
        return _event(state, "Retrieval recovery query rewritten", retrieval_query=query, rewrite_query=query,
                      rewrite_purpose="recovery", retrieval_retry_count=state.get("retrieval_retry_count", 0) + 1,
                      iteration_count=state.get("iteration_count", 0) + 1,
                      llm_calls=state.get("llm_calls", 0) + 1)
    except Exception as exc:
        return _error(state, f"Recovery rewrite unavailable: {exc}", retrieval_query=state["standalone_question"],
                      retrieval_retry_count=state.get("retrieval_retry_count", 0) + 1,
                      iteration_count=state.get("iteration_count", 0) + 1)


def rewrite_for_web(state: GraphState, services: GraphServices) -> dict[str, Any]:
    prompt = (
        "Rewrite this as a concise search-engine query. Preserve PEFT technical entities, model names and "
        "acronyms; include missing context. Do not answer.\n"
        f"QUESTION: {state['standalone_question']}\nMISSING EVIDENCE: {state.get('missing_information') or state.get('retrieval_reasoning', '')}"
    )
    try:
        query = _text(services.llm_factory().invoke(prompt)).strip() or state["standalone_question"]
        return _event(state, "Web-search query rewritten", retrieval_query=query, rewrite_query=query,
                      rewrite_purpose="web", iteration_count=state.get("iteration_count", 0) + 1,
                      llm_calls=state.get("llm_calls", 0) + 1)
    except Exception as exc:
        return _error(state, f"Web rewrite unavailable: {exc}", retrieval_query=state["standalone_question"],
                      iteration_count=state.get("iteration_count", 0) + 1)


def web_search(state: GraphState, services: GraphServices) -> dict[str, Any]:
    if state.get("source_policy") == "corpus_only":
        return _event(state, "Web search skipped by corpus-only policy", web_documents=[])
    if not TAVILY_API_KEY:
        return _error(state, "Web search unavailable: TAVILY_API_KEY is not configured", web_documents=[])
    try:
        # langchain-tavily's current Runnable integration.  Results retain
        # URLs/content and are converted to normal Documents for downstream use.
        from langchain_tavily import TavilySearch

        result = TavilySearch(max_results=TAVILY_MAX_RESULTS).invoke({"query": state["retrieval_query"]})
        rows = result.get("results", result) if isinstance(result, dict) else result
        docs = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            content = row.get("content") or row.get("snippet") or ""
            if content:
                docs.append(Document(page_content=content, metadata={
                    "source_type": "web", "title": row.get("title", "Web source"),
                    "url": row.get("url", ""), "snippet": content[:500],
                }))
        return _event(state, f"Web search added {len(docs)} sources", web_documents=docs,
                      used_web_search=True)
    except Exception as exc:
        return _error(state, f"Web search failed: {exc}", web_documents=[], used_web_search=True)


def merge_knowledge(state: GraphState, services: GraphServices) -> dict[str, Any]:
    local = state.get("relevant_documents") or state.get("retrieved_documents", [])
    merged: list[Document] = []
    seen: set[tuple[str, str]] = set()
    for doc in [*local, *state.get("web_documents", [])]:
        meta = dict(doc.metadata)
        meta.setdefault("source_type", "local")
        key = (str(meta.get("url") or meta.get("chunk_id") or meta.get("paper_title")), doc.page_content[:180])
        if key not in seen:
            seen.add(key)
            merged.append(Document(page_content=doc.page_content, metadata=meta))
    return _event(state, f"Merged {len(local)} local and {len(state.get('web_documents', []))} web evidence items",
                  relevant_documents=[d for d in merged if d.metadata.get("source_type") == "local"],
                  web_documents=[d for d in merged if d.metadata.get("source_type") == "web"],
                  source_types=sorted({d.metadata.get("source_type", "local") for d in merged}))


def _strips(docs: list[Document]) -> list[Document]:
    output: list[Document] = []
    for doc_index, doc in enumerate(docs, 1):
        sentences = re.split(r"(?<=[.!?])\s+", doc.page_content.strip())
        for sentence_index, sentence in enumerate(sentences, 1):
            if len(sentence.strip()) < 25:
                continue
            meta = dict(doc.metadata)
            meta["strip_id"] = f"{document_id(doc, doc_index)}::s{sentence_index}"
            meta["source_document_id"] = document_id(doc, doc_index)
            output.append(Document(page_content=sentence.strip(), metadata=meta))
            if len(output) >= CRAG_MAX_STRIPS:
                return output
    return output


def refine_knowledge(state: GraphState, services: GraphServices) -> dict[str, Any]:
    docs = [*state.get("relevant_documents", []), *state.get("web_documents", [])]
    if not docs:
        return _event(state, "Knowledge refinement found no evidence", final_context=[])
    if (state.get("retrieval_confidence", 0.0) >= CRAG_SKIP_REFINEMENT_CONFIDENCE
            and len(docs) <= CRAG_SKIP_REFINEMENT_MAX_DOCS and not state.get("web_documents")):
        return _event(state, "Knowledge refinement skipped for compact high-confidence evidence", final_context=docs,
                      source_types=sorted({d.metadata.get("source_type", "local") for d in docs}))
    strips = _strips(docs)
    prompt = (
        "Select only evidence strips that help answer the question. Return IDs only through the schema; do not "
        "infer facts or select topically related but non-evidentiary material.\nQUESTION:\n"
        + state["standalone_question"] + "\n\nSTRIPS:\n" + "\n".join(
            f"{d.metadata['strip_id']}: {d.page_content}" for d in strips
        )
    )
    try:
        selected: StripSelection = _structured(services.judge_llm_factory(), StripSelection, prompt)
        ids = set(selected.relevant_strip_ids)
        kept = [d for d in strips if d.metadata["strip_id"] in ids]
        return _event(state, f"Knowledge refinement retained {len(kept)} evidence strips", final_context=kept,
                      source_types=sorted({d.metadata.get("source_type", "local") for d in kept}),
                      llm_calls=state.get("llm_calls", 0) + 1)
    except Exception as exc:
        # Keeping original passages is safer than losing all usable evidence.
        return _error(state, f"Knowledge refinement unavailable: {exc}", final_context=docs,
                      source_types=sorted({d.metadata.get("source_type", "local") for d in docs}))


def _render_context(docs: list[Document]) -> str:
    rendered = []
    for doc in docs:
        m = doc.metadata
        if m.get("source_type") == "web":
            header = f"[Web: {m.get('title', 'Web source')} | URL: {m.get('url', '')}]"
        else:
            header = f"[Source: {m.get('paper_title', 'Unknown Paper')} | Page {m.get('page_number', '?')}]"
        rendered.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(rendered) or "(No reliable evidence is available.)"


def generate_answer(state: GraphState, services: GraphServices) -> dict[str, Any]:
    feedback = "\nVerifier feedback to address: " + "; ".join(state.get("unsupported_claims", []) + state.get("missing_information", [])) if state.get("unsupported_claims") or state.get("missing_information") else ""
    prompt = (
        "You are a PEFT research assistant. Answer only from the supplied evidence. Do not invent facts, numbers "
        "or citations. Cite each important local claim as [Paper Title, Page X] and web claims as [Web: Source Title]. "
        "Clearly distinguish web from paper evidence. If evidence is insufficient, say so plainly.\n\n"
        f"QUESTION:\n{state['question']}\n\nEVIDENCE:\n{_render_context(state.get('final_context', []))}{feedback}"
    )
    try:
        answer = _text(services.llm_factory().invoke(prompt)).strip()
        return _event(state, f"Answer generated (attempt {state.get('generation_retry_count', 0) + 1})",
                      draft_answer=answer, llm_calls=state.get("llm_calls", 0) + 1)
    except Exception as exc:
        return _error(state, f"Answer generation failed: {exc}", draft_answer="I cannot answer because generation failed.")


def _programmatic_citations_valid(answer: str, docs: list[Document]) -> bool:
    if "cannot" in answer.lower() and not docs:
        return True
    local = {(str(d.metadata.get("paper_title", "")).lower(), str(d.metadata.get("page_number")))
             for d in docs if d.metadata.get("source_type", "local") == "local"}
    web = {str(d.metadata.get("title", "")).lower() for d in docs if d.metadata.get("source_type") == "web"}
    citations = re.findall(r"\[([^\[\]]+?)\]", answer)
    if not citations:
        return False
    for citation in citations:
        if citation.lower().startswith("web:"):
            title = citation.split(":", 1)[1].strip().lower()
            if not any(title in candidate or candidate in title for candidate in web):
                return False
        else:
            match = re.match(r"(.+?),\s*Page\s+(\d+)$", citation, re.I)
            if not match:
                return False
            title, page = match.group(1).lower(), match.group(2)
            if not any(title in candidate or candidate in title for candidate, p in local if p == page):
                return False
    return True


def verify_answer(state: GraphState, services: GraphServices) -> dict[str, Any]:
    context = _render_context(state.get("final_context", []))
    prompt = (
        "Verify the draft answer against only the evidence. Check grounding, completeness, answerability and that every "
        "citation names an available source. Choose accept, regenerate, web_search, or abstain. Do not expose reasoning "
        "beyond a concise operational rationale.\n\nQUESTION:\n" + state["question"] + "\n\nEVIDENCE:\n" + context
        + "\n\nDRAFT ANSWER:\n" + state.get("draft_answer", "")
    )
    try:
        grade: AnswerGrade = _structured(services.judge_llm_factory(), AnswerGrade, prompt)
        valid = grade.citation_valid and _programmatic_citations_valid(state.get("draft_answer", ""), state.get("final_context", []))
        if not valid:
            grade = grade.model_copy(update={"citation_valid": False,
                                             "action": "regenerate" if grade.answerable else "abstain"})
        updates = dict(answer_grade=grade, unsupported_claims=grade.unsupported_claims,
                       missing_information=grade.missing_information,
                       llm_calls=state.get("llm_calls", 0) + 1)
        if grade.action == "accept" and grade.grounded and grade.complete and valid:
            updates["final_answer"] = state.get("draft_answer", "")
        return _event(state, f"Self-verification: {grade.action.upper()}", **updates)
    except Exception as exc:
        grade = AnswerGrade(grounded=False, complete=False, answerable=False, citation_valid=False,
                            confidence=0.0, reasoning="Verifier unavailable.", action="abstain")
        return _error(state, f"Answer verifier unavailable: {exc}", answer_grade=grade)


def prepare_regeneration(state: GraphState, services: GraphServices) -> dict[str, Any]:
    return _event(state, "Regenerating with verifier feedback",
                  generation_retry_count=state.get("generation_retry_count", 0) + 1)


def abstain(state: GraphState, services: GraphServices) -> dict[str, Any]:
    reason = (state.get("answer_grade").reasoning if state.get("answer_grade") else state.get("retrieval_reasoning", ""))
    return _event(state, "Final route: ABSTAIN", final_answer=(
        "I cannot answer this reliably from the available evidence. "
        + (f"Reason: {reason}" if reason else "")
    ), routing_decision="abstain")


def final_sources(state: GraphState) -> list[dict[str, Any]]:
    output = []
    seen: set[tuple[str, str, str, str]] = set()
    for d in state.get("final_context", []):
        m = d.metadata
        source_type = m.get("source_type", "local")
        title = m.get("paper_title") or m.get("title", "Unknown source")
        key = (source_type, str(title), str(m.get("page_number", "")), str(m.get("url", "")))
        if key in seen:
            continue
        seen.add(key)
        output.append({"source_type": source_type, "title": title,
                       "file_name": m.get("file_name"), "page": m.get("page_number"), "chunk_id": m.get("chunk_id", ""),
                       "url": m.get("url", ""), "score": float(m.get("rrf_score", 0.0)),
                       "rank_dense": m.get("rank_dense"), "rank_sparse": m.get("rank_sparse"),
                       "snippet": d.page_content[:300]})
    return output
