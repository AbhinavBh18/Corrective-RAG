"""Pure routing functions. Keeping them separate makes loop bounds testable."""

from __future__ import annotations

from src.config import MAX_GENERATION_RETRIES, MAX_RETRIEVAL_RETRIES, MAX_TOTAL_ITERATIONS
from src.graph.state import GraphState


def retrieval_route(state: GraphState) -> str:
    verdict = state.get("retrieval_verdict", "incorrect")
    policy = state.get("source_policy", "adaptive")
    retries = state.get("retrieval_retry_count", 0)
    iterations = state.get("iteration_count", 0)
    if verdict == "correct":
        return "refine"
    if verdict == "incorrect" and retries < MAX_RETRIEVAL_RETRIES and iterations < MAX_TOTAL_ITERATIONS:
        return "recover"
    if policy == "adaptive" and iterations < MAX_TOTAL_ITERATIONS:
        return "web"
    return "refine"


def answer_route(state: GraphState) -> str:
    grade = state.get("answer_grade")
    if grade is None:
        return "abstain"
    if grade.action == "accept" and grade.grounded and grade.complete and grade.citation_valid:
        return "accept"
    if grade.action == "regenerate" and state.get("generation_retry_count", 0) < MAX_GENERATION_RETRIES:
        return "regenerate"
    if (
        grade.action == "web_search"
        and state.get("source_policy") == "adaptive"
        and state.get("iteration_count", 0) < MAX_TOTAL_ITERATIONS
    ):
        return "web"
    return "abstain"
