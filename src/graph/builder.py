"""Construction and visualization helpers for the bounded adaptive RAG graph."""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.config import DEFAULT_SOURCE_POLICY, RETRIEVAL_MODE
from src.graph import nodes
from src.graph.nodes import GraphServices
from src.graph.routers import answer_route, retrieval_route
from src.graph.state import GraphState
from src.pipeline_support import get_judge_llm, get_llm


def default_services(mode: str = RETRIEVAL_MODE) -> GraphServices:
    """Production dependencies; tests can inject deterministic alternatives."""
    # Lazy import keeps graph construction and Streamlit startup independent
    # from Qdrant/HuggingFace loading until a user actually submits a query.
    def build_retriever(**kwargs: Any):
        from src.retrieval import get_retriever
        return get_retriever(mode=mode, **kwargs)

    return GraphServices(
        llm_factory=get_llm,
        judge_llm_factory=get_judge_llm,
        retriever_factory=build_retriever,
    )


def build_adaptive_rag_graph(services: GraphServices | None = None, mode: str = RETRIEVAL_MODE):
    services = services or default_services(mode)
    graph = StateGraph(GraphState)
    graph.add_node("contextualize", partial(nodes.contextualize_query, services=services))
    graph.add_node("retrieve", partial(nodes.retrieve, services=services))
    graph.add_node("grade_retrieval", partial(nodes.grade_retrieval, services=services))
    graph.add_node("recover_retrieval", partial(nodes.rewrite_for_recovery, services=services))
    graph.add_node("rewrite_for_web", partial(nodes.rewrite_for_web, services=services))
    graph.add_node("web_search", partial(nodes.web_search, services=services))
    graph.add_node("merge_knowledge", partial(nodes.merge_knowledge, services=services))
    graph.add_node("refine_knowledge", partial(nodes.refine_knowledge, services=services))
    graph.add_node("generate", partial(nodes.generate_answer, services=services))
    graph.add_node("verify", partial(nodes.verify_answer, services=services))
    graph.add_node("regenerate", partial(nodes.prepare_regeneration, services=services))
    graph.add_node("abstain", partial(nodes.abstain, services=services))

    graph.add_edge(START, "contextualize")
    graph.add_edge("contextualize", "retrieve")
    graph.add_edge("retrieve", "grade_retrieval")
    graph.add_conditional_edges("grade_retrieval", retrieval_route, {
        "refine": "refine_knowledge", "recover": "recover_retrieval", "web": "rewrite_for_web",
    })
    graph.add_edge("recover_retrieval", "retrieve")
    graph.add_edge("rewrite_for_web", "web_search")
    graph.add_edge("web_search", "merge_knowledge")
    graph.add_edge("merge_knowledge", "refine_knowledge")
    graph.add_edge("refine_knowledge", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", answer_route, {
        "accept": END, "regenerate": "regenerate", "web": "rewrite_for_web", "abstain": "abstain",
    })
    graph.add_edge("regenerate", "generate")
    graph.add_edge("abstain", END)
    return graph.compile()


def initial_state(question: str, chat_history: list[Any] | None = None,
                  source_policy: str = DEFAULT_SOURCE_POLICY) -> GraphState:
    policy = source_policy if source_policy in {"adaptive", "corpus_only"} else DEFAULT_SOURCE_POLICY
    return {
        "question": question, "chat_history": list(chat_history or []), "standalone_question": question,
        "retrieval_query": question, "retrieved_documents": [], "relevant_documents": [], "web_documents": [],
        "final_context": [], "retrieval_retry_count": 0, "generation_retry_count": 0,
        "iteration_count": 0, "used_web_search": False, "source_policy": policy,
        "trace": [], "errors": [], "llm_calls": 0, "source_types": [], "sources": [],
    }


def graph_mermaid() -> str:
    """Export a demo-friendly Mermaid diagram without requiring graphviz."""
    return """flowchart TD
    A[User query] --> B[Contextualize query]
    B --> C[Hybrid dense + BM25 + RRF retrieval]
    C --> D[Structured retrieval evaluator]
    D -->|correct| H[Knowledge refinement]
    D -->|incorrect; retry available| E[Recovery query rewrite]
    E --> C
    D -->|ambiguous/insufficient| F[Web query rewrite]
    F --> G[Conditional Tavily search]
    G --> M[Merge local + web evidence]
    M --> H
    H --> I[Grounded answer generation]
    I --> J[Structured self-verification]
    J -->|accept| K[End]
    J -->|regenerate; bounded| I
    J -->|need web; bounded| F
    J -->|abstain/exhausted| L[Transparent abstention]
    L --> K"""
