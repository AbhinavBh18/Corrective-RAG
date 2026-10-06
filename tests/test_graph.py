"""Fast, dependency-injected safety tests for adaptive graph routing."""

from __future__ import annotations

import unittest

from langchain_core.documents import Document

from src.graph.builder import build_adaptive_rag_graph, initial_state
from src.graph.nodes import GraphServices, _programmatic_citations_valid, contextualize_query, verify_answer
from src.graph.routers import answer_route, retrieval_route
from src.graph.state import AnswerGrade, RetrievalGrade, StripSelection


class _Reply:
    def __init__(self, content: str): self.content = content


class _Structured:
    def __init__(self, schema): self.schema = schema
    def invoke(self, prompt):
        if self.schema is RetrievalGrade:
            return RetrievalGrade(verdict="correct", confidence=0.95, relevant_document_ids=["lora-1"],
                                  reasoning="Direct evidence is present.", needs_web=False)
        if self.schema is StripSelection:
            return StripSelection(relevant_strip_ids=[])
        return AnswerGrade(grounded=True, complete=True, answerable=True, citation_valid=True,
                           confidence=0.95, reasoning="Supported.", action="accept")


class _LLM:
    def invoke(self, prompt): return _Reply("LoRA freezes base weights. [LoRA, Page 1]")
    def with_structured_output(self, schema): return _Structured(schema)


class _Retriever:
    def invoke(self, query):
        return [Document(page_content="LoRA freezes the pretrained weights and learns low-rank updates.", metadata={
            "chunk_id": "lora-1", "paper_title": "LoRA", "page_number": 1, "file_name": "lora.pdf",
        })]


class GraphTests(unittest.TestCase):
    def test_local_evidence_path_accepts_without_web(self):
        services = GraphServices(llm_factory=_LLM, judge_llm_factory=_LLM,
                                 retriever_factory=lambda **_: _Retriever())
        result = build_adaptive_rag_graph(services=services).invoke(
            initial_state("What does LoRA train?", source_policy="corpus_only")
        )
        self.assertFalse(result["used_web_search"])
        self.assertIn("[LoRA, Page 1]", result["final_answer"])
        self.assertEqual(result["retrieval_verdict"], "correct")

    def test_irrelevant_retrieval_has_only_one_recovery(self):
        self.assertEqual(retrieval_route({"retrieval_verdict": "incorrect", "source_policy": "adaptive",
                                          "retrieval_retry_count": 0, "iteration_count": 0}), "recover")
        self.assertEqual(retrieval_route({"retrieval_verdict": "incorrect", "source_policy": "adaptive",
                                          "retrieval_retry_count": 1, "iteration_count": 1}), "web")

    def test_ambiguous_evidence_preserves_route_to_web(self):
        self.assertEqual(retrieval_route({"retrieval_verdict": "ambiguous", "source_policy": "adaptive",
                                          "iteration_count": 0}), "web")

    def test_corpus_only_never_routes_to_web(self):
        self.assertEqual(retrieval_route({"retrieval_verdict": "incorrect", "source_policy": "corpus_only",
                                          "retrieval_retry_count": 1, "iteration_count": 1}), "refine")

    def test_answer_repair_is_bounded(self):
        grade = AnswerGrade(grounded=False, complete=False, answerable=True, citation_valid=False,
                            confidence=0.3, reasoning="Unsupported", action="regenerate")
        self.assertEqual(answer_route({"answer_grade": grade, "generation_retry_count": 0}), "regenerate")
        self.assertEqual(answer_route({"answer_grade": grade, "generation_retry_count": 1}), "abstain")

    def test_verifier_can_require_web_or_abstention(self):
        grade = AnswerGrade(grounded=False, complete=False, answerable=False, citation_valid=False,
                            confidence=0.1, reasoning="Missing evidence", action="web_search")
        self.assertEqual(answer_route({"answer_grade": grade, "source_policy": "adaptive", "iteration_count": 0}), "web")
        self.assertEqual(answer_route({"answer_grade": grade, "source_policy": "corpus_only", "iteration_count": 0}), "abstain")

    def test_outside_corpus_routes_to_web_after_recovery_budget(self):
        self.assertEqual(retrieval_route({"retrieval_verdict": "incorrect", "source_policy": "adaptive",
                                          "retrieval_retry_count": 1, "iteration_count": 1}), "web")

    def test_follow_up_is_contextualized_before_retrieval(self):
        class RewriteLLM:
            def invoke(self, prompt): return _Reply("What is QLoRA's memory usage?")
        services = GraphServices(llm_factory=RewriteLLM, judge_llm_factory=_LLM,
                                 retriever_factory=lambda **_: _Retriever())
        from langchain_core.messages import HumanMessage
        result = contextualize_query({"question": "What about its memory usage?", "chat_history": [
            HumanMessage(content="Tell me about QLoRA"),
        ], "trace": [], "errors": [], "llm_calls": 0}, services)
        self.assertEqual(result["standalone_question"], "What is QLoRA's memory usage?")

    def test_invalid_citation_is_rejected_against_graph_context(self):
        docs = [Document(page_content="Evidence", metadata={"paper_title": "LoRA", "page_number": 1})]
        self.assertFalse(_programmatic_citations_valid("Claim [LoRA, Page 9]", docs))

    def test_unsupported_draft_requires_repair(self):
        class RejectJudge:
            def with_structured_output(self, schema):
                class Call:
                    def invoke(self, prompt):
                        return AnswerGrade(grounded=False, complete=False, answerable=True, citation_valid=True,
                                           confidence=0.2, unsupported_claims=["invented number"],
                                           missing_information=[], reasoning="Unsupported number.", action="regenerate")
                return Call()
        services = GraphServices(llm_factory=_LLM, judge_llm_factory=RejectJudge,
                                 retriever_factory=lambda **_: _Retriever())
        result = verify_answer({"question": "What does LoRA train?", "draft_answer": "It saves 99%. [LoRA, Page 1]",
                                "final_context": [Document(page_content="LoRA trains low-rank updates.", metadata={
                                    "paper_title": "LoRA", "page_number": 1
                                })], "llm_calls": 0, "trace": [], "errors": []}, services)
        self.assertEqual(result["answer_grade"].action, "regenerate")


if __name__ == "__main__":
    unittest.main()
