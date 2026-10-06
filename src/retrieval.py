"""
Phase 3b (NEW): Hybrid Retrieval

The old pipeline used dense vector search only. Dense embeddings are great at
paraphrase ("how does it cut memory use?" -> a passage about GPU footprint) but
weak at rare literal tokens — model names, hyperparameters, acronyms like
"NF4", "IA3", "r=8". BM25 is the mirror image: excellent on exact terms, blind
to paraphrase. Running both and fusing the rankings gets the strengths of each.

FUSION STRATEGY: Weighted Reciprocal Rank Fusion (RRF)
------------------------------------------------------
    score(doc) = Σ_retriever  weight_r * 1 / (RRF_K + rank_r(doc))

Why RRF instead of just adding the raw scores together?
  * Cosine similarity (0-1, tightly clustered around 0.3-0.7) and BM25 scores
    (unbounded, corpus-dependent, often 0-30) live on completely different
    scales. Summing them means BM25 silently dominates.
  * Min-max normalising per query is unstable — one outlier rescales everything.
  * RRF throws away the magnitudes and keeps only the ordering, which is the
    part both retrievers actually agree on the meaning of. It needs no tuning
    and no training data.
  * A document ranked highly by BOTH retrievers accumulates from both terms, so
    consensus results naturally float to the top.

RRF_K (default 60) damps the head of the curve: with K=60, rank 1 scores
1/61 and rank 2 scores 1/62 — close together — so a document only wins by
appearing in several lists, not by narrowly topping one.

IMPLEMENTATION NOTE
-------------------
LangChain ships `EnsembleRetriever`, which does essentially this. We implement
it explicitly here for two reasons: (1) it moved packages between LangChain
0.3 and 1.x (`langchain.retrievers` -> `langchain_classic.retrievers`), and
(2) writing it out lets us attach per-retriever ranks to each document's
metadata, which the evaluation harness uses to attribute wins to dense vs
sparse. It is ~40 lines and removes a moving dependency.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.config import (
    DENSE_TOP_K,
    HYBRID_DENSE_WEIGHT,
    HYBRID_SPARSE_WEIGHT,
    RRF_K,
    SPARSE_TOP_K,
    TOP_K,
    QDRANT_COLLECTION_NAME,
)
from src.indexing import get_vector_store, get_qdrant_client

logger = logging.getLogger(__name__)

# BM25 is built in-process from the chunk store; cache it so we don't rebuild
# the term statistics on every Streamlit rerun.
_bm25_singleton: BaseRetriever | None = None


def _doc_key(doc: Document) -> str:
    """Stable identity for a chunk, used to dedupe across the two retrievers."""
    return doc.metadata.get("chunk_id") or f"hash::{hash(doc.page_content)}"

#fusion
class HybridRRFRetriever(BaseRetriever):
    retrievers: list[BaseRetriever]
    names: list[str]
    weights: list[float]
    top_k: int = TOP_K
    rrf_k: int = RRF_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        fused_scores: dict[str, float] = {}
        best_doc: dict[str, Document] = {}
        rank_info: dict[str, dict[str, int]] = {}

        for retriever, name, weight in zip(self.retrievers, self.names, self.weights):
            try:
                docs = retriever.invoke(query)
            except Exception as exc:
                logger.warning(f"Retriever '{name}' failed: {exc}")
                continue

            for rank, doc in enumerate(docs, start=1):
                key = _doc_key(doc)
                fused_scores[key] = fused_scores.get(key, 0.0) + weight / (
                    self.rrf_k + rank
                )
                rank_info.setdefault(key, {})[f"rank_{name}"] = rank
                if key not in best_doc:
                    best_doc[key] = Document(
                        page_content=doc.page_content, metadata=dict(doc.metadata)
                    )

        ordered = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)

        results: list[Document] = []
        for position, (key, score) in enumerate(ordered[: self.top_k], start=1):
            doc = best_doc[key]
            doc.metadata["rrf_score"] = round(score, 6)
            doc.metadata["final_rank"] = position
            doc.metadata.update(rank_info.get(key, {}))
            results.append(doc)

        return results


def build_dense_retriever(k: int = DENSE_TOP_K) -> BaseRetriever:
    return get_vector_store().as_retriever(search_kwargs={"k": k})


def load_corpus_from_qdrant() -> list[Document]:
    client = get_qdrant_client()
    docs = []
    next_page_offset = None
    
    if not client.collection_exists(QDRANT_COLLECTION_NAME):
        return []

    while True:
        records, next_page_offset = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=1000,
            with_payload=True,
            with_vectors=False,
            offset=next_page_offset
        )
        for r in records:
            payload = r.payload or {}
            # Langchain Qdrant uses page_content and metadata keys by default
            content = payload.get("page_content", "")
            metadata = payload.get("metadata", {})
            docs.append(Document(page_content=content, metadata=metadata))
        if next_page_offset is None:
            break
    logger.info(f"Loaded {len(docs)} chunks from Qdrant payloads for BM25.")
    return docs


def build_sparse_retriever(
    k: int = SPARSE_TOP_K, docs: Iterable[Document] | None = None
) -> BaseRetriever:
    global _bm25_singleton
    if _bm25_singleton is not None and docs is None:
        _bm25_singleton.k = k
        return _bm25_singleton

    from langchain_community.retrievers import BM25Retriever  # needs `rank_bm25`

    corpus = list(docs) if docs is not None else load_corpus_from_qdrant()
    if not corpus:
        corpus = [Document(page_content="empty", metadata={})]

    retriever = BM25Retriever.from_documents(corpus)
    retriever.k = k
    logger.info(f"Built BM25 index over {len(corpus)} chunks.")

    if docs is None:
        _bm25_singleton = retriever
    return retriever


def get_retriever(
    mode: str = "hybrid",
    top_k: int = TOP_K,
    dense_k: int = DENSE_TOP_K,
    sparse_k: int = SPARSE_TOP_K,
) -> BaseRetriever:
    mode = mode.lower()

    if mode == "dense":
        return build_dense_retriever(k=top_k)
    if mode == "sparse":
        return build_sparse_retriever(k=top_k)
    if mode == "hybrid":
        return HybridRRFRetriever(
            retrievers=[build_dense_retriever(dense_k), build_sparse_retriever(sparse_k)],
            names=["dense", "sparse"],
            weights=[HYBRID_DENSE_WEIGHT, HYBRID_SPARSE_WEIGHT],
            top_k=top_k,
        )

    raise ValueError(f"Unknown retrieval mode '{mode}'. Use dense | sparse | hybrid.")


# ──────────────────────────────────────────────
# Standalone verification
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("\nHybrid Retrieval Check")
    print("=" * 60)

    query = "What quantization data type does QLoRA introduce?"
    print(f"Query: {query}\n")

    for mode in ("dense", "sparse", "hybrid"):
        print(f"--- {mode.upper()} ---")
        try:
            docs = get_retriever(mode=mode).invoke(query)
            for i, d in enumerate(docs, 1):
                extra: dict[str, Any] = {
                    k: v for k, v in d.metadata.items() if k.startswith(("rank_", "rrf_"))
                }
                print(
                    f"  {i}. {d.metadata.get('file_name')} "
                    f"p{d.metadata.get('page_number')} {extra}"
                )
        except Exception as e:
            print(f"  Failed: {e}")
        print()
