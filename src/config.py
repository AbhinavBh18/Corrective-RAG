"""
Centralized configuration for the RAG pipeline.

All tunable parameters live here so experiments (chunk sizes, top-k, hybrid
weights) only require edits in one file.

CHANGED IN THIS REFACTOR
------------------------
* Added Qdrant settings (replaces ChromaDB).
* Added hybrid-retrieval settings (dense k, sparse k, RRF constant, weights).
* Added conversation-memory settings.
* Added evaluation settings (judge model, k-cutoffs).
* CHUNK_SIZE is now measured with the *embedding model's own tokenizer*
  instead of LlamaIndex's default token counter — see note below.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


EVAL_DIR = PROJECT_ROOT / "eval"
EVAL_RESULTS_DIR = EVAL_DIR / "results"

# ──────────────────────────────────────────────
# Gemini API
# ──────────────────────────────────────────────
# Credentials are deliberately environment-only.  Never add a fallback key
# here: source code is routinely copied into demos, notebooks, and forks.
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gemini-3.5-flash")

# NOTE: Google deprecated `temperature` / `top_p` / `top_k` on the newest
# Gemini 3.x endpoints. Set LLM_TEMPERATURE to an empty string in .env to omit
# the parameter entirely if your model rejects it.
_raw_temp = os.getenv("LLM_TEMPERATURE", "0.1")
LLM_TEMPERATURE: float | None = float(_raw_temp) if _raw_temp.strip() else None

# A separate (cheap, deterministic) model used as the LLM-as-judge in Phase 5.
# Keeping the judge distinct from the generator avoids self-preference bias.
JUDGE_MODEL_NAME: str = os.getenv("JUDGE_MODEL_NAME", "gemini-3.5-flash")
JUDGE_TEMPERATURE: float | None = 0.0

# ──────────────────────────────────────────────
# Embedding Model (local — sentence-transformers)
# ──────────────────────────────────────────────
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384  # Output dimension for all-MiniLM-L6-v2

# ──────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────
# IMPORTANT FIX: all-MiniLM-L6-v2 has a hard max sequence length of 256 tokens.
# The old config used 512-token chunks, which meant roughly the second half of
# every chunk was silently truncated before embedding — the text was retrievable
# only by whatever happened to be in its first 256 tokens. We now chunk at 256
# tokens using the model's own tokenizer so nothing is dropped.
CHUNK_SIZE: int = 256   # tokens per chunk (measured with the MiniLM tokenizer)
CHUNK_OVERLAP: int = 40  # token overlap between consecutive chunks

# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────
TOP_K: int = 5           # final number of chunks handed to the LLM
DENSE_TOP_K: int = 10    # candidates pulled from Qdrant before fusion
SPARSE_TOP_K: int = 10   # candidates pulled from BM25 before fusion

# Reciprocal Rank Fusion constant. Standard value from the original RRF paper
# (Cormack et al., 2009). Larger k => flatter curve => deep ranks matter more.
RRF_K: int = 60

# Relative trust in each retriever during fusion. Dense is weighted slightly
# higher because paraphrased questions ("how does it save memory?") do not
# share vocabulary with the papers; BM25 earns its keep on exact identifiers
# ("NF4", "rank r=8", "IA3").
HYBRID_DENSE_WEIGHT: float = 0.6
HYBRID_SPARSE_WEIGHT: float = 0.4

RETRIEVAL_MODE: str = os.getenv("RETRIEVAL_MODE", "hybrid")  # hybrid | dense | sparse

# ──────────────────────────────────────────────
# Adaptive LangGraph / CRAG orchestration
# ──────────────────────────────────────────────
# These are evaluator signals, not calibrated probabilities and never raw RRF
# scores.  Tune them with the evaluation suite, not by interpreting RRF as a
# confidence value.
CRAG_CANDIDATE_K: int = int(os.getenv("CRAG_CANDIDATE_K", "12"))
CRAG_CORRECT_THRESHOLD: float = float(os.getenv("CRAG_CORRECT_THRESHOLD", "0.70"))
CRAG_INCORRECT_THRESHOLD: float = float(os.getenv("CRAG_INCORRECT_THRESHOLD", "0.30"))
CRAG_SKIP_REFINEMENT_CONFIDENCE: float = float(
    os.getenv("CRAG_SKIP_REFINEMENT_CONFIDENCE", "0.85")
)
CRAG_SKIP_REFINEMENT_MAX_DOCS: int = int(
    os.getenv("CRAG_SKIP_REFINEMENT_MAX_DOCS", "4")
)
CRAG_MAX_STRIPS: int = int(os.getenv("CRAG_MAX_STRIPS", "48"))

MAX_RETRIEVAL_RETRIES: int = int(os.getenv("MAX_RETRIEVAL_RETRIES", "1"))
MAX_GENERATION_RETRIES: int = int(os.getenv("MAX_GENERATION_RETRIES", "1"))
MAX_TOTAL_ITERATIONS: int = int(os.getenv("MAX_TOTAL_ITERATIONS", "3"))
DEFAULT_SOURCE_POLICY: str = os.getenv("SOURCE_POLICY", "adaptive")  # adaptive | corpus_only
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
TAVILY_MAX_RESULTS: int = int(os.getenv("TAVILY_MAX_RESULTS", "4"))

# ──────────────────────────────────────────────
# Qdrant (replaces ChromaDB; chosen over Pinecone for Python 3.14 support)
# ──────────────────────────────────────────────
# Qdrant runs in two modes and the code supports both:
#   * EMBEDDED  (QDRANT_URL empty) — qdrant-client writes to a local folder.
#     No server, no Docker, no API key. This is the default.
#   * SERVER    (QDRANT_URL set)   — a real Qdrant instance, either
#     `docker run -p 6333:6333 qdrant/qdrant` or Qdrant Cloud.
# The only reason to move to server mode is concurrency — see the note in
# indexing.py about the embedded-mode directory lock.
QDRANT_URL: str = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")  # Qdrant Cloud only
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant_db"      # embedded-mode storage
QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "peft_papers")
QDRANT_DISTANCE: str = "COSINE"
UPSERT_BATCH_SIZE: int = 100

# ──────────────────────────────────────────────
# Conversation memory (session-scoped only)
# ──────────────────────────────────────────────
# Number of past *turns* (1 turn = user msg + assistant msg) kept in context.
# Bounded so the prompt cannot grow without limit over a long chat.
MAX_HISTORY_TURNS: int = 6

# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
EVAL_K_CUTOFFS: tuple[int, ...] = (1, 3, 5, 10)
EVAL_RETRIEVAL_CANDIDATES: int = 10  # depth to score retrieval metrics at
EVAL_MODES: tuple[str, ...] = ("dense", "sparse", "hybrid")
