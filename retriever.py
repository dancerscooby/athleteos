"""
AthleteOS - Retrieval (RAG stage, Step 3 - FAISS-backed, now with reranking).

Given a text query, finds the most relevant chunks in two passes:
  1. FAISS (unchanged from the previous step) quickly retrieves a broad set
     of FAISS_CANDIDATE_COUNT candidates using the embedding model
     (all-MiniLM-L6-v2, unchanged).
  2. The reranker (reranker.py) re-scores those candidates more precisely
     and narrows them down to the best `top_k`.

The PUBLIC INTERFACE is UNCHANGED on purpose, so every caller keeps working
with zero changes:

    retrieve_relevant_chunks(query, top_k=3)
        -> [{"id", "source", "text", "similarity"}, ...]   (best match first)

rag.py, llm_provider.py, and tests/test_scenarios.py only ever call this one
function - none of them need to know retrieval now involves two passes.

Embedding runtime (deployment change): loads the SAME model
(sentence-transformers/all-MiniLM-L6-v2) via `fastembed` (ONNX Runtime)
instead of the `sentence-transformers`/torch package. This was a deliberate
swap to fit Render's free-tier 512MB RAM limit - torch + sentence-
transformers alone cost ~360MB of resident memory before handling a single
request, which left no real headroom; fastembed's onnxruntime-based
pipeline costs a fraction of that. Verified before switching, not assumed:
embedding the same text with both libraries produces vectors with cosine
similarity 1.0 (numerically identical, within float32 precision), so
embeddings.json and faiss_index.bin below did NOT need to be regenerated -
they were built with the old library and remain valid with the new one.

Requires:
  - embeddings.json (see embeddings.py) - still the source of chunk
    id/source/text; UNCHANGED, still produced the same way it always was.
  - faiss_index.bin (see build_faiss_index.py) - the FAISS vector index,
    UNCHANGED from the previous step.
  - reranker.py - the cross-encoder reranking step (also fastembed-backed).
  - the 'fastembed' and 'faiss-cpu' packages.

If faiss_index.bin is missing or out of date, build/rebuild it with:
    python3 build_faiss_index.py

Run this file directly with:
    python3 retriever.py
"""

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from fastembed import TextEmbedding

from reranker import rerank

# Full fastembed model id (not the short "all-MiniLM-L6-v2" alias) - must
# match embeddings.py / build_faiss_index.py's model choice, since this
# name is what selects which ONNX weights fastembed downloads/loads.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS_FILE = Path(__file__).parent / "embeddings.json"
FAISS_INDEX_FILE = Path(__file__).parent / "faiss_index.bin"
FAISS_CANDIDATE_COUNT = 10  # how many candidates FAISS retrieves before reranking narrows them down

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("athleteos.retriever")

_model = None    # embedding model, loaded once and reused (see load_model())
_index = None    # FAISS index, loaded once and reused (see load_faiss_index())
_records = None  # chunk id/source/text records, loaded once and reused (see load_embeddings())


def load_model(model_name=MODEL_NAME):
    """Load the embedding model once and cache it for reuse."""
    global _model
    if _model is None:
        logger.info("Loading embedding model '%s'...", model_name)
        _model = TextEmbedding(model_name)
        logger.info("Embedding model loaded.")
    return _model


def load_embeddings(path=EMBEDDINGS_FILE):
    """
    Load the chunk records (id/source/text/embedding) from disk. Still used
    to map a FAISS result row back to its id/source/text - cached so
    repeated queries don't re-read the file every time.
    """
    global _records
    if _records is None:
        logger.info("Loading chunk records from %s", path)
        with open(path, encoding="utf-8") as f:
            _records = json.load(f)
        logger.info("Loaded %d chunk records.", len(_records))
    return _records


def load_faiss_index(path=FAISS_INDEX_FILE):
    """Load the local FAISS index once and cache it for reuse."""
    global _index
    if _index is None:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"FAISS index not found at {path}. Build it first with: "
                f"python3 build_faiss_index.py"
            )
        logger.info("Loading FAISS index from %s", path)
        _index = faiss.read_index(str(path))
        logger.info(
            "FAISS index loaded: %d vectors, dimension %d", _index.ntotal, _index.d
        )
    return _index


def search_faiss_only(query, top_k=3):
    """
    Raw FAISS search - UNCHANGED from the previous step, no reranking.

    Kept as its own function for two reasons: (1) retrieve_relevant_chunks()
    below reuses it as the first pass, and (2) it's used as the "before
    reranking" baseline in benchmark_reranking.py, so the two can be
    compared fairly using the exact same search code.
    """
    model = load_model()
    index = load_faiss_index()
    records = load_embeddings()

    # fastembed's embed() takes an iterable of texts and returns a generator
    # of numpy arrays, one per input - pass a single-item list, take the
    # first (only) result.
    query_vector = np.array(next(iter(model.embed([query]))), dtype="float32").reshape(1, -1)
    faiss.normalize_L2(query_vector)  # unit-normalize -> inner product = cosine similarity

    similarities, row_indices = index.search(query_vector, top_k)

    results = []
    for similarity, row in zip(similarities[0], row_indices[0]):
        if row == -1:  # FAISS pads with -1 if the index has fewer than top_k vectors
            continue
        record = records[row]
        results.append({
            "id": record["id"],
            "source": record["source"],
            "text": record["text"],
            "similarity": float(similarity),
        })

    return results


def retrieve_relevant_chunks(query, top_k=3):
    """
    Two-pass retrieval:
      1. FAISS retrieves FAISS_CANDIDATE_COUNT candidates (broad, fast).
      2. The reranker narrows them down to the best top_k (precise, slower,
         but only run over the small candidate set).

    Returns a list of dicts: {"id", "source", "text", "similarity"} sorted
    from most to least relevant after reranking - identical shape to the
    previous FAISS-only implementation, so callers are unaffected.
    """
    logger.info("Query: %s", query)

    candidate_k = max(FAISS_CANDIDATE_COUNT, top_k)  # never fetch fewer than requested
    candidates = search_faiss_only(query, top_k=candidate_k)
    logger.info("STAGE A - FAISS retrieval: %d candidates", len(candidates))
    for rank, c in enumerate(candidates, start=1):
        logger.info(
            "  faiss candidate #%d: %s (source=%s, similarity=%.3f)",
            rank, c["id"], c["source"], c["similarity"],
        )

    logger.info("STAGE B - Reranking %d candidates down to top %d", len(candidates), top_k)
    results = rerank(query, candidates, top_k=top_k)  # logs each rerank score itself

    logger.info("Returning %d chunks after reranking", len(results))
    return results


# ----------------------------------------------------------------------
# TEST SECTION  -  run with: python retriever.py
# ----------------------------------------------------------------------

if __name__ == "__main__":
    test_queries = [
        "What should I eat before a football match?",
        "How much water should I drink during training?",
        "cheap vegetarian protein sources in India",
    ]

    for query in test_queries:
        print("=" * 64)
        print(f"QUERY: {query}")
        print("=" * 64)
        results = retrieve_relevant_chunks(query, top_k=3)
        for rank, result in enumerate(results, start=1):
            preview = result["text"][:150].replace("\n", " ")
            print(f"\n#{rank}  similarity={result['similarity']:.3f}  "
                  f"source={result['source']}  id={result['id']}")
            print(f"    {preview}...")
        print()
