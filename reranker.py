"""
AthleteOS - Reranking layer (RAG retrieval upgrade).

Adds a second, more precise scoring pass on top of FAISS's fast broad
search. The existing embedding model (all-MiniLM-L6-v2, UNCHANGED) is a
"bi-encoder": it embeds the query and every chunk SEPARATELY, so comparing
them is just a fast vector-math lookup (what FAISS does). A "cross-encoder"
instead reads the query and ONE candidate chunk TOGETHER in a single pass,
which lets it actually weigh how well they relate to each other - much more
accurate, but too slow to run against all 39 chunks on every request. So the
two are combined: FAISS retrieves a broad set of candidates quickly, the
cross-encoder re-scores just that small set precisely.

Reranking runtime (deployment change): uses fastembed's TextCrossEncoder
(ONNX Runtime) with 'Xenova/ms-marco-MiniLM-L-6-v2' - the ONNX port of the
EXACT SAME model this file used before ('cross-encoder/ms-marco-MiniLM-L-
6-v2'), not a different reranker. This was part of the same memory-driven
swap described in retriever.py's docstring (fastembed instead of torch/
sentence-transformers, to fit Render's 512MB free-tier limit). Verified
before switching: scoring the same (query, candidates) with both the old
and new model produced numerically matching scores and an IDENTICAL
ranking order, so reranking QUALITY is unchanged, only the runtime under
it is lighter.
"""

import logging

from fastembed.rerank.cross_encoder import TextCrossEncoder

RERANKER_MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("athleteos.reranker")

_reranker = None  # loaded once and reused (see load_reranker())


def load_reranker(model_name=RERANKER_MODEL_NAME):
    """Load the cross-encoder reranking model once and cache it for reuse."""
    global _reranker
    if _reranker is None:
        logger.info("Loading reranker model '%s'...", model_name)
        _reranker = TextCrossEncoder(model_name)
        logger.info("Reranker model loaded.")
    return _reranker


def rerank(query, candidates, top_k=3):
    """
    Re-score `candidates` (chunk dicts as returned by FAISS search) against
    `query` using the cross-encoder, and return the best `top_k`.

    IMPORTANT for backward compatibility: the returned dicts keep the EXACT
    SAME shape as the input candidates ({"id","source","text","similarity"}).
    The "similarity" field still holds the ORIGINAL FAISS cosine similarity,
    completely unchanged - reranking only affects WHICH candidates are kept
    and in WHAT ORDER, never the meaning of any existing field. This is what
    preserves the existing output format exactly.
    """
    if not candidates:
        return []

    model = load_reranker()
    documents = [c["text"] for c in candidates]
    logger.info("Reranking %d candidates for query: %s", len(candidates), query)

    # fastembed's rerank() takes the query once plus a list of documents
    # (not (query, doc) pairs) and returns one score per document, in the
    # same order as `documents` - higher score = more relevant, same
    # convention the previous sentence-transformers CrossEncoder used.
    scores = list(model.rerank(query, documents))

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    for rank, (candidate, score) in enumerate(scored, start=1):
        logger.info(
            "  rerank #%d: %s (rerank_score=%.3f, faiss_similarity=%.3f)",
            rank, candidate["id"], score, candidate["similarity"],
        )

    return [candidate for candidate, _score in scored[:top_k]]
