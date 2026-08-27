"""
AthleteOS - FAISS index builder (RAG retrieval upgrade).

Builds a local FAISS vector index from embeddings.json (produced by
embeddings.py - completely unchanged) so retriever.py can search using FAISS
instead of a manual cosine-similarity loop. ONLY the search mechanism
changes: the embedding model (all-MiniLM-L6-v2), the chunking pipeline
(ingest.py), and embeddings.json's format are all untouched.

Why this needs to be a separate, one-time-ish script rather than something
retriever.py does on every request: building the index is a bulk operation
over every chunk (fine to redo occasionally, wasteful to redo on every
question), while searching it is a per-query operation. Keeping "build" and
"search" separate mirrors the existing ingest.py / embeddings.py / retriever.py
split, where each file has one job.

Run with:
    python3 build_faiss_index.py
whenever embeddings.json changes (i.e. after editing knowledge_base/*.md and
re-running ingest.py + embeddings.py).
"""

import json
import logging
from pathlib import Path

import faiss
import numpy as np

EMBEDDINGS_FILE = Path(__file__).parent / "embeddings.json"
FAISS_INDEX_FILE = Path(__file__).parent / "faiss_index.bin"  # stored locally, next to embeddings.json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("athleteos.faiss_index")


def build_index(embeddings_path=EMBEDDINGS_FILE, index_path=FAISS_INDEX_FILE):
    """
    Read every chunk's embedding from embeddings.json, build a FAISS index
    over them, and save it to disk at `index_path`.
    """
    logger.info("Loading chunk embeddings from %s", embeddings_path)
    with open(embeddings_path, encoding="utf-8") as f:
        records = json.load(f)

    vectors = np.array([r["embedding"] for r in records], dtype="float32")
    logger.info("Loaded %d vectors, dimension %d", vectors.shape[0], vectors.shape[1])

    # Unit-normalize every vector so a plain inner product equals cosine
    # similarity - the exact same formula the old manual cosine_similarity()
    # used, just computed by FAISS instead of a Python loop.
    faiss.normalize_L2(vectors)
    logger.info("Normalized vectors to unit length (for cosine similarity via inner product)")

    index = faiss.IndexFlatIP(vectors.shape[1])  # exact search, fine at this scale (39 chunks)
    index.add(vectors)
    logger.info("Built FAISS IndexFlatIP with %d vectors", index.ntotal)

    faiss.write_index(index, str(index_path))
    logger.info("Saved FAISS index to %s", index_path)

    return index


if __name__ == "__main__":
    index = build_index()
    print(f"\nFAISS index built: {index.ntotal} vectors, dimension {index.d}")
    print(f"Saved to: {FAISS_INDEX_FILE}")
