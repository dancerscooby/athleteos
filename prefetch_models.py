"""
AthleteOS - download the RAG models at BUILD time (deployment helper).

fastembed downloads its ONNX models (the embedding model in retriever.py and
the cross-encoder in reranker.py) on first use, into a temp-dir cache by
default. On Render that meant every cold start - and the free tier cold-
starts after every 15 idle minutes - re-downloaded both models from Hugging
Face inside the first athlete's request, before any LLM call even began.

Running this from the build command, with FASTEMBED_CACHE_PATH pointing
inside the project directory (see render.yaml), bakes the models into the
deployed build so a cold start only has to load them from local disk.

Run (from this folder):
    python3 prefetch_models.py
"""

import logging

from retriever import load_model
from reranker import load_reranker

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_model()
    load_reranker()
    print("RAG models downloaded and cached.")
