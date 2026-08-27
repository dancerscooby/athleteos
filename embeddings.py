"""
AthleteOS - Embeddings (RAG stage, Step 2).

Takes the chunk records built by ingest.py and converts each chunk's text
into an embedding - a list of numbers that represents its meaning, produced
by a local Sentence Transformers model (no API calls, runs on your machine).
Results are saved to embeddings.json so later steps (retrieval / RAG) can
load them without re-computing anything.

Requires the 'sentence-transformers' package, which is NOT yet part of this
project (see requirements.txt). Install it separately:

    python3 -m pip install sentence-transformers

Run with:
    python3 embeddings.py
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from ingest import load_documents, create_chunk_records

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, runs locally - good for a first RAG pass
OUTPUT_FILE = Path(__file__).parent / "embeddings.json"


def generate_embeddings(records, model_name=MODEL_NAME):
    """
    Given the chunk records from ingest.create_chunk_records(), return a new
    list of dicts with an added "embedding" field:
        {"id": ..., "source": ..., "text": ..., "embedding": [0.01, -0.03, ...]}
    """
    model = SentenceTransformer(model_name)
    texts = [record["text"] for record in records]
    vectors = model.encode(texts, show_progress_bar=False)

    enriched = []
    for record, vector in zip(records, vectors):
        enriched.append({
            "id": record["id"],
            "source": record["source"],
            "text": record["text"],
            "embedding": vector.tolist(),  # numpy array -> plain list for JSON
        })
    return enriched


def save_embeddings(records_with_embeddings, path=OUTPUT_FILE):
    """Write the enriched records to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records_with_embeddings, f, ensure_ascii=False, indent=2)


# ----------------------------------------------------------------------
# TEST SECTION  -  run with: python embeddings.py
# ----------------------------------------------------------------------

if __name__ == "__main__":
    documents = load_documents()
    records = create_chunk_records(documents)

    print("=" * 64)
    print(f"Generating embeddings for {len(records)} chunks using '{MODEL_NAME}'...")
    print("=" * 64)
    enriched = generate_embeddings(records)

    save_embeddings(enriched)

    dims = len(enriched[0]["embedding"]) if enriched else 0
    print(f"Saved {len(enriched)} embeddings to {OUTPUT_FILE}")
    print(f"Embedding size (dimensions per chunk): {dims}")
    print(f"\nExample - {enriched[0]['id']}:")
    print(f"  first 5 numbers: {enriched[0]['embedding'][:5]}")
