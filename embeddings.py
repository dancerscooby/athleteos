"""
AthleteOS - Embeddings (RAG stage, Step 2).

Takes the chunk records built by ingest.py and converts each chunk's text
into an embedding - a list of numbers that represents its meaning, produced
by a local embedding model (no API calls, runs on your machine). Results
are saved to embeddings.json so later steps (retrieval / RAG) can load them
without re-computing anything.

Runtime (deployment change): uses 'fastembed' (ONNX Runtime) rather than
'sentence-transformers'/torch - see retriever.py's docstring for the full
memory-driven rationale. Same underlying model
(sentence-transformers/all-MiniLM-L6-v2), verified to produce numerically
identical embeddings (cosine similarity 1.0) to the previous library, so
re-running this script reproduces the SAME embeddings.json that is already
checked in - it does not need to be regenerated for the runtime swap alone.

Requires the 'fastembed' package (see requirements.txt). Install it
separately if needed:

    python3 -m pip install fastembed

Run with:
    python3 embeddings.py
"""

import json
from pathlib import Path

from fastembed import TextEmbedding

from ingest import load_documents, create_chunk_records

# Full fastembed model id - must match retriever.py's MODEL_NAME.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_FILE = Path(__file__).parent / "embeddings.json"


def generate_embeddings(records, model_name=MODEL_NAME):
    """
    Given the chunk records from ingest.create_chunk_records(), return a new
    list of dicts with an added "embedding" field:
        {"id": ..., "source": ..., "text": ..., "embedding": [0.01, -0.03, ...]}
    """
    model = TextEmbedding(model_name)
    texts = [record["text"] for record in records]
    vectors = list(model.embed(texts))

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
