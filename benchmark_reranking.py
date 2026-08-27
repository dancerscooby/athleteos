"""
AthleteOS - Retrieval quality benchmark: FAISS-only vs FAISS + reranking.

A small, hand-labeled benchmark, honestly scaled to this project: 5 queries,
one deliberately aimed at each of the 5 knowledge_base documents (since we
wrote those documents, we know which source file SHOULD be the best match
for each query). This is not a formal IR benchmark with thousands of labeled
examples - it's a lightweight, appropriate-sized sanity check for a
5-document, 39-chunk knowledge base.

For each query we compare:
  - "FAISS only"        : top-3 straight from FAISS (search_faiss_only)
  - "FAISS + reranking" : top-3 after the reranking step (retrieve_relevant_chunks)

and report, for each side, whether the #1 result came from the expected
source file (top-1 accuracy) - plus the full ranking so you can see exactly
how (or whether) reranking changed the order.

Run with:
    python3 benchmark_reranking.py
"""

import logging

from retriever import retrieve_relevant_chunks, search_faiss_only

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("athleteos.benchmark")

# (query, expected best source file) - one per knowledge_base document.
BENCHMARK_QUERIES = [
    ("What should I eat before a football match?", "match_day_nutrition.md"),
    ("How much water should I drink during training?", "hydration_guide.md"),
    ("cheap vegetarian protein sources in India", "indian_protein_sources.md"),
    ("How should I recover after a match, including sleep and stretching?", "recovery_guide.md"),
    ("carbohydrate and protein needs for football players", "football_nutrition.md"),
]


def run_benchmark():
    faiss_hits = 0
    reranked_hits = 0
    order_changed_count = 0

    for query, expected_source in BENCHMARK_QUERIES:
        print("=" * 70)
        print(f"QUERY: {query}")
        print(f"EXPECTED BEST SOURCE: {expected_source}")
        print("=" * 70)

        faiss_top3 = search_faiss_only(query, top_k=3)
        reranked_top3 = retrieve_relevant_chunks(query, top_k=3)

        faiss_top1_source = faiss_top3[0]["source"] if faiss_top3 else None
        reranked_top1_source = reranked_top3[0]["source"] if reranked_top3 else None

        faiss_correct = faiss_top1_source == expected_source
        reranked_correct = reranked_top1_source == expected_source
        faiss_hits += faiss_correct
        reranked_hits += reranked_correct

        faiss_ids = [r["id"] for r in faiss_top3]
        reranked_ids = [r["id"] for r in reranked_top3]
        order_changed = faiss_ids != reranked_ids
        order_changed_count += order_changed

        print("\nFAISS ONLY (before reranking):")
        for rank, r in enumerate(faiss_top3, start=1):
            print(f"  #{rank} {r['id']}  similarity={r['similarity']:.3f}")
        print(f"  top-1 correct: {faiss_correct}")

        print("\nFAISS + RERANKING:")
        for rank, r in enumerate(reranked_top3, start=1):
            print(f"  #{rank} {r['id']}  similarity={r['similarity']:.3f}")
        print(f"  top-1 correct: {reranked_correct}")

        print(f"\nSelection/order changed by reranking: {order_changed}")
        print()

    total = len(BENCHMARK_QUERIES)
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"FAISS-only top-1 accuracy:        {faiss_hits}/{total}")
    print(f"FAISS + reranking top-1 accuracy: {reranked_hits}/{total}")
    print(f"Queries where reranking changed the top-3 set/order: {order_changed_count}/{total}")


if __name__ == "__main__":
    run_benchmark()
