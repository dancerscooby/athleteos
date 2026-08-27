"""
AthleteOS - Document ingestion (RAG stage, Step 1).

A simple pipeline that turns the .md files in knowledge_base/ into small,
labeled text chunks - the raw material a future retrieval/embedding system
will search over. No AI, no embeddings, no external packages: this step is
purely about reading files and splitting text.

Pipeline:
    load_documents()       -> read every .md file into {source, text}
    split_into_chunks()    -> break one document's text into ~500-char pieces
    create_chunk_records() -> combine the two into {id, source, text} dicts
"""

from pathlib import Path

KNOWLEDGE_BASE_DIR = Path(__file__).parent / "knowledge_base"
CHUNK_SIZE = 500  # approximate characters per chunk


# ----------------------------------------------------------------------
# 1. LOAD  -  read every .md file into memory
# ----------------------------------------------------------------------

def load_documents(folder=KNOWLEDGE_BASE_DIR):
    """
    Read every .md file in `folder` and return a list of documents:
        [{"source": "football_nutrition.md", "text": "<full file text>"}, ...]
    """
    documents = []
    for path in sorted(Path(folder).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        documents.append({"source": path.name, "text": text})
    return documents


# ----------------------------------------------------------------------
# 2. SPLIT  -  break one document's text into ~500-character chunks
# ----------------------------------------------------------------------

def _is_heading(block):
    """True for a markdown heading block ("# Title", "## Section", ...)."""
    return block.lstrip().startswith("#")


def split_into_chunks(text, chunk_size=CHUNK_SIZE):
    """
    Split `text` into chunks of roughly `chunk_size` characters, keeping each
    section's heading attached to the content beneath it.

    Splits on paragraphs (blank-line-separated blocks) and groups consecutive
    paragraphs together, so whole sentences/paragraphs stay intact instead of
    being cut mid-word. Two heading-aware rules sit on top of that:

      1. A heading always starts a new chunk (once the current chunk has real
         content), so one chunk does not straddle two sections.
      2. A chunk is never flushed while it still contains only heading(s) -
         the heading is carried forward and merged with the body that
         follows it.

    Rule 2 is the important one. Without it, a section like

        ## South India
        *Tamil Nadu, Kerala, ...*

        - **Staples:** rice dominates ...

    split into a ~70-character heading-only chunk plus a ~640-character body
    chunk that never named the region at all. Neither was retrievable for a
    "South India" query: the heading had no substance to match, and the
    substantive text had no region name in it. Keeping them together makes
    each regional section independently retrievable, which is the whole
    point of having per-region sections.

    A section longer than chunk_size still splits into multiple chunks as
    before; only the first of those carries the heading.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current = []  # list of blocks making up the chunk being built

    def has_body(blocks):
        return any(not _is_heading(b) for b in blocks)

    def length(blocks):
        return sum(len(b) for b in blocks) + 2 * max(len(blocks) - 1, 0)

    for paragraph in paragraphs:
        if not current:
            current = [paragraph]
            continue

        starts_new_section = _is_heading(paragraph) and has_body(current)
        too_long = has_body(current) and length(current) + len(paragraph) + 2 > chunk_size

        if starts_new_section or too_long:
            chunks.append("\n\n".join(current))
            current = [paragraph]
        else:
            current.append(paragraph)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ----------------------------------------------------------------------
# 3. RECORD  -  turn (document, chunks) into the final {id, source, text} dicts
# ----------------------------------------------------------------------

def create_chunk_records(documents, chunk_size=CHUNK_SIZE):
    """
    Given the documents from load_documents(), split each one and return a
    flat list of chunk records:
        {"id": "football_nutrition.md#0", "source": "football_nutrition.md", "text": "..."}
    """
    records = []
    for doc in documents:
        chunks = split_into_chunks(doc["text"], chunk_size)
        for i, chunk_text in enumerate(chunks):
            records.append({
                "id": f"{doc['source']}#{i}",
                "source": doc["source"],
                "text": chunk_text,
            })
    return records


# ----------------------------------------------------------------------
# TEST SECTION  -  run with: python ingest.py
# ----------------------------------------------------------------------

if __name__ == "__main__":
    documents = load_documents()
    records = create_chunk_records(documents)

    print("=" * 64)
    print("INGESTION SUMMARY")
    print("=" * 64)
    print(f"Documents loaded: {len(documents)}")
    for doc in documents:
        print(f"  - {doc['source']} ({len(doc['text'])} characters)")
    print(f"Total chunks created: {len(records)}")

    print("\n" + "=" * 64)
    print("EXAMPLE CHUNKS (first 3)")
    print("=" * 64)
    for record in records[:3]:
        print(f"\nid:     {record['id']}")
        print(f"source: {record['source']}")
        preview = record["text"][:200].replace("\n", " ")
        print(f"text:   {preview}{'...' if len(record['text']) > 200 else ''}")
