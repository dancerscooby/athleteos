# AthleteOS — Architecture

## 1. High-Level Overview

AthleteOS turns an athlete's profile (age, diet, training load, region, budget, etc.) into a personalized daily meal plan. The system is built in layers, each with one job, communicating through a small number of fixed contracts:

```
 Browser (index.html)
        │  HTTP (fetch)
        ▼
 FastAPI (api.py)
        │  validated dict
        ▼
 Core logic (meal_planner.py)
        │  provider(profile) -> plan
        ▼
 Provider layer  ──┬── mock_provider (rule-based, free, offline)
                    └── claude_provider (LLM-based, via llm_provider.py)
                             │
                             ▼
                   RAG layer (rag.py + retriever.py + embeddings.json)
                             │
                             ▼
                   knowledge_base/*.md  (source documents)
```

Every layer only knows about the layer directly below it. The frontend never talks to `meal_planner.py` directly; FastAPI never knows whether a mock or a real LLM produced the plan; the LLM provider never knows how retrieval works internally. This is what lets each piece be replaced or extended without rewriting the rest — e.g. swapping the mock for Claude was a one-line change, and adding RAG only touched the LLM provider.

## 2. Major Files

| File | Role |
|---|---|
| `index.html` | The entire frontend: one HTML page, CSS, and vanilla JS. Collects the athlete profile and renders the returned plan. |
| `api.py` | FastAPI app. Defines the HTTP request/response contracts and one endpoint, `POST /meal-plan`. Also serves `index.html` at `/app`. |
| `meal_planner.py` | The core of the app. Defines `generate_meal_plan(profile)`, the CLI developer tool, the deterministic `mock_provider`, and `build_prompt()` (shared by both providers). |
| `llm_provider.py` | The real-LLM provider (`claude_provider`). Builds the final prompt (base prompt + RAG context), calls Claude with structured output, and returns the same JSON shape as the mock. |
| `rag.py` | Turns an athlete profile into a search query, calls the retriever, and formats the results into a "context block" for the LLM provider to inject into its prompt. |
| `retriever.py` | Given a text query, embeds it and finds the most similar chunks in `embeddings.json` via cosine similarity. |
| `embeddings.py` | One-time (or re-run-when-content-changes) script: reads chunks from `ingest.py`, generates an embedding per chunk with a local Sentence Transformers model, and writes `embeddings.json`. |
| `ingest.py` | Reads every `.md` file in `knowledge_base/`, splits it into ~500-character chunks, and returns `{id, source, text}` records. |
| `embeddings.json` | Generated data file: every chunk plus its embedding vector. The retriever's search index. |
| `knowledge_base/*.md` | Hand-written reference documents (football nutrition, hydration, recovery, Indian protein sources, match-day nutrition) — the raw knowledge RAG draws from. |
| `tools/` | Tool-calling foundation (this stage): a base tool interface, a registry, and two concrete tools (nutrition calculations, budget-based food suggestions). Not yet wired into the LLM provider. |
| `requirements.txt` | Declares `fastapi`, `uvicorn`, and the optional `anthropic` / `sentence-transformers` packages needed for the LLM and RAG stages. |

## 3. Data Flow

### 3.1 A meal-plan request, end to end

```
User fills form in browser
        │
        ▼
fetch() POST /meal-plan  (JSON body)
        │
        ▼
FastAPI validates body -> MealPlanRequest
        │  .model_dump()
        ▼
generate_meal_plan(profile)          <- meal_planner.py, the ONE entry point
        │
        ▼
   PROVIDER == "mock"?  ──yes──> mock_provider(profile)  (rule-based, offline)
        │no
        ▼
   claude_provider(profile)          <- llm_provider.py
        │
        ├─ build_prompt(profile)              (meal_planner.py - unchanged)
        ├─ get_context_block(profile)         (rag.py - RAG injection)
        │        │
        │        ├─ build_retrieval_query(profile)
        │        ├─ retrieve_relevant_chunks(query, top_k=3)   (retriever.py)
        │        │        │
        │        │        ├─ embed query (Sentence Transformers, local)
        │        │        └─ cosine similarity vs embeddings.json
        │        └─ format_context_block(chunks)
        │
        ├─ final prompt = context_block + base_prompt
        └─ Claude API call (structured output, JSON schema enforced)
        │
        ▼
plan (dict)  ──same shape regardless of provider──>  FastAPI response_model
        │
        ▼
Browser renders meal cards, macros, confidence, resources
```

### 3.2 Building the knowledge base (offline, run manually / occasionally)

```
knowledge_base/*.md
        │  load_documents()          (ingest.py)
        ▼
[{source, text}, ...]
        │  split_into_chunks() + create_chunk_records()
        ▼
[{id, source, text}, ...]
        │  generate_embeddings()     (embeddings.py)
        ▼
[{id, source, text, embedding}, ...]
        │  save_embeddings()
        ▼
embeddings.json    <- what retriever.py searches at request time
```

This pipeline runs **separately** from the live request path — it only needs to re-run when `knowledge_base/` content changes.

## 4. Role of the Frontend (`index.html`)

A single static page, no framework, no build step. Its only jobs: render the profile form, `fetch()` a POST to `/meal-plan`, and render whatever JSON comes back (meal cards, macro badges, confidence panel, error messages). It has no knowledge of providers, RAG, or how the plan was produced — it only knows the response shape (`MealPlanResponse`).

## 5. Role of FastAPI (`api.py`)

The web boundary. It does exactly three things: serve the frontend file, validate incoming requests against `MealPlanRequest` (mirroring the CLI's own validation rules), and call `generate_meal_plan()` with the validated data. It has **no business logic** — the endpoint is three lines. It doesn't know or care whether the plan came from the mock or from Claude+RAG.

## 6. Role of the Provider Layer

The seam that makes the AI implementation swappable. `generate_meal_plan(profile)` in `meal_planner.py` is the single call site every caller (CLI, FastAPI) uses; internally it dispatches to whichever provider is selected (`mock_provider` or `claude_provider`), and both must return the identical JSON shape. Adding RAG, structured output, or (later) tool calling only ever changes what happens *inside* `claude_provider` — the contract with the rest of the app never changes.

## 7. Role of the RAG Pipeline

RAG (Retrieval-Augmented Generation) lets Claude answer using real reference documents instead of only its own training knowledge. It has three moving parts, kept in three files on purpose (ingest → embed → retrieve is a different concern at each stage):

1. **`ingest.py`** turns raw markdown into small, labeled chunks.
2. **`embeddings.py`** turns each chunk into a vector (a list of numbers capturing meaning), stored in `embeddings.json`.
3. **`retriever.py`** turns a live query into a vector and finds the closest stored chunks by cosine similarity.

**`rag.py`** is the glue that connects this pipeline to the LLM provider: profile → query → top-3 chunks → formatted context block → injected into the prompt. It also owns the logging for this stage (query, retrieved chunk IDs/similarity, final prompt length).

## 8. Role of the Knowledge Base (`knowledge_base/*.md`)

Five short, hand-written Markdown documents summarizing football nutrition, hydration, recovery, Indian protein sources, and match-day nutrition. This is intentionally a *small, curated* library, not a comprehensive nutrition database — enough to prove retrieval works and to give Claude grounded, India-specific reference material.

## 9. Role of the Embedding Model

`all-MiniLM-L6-v2`, a small local Sentence Transformers model, converts text (chunks or queries) into 384-dimension vectors. It runs entirely on the local machine — no API calls, no cost, no external dependency at request time beyond the model weights (downloaded once, then cached). Both the knowledge-base chunks and every live query are embedded with the **same** model, which is what makes their vectors comparable.

## 10. Role of the Retriever (`retriever.py`)

Given a query, it (1) embeds the query with the same model used for the chunks, (2) computes cosine similarity between the query vector and every stored chunk vector, and (3) returns the top-k most similar chunks. At the current scale (39 chunks) this is a simple linear scan held in memory — no vector database is needed yet (see Future Additions).

## 11. Role of the Prompt Builder

Two prompt-related responsibilities are deliberately split:

- **`build_prompt(profile)`** (in `meal_planner.py`) turns the athlete's profile into the base instructions — persona, constraints, required JSON shape. This is shared by *every* provider that talks to an LLM, so it lives with the core logic, not inside `llm_provider.py`.
- **`get_context_block(profile)`** (in `rag.py`) produces the *retrieved* reference material.

`llm_provider.py` combines the two (`context_block + base_prompt`) into the final prompt sent to Claude. Neither piece knows about the other's internals.

## 12. Role of Claude

Claude is the model actually generating the meal plan when `PROVIDER == "claude"`. It receives the combined prompt (instructions + retrieved knowledge-base context) and is constrained via **structured output** (a JSON schema matching `MealPlanResponse`) to return exactly the shape the rest of the app expects. It never sees the frontend, the FastAPI layer, or the retrieval mechanics — only the final assembled prompt.

## 13. Future Additions

These are **not implemented yet** — noted here so the current architecture is understood as a foundation, not a finished system:

- **FAISS (or another vector index):** at 39 chunks, a linear scan is fast enough. If the knowledge base grows into the thousands of chunks, `retriever.py`'s in-memory loop would be replaced with a proper vector index (e.g. FAISS) — the `retrieve_relevant_chunks(query, top_k)` function signature would stay the same, so nothing above it needs to change.
- **Tool calling (this stage's foundation):** `tools/` introduces a base tool interface, a registry, and two standalone tools (nutrition math, budget-based food suggestions) that Claude could eventually call mid-conversation instead of relying purely on its own reasoning. Not yet wired into `llm_provider.py`.
- **Agents:** a future step where the model autonomously decides *which* tools to call and in what order, rather than tools being invoked directly by application code. Deliberately out of scope for now.
- **MCP (Model Context Protocol):** a standard way to expose tools/data to LLMs across applications. Only worth adopting once there are multiple external tools/services to standardize — not needed for two local tools.

*Consistent with the project's MVP principle: each of the above gets added only when a real limitation of the current, simpler approach actually appears.*
