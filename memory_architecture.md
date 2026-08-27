# AthleteOS — Memory Layer Design

**Status: design only. No code in this document has been implemented yet.**
This is a companion to `architecture.md`, describing a layer that does not
exist in the codebase yet. Nothing in the current app changes because of
this document.

## 1. What "memory" means here, in plain terms

Today, AthleteOS is **stateless**: every request to `/meal-plan` is a stranger
walking in fresh — the app has no idea it has ever seen this athlete before,
even if they generated a plan five minutes ago. "Memory" means selectively
remembering things across requests, so the app can eventually do things like
*"don't ask for the same details again"* or *"here's how your plan changed
since last week"* — without turning into a full user-account system before
that's actually needed (the project's MVP principle, applied here too: no
memory feature is added until a real limitation of "no memory" shows up).

## 2. Memory categories

Five categories, each with a different lifetime and a different reason to
exist. This is the core design decision — get the categories right and
everything else (expiry, storage, migration) follows naturally.

| # | Category | Lifetime | Updated how | Example content |
|---|---|---|---|---|
| 1 | **Profile Memory** | Permanent | Overwritten in place | age, height, weight, sex, diet, region, allergies |
| 2 | **Preference Memory** | Permanent | Overwritten in place | food preferences, foods to avoid, typical budget |
| 3 | **Episodic Memory** | Permanent, append-only, pruned | Appended, never edited | a log entry per generated meal plan (timestamp, goal, plan summary) |
| 4 | **Session Memory** | Temporary | Overwritten during the session | the in-progress request's RAG context, tool calls made, partial state |
| 5 | **Cache Memory** | Expiring (TTL) | Overwritten on cache miss | a previously-computed RAG context block or tool result, keyed by input hash |

### 2.1 Profile Memory — **permanent**

**What:** the stable facts about an athlete that rarely change: age, height,
weight, sex, diet type, region, allergies.

**Why permanent:** these are exactly the fields the current `MealPlanRequest`
already asks for on every single request (see `api.py`). The whole point of
remembering them is to stop asking a returning athlete to retype their body
stats every time — that's the most obvious, lowest-risk win memory can offer.

**Why updatable, not append-only:** weight and training details genuinely
change over weeks/months. We want the *current* value, not a growing pile of
old ones — that history belongs in Episodic Memory instead (§2.3), not mixed
into the profile itself.

### 2.2 Preference Memory — **permanent, but soft**

**What:** food preferences, foods to avoid, typical daily budget.

**Why separate from Profile Memory:** these are *opinions*, not *facts* — an
athlete might say "no more paneer" one month and mean it loosely. Keeping
them in a separate category means we can later attach different confidence
handling to preferences (e.g. "last confirmed 3 months ago, double-check")
without touching the athlete's hard body-stat facts. This is exactly why
`meal_planner.py`'s prompt-builder already treats `budget_per_day`,
`food_preferences`, `foods_to_avoid`, and `notes` as optional fields distinct
from the mandatory profile fields — the memory design mirrors a distinction
the app already makes.

### 2.3 Episodic Memory — **permanent, append-only, with pruning**

**What:** a timestamped log of things that happened — one entry per
generated meal plan (goal, a short summary, confidence score), not the full
verbose plan text.

**Why append-only:** the value of this category *is* the history — "what did
we recommend last time" only works if old entries aren't overwritten. This is
the piece of memory that would eventually answer "show me my last 5 plans"
or "has my protein target changed over time."

**Why pruned (not infinite):** an append-only local JSON log grows forever if
nothing bounds it. The design caps it — keep the most recent N entries (e.g.
30) or entries within the last 90 days, whichever limit is hit first — so a
single local JSON file never becomes unwieldy. This is the "what should
expire" decision applied to a *permanent* category: permanent doesn't mean
"unbounded," it means "not simply thrown away after the session."

### 2.4 Session Memory — **temporary**

**What:** whatever is only relevant to the *current* request/session in
progress — the RAG context block that was just retrieved, which tools were
called and their results, any partial state.

**Why temporary:** this is exactly the kind of thing that already exists
today as ordinary Python variables inside `claude_provider()` and
`run_with_tools()` — it lives for the duration of one function call and
disappears. Formalizing it as a memory category matters only once sessions
span *multiple* requests (e.g. a future multi-turn chat UI). Until then, it
should stay ephemeral by design — persisting it would be storing information
nobody will ever read again, which is waste, not memory.

### 2.5 Cache Memory — **expiring**

**What:** a previously-computed result keyed by its inputs — e.g. "the RAG
context block for this exact query" or "this tool's result for these exact
arguments" — so an identical repeated request doesn't redo the same local
model inference.

**Why it must expire:** the whole point of a cache is that it can go stale.
If `knowledge_base/*.md` changes and `embeddings.json` / `faiss_index.bin`
are rebuilt (see `architecture.md` §3.2), a cached RAG context computed
*before* that rebuild is now wrong and must not be served. Two expiry
triggers, either is sufficient: **time-based** (e.g. 24-hour TTL) and
**content-based** (invalidate whenever the knowledge base's content hash
changes). A cache entry with no expiry isn't a cache — it's a silent source
of stale answers.

## 3. What should **never** be stored

This list matters as much as what *is* stored. Explicit non-goals:

| Never store | Why |
|---|---|
| `ANTHROPIC_API_KEY` or any credential/secret | Already env-var only (see `llm_provider.py`); a memory file is a durable, readable artifact — exactly the wrong place for a secret, the same reasoning already applied to vault-based credentials in this project's tool-use research |
| Full raw LLM prompts/responses, verbatim, long-term | The RAG context block alone can be 1,000+ characters (see the RAG stage's logged prompt lengths); storing every full prompt forever is pure bloat with no benefit — Episodic Memory stores a short *summary*, not a transcript |
| Medical or diagnostic data | The app collects nutrition-relevant self-reported facts (allergies, diet), never medical records, lab results, or diagnoses — out of scope and a real liability if stored |
| Precise location (GPS coordinates) | The app only ever needs city/state-level `region` for regional food logic (`meal_planner.py`'s `detect_region()`) — anything more precise is collected data with no feature behind it |
| Payment or financial account details | AthleteOS has no payment feature; `budget_per_day` is a spending *target* in rupees, not a financial account |
| Any identifying info beyond what's needed to run the app | No name, email, phone, or government ID is collected today, and none should be added just because a memory system exists to put it in |
| Internal error stack traces / debug logs | Those belong in the existing `logging` output (`athleteos.rag`, `athleteos.tools`, etc.), which is ephemeral console output — not the persisted memory layer. Mixing debug logs into user memory conflates two very different retention needs |

## 4. How memory fits the existing architecture

### 4.1 Today (no memory layer) — for comparison

```
Browser → FastAPI → generate_meal_plan(profile) → provider → (RAG, tools) → plan
```

Every arrow above is stateless — nothing persists between requests.

### 4.2 With the memory layer added (future state, not implemented)

```
                         ┌───────────────────────────┐
                         │        Memory Layer         │
                         │   (new, optional, additive)  │
                         │                              │
                         │  Profile │ Preference │       │
                         │  Episodic │ Session │ Cache   │
                         └─────┬───────────┬─────────────┘
                               │           │
                    load-on-request   save-after-response
                               │           │
                               ▼           │
 Browser → FastAPI → generate_meal_plan(profile) → provider → (RAG, tools) → plan
                               │                                              │
                               └──────────────── (unchanged) ─────────────────┘
```

Two things this diagram is intentionally showing:

1. **Memory sits beside the existing pipeline, not inside it.** `generate_meal_plan()`, the provider layer, RAG, and tools are all untouched — they still take a profile dict in and return a plan dict out, exactly as today. Memory only ever *reads before* and *writes after* that unchanged core, the same non-invasive pattern RAG and tool-calling used when they were added.
2. **Only two integration points would ever be needed later:** load relevant memory before building the profile/prompt, and save relevant memory after the plan is generated. This keeps the blast radius of adding memory small and reversible.

### 4.3 Where each category would plug in

```
 REQUEST ARRIVES
       │
       ▼
 [Profile Memory]  ──▶ pre-fill missing profile fields (if athlete is known)
       │
       ▼
 generate_meal_plan(profile)
       │
       ├─▶ [Session Memory]   (RAG context + tool calls, this request only)
       │
       ▼
 plan returned to browser
       │
       ▼
 [Episodic Memory]  ◀── append a short summary entry (timestamp, goal, confidence)
 [Preference Memory] ◀── update if the athlete's stated preferences changed
 [Cache Memory]      ◀── store the RAG/tool result, keyed by input hash, with a TTL
```

## 5. Local storage design (today's target: JSON only)

No database, no server process — a `memory/` directory of small JSON files,
one concern per file, mirroring the project's existing "one file, one job"
convention (`ingest.py` / `embeddings.py` / `retriever.py`):

```
memory/
├── profiles.json         # {athlete_id: {age, height_cm, weight_kg, ...}}
├── preferences.json       # {athlete_id: {food_preferences, foods_to_avoid, budget_per_day}}
├── episodes/
│   └── <athlete_id>.json  # append-only list of past-plan summaries, pruned to last N
└── cache/
    └── <hash>.json         # one file per cached result, {value, created_at, ttl_seconds}
```

Session Memory is deliberately **absent** from this list — per §2.4, it stays
as in-memory Python state, not a file, until a real multi-request session
concept exists.

**On `athlete_id`:** the project has no authentication yet (by design — see
the original MVP principles: *"no authentication until multiple users become
necessary"*). So this design assumes a single default `athlete_id` (e.g.
`"default"`) for now, but every schema is keyed by `athlete_id` from day one
— so introducing real accounts later is a matter of generating real IDs, not
redesigning the storage shape.

## 6. The migration-proof interface

The requirement that this be "reusable and migratable to SQLite or Supabase
without changing the rest of the application" is a single design rule: **no
file outside the memory layer ever touches a JSON file, a SQL table, or a
Supabase client directly.** Every other file talks to a small set of
*operations*, not a storage mechanism:

```
get(category, athlete_id)              -> current value (Profile/Preference)
set(category, athlete_id, value)       -> overwrite current value
append(category, athlete_id, entry)    -> add one entry (Episodic)
list(category, athlete_id, limit)      -> recent entries (Episodic)
cache_get(key)                          -> value or None if missing/expired
cache_set(key, value, ttl_seconds)      -> store with expiry
```

```
 meal_planner.py / api.py / rag.py / tools/          <- callers, unaware of storage
              │
              ▼
      ┌───────────────┐
      │  Memory        │   <- the ONLY file(s) that know the storage details
      │  operations    │
      │  (interface)    │
      └───────┬────────┘
              │
   ┌──────────┼──────────────┐
   ▼          ▼              ▼
 JSON       SQLite         Supabase
 (today)   (later, if      (later, if
           local scale     multi-device /
           needs it)       multi-user needs it)
```

Because every caller only ever calls `get` / `set` / `append` / `list` /
`cache_get` / `cache_set`, swapping the JSON files for SQLite tables (same
six operations, backed by `SELECT`/`INSERT` instead of file reads/writes) or
for a Supabase client (same six operations, backed by network calls) is a
change **entirely inside the memory layer** — `meal_planner.py`, `api.py`,
`rag.py`, and `tools/` would not need a single line changed.

## 7. Summary of decisions

| Question | Decision |
|---|---|
| Permanent categories | Profile, Preference, Episodic |
| Temporary category | Session (in-memory only, never written to disk) |
| Expiring category | Cache (time-based TTL + content-hash invalidation) |
| Updated-in-place categories | Profile, Preference |
| Append-only category | Episodic (pruned by count/age, not infinite) |
| Never stored | Secrets, full raw prompts/transcripts, medical data, precise location, payment info, unnecessary PII, debug logs |
| Storage today | Local JSON files under `memory/`, one file per concern |
| Migration path | A fixed small operation interface (`get`/`set`/`append`/`list`/`cache_get`/`cache_set`) that every other file calls — only the memory layer's internals change when moving to SQLite or Supabase |
| Impact on existing architecture | None yet — this is a design document; `api.py`, `meal_planner.py`, the provider layer, RAG, and tools are all unchanged |
