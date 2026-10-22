"""
AthleteOS - Postgres/Supabase storage backend (Phase 1 storage abstraction).

Implements the exact same six-operation contract memory_manager.py already
exposes for its "json" backend - see that module's own docstring and
memory_architecture.md Section 6 ("the migration-proof interface"). This
module is ONLY imported when ATHLETEOS_STORAGE_BACKEND=postgres is set
(memory_manager.py's dispatch does this lazily, inside each function); with
the default "json" backend this file is never imported, so psycopg is never
required to be installed and no database connection is ever attempted.

CACHE IS NOT IMPLEMENTED HERE, ON PURPOSE (approved Phase 1 design decision
H.3): cache_get()/cache_set() keep running against local JSON files under
every ATHLETEOS_STORAGE_BACKEND setting - memory_manager.py's dispatch never
routes those two operations here. The cache is disposable, currently unused
by any production code path, and explicitly out of scope for this
migration. This module therefore defines no cache_get()/cache_set() at all.

SCHEMA (approved Phase 1 design - see the design document for full
rationale; none of this has been created in any real database yet):

    profiles(athlete_id text PRIMARY KEY, data jsonb NOT NULL,
             updated_at timestamptz NOT NULL DEFAULT now())
    preferences(athlete_id text PRIMARY KEY, data jsonb NOT NULL,
             updated_at timestamptz NOT NULL DEFAULT now())
    latest_plans(athlete_id text PRIMARY KEY, data jsonb NOT NULL,
             updated_at timestamptz NOT NULL DEFAULT now())
    recipes(athlete_id text PRIMARY KEY, data jsonb NOT NULL,
             updated_at timestamptz NOT NULL DEFAULT now())
        -- Option A (approved): `data` holds the athlete's WHOLE recipe
        -- list, exactly like recipes.json today - not one row per recipe.
        -- personal_recipes.py's add_recipe()/delete_recipe() already do a
        -- read-modify-write of the whole list via generic get()/set(), so
        -- this table needs no special-casing anywhere else in the app.
    episodes(id bigserial PRIMARY KEY, athlete_id text NOT NULL,
             stored_at double precision NOT NULL, data jsonb NOT NULL)
        -- one row per episode. stored_at is an epoch-seconds float
        -- (approved: NOT timestamptz), matching memory_manager's
        -- "_stored_at" field exactly so pruning math is identical to the
        -- JSON backend. `data` holds the rest of the episode dict; the
        -- "_stored_at" key is never duplicated inside it - list() below
        -- reconstructs it from the stored_at column on the way out.

    CREATE INDEX ON episodes (athlete_id, stored_at DESC);
        -- supports list()'s "most recent N for this athlete" query.

No athletes table, no foreign keys (approved H.5): athlete_id is a plain
text column everywhere, exactly matching today's "any UUID (or 'default')
the client sends is valid" behaviour. Revisited only when real
authentication exists - not in scope for this phase.

CONNECTION: a single, lazily-created, module-level connection (the same
lazy-singleton pattern retriever.py/reranker.py already use for their
models), opened from the DATABASE_URL env var on first use - not at import
time. No Supabase project has been created and this code has not been run
against a real database as part of this change.
"""

import logging
import os
import time

logger = logging.getLogger("athleteos.memory.postgres")

DATABASE_URL = os.environ.get("DATABASE_URL")

# Mirrors memory_manager.EPISODIC_MAX_ENTRIES / EPISODIC_MAX_AGE_DAYS exactly
# (same values, same "whichever hits first" pruning rule) - kept as this
# module's own constants rather than importing memory_manager's, since
# memory_manager.py is what imports THIS module (importing back would be
# circular); the two are duplicated on purpose, not diverged from.
EPISODIC_MAX_ENTRIES = 30
EPISODIC_MAX_AGE_DAYS = 90

# category -> table name, for the four update-in-place categories. Table
# names come from this fixed, hardcoded mapping only - never from caller
# input - so building SQL with an f-string here is not an injection risk;
# the athlete_id/value themselves are always passed as query parameters.
_UPDATE_IN_PLACE_TABLES = {
    "profile": "profiles",
    "preference": "preferences",
    "latest_plan": "latest_plans",
    "recipes": "recipes",
}

_conn = None


def _get_connection():
    """
    Return the shared Postgres connection, opening it on first use. psycopg
    is imported HERE, not at module level, so merely importing this file
    (which memory_manager.py only does when ATHLETEOS_STORAGE_BACKEND=
    postgres) doesn't require psycopg to be installed until a database
    operation is actually attempted.
    """
    global _conn
    if _conn is None or _conn.closed:
        if not DATABASE_URL:
            raise RuntimeError(
                "ATHLETEOS_STORAGE_BACKEND=postgres requires DATABASE_URL to "
                "be set (a Postgres/Supabase connection string)."
            )
        import psycopg
        logger.info("Connecting to Postgres backend...")
        # prepare_threshold=None: Supabase's transaction pooler (port 6543)
        # can't keep server-side prepared statements across transactions,
        # and psycopg auto-prepares a query after 5 runs - which then fails
        # with "prepared statement ... does not exist". Harmless on a direct
        # or session-pooler connection.
        _conn = psycopg.connect(
            DATABASE_URL, autocommit=True, prepare_threshold=None, connect_timeout=10,
        )
        logger.info("Postgres connection established.")
    return _conn


def _drop_connection():
    """Close and forget the cached connection so the next call opens a fresh one."""
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
    _conn = None


def _reset_connection_for_tests():
    """
    Test-only hook: closes and forgets the cached connection so the next
    call opens a fresh one - mirrors the reset pattern already used for
    retriever.py's lazily-cached _model/_index globals in tests.
    """
    _drop_connection()


def _with_connection(operation):
    """
    Run `operation(conn)` on the shared connection; if the connection turns
    out to be dead, reconnect and run it once more. Hosted Postgres (and
    Supabase's pooler especially) closes idle connections, and Render's free
    tier idles between visitors - so the first request after a quiet spell
    used to fail on the stale connection instead of just reconnecting.
    """
    import psycopg

    try:
        return operation(_get_connection())
    except psycopg.OperationalError as exc:
        logger.warning("Postgres connection lost (%s) - reconnecting and retrying once", exc)
        _drop_connection()
        return operation(_get_connection())


# ----------------------------------------------------------------------
# 1/2. get / set  -  update-in-place categories
# ----------------------------------------------------------------------

def get(category, athlete_id):
    """
    Return the current stored value for `athlete_id` in `category`, or None
    if nothing is stored yet. `category` has already been validated by
    memory_manager.py's dispatch before this is ever called.
    """
    table = _UPDATE_IN_PLACE_TABLES[category]

    def select(conn):
        with conn.cursor() as cur:
            cur.execute(f"SELECT data FROM {table} WHERE athlete_id = %s", (athlete_id,))
            return cur.fetchone()

    row = _with_connection(select)
    value = row[0] if row else None
    logger.info(
        "get(category=%r, athlete_id=%r) -> %s",
        category, athlete_id, "found" if value is not None else "not found",
    )
    return value


def set(category, athlete_id, value):
    """
    Overwrite the stored value for `athlete_id` in `category` with `value` -
    an UPSERT, matching the JSON backend's "always replaces, never
    accumulates" set() semantics exactly.
    """
    from psycopg.types.json import Jsonb

    table = _UPDATE_IN_PLACE_TABLES[category]

    def upsert(conn):
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {table} (athlete_id, data, updated_at)
                VALUES (%s, %s, now())
                ON CONFLICT (athlete_id) DO UPDATE
                    SET data = EXCLUDED.data, updated_at = EXCLUDED.updated_at
                """,
                (athlete_id, Jsonb(value)),
            )

    _with_connection(upsert)
    logger.info("set(category=%r, athlete_id=%r) -> stored", category, athlete_id)


# ----------------------------------------------------------------------
# 3/4. append / list  -  append-only category (episodic)
# ----------------------------------------------------------------------

def append(category, athlete_id, entry):
    """
    Add one entry to `athlete_id`'s episodic history (one new row in
    `episodes`), then prune synchronously - same timing as the JSON
    backend's every-append pruning, age first then count, whichever limit
    is hit first (mirrors memory_manager._prune_episodes() exactly, just
    expressed as two DELETE statements instead of a Python list filter).

    `entry`'s own "_stored_at" (if any) is never used - stored_at lives in
    its own indexed column, always set to time.time() here, exactly like
    memory_manager.append() always overwriting any caller-supplied value.
    """
    from psycopg.types.json import Jsonb

    stamped_entry = {k: v for k, v in dict(entry).items() if k != "_stored_at"}
    stored_at = time.time()

    def insert_and_prune(conn):
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO episodes (athlete_id, stored_at, data) VALUES (%s, %s, %s)",
                (athlete_id, stored_at, Jsonb(stamped_entry)),
            )

            cutoff = time.time() - (EPISODIC_MAX_AGE_DAYS * 86400)
            cur.execute(
                "DELETE FROM episodes WHERE athlete_id = %s AND stored_at < %s",
                (athlete_id, cutoff),
            )
            cur.execute(
                """
                DELETE FROM episodes WHERE id IN (
                    SELECT id FROM episodes WHERE athlete_id = %s
                    ORDER BY stored_at DESC OFFSET %s
                )
                """,
                (athlete_id, EPISODIC_MAX_ENTRIES),
            )

    _with_connection(insert_and_prune)
    logger.info("append(category=%r, athlete_id=%r) -> stored", category, athlete_id)


def list(category, athlete_id, limit=None):
    """
    Return `athlete_id`'s episodic history, most recent first - an indexed
    query (see the CREATE INDEX in this module's docstring) rather than the
    JSON backend's "read the whole file, reverse it in Python" approach,
    though the observable result is identical.

    Each returned dict gets "_stored_at" merged back in from the `stored_at`
    column, so callers see the exact same shape the JSON backend returns
    (memory_manager's tests check for this field explicitly).
    """
    query = "SELECT stored_at, data FROM episodes WHERE athlete_id = %s ORDER BY stored_at DESC"
    params = [athlete_id]
    if limit is not None:
        query += " LIMIT %s"
        params.append(limit)

    def select(conn):
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    rows = _with_connection(select)

    result = []
    for stored_at, data in rows:
        entry = dict(data)
        entry["_stored_at"] = stored_at
        result.append(entry)

    logger.info(
        "list(category=%r, athlete_id=%r, limit=%s) -> %d entries",
        category, athlete_id, limit, len(result),
    )
    return result
