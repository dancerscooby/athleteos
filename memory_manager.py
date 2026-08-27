"""
AthleteOS - Memory Manager (local JSON backend, with an optional Postgres
backend - Phase 1 storage abstraction).

Implements exactly the six operations described in memory_architecture.md's
migration-proof interface (Section 6). Every caller uses ONLY these
functions and never touches a JSON file (or a database table) directly - so
swapping backends only ever changes the INSIDE of this one file:

    get(category, athlete_id)            -> current value, or None
    set(category, athlete_id, value)     -> overwrite current value
    append(category, athlete_id, entry)  -> add one entry (episodic only)
    list(category, athlete_id, limit)    -> recent entries, newest first
    cache_get(key)                       -> cached value, or None if missing/expired
    cache_set(key, value, ttl_seconds)   -> store a value with an expiry

Per memory_architecture.md:
  - "profile" and "preference" are UPDATE-IN-PLACE categories -> get/set.
  - "episodic" is APPEND-ONLY and pruned to stay bounded -> append/list.
  - Session Memory is DELIBERATELY ABSENT here - per the design doc it stays
    as plain in-process state and is never written to disk.
  - Cache entries expire on a TTL, checked on every cache_get().

Storage backend (Phase 1): selected once via the ATHLETEOS_STORAGE_BACKEND
env var - "json" (default, unchanged local-file behaviour) or "postgres"
(delegates get/set/append/list to postgres_memory_backend.py, see below).
Every function's category validation happens HERE, before the dispatch, so
both backends enforce the exact same rules (unknown/wrong category ->
ValueError) without duplicating that check in each backend.

cache_get()/cache_set() are DELIBERATELY NOT dispatched to Postgres, even
when ATHLETEOS_STORAGE_BACKEND=postgres - an explicit, approved Phase 1
design decision (see the Phase 1 design doc, H.3): the cache is disposable,
currently unused by any production code path, and out of scope for this
migration. It keeps using local JSON files under every backend setting.

The Postgres client library (psycopg) is imported lazily, INSIDE
postgres_memory_backend.py's own functions - never at this module's import
time - so running with the default ATHLETEOS_STORAGE_BACKEND=json (what
every existing test and local dev session uses) never requires psycopg to
be installed and never attempts a database connection.

Design note (Python gotcha): two of the six required names - "set" and
"list" - are also Python's built-in types. Defining module-level functions
with those names shadows the built-ins INSIDE THIS FILE ONLY (other files
that `import memory_manager` and call `memory_manager.set(...)` are
unaffected). To avoid accidentally calling this module's own set()/list()
functions where the built-in type was intended, this file uses {...} and
[...] literals instead of set(...)/list(...) constructor calls throughout.
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("athleteos.memory")

# "json" (default) preserves today's exact local-file behaviour for every
# existing caller/test that never sets this env var. "postgres" delegates
# get/set/append/list to postgres_memory_backend.py - see that module's
# docstring for the schema. Read as a plain module attribute (not captured
# into a local at call time) so tests can flip it with
# mock.patch.object(memory_manager, "_BACKEND", "postgres") the same way
# other env-driven constants in this codebase are already tested.
_BACKEND = os.environ.get("ATHLETEOS_STORAGE_BACKEND", "json")

MEMORY_DIR = Path(__file__).parent / "memory"
PROFILES_FILE = MEMORY_DIR / "profiles.json"
PREFERENCES_FILE = MEMORY_DIR / "preferences.json"
# "latest_plan" (Step 9): the most recent meal-plan generation's compact
# summary (see episodic_memory.record_latest_plan_summary()) - current
# STATE about the last generation, not a history log, so it belongs with
# profile/preference's update-in-place pattern (a new plan overwrites the
# old one via set(), exactly like a profile edit does) rather than with
# episodic's append-only history. No new access pattern was added to
# support this - it reuses get()/set() exactly as they already exist.
LATEST_PLAN_FILE = MEMORY_DIR / "latest_plans.json"
# Personal recipes (athlete-saved homemade/family recipes - see
# personal_recipes.py): the value stored per athlete_id is their WHOLE
# recipe list, overwritten in place on every add/delete - the same
# update-in-place pattern as profile/preference/latest_plan, not a new
# access pattern. Adding/removing one recipe is a read-modify-write of the
# full list, done in personal_recipes.py, not here.
RECIPES_FILE = MEMORY_DIR / "recipes.json"
EPISODES_DIR = MEMORY_DIR / "episodes"
CACHE_DIR = MEMORY_DIR / "cache"

# Only these categories exist, and each supports exactly ONE access pattern.
# This is deliberate: it's what stops episodic history from being silently
# overwritten with set(), or a profile from growing into an unbounded list.
UPDATABLE_CATEGORIES = {
    "profile": PROFILES_FILE,
    "preference": PREFERENCES_FILE,
    "latest_plan": LATEST_PLAN_FILE,
    "recipes": RECIPES_FILE,
}
APPEND_ONLY_CATEGORIES = {"episodic"}

EPISODIC_MAX_ENTRIES = 30    # keep at most this many entries per athlete
EPISODIC_MAX_AGE_DAYS = 90   # AND drop anything older than this, whichever hits first

__all__ = ["get", "set", "append", "list", "cache_get", "cache_set"]


# ----------------------------------------------------------------------
# Internal helpers (not part of the public interface)
# ----------------------------------------------------------------------

def _read_json(path, default):
    """Read a JSON file, returning `default` if it doesn't exist yet."""
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, data):
    """
    Write JSON atomically: write to a temp file, then rename it over the
    real target. This avoids ever leaving a half-written, corrupted file
    behind if the process is interrupted mid-write - a real risk for a
    local JSON store that may be written on every request.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def _cache_path(key):
    """Cache keys can be arbitrary strings (queries, tool args, ...) - hash
    them into a safe, fixed-length filename."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{digest}.json"


def _episode_path(athlete_id):
    return EPISODES_DIR / f"{athlete_id}.json"


def _prune_episodes(entries):
    """
    Keep only the most recent EPISODIC_MAX_ENTRIES entries, AND drop
    anything older than EPISODIC_MAX_AGE_DAYS - whichever limit is hit
    first (memory_architecture.md Section 2.3: "permanent... but pruned").
    """
    cutoff = time.time() - (EPISODIC_MAX_AGE_DAYS * 86400)
    not_too_old = [e for e in entries if e.get("_stored_at", 0) >= cutoff]
    return not_too_old[-EPISODIC_MAX_ENTRIES:]


# ----------------------------------------------------------------------
# 1/2. get / set  -  update-in-place categories (profile, preference)
# ----------------------------------------------------------------------

def get(category, athlete_id):
    """
    Return the current stored value for `athlete_id` in `category`
    ("profile" or "preference"), or None if nothing is stored yet.
    """
    if category not in UPDATABLE_CATEGORIES:
        raise ValueError(
            f"get() only supports {sorted(UPDATABLE_CATEGORIES)!r}; "
            f"got {category!r}. Use list() for 'episodic'."
        )
    if _BACKEND == "postgres":
        import postgres_memory_backend as _pg
        return _pg.get(category, athlete_id)
    path = UPDATABLE_CATEGORIES[category]
    data = _read_json(path, {})
    value = data.get(athlete_id)
    logger.info(
        "get(category=%r, athlete_id=%r) -> %s",
        category, athlete_id, "found" if value is not None else "not found",
    )
    return value


def set(category, athlete_id, value):
    """
    Overwrite the stored value for `athlete_id` in `category`
    ("profile" or "preference") with `value`.
    """
    if category not in UPDATABLE_CATEGORIES:
        raise ValueError(
            f"set() only supports {sorted(UPDATABLE_CATEGORIES)!r}; "
            f"got {category!r}. Use append() for 'episodic'."
        )
    if _BACKEND == "postgres":
        import postgres_memory_backend as _pg
        return _pg.set(category, athlete_id, value)
    path = UPDATABLE_CATEGORIES[category]
    data = _read_json(path, {})
    data[athlete_id] = value
    _write_json(path, data)
    logger.info("set(category=%r, athlete_id=%r) -> stored", category, athlete_id)


# ----------------------------------------------------------------------
# 3/4. append / list  -  append-only category (episodic)
# ----------------------------------------------------------------------

def append(category, athlete_id, entry):
    """
    Add one entry to `athlete_id`'s episodic history. `entry` should be a
    JSON-serializable dict. A "_stored_at" timestamp (epoch seconds) is
    added automatically - any caller-supplied "_stored_at" is overwritten,
    since this field is memory-manager metadata, not caller data.

    After appending, the history is pruned - see _prune_episodes().
    """
    if category not in APPEND_ONLY_CATEGORIES:
        raise ValueError(
            f"append() only supports {sorted(APPEND_ONLY_CATEGORIES)!r}; "
            f"got {category!r}. Use set() for 'profile'/'preference'."
        )
    if _BACKEND == "postgres":
        import postgres_memory_backend as _pg
        return _pg.append(category, athlete_id, entry)
    path = _episode_path(athlete_id)
    entries = _read_json(path, [])

    stamped_entry = dict(entry)
    stamped_entry["_stored_at"] = time.time()
    entries.append(stamped_entry)

    pruned = _prune_episodes(entries)
    _write_json(path, pruned)
    logger.info(
        "append(category=%r, athlete_id=%r) -> stored (history now %d/%d entries)",
        category, athlete_id, len(pruned), EPISODIC_MAX_ENTRIES,
    )


def list(category, athlete_id, limit=None):
    """
    Return `athlete_id`'s episodic history, most recent first. `limit`
    caps how many entries are returned (None = everything currently kept,
    which is already bounded by the pruning applied in append()).
    """
    if category not in APPEND_ONLY_CATEGORIES:
        raise ValueError(
            f"list() only supports {sorted(APPEND_ONLY_CATEGORIES)!r}; "
            f"got {category!r}. Use get() for 'profile'/'preference'."
        )
    if _BACKEND == "postgres":
        import postgres_memory_backend as _pg
        return _pg.list(category, athlete_id, limit)
    path = _episode_path(athlete_id)
    entries = _read_json(path, [])
    newest_first = entries[::-1]
    result = newest_first[:limit] if limit is not None else newest_first
    logger.info(
        "list(category=%r, athlete_id=%r, limit=%s) -> %d entries",
        category, athlete_id, limit, len(result),
    )
    return result


# ----------------------------------------------------------------------
# 5/6. cache_get / cache_set  -  expiring category (cache)
# ----------------------------------------------------------------------

def cache_get(key):
    """
    Return the cached value for `key`, or None if nothing is cached or the
    entry has expired. An expired entry is deleted on read, so stale
    cache files don't linger on disk.
    """
    path = _cache_path(key)
    if not path.exists():
        logger.info("cache_get(key=%r) -> miss", key)
        return None

    entry = _read_json(path, None)
    age = time.time() - entry["created_at"]
    if age > entry["ttl_seconds"]:
        path.unlink(missing_ok=True)
        logger.info(
            "cache_get(key=%r) -> expired (age=%.0fs, ttl=%ds)",
            key, age, entry["ttl_seconds"],
        )
        return None

    logger.info(
        "cache_get(key=%r) -> hit (age=%.0fs, ttl=%ds)",
        key, age, entry["ttl_seconds"],
    )
    return entry["value"]


def cache_set(key, value, ttl_seconds):
    """Store `value` under `key`, expiring after `ttl_seconds` seconds."""
    path = _cache_path(key)
    _write_json(path, {
        "value": value,
        "created_at": time.time(),
        "ttl_seconds": ttl_seconds,
    })
    logger.info("cache_set(key=%r, ttl_seconds=%d) -> stored", key, ttl_seconds)
