"""
AthleteOS - one-time JSON -> Postgres migration (Phase 1).

NOT run automatically by anything, and NOT run as part of writing this
script. Dry-run by default: prints exactly what would be migrated (per-
athlete row counts per table) and writes nothing. Pass --apply to actually
write to Postgres.

Reuses both backends' already-tested code rather than inventing new read/
write logic:
  - reads via memory_manager's get()/list(), forced against the JSON files
    directly (see _discover_athlete_ids()) - the exact functions every
    existing test already exercises.
  - writes via postgres_memory_backend's set()/append() directly (not
    through memory_manager's dispatch), so this script always reads JSON
    and always writes Postgres regardless of the running process's own
    ATHLETEOS_STORAGE_BACKEND setting.

Idempotent for profile/preference/latest_plan/recipes (set() is an UPSERT
already). Episodes are de-duplicated by comparing "_stored_at" against
what's already in Postgres, so a re-run never double-inserts history.

Requires DATABASE_URL to be set (see .env.example) and the target tables to
already exist (see schema.sql) - this script does not create tables itself,
on purpose: table creation is a one-time, explicit, reviewed DDL step (run
schema.sql by hand), not something a data-migration script should do
implicitly.

Usage:
    python migrate_json_to_postgres.py            # dry run, writes nothing
    python migrate_json_to_postgres.py --apply     # actually writes

This script has NOT been run against any real database as part of writing
it. No Supabase project exists yet.
"""

import argparse
import json
import logging

import memory_manager as mm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("athleteos.migrate")

UPDATE_IN_PLACE_CATEGORIES = ["profile", "preference", "latest_plan", "recipes"]


def _discover_athlete_ids():
    """
    Every athlete_id that appears ANYWHERE in the local JSON store - the
    union of every update-in-place category's keys, plus every
    episodes/<athlete_id>.json file's stem. This is the one place this
    script reads a JSON file's raw contents itself (rather than through
    memory_manager.get(), which needs an athlete_id supplied up front) -
    purely to enumerate which athletes exist at all.
    """
    athlete_ids = set()
    for category in UPDATE_IN_PLACE_CATEGORIES:
        path = mm.UPDATABLE_CATEGORIES[category]
        if path.exists():
            with open(path, encoding="utf-8") as f:
                athlete_ids.update(json.load(f).keys())
    if mm.EPISODES_DIR.exists():
        athlete_ids.update(p.stem for p in mm.EPISODES_DIR.glob("*.json"))
    return sorted(athlete_ids)


def _plan_for_athlete(athlete_id):
    """
    Describe what this athlete has in the JSON store, without writing
    anything - {"profile": 0 or 1, ..., "recipes": <recipe count>,
    "episodic": <episode count>}.
    """
    plan = {}
    for category in UPDATE_IN_PLACE_CATEGORIES:
        value = mm.get(category, athlete_id)
        if category == "recipes":
            plan[category] = len(value) if value else 0
        else:
            plan[category] = 1 if value is not None else 0
    plan["episodic"] = len(mm.list("episodic", athlete_id))
    return plan


def _apply_for_athlete(athlete_id, plan):
    """Write this athlete's JSON-stored data into Postgres."""
    import postgres_memory_backend as pg

    for category in UPDATE_IN_PLACE_CATEGORIES:
        if plan[category]:
            pg.set(category, athlete_id, mm.get(category, athlete_id))

    if plan["episodic"]:
        already_migrated = {e["_stored_at"] for e in pg.list("episodic", athlete_id)}
        for entry in reversed(mm.list("episodic", athlete_id)):  # oldest first
            if entry["_stored_at"] in already_migrated:
                continue  # already present from a previous --apply run
            pg.append("episodic", athlete_id, entry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually write to Postgres. Without this flag, only prints the plan.",
    )
    args = parser.parse_args()

    if mm._BACKEND != "json":
        logger.warning(
            "This process's ATHLETEOS_STORAGE_BACKEND is %r, not 'json' - "
            "this script still always reads the JSON files directly, "
            "regardless of that setting.",
            mm._BACKEND,
        )

    athlete_ids = _discover_athlete_ids()
    if not athlete_ids:
        print("No athletes found in the local JSON store (memory/). Nothing to migrate.")
        return

    print(f"{'DRY RUN' if not args.apply else 'APPLYING'} "
          f"- {len(athlete_ids)} athlete(s) found in memory/\n")

    totals = {c: 0 for c in UPDATE_IN_PLACE_CATEGORIES + ["episodic"]}
    for athlete_id in athlete_ids:
        plan = _plan_for_athlete(athlete_id)
        parts = ", ".join(f"{k}={v}" for k, v in plan.items() if v)
        print(f"  {athlete_id}: {parts or '(nothing)'}")
        for k, v in plan.items():
            totals[k] += v
        if args.apply:
            _apply_for_athlete(athlete_id, plan)

    print("\nTotals:", ", ".join(f"{k}={v}" for k, v in totals.items()))
    if not args.apply:
        print("\nDry run only - nothing was written. Re-run with --apply to migrate for real.")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
