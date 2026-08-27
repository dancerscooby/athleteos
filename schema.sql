-- AthleteOS - Postgres schema for the Phase 1 storage backend.
--
-- This file has NOT been run against any database. No Supabase project
-- exists yet. It documents the exact schema postgres_memory_backend.py
-- expects, per the approved Phase 1 design (recipes = Option A JSONB blob,
-- episodes.stored_at = double precision epoch seconds, no athletes table,
-- no foreign keys - see that module's own docstring for the full
-- rationale). Run this manually against a real Postgres/Supabase database
-- only when you're ready to actually start Phase 1 execution.

CREATE TABLE IF NOT EXISTS profiles (
    athlete_id  text        PRIMARY KEY,
    data        jsonb       NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS preferences (
    athlete_id  text        PRIMARY KEY,
    data        jsonb       NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS latest_plans (
    athlete_id  text        PRIMARY KEY,
    data        jsonb       NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- Option A (approved): the whole recipe list per athlete, exactly like
-- recipes.json today - not one row per recipe. personal_recipes.py needs
-- no changes either way, since it only ever calls generic get()/set().
CREATE TABLE IF NOT EXISTS recipes (
    athlete_id  text        PRIMARY KEY,
    data        jsonb       NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- One row per episode (append-only). stored_at is an epoch-seconds float,
-- matching memory_manager's "_stored_at" field exactly - NOT timestamptz,
-- so pruning math is identical to the JSON backend's.
CREATE TABLE IF NOT EXISTS episodes (
    id         bigserial         PRIMARY KEY,
    athlete_id text              NOT NULL,
    stored_at  double precision  NOT NULL,
    data       jsonb             NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_episodes_athlete_stored_at
    ON episodes (athlete_id, stored_at DESC);

-- No "cache" table: cache_get()/cache_set() intentionally stay local-JSON-
-- backed under every ATHLETEOS_STORAGE_BACKEND setting (approved H.3).
-- No "athletes" table, no foreign keys anywhere (approved H.5) - athlete_id
-- is a plain string column, matching today's "any UUID is valid" behaviour.
-- Revisited only when real authentication exists.
