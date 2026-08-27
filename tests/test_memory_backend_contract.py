"""
AthleteOS - backend-agnostic contract tests (Phase 1 storage abstraction).

Unlike tests/test_memory_manager.py (which deliberately reaches into
JSON-specific internals - mm.PROFILES_FILE, mm._cache_path(), the .tmp
file, etc. - and is right to, since it's testing the JSON backend
specifically), this file exercises ONLY the six public functions
(get/set/append/list/cache_get/cache_set) and never touches a file path or
a database table directly. The same test bodies run against BOTH backends
via one shared mixin, so a passing run is literal proof the two backends
satisfy the identical contract.

    TestJSONBackendContract      - runs always, isolated to a temp dir,
                                    exactly like every other JSON-backend
                                    test in this project.
    TestPostgresBackendContract  - SKIPPED unless DATABASE_URL is set. No
                                    Supabase project exists yet, so this
                                    currently skips in full - that is
                                    expected and correct, not a failure. Do
                                    NOT mock the database to force these to
                                    "pass" - an untested claim of Postgres
                                    correctness is worse than an honest
                                    skip. Point DATABASE_URL at a real
                                    Postgres/Supabase database (with
                                    schema.sql already applied) to actually
                                    run these.

Run with:
    python -m unittest tests.test_memory_backend_contract -v
"""

import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import memory_manager as mm

DATABASE_URL = os.environ.get("DATABASE_URL")


class MemoryBackendContractTests:
    """
    Shared test bodies, mixed into one TestCase per backend below. Every
    test calls ONLY mm.get/set/append/list - never a backend-specific
    internal - so the exact same assertions apply to both backends.

    Subclasses provide:
      _athlete_id()       a fresh, isolated athlete_id for this test
      _set_max_entries(n) patch whichever module's EPISODIC_MAX_ENTRIES
                           this backend actually reads, for the pruning test
    """

    # -- get / set --------------------------------------------------

    def test_get_returns_none_when_nothing_stored(self):
        self.assertIsNone(mm.get("profile", self._athlete_id()))

    def test_set_then_get_returns_stored_value(self):
        athlete_id = self._athlete_id()
        mm.set("profile", athlete_id, {"age": 22, "diet": "Vegetarian"})
        self.assertEqual(mm.get("profile", athlete_id), {"age": 22, "diet": "Vegetarian"})

    def test_set_overwrites_previous_value(self):
        athlete_id = self._athlete_id()
        mm.set("profile", athlete_id, {"age": 22})
        mm.set("profile", athlete_id, {"age": 23})
        self.assertEqual(mm.get("profile", athlete_id), {"age": 23})

    def test_categories_are_isolated_from_each_other(self):
        athlete_id = self._athlete_id()
        mm.set("profile", athlete_id, {"age": 22})
        mm.set("preference", athlete_id, {"budget_per_day": "Rs 300"})
        self.assertEqual(mm.get("profile", athlete_id), {"age": 22})
        self.assertEqual(mm.get("preference", athlete_id), {"budget_per_day": "Rs 300"})

    def test_athletes_are_isolated_from_each_other(self):
        a, b = self._athlete_id(), self._athlete_id()
        mm.set("profile", a, {"age": 22})
        mm.set("profile", b, {"age": 30})
        self.assertEqual(mm.get("profile", a), {"age": 22})
        self.assertEqual(mm.get("profile", b), {"age": 30})

    def test_recipes_category_round_trips_a_whole_list(self):
        """Option A (approved): the stored value for "recipes" is the
        athlete's WHOLE recipe list, not one row per recipe."""
        athlete_id = self._athlete_id()
        recipes = [
            {"id": "abc123", "name": "Mom's Chicken Curry", "ingredients": ["Chicken", "Onion"]},
            {"id": "def456", "name": "Rajma Bowl", "ingredients": ["Rajma", "Rice"]},
        ]
        mm.set("recipes", athlete_id, recipes)
        self.assertEqual(mm.get("recipes", athlete_id), recipes)

    def test_get_rejects_episodic_category(self):
        with self.assertRaises(ValueError):
            mm.get("episodic", self._athlete_id())

    def test_set_rejects_episodic_category(self):
        with self.assertRaises(ValueError):
            mm.set("episodic", self._athlete_id(), {})

    def test_get_rejects_unknown_category(self):
        with self.assertRaises(ValueError):
            mm.get("not_a_real_category", self._athlete_id())

    # -- append / list ------------------------------------------------

    def test_list_returns_empty_when_nothing_stored(self):
        self.assertEqual(mm.list("episodic", self._athlete_id()), [])

    def test_append_then_list_returns_the_entry(self):
        athlete_id = self._athlete_id()
        mm.append("episodic", athlete_id, {"goal": "muscle gain"})
        entries = mm.list("episodic", athlete_id)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["goal"], "muscle gain")

    def test_entries_get_an_automatic_stored_at_timestamp(self):
        athlete_id = self._athlete_id()
        mm.append("episodic", athlete_id, {"goal": "muscle gain"})
        entries = mm.list("episodic", athlete_id)
        self.assertIn("_stored_at", entries[0])
        self.assertIsInstance(entries[0]["_stored_at"], float)

    def test_caller_supplied_stored_at_is_overwritten(self):
        athlete_id = self._athlete_id()
        mm.append("episodic", athlete_id, {"goal": "x", "_stored_at": "not a real timestamp"})
        entries = mm.list("episodic", athlete_id)
        self.assertIsInstance(entries[0]["_stored_at"], float)

    def test_list_returns_newest_first(self):
        athlete_id = self._athlete_id()
        mm.append("episodic", athlete_id, {"goal": "first"})
        mm.append("episodic", athlete_id, {"goal": "second"})
        mm.append("episodic", athlete_id, {"goal": "third"})
        entries = mm.list("episodic", athlete_id)
        self.assertEqual([e["goal"] for e in entries], ["third", "second", "first"])

    def test_list_respects_limit(self):
        athlete_id = self._athlete_id()
        for i in range(5):
            mm.append("episodic", athlete_id, {"goal": f"plan {i}"})
        entries = mm.list("episodic", athlete_id, limit=2)
        self.assertEqual(len(entries), 2)
        self.assertEqual([e["goal"] for e in entries], ["plan 4", "plan 3"])

    def test_append_rejects_profile_category(self):
        with self.assertRaises(ValueError):
            mm.append("profile", self._athlete_id(), {})

    def test_list_rejects_profile_category(self):
        with self.assertRaises(ValueError):
            mm.list("profile", self._athlete_id())

    def test_pruning_by_count(self):
        athlete_id = self._athlete_id()
        self._set_max_entries(3)
        for i in range(5):
            mm.append("episodic", athlete_id, {"goal": f"plan {i}"})
        entries = mm.list("episodic", athlete_id)
        self.assertEqual(len(entries), 3)
        self.assertEqual([e["goal"] for e in entries], ["plan 4", "plan 3", "plan 2"])


# ----------------------------------------------------------------------
# JSON backend - runs always (this is the default/dev/CI backend)
# ----------------------------------------------------------------------

class TestJSONBackendContract(MemoryBackendContractTests, unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        tmp = Path(self._tmpdir.name)
        self._orig = {n: getattr(mm, n) for n in
                      ["MEMORY_DIR", "PROFILES_FILE", "PREFERENCES_FILE", "LATEST_PLAN_FILE",
                       "RECIPES_FILE", "EPISODES_DIR", "CACHE_DIR", "UPDATABLE_CATEGORIES",
                       "EPISODIC_MAX_ENTRIES", "_BACKEND"]}
        mm.MEMORY_DIR = tmp
        mm.PROFILES_FILE = tmp / "profiles.json"
        mm.PREFERENCES_FILE = tmp / "preferences.json"
        mm.LATEST_PLAN_FILE = tmp / "latest_plans.json"
        mm.RECIPES_FILE = tmp / "recipes.json"
        mm.EPISODES_DIR = tmp / "episodes"
        mm.CACHE_DIR = tmp / "cache"
        mm.UPDATABLE_CATEGORIES = {"profile": mm.PROFILES_FILE,
                                   "preference": mm.PREFERENCES_FILE,
                                   "latest_plan": mm.LATEST_PLAN_FILE,
                                   "recipes": mm.RECIPES_FILE}
        mm._BACKEND = "json"

    def tearDown(self):
        for name, value in self._orig.items():
            setattr(mm, name, value)
        self._tmpdir.cleanup()

    def _athlete_id(self):
        return uuid.uuid4().hex

    def _set_max_entries(self, n):
        mm.EPISODIC_MAX_ENTRIES = n


# ----------------------------------------------------------------------
# Postgres backend - SKIPPED unless DATABASE_URL is set. No database
# exists yet as of this change, so this class currently skips entirely.
# That is correct, expected behaviour - not something to mock around.
# ----------------------------------------------------------------------

@unittest.skipUnless(DATABASE_URL, "DATABASE_URL not set - no Postgres/Supabase database configured")
class TestPostgresBackendContract(MemoryBackendContractTests, unittest.TestCase):
    def setUp(self):
        self._backend_patch = mock.patch.object(mm, "_BACKEND", "postgres")
        self._backend_patch.start()
        self._athlete_ids = []

    def tearDown(self):
        self._backend_patch.stop()
        # Best-effort cleanup so repeated test runs against a real database
        # don't accumulate rows - every athlete_id this test created is
        # deleted from every table directly.
        if self._athlete_ids:
            import postgres_memory_backend as pg
            conn = pg._get_connection()
            with conn.cursor() as cur:
                for table in ("profiles", "preferences", "latest_plans", "recipes", "episodes"):
                    cur.execute(
                        f"DELETE FROM {table} WHERE athlete_id = ANY(%s)",
                        (self._athlete_ids,),
                    )

    def _athlete_id(self):
        new_id = f"contract-test-{uuid.uuid4().hex}"
        self._athlete_ids.append(new_id)
        return new_id

    def _set_max_entries(self, n):
        import postgres_memory_backend as pg
        pg.EPISODIC_MAX_ENTRIES = n
        self.addCleanup(setattr, pg, "EPISODIC_MAX_ENTRIES", 30)


# ----------------------------------------------------------------------
# The cache category always stays local-JSON-backed (approved H.3), even
# when ATHLETEOS_STORAGE_BACKEND=postgres - this proves the dispatch never
# routes cache_get/cache_set to Postgres. Runs unconditionally: it never
# actually connects to a database, since cache is never dispatched there.
# ----------------------------------------------------------------------

class TestCacheStaysLocalRegardlessOfBackend(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        tmp = Path(self._tmpdir.name)
        self._orig_cache_dir = mm.CACHE_DIR
        self._orig_backend = mm._BACKEND
        mm.CACHE_DIR = tmp / "cache"
        mm._BACKEND = "postgres"  # no DATABASE_URL needed - cache must never reach it

    def tearDown(self):
        mm.CACHE_DIR = self._orig_cache_dir
        mm._BACKEND = self._orig_backend
        self._tmpdir.cleanup()

    def test_cache_works_without_any_database_configured(self):
        mm.cache_set("some query", {"chunks": ["a", "b"]}, ttl_seconds=60)
        self.assertEqual(mm.cache_get("some query"), {"chunks": ["a", "b"]})


if __name__ == "__main__":
    unittest.main(verbosity=2)
