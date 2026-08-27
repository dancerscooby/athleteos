"""
AthleteOS - Unit tests for memory_manager.py.

Every test runs against a temporary directory, never the project's real
memory/ folder - setUp() redirects memory_manager's module-level path
constants into a fresh tempfile.TemporaryDirectory() per test, and tearDown()
restores them. This keeps the tests fully isolated and repeatable.

Run with:
    python3 -m unittest tests.test_memory_manager -v
"""

import json
import tempfile
import time
import unittest
from pathlib import Path

import memory_manager as mm


class MemoryManagerTestCase(unittest.TestCase):
    """Base class that isolates every test into its own temp memory/ dir."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmpdir.name)

        # Save originals so tearDown can restore them exactly.
        self._originals = {
            "MEMORY_DIR": mm.MEMORY_DIR,
            "PROFILES_FILE": mm.PROFILES_FILE,
            "PREFERENCES_FILE": mm.PREFERENCES_FILE,
            "LATEST_PLAN_FILE": mm.LATEST_PLAN_FILE,
            "RECIPES_FILE": mm.RECIPES_FILE,
            "EPISODES_DIR": mm.EPISODES_DIR,
            "CACHE_DIR": mm.CACHE_DIR,
            "UPDATABLE_CATEGORIES": mm.UPDATABLE_CATEGORIES,
            "EPISODIC_MAX_ENTRIES": mm.EPISODIC_MAX_ENTRIES,
            "EPISODIC_MAX_AGE_DAYS": mm.EPISODIC_MAX_AGE_DAYS,
        }

        mm.MEMORY_DIR = tmp_path
        mm.PROFILES_FILE = tmp_path / "profiles.json"
        mm.PREFERENCES_FILE = tmp_path / "preferences.json"
        mm.LATEST_PLAN_FILE = tmp_path / "latest_plans.json"
        mm.RECIPES_FILE = tmp_path / "recipes.json"
        mm.EPISODES_DIR = tmp_path / "episodes"
        mm.CACHE_DIR = tmp_path / "cache"
        mm.UPDATABLE_CATEGORIES = {
            "profile": mm.PROFILES_FILE,
            "preference": mm.PREFERENCES_FILE,
            "latest_plan": mm.LATEST_PLAN_FILE,
            "recipes": mm.RECIPES_FILE,
        }

    def tearDown(self):
        for name, value in self._originals.items():
            setattr(mm, name, value)
        self._tmpdir.cleanup()


# ----------------------------------------------------------------------
# get / set  (profile, preference)
# ----------------------------------------------------------------------

class TestGetSet(MemoryManagerTestCase):
    def test_get_returns_none_when_nothing_stored(self):
        self.assertIsNone(mm.get("profile", "athlete_1"))

    def test_set_then_get_returns_stored_value(self):
        mm.set("profile", "athlete_1", {"age": 22, "diet": "Vegetarian"})
        self.assertEqual(mm.get("profile", "athlete_1"), {"age": 22, "diet": "Vegetarian"})

    def test_set_overwrites_previous_value(self):
        mm.set("profile", "athlete_1", {"age": 22})
        mm.set("profile", "athlete_1", {"age": 23})
        self.assertEqual(mm.get("profile", "athlete_1"), {"age": 23})

    def test_categories_are_isolated_from_each_other(self):
        mm.set("profile", "athlete_1", {"age": 22})
        mm.set("preference", "athlete_1", {"budget_per_day": "Rs 300"})
        self.assertEqual(mm.get("profile", "athlete_1"), {"age": 22})
        self.assertEqual(mm.get("preference", "athlete_1"), {"budget_per_day": "Rs 300"})

    def test_athletes_are_isolated_from_each_other(self):
        mm.set("profile", "athlete_1", {"age": 22})
        mm.set("profile", "athlete_2", {"age": 30})
        self.assertEqual(mm.get("profile", "athlete_1"), {"age": 22})
        self.assertEqual(mm.get("profile", "athlete_2"), {"age": 30})

    def test_get_rejects_episodic_category(self):
        with self.assertRaises(ValueError):
            mm.get("episodic", "athlete_1")

    def test_set_rejects_episodic_category(self):
        with self.assertRaises(ValueError):
            mm.set("episodic", "athlete_1", {})

    def test_get_rejects_unknown_category(self):
        with self.assertRaises(ValueError):
            mm.get("not_a_real_category", "athlete_1")

    def test_data_is_actually_written_to_disk_as_json(self):
        mm.set("profile", "athlete_1", {"age": 22})
        raw = json.loads(mm.PROFILES_FILE.read_text(encoding="utf-8"))
        self.assertEqual(raw, {"athlete_1": {"age": 22}})

    def test_no_leftover_tmp_file_after_a_successful_write(self):
        mm.set("profile", "athlete_1", {"age": 22})
        tmp_file = mm.PROFILES_FILE.with_name(mm.PROFILES_FILE.name + ".tmp")
        self.assertFalse(tmp_file.exists())


# ----------------------------------------------------------------------
# append / list  (episodic)
# ----------------------------------------------------------------------

class TestAppendList(MemoryManagerTestCase):
    def test_list_returns_empty_when_nothing_stored(self):
        self.assertEqual(mm.list("episodic", "athlete_1"), [])

    def test_append_then_list_returns_the_entry(self):
        mm.append("episodic", "athlete_1", {"goal": "muscle gain"})
        entries = mm.list("episodic", "athlete_1")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["goal"], "muscle gain")

    def test_entries_get_an_automatic_stored_at_timestamp(self):
        mm.append("episodic", "athlete_1", {"goal": "muscle gain"})
        entries = mm.list("episodic", "athlete_1")
        self.assertIn("_stored_at", entries[0])
        self.assertIsInstance(entries[0]["_stored_at"], float)

    def test_caller_supplied_stored_at_is_overwritten(self):
        mm.append("episodic", "athlete_1", {"goal": "x", "_stored_at": "not a real timestamp"})
        entries = mm.list("episodic", "athlete_1")
        self.assertIsInstance(entries[0]["_stored_at"], float)

    def test_list_returns_newest_first(self):
        mm.append("episodic", "athlete_1", {"goal": "first"})
        mm.append("episodic", "athlete_1", {"goal": "second"})
        mm.append("episodic", "athlete_1", {"goal": "third"})
        entries = mm.list("episodic", "athlete_1")
        self.assertEqual([e["goal"] for e in entries], ["third", "second", "first"])

    def test_list_respects_limit(self):
        for i in range(5):
            mm.append("episodic", "athlete_1", {"goal": f"plan {i}"})
        entries = mm.list("episodic", "athlete_1", limit=2)
        self.assertEqual(len(entries), 2)
        self.assertEqual([e["goal"] for e in entries], ["plan 4", "plan 3"])

    def test_append_rejects_profile_category(self):
        with self.assertRaises(ValueError):
            mm.append("profile", "athlete_1", {})

    def test_list_rejects_profile_category(self):
        with self.assertRaises(ValueError):
            mm.list("profile", "athlete_1")

    def test_pruning_by_count(self):
        mm.EPISODIC_MAX_ENTRIES = 3
        for i in range(5):
            mm.append("episodic", "athlete_1", {"goal": f"plan {i}"})
        entries = mm.list("episodic", "athlete_1")
        self.assertEqual(len(entries), 3)
        # Should keep the 3 MOST RECENT (plan 2, 3, 4), newest first.
        self.assertEqual([e["goal"] for e in entries], ["plan 4", "plan 3", "plan 2"])

    def test_pruning_by_age(self):
        mm.EPISODIC_MAX_AGE_DAYS = 90
        old_entry = {"goal": "very old plan", "_stored_at": time.time() - (200 * 86400)}
        mm.EPISODES_DIR.mkdir(parents=True, exist_ok=True)
        (mm.EPISODES_DIR / "athlete_1.json").write_text(
            json.dumps([old_entry]), encoding="utf-8"
        )

        mm.append("episodic", "athlete_1", {"goal": "new plan"})

        entries = mm.list("episodic", "athlete_1")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["goal"], "new plan")

    def test_episodic_history_is_per_athlete_file(self):
        mm.append("episodic", "athlete_1", {"goal": "x"})
        expected_path = mm.EPISODES_DIR / "athlete_1.json"
        self.assertTrue(expected_path.exists())


# ----------------------------------------------------------------------
# cache_get / cache_set
# ----------------------------------------------------------------------

class TestCache(MemoryManagerTestCase):
    def test_cache_get_returns_none_for_missing_key(self):
        self.assertIsNone(mm.cache_get("no-such-key"))

    def test_cache_set_then_get_returns_the_value(self):
        mm.cache_set("rag:some query", {"chunks": ["a", "b"]}, ttl_seconds=60)
        self.assertEqual(mm.cache_get("rag:some query"), {"chunks": ["a", "b"]})

    def test_cache_entry_expires_after_ttl(self):
        mm.cache_set("rag:some query", "cached value", ttl_seconds=100)

        # Simulate time passing by rewriting created_at into the past,
        # rather than sleeping in the test.
        path = mm._cache_path("rag:some query")
        data = json.loads(path.read_text(encoding="utf-8"))
        data["created_at"] = time.time() - 200  # older than the 100s ttl
        path.write_text(json.dumps(data), encoding="utf-8")

        self.assertIsNone(mm.cache_get("rag:some query"))

    def test_expired_entry_is_deleted_from_disk(self):
        mm.cache_set("rag:some query", "cached value", ttl_seconds=0)
        path = mm._cache_path("rag:some query")
        data = json.loads(path.read_text(encoding="utf-8"))
        data["created_at"] = time.time() - 1  # already older than ttl_seconds=0
        path.write_text(json.dumps(data), encoding="utf-8")

        mm.cache_get("rag:some query")  # triggers the expiry + delete

        self.assertFalse(path.exists())

    def test_different_keys_do_not_collide(self):
        mm.cache_set("query A", "value A", ttl_seconds=60)
        mm.cache_set("query B", "value B", ttl_seconds=60)
        self.assertEqual(mm.cache_get("query A"), "value A")
        self.assertEqual(mm.cache_get("query B"), "value B")


if __name__ == "__main__":
    unittest.main(verbosity=2)
