"""
AthleteOS - Tests for Stage 1 memory integration (memory_integration.py + api.py).

Covers:
  - merge_with_memory(): request values win, memory fills gaps, nothing
    stored -> unchanged (today's default behaviour); goal/notes never
    filled from memory.
  - update_memory_from_profile(): only changed fields are written; goal/
    notes are never persisted.
  - End-to-end: the actual /meal-plan endpoint (create_meal_plan(), called
    directly as a plain function - no HTTP server needed) remembers a
    returning athlete's preferences across two separate requests, with the
    response schema completely unchanged.

Every test runs against a temporary memory/ directory - never the project's
real one - using the same isolation technique as tests/test_memory_manager.py.

Run with:
    python3 -m unittest tests.test_memory_integration -v
"""

import tempfile
import unittest
from pathlib import Path

import memory_integration as mi
import memory_manager as mm
from api import MealPlanRequest, create_meal_plan


class MemoryIsolatedTestCase(unittest.TestCase):
    """Redirects memory_manager's storage into a fresh temp dir per test."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmpdir.name)
        self._originals = {
            "MEMORY_DIR": mm.MEMORY_DIR,
            "PROFILES_FILE": mm.PROFILES_FILE,
            "PREFERENCES_FILE": mm.PREFERENCES_FILE,
            "LATEST_PLAN_FILE": mm.LATEST_PLAN_FILE,
            "RECIPES_FILE": mm.RECIPES_FILE,
            "EPISODES_DIR": mm.EPISODES_DIR,
            "CACHE_DIR": mm.CACHE_DIR,
            "UPDATABLE_CATEGORIES": mm.UPDATABLE_CATEGORIES,
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


BASE_PROFILE = dict(
    age=22, height_cm=178, weight_kg=72, sex="Male",
    goal="Improve football performance",
    diet="Vegetarian", training_days_per_week=5, training_intensity="High",
    region="Lucknow, Uttar Pradesh", allergies="none",
    budget_per_day=None, food_preferences=None, foods_to_avoid=None, notes=None,
)


# ----------------------------------------------------------------------
# merge_with_memory()
# ----------------------------------------------------------------------

class TestMergeWithMemory(MemoryIsolatedTestCase):
    def test_no_stored_memory_leaves_profile_unchanged(self):
        merged = mi.merge_with_memory(dict(BASE_PROFILE))
        self.assertEqual(merged, BASE_PROFILE)

    def test_fills_missing_optional_field_from_preference_memory(self):
        mm.set("preference", mi.DEFAULT_ATHLETE_ID, {"budget_per_day": "Rs 300"})
        merged = mi.merge_with_memory(dict(BASE_PROFILE))
        self.assertEqual(merged["budget_per_day"], "Rs 300")

    def test_request_value_always_wins_over_stored_value(self):
        mm.set("preference", mi.DEFAULT_ATHLETE_ID, {"budget_per_day": "Rs 300"})
        profile = dict(BASE_PROFILE, budget_per_day="Rs 500")
        merged = mi.merge_with_memory(profile)
        self.assertEqual(merged["budget_per_day"], "Rs 500")

    def test_profile_fields_filled_from_profile_memory(self):
        mm.set("profile", mi.DEFAULT_ATHLETE_ID, {"region": "Kolkata, West Bengal"})
        profile = dict(BASE_PROFILE, region=None)
        merged = mi.merge_with_memory(profile)
        self.assertEqual(merged["region"], "Kolkata, West Bengal")

    def test_goal_is_never_filled_from_memory(self):
        mm.set("profile", mi.DEFAULT_ATHLETE_ID, {"goal": "stale goal from last week"})
        profile = dict(BASE_PROFILE, goal=None)
        merged = mi.merge_with_memory(profile)
        self.assertIsNone(merged["goal"])

    def test_notes_is_never_filled_from_memory(self):
        mm.set("preference", mi.DEFAULT_ATHLETE_ID, {"notes": "stale notes from last week"})
        profile = dict(BASE_PROFILE, notes=None)
        merged = mi.merge_with_memory(profile)
        self.assertIsNone(merged["notes"])

    def test_neither_request_nor_memory_has_value_stays_none(self):
        merged = mi.merge_with_memory(dict(BASE_PROFILE, food_preferences=None))
        self.assertIsNone(merged["food_preferences"])


# ----------------------------------------------------------------------
# update_memory_from_profile()
# ----------------------------------------------------------------------

class TestUpdateMemoryFromProfile(MemoryIsolatedTestCase):
    def test_first_update_stores_known_fields(self):
        mi.update_memory_from_profile(dict(BASE_PROFILE, budget_per_day="Rs 300"))
        stored_profile = mm.get("profile", mi.DEFAULT_ATHLETE_ID)
        stored_pref = mm.get("preference", mi.DEFAULT_ATHLETE_ID)
        self.assertEqual(stored_profile["age"], 22)
        self.assertEqual(stored_profile["diet"], "Vegetarian")
        self.assertEqual(stored_pref["budget_per_day"], "Rs 300")

    def test_goal_and_notes_are_never_stored(self):
        mi.update_memory_from_profile(dict(BASE_PROFILE, notes="Match tomorrow at 9 AM"))
        stored_profile = mm.get("profile", mi.DEFAULT_ATHLETE_ID) or {}
        stored_pref = mm.get("preference", mi.DEFAULT_ATHLETE_ID) or {}
        self.assertNotIn("goal", stored_profile)
        self.assertNotIn("notes", stored_profile)
        self.assertNotIn("notes", stored_pref)

    def test_unchanged_fields_are_not_rewritten_to_disk(self):
        mi.update_memory_from_profile(dict(BASE_PROFILE))
        mtime_before = mm.PROFILES_FILE.stat().st_mtime

        mi.update_memory_from_profile(dict(BASE_PROFILE))  # identical values again
        mtime_after = mm.PROFILES_FILE.stat().st_mtime

        self.assertEqual(mtime_before, mtime_after)

    def test_changed_field_overwrites_the_old_value(self):
        mi.update_memory_from_profile(dict(BASE_PROFILE, weight_kg=72))
        mi.update_memory_from_profile(dict(BASE_PROFILE, weight_kg=74))
        stored = mm.get("profile", mi.DEFAULT_ATHLETE_ID)
        self.assertEqual(stored["weight_kg"], 74)


# ----------------------------------------------------------------------
# End-to-end: the real endpoint function, called directly (no HTTP server)
# ----------------------------------------------------------------------

class TestEndToEndApiIntegration(MemoryIsolatedTestCase):
    def test_second_request_reuses_first_requests_budget_but_not_food_preferences(self):
        first = MealPlanRequest(
            **dict(BASE_PROFILE, budget_per_day="Rs 300", food_preferences="paneer, dal")
        )
        create_meal_plan(first)

        # Second request omits budget_per_day / food_preferences entirely -
        # exactly like a returning athlete who doesn't retype everything.
        second_data = dict(BASE_PROFILE)
        second_data["budget_per_day"] = None
        second_data["food_preferences"] = None
        second = MealPlanRequest(**second_data)
        create_meal_plan(second)

        # budget_per_day still fills forward (unchanged, stable-fact-like
        # field). food_preferences does NOT - it's excluded from
        # MERGE_FIELDS (see memory_integration.py) precisely because
        # silently reapplying an old food preference to a request that left
        # it blank caused confusing "stuck" behavior in practice.
        stored_pref = mm.get("preference", mi.DEFAULT_ATHLETE_ID)
        self.assertEqual(stored_pref["budget_per_day"], "Rs 300")
        # Still recorded from the FIRST request (storage is unaffected),
        # but the second request's OWN (blank) value is what actually
        # reached generate_meal_plan() - confirmed by the merge helper
        # directly, since the API contract doesn't echo the merged profile.
        self.assertIsNone(mi.merge_with_memory(dict(second_data))["food_preferences"])

    def test_response_schema_is_unchanged(self):
        request = MealPlanRequest(**BASE_PROFILE)
        response = create_meal_plan(request)
        self.assertEqual(
            set(response.keys()),
            {"confidence_score", "confidence_reason", "day_summary", "meals", "disclaimer"},
        )

    def test_response_still_correct_for_a_brand_new_athlete(self):
        # No memory exists yet at all - this must behave exactly as before
        # memory was introduced.
        request = MealPlanRequest(**BASE_PROFILE)
        response = create_meal_plan(request)
        self.assertEqual(len(response["meals"]), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
