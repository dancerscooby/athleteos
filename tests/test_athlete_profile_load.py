"""
AthleteOS - tests for GET /athlete/{athlete_id}/profile.

This endpoint exists so the frontend can populate the form when an athlete
is switched to, without redesigning memory_manager.py or introducing any
new storage. It is a straight passthrough of memory_manager.get("profile", ...)
and memory_manager.get("preference", ...) - no merge logic, no new writes.

Deliberately NOT covered here (pure frontend/DOM behaviour, no JS test
harness in this repo - verified manually in the browser instead):
  - switching athletes clears chatHistory/chat log
  - switching athletes clears the ingredients textarea
  - switching athletes clears the previous meal-plan result panel
  - reload preserves the athlete list/active athlete but starts a fresh
    session and empty ingredients
These are exercised by index.html's existing startNewSession()/
clearIngredients()/form.reset() logic, none of which this change touches.

No live OpenRouter calls - this endpoint never reaches llm_provider.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import memory_manager as mm
from api import app

ALICE_ID = "11111111-1111-4111-8111-111111111111"
BOB_ID = "22222222-2222-4222-8222-222222222222"
SHARED_NAME_ALICE_ID = "33333333-3333-4333-8333-333333333333"
SHARED_NAME_BOB_ID = "44444444-4444-4444-8444-444444444444"


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

    def _client(self):
        from fastapi.testclient import TestClient
        return TestClient(app)


class TestAthleteProfileLoad(MemoryIsolatedTestCase):
    def test_saved_profile_loads_for_the_correct_athlete(self):
        mm.set("profile", ALICE_ID, {
            "age": 26, "height_cm": 175, "weight_kg": 70, "sex": "Male",
            "diet": "Vegetarian", "region": "Lucknow, Uttar Pradesh", "allergies": "none",
        })
        mm.set("preference", ALICE_ID, {
            "budget_per_day": "Rs 300", "food_preferences": "paneer", "foods_to_avoid": "fried food",
        })

        response = self._client().get(f"/athlete/{ALICE_ID}/profile")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["age"], 26)
        self.assertEqual(body["height_cm"], 175)
        self.assertEqual(body["weight_kg"], 70)
        self.assertEqual(body["sex"], "Male")
        self.assertEqual(body["diet"], "Vegetarian")
        self.assertEqual(body["region"], "Lucknow, Uttar Pradesh")
        self.assertEqual(body["allergies"], "none")
        self.assertEqual(body["budget_per_day"], "Rs 300")
        self.assertEqual(body["food_preferences"], "paneer")
        self.assertEqual(body["foods_to_avoid"], "fried food")

    def test_two_same_name_athletes_load_different_profiles(self):
        """The frontend only ever sends the UUID, but this proves the
        endpoint itself keeps two 'Athlete 1'-labelled UUIDs fully separate -
        the same guarantee test_athlete_identity_sessions.py already proves
        for the meal-plan/chat endpoints."""
        mm.set("profile", SHARED_NAME_ALICE_ID, {"weight_kg": 70, "diet": "Vegetarian"})
        mm.set("profile", SHARED_NAME_BOB_ID, {"weight_kg": 95, "diet": "Non-Vegetarian"})

        client = self._client()
        alice_body = client.get(f"/athlete/{SHARED_NAME_ALICE_ID}/profile").json()
        bob_body = client.get(f"/athlete/{SHARED_NAME_BOB_ID}/profile").json()

        self.assertEqual(alice_body["weight_kg"], 70)
        self.assertEqual(alice_body["diet"], "Vegetarian")
        self.assertEqual(bob_body["weight_kg"], 95)
        self.assertEqual(bob_body["diet"], "Non-Vegetarian")

    def test_new_athlete_with_no_memory_gets_all_defaults(self):
        response = self._client().get(f"/athlete/{BOB_ID}/profile")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        for field in ["age", "height_cm", "weight_kg", "sex", "diet", "region",
                       "allergies", "budget_per_day", "food_preferences", "foods_to_avoid"]:
            self.assertIsNone(body[field])

    def test_goal_training_and_notes_are_never_returned(self):
        """These are deliberately excluded from Profile/Preference memory by
        memory_integration.py (they describe one request, not a stable
        athlete fact) - this endpoint must not invent a way to surface them."""
        mm.set("profile", ALICE_ID, {"age": 26, "diet": "Vegetarian"})

        body = self._client().get(f"/athlete/{ALICE_ID}/profile").json()

        for field in ["goal", "training_days_per_week", "training_intensity", "notes"]:
            self.assertNotIn(field, body)

    def test_default_athlete_id_is_accepted(self):
        """Migration safety, same as the other athlete_id-scoped endpoints."""
        mm.set("profile", "default", {"age": 30})
        response = self._client().get("/athlete/default/profile")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["age"], 30)

    def test_malformed_athlete_id_is_rejected(self):
        """A single-path-segment value that isn't a UUID or 'default' must
        never resolve to real memory. (Multi-segment traversal attempts like
        "../../etc/passwd" don't even reach this handler - the router itself
        treats each "/" as a path boundary - so they're not exercised here;
        this endpoint reads from a JSON dict keyed by athlete_id anyway, not
        a filesystem path, so there is no traversal surface to begin with.)"""
        response = self._client().get("/athlete/not-a-uuid/profile")
        self.assertEqual(response.status_code, 422)

    def test_profile_write_then_read_round_trips_through_the_real_endpoints(self):
        """End-to-end: generate a plan (which persists memory via the
        existing /meal-plan flow), then confirm the new GET endpoint sees
        exactly what was written - no separate/duplicated storage path."""
        from api import MealPlanRequest, create_meal_plan

        request = MealPlanRequest(
            age=26, height_cm=175, weight_kg=70, sex="Male",
            goal="Improve football performance", diet="Vegetarian",
            training_days_per_week=5, training_intensity="High",
            region="Mumbai, Maharashtra", allergies="none",
            budget_per_day="Rs 250", food_preferences="dal, rice",
            foods_to_avoid=None, notes=None, athlete_id=ALICE_ID,
        )
        create_meal_plan(request)

        body = self._client().get(f"/athlete/{ALICE_ID}/profile").json()
        self.assertEqual(body["weight_kg"], 70)
        self.assertEqual(body["region"], "Mumbai, Maharashtra")
        self.assertEqual(body["budget_per_day"], "Rs 250")
        self.assertEqual(body["food_preferences"], "dal, rice")


if __name__ == "__main__":
    unittest.main(verbosity=2)
