"""
AthleteOS - Tests for Stage 2 memory integration (episodic_memory.py + api.py).

Covers:
  - create_episode(): the summary dict contains exactly the required
    fields, derived correctly from a sample profile/plan, and contains
    NONE of the prohibited raw data (full ingredients, resources, etc.).
  - record_episode(): stores via the existing memory_manager.append(), and
    the entry is retrievable via the existing memory_manager.list().
  - End-to-end: the real /meal-plan endpoint (create_meal_plan(), called
    directly - no HTTP server needed) records one episode per successful
    generation, and episodes correctly capture per-request data (like
    `goal`) that Stage 1's Profile/Preference memory deliberately excludes.

Every test runs against a temporary memory/ directory - never the project's
real one - using the same isolation technique as the Stage 1 memory tests.

Run with:
    python3 -m unittest tests.test_episodic_memory -v
"""

import tempfile
import unittest
from pathlib import Path

import episodic_memory as em
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

SAMPLE_PLAN = {
    "confidence_score": 9.3,
    "confidence_reason": ["Budget provided", "Training intensity known"],
    "day_summary": {
        "total_calories": 3479, "total_protein_g": 145,
        "total_carbs_g": 481, "total_fats_g": 88,
        "notes": "Balanced energy to support high-intensity football performance.",
    },
    "meals": [
        {
            "meal": "Breakfast",
            "ingredients": [{"item": "Poha with peanuts", "quantity": "1 large plate"},
                             {"item": "Curd", "quantity": "1 large bowl"}],
            "calories": 870, "protein_g": 36, "carbs_g": 120, "fats_g": 22,
            "reasoning": "Carb-rich start to fuel morning training",
            "resources": [{"type": "youtube", "title": "How to make Poha",
                            "url": "https://example.com/poha"}],
        },
        {
            "meal": "Lunch",
            "ingredients": [{"item": "Dal", "quantity": "1 large bowl"}],
            "calories": 1148, "protein_g": 48, "carbs_g": 159, "fats_g": 29,
            "reasoning": "Largest meal around training for steady energy and protein",
            "resources": [],
        },
    ],
    "disclaimer": "This meal plan is AI-generated for educational purposes...",
}


# ----------------------------------------------------------------------
# create_episode()
# ----------------------------------------------------------------------

class TestCreateEpisode(unittest.TestCase):
    def test_contains_exactly_the_required_fields(self):
        episode = em.create_episode(BASE_PROFILE, SAMPLE_PLAN)
        self.assertEqual(
            set(episode.keys()),
            {"timestamp", "goal", "training_intensity", "training_days_per_week",
             "calories", "confidence_score", "summary"},
        )

    def test_field_values_are_correctly_derived(self):
        episode = em.create_episode(BASE_PROFILE, SAMPLE_PLAN)
        self.assertEqual(episode["goal"], "Improve football performance")
        self.assertEqual(episode["training_intensity"], "High")
        self.assertEqual(episode["training_days_per_week"], 5)
        self.assertEqual(episode["calories"], 3479)
        self.assertEqual(episode["confidence_score"], 9.3)

    def test_timestamp_is_a_number(self):
        episode = em.create_episode(BASE_PROFILE, SAMPLE_PLAN)
        self.assertIsInstance(episode["timestamp"], float)

    def test_summary_mentions_each_meal_and_its_main_ingredient(self):
        episode = em.create_episode(BASE_PROFILE, SAMPLE_PLAN)
        self.assertIn("Breakfast: Poha with peanuts", episode["summary"])
        self.assertIn("Lunch: Dal", episode["summary"])

    def test_summary_does_not_include_full_ingredient_details(self):
        episode = em.create_episode(BASE_PROFILE, SAMPLE_PLAN)
        # "Curd" is a second ingredient in Breakfast - only the FIRST
        # ingredient per meal should appear in the summary.
        self.assertNotIn("Curd", episode["summary"])

    def test_episode_never_contains_raw_plan_structures(self):
        episode = em.create_episode(BASE_PROFILE, SAMPLE_PLAN)
        # None of the prohibited raw/verbose data should be present anywhere.
        self.assertNotIn("meals", episode)
        self.assertNotIn("ingredients", episode)
        self.assertNotIn("resources", episode)
        self.assertNotIn("reasoning", episode)
        self.assertNotIn("disclaimer", episode)
        self.assertNotIn("confidence_reason", episode)

    def test_episode_is_much_smaller_than_the_full_plan(self):
        import json
        episode_size = len(json.dumps(em.create_episode(BASE_PROFILE, SAMPLE_PLAN)))
        plan_size = len(json.dumps(SAMPLE_PLAN))
        self.assertLess(episode_size, plan_size)


# ----------------------------------------------------------------------
# record_episode()
# ----------------------------------------------------------------------

class TestRecordEpisode(MemoryIsolatedTestCase):
    def test_record_episode_stores_via_memory_manager_append(self):
        em.record_episode(BASE_PROFILE, SAMPLE_PLAN, athlete_id=mi.DEFAULT_ATHLETE_ID)
        entries = mm.list("episodic", mi.DEFAULT_ATHLETE_ID)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["goal"], "Improve football performance")

    def test_stored_entry_has_memory_managers_own_stored_at_too(self):
        # memory_manager.append() always adds its own bookkeeping field,
        # independent of episodic_memory's own "timestamp" field.
        em.record_episode(BASE_PROFILE, SAMPLE_PLAN, athlete_id=mi.DEFAULT_ATHLETE_ID)
        entries = mm.list("episodic", mi.DEFAULT_ATHLETE_ID)
        self.assertIn("_stored_at", entries[0])
        self.assertIn("timestamp", entries[0])

    def test_multiple_episodes_accumulate_and_stay_newest_first(self):
        em.record_episode(dict(BASE_PROFILE, goal="first plan"), SAMPLE_PLAN)
        em.record_episode(dict(BASE_PROFILE, goal="second plan"), SAMPLE_PLAN)
        entries = mm.list("episodic", mi.DEFAULT_ATHLETE_ID)
        self.assertEqual(len(entries), 2)
        self.assertEqual([e["goal"] for e in entries], ["second plan", "first plan"])

    def test_record_episode_returns_the_stored_episode(self):
        episode = em.record_episode(BASE_PROFILE, SAMPLE_PLAN)
        self.assertEqual(episode["calories"], 3479)


# ----------------------------------------------------------------------
# End-to-end: the real endpoint, called directly (no HTTP server)
# ----------------------------------------------------------------------

class TestEndToEndApiIntegration(MemoryIsolatedTestCase):
    def test_successful_generation_creates_one_episode(self):
        request = MealPlanRequest(**BASE_PROFILE)
        create_meal_plan(request)
        entries = mm.list("episodic", mi.DEFAULT_ATHLETE_ID)
        self.assertEqual(len(entries), 1)

    def test_two_requests_create_two_episodes(self):
        create_meal_plan(MealPlanRequest(**BASE_PROFILE))
        create_meal_plan(MealPlanRequest(**dict(BASE_PROFILE, goal="Fat loss")))
        entries = mm.list("episodic", mi.DEFAULT_ATHLETE_ID)
        self.assertEqual(len(entries), 2)

    def test_episode_captures_per_request_goal_unlike_profile_memory(self):
        # Stage 1 deliberately never stores "goal" in Profile/Preference
        # memory. Episodic memory is the one place a request's goal IS
        # remembered - as part of that request's own history entry.
        create_meal_plan(MealPlanRequest(**dict(BASE_PROFILE, goal="Match day fueling")))
        create_meal_plan(MealPlanRequest(**dict(BASE_PROFILE, goal="Muscle gain")))

        entries = mm.list("episodic", mi.DEFAULT_ATHLETE_ID)
        self.assertEqual(entries[0]["goal"], "Muscle gain")   # most recent first
        self.assertEqual(entries[1]["goal"], "Match day fueling")

        # Profile memory itself must still have no "goal" field at all.
        stored_profile = mm.get("profile", mi.DEFAULT_ATHLETE_ID) or {}
        self.assertNotIn("goal", stored_profile)

    def test_response_schema_is_still_unchanged(self):
        response = create_meal_plan(MealPlanRequest(**BASE_PROFILE))
        self.assertEqual(
            set(response.keys()),
            {"confidence_score", "confidence_reason", "day_summary", "meals", "disclaimer"},
        )

    def test_episode_reflects_actual_generated_calories(self):
        response = create_meal_plan(MealPlanRequest(**BASE_PROFILE))
        entries = mm.list("episodic", mi.DEFAULT_ATHLETE_ID)
        self.assertEqual(entries[0]["calories"], response["day_summary"]["total_calories"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
