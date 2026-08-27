"""
AthleteOS - Phase 4B tests: chat awareness of the athlete's most recently
generated meal plan (1-day or multi-day), via the new "latest_plan" memory
category (see memory_manager.py and episodic_memory.record_latest_plan_summary()).

This revisits the Phase 2 decision not to record multi-day plans to
episodic memory - that decision stands UNCHANGED for episodic history
(still single-day-shaped, still skipped for duration_days > 1). What's new
is a SEPARATE, bounded, update-in-place summary that covers every
generation regardless of duration, always reflects only the MOST RECENT
plan (never accumulates), and is fed into chat's system prompt as its own
section.

Uses the real /meal-plan endpoint (create_meal_plan) to generate plans
exactly as the app would, then the real chat.chat_reply() to inspect what
reaches the system prompt sent to OpenRouter. No live OpenRouter calls -
httpx.post is mocked at the module boundary, same technique as every other
chat test file.
"""

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import chat
import memory_manager as mm
from api import MealPlanRequest, create_meal_plan


class MemoryIsolatedTestCase(unittest.TestCase):
    """Redirects memory_manager's storage into a fresh temp dir per test -
    same isolation technique used by the other memory-touching test suites."""

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
        chat.reset_sessions()

    def tearDown(self):
        for name, value in self._originals.items():
            setattr(mm, name, value)
        self._tmpdir.cleanup()
        chat.reset_sessions()


BASE = dict(
    age=25, height_cm=178, weight_kg=72, sex="Male",
    goal="Improve football performance",
    diet="Vegetarian", training_days_per_week=5, training_intensity="High",
    region="Lucknow, Uttar Pradesh", allergies="none",
    budget_per_day=None, food_preferences=None, foods_to_avoid=None, notes=None,
)

ALICE_ID = "11111111-1111-4111-8111-111111111111"
BOB_ID = "22222222-2222-4222-8222-222222222222"


def _fake_httpx_response(json_body):
    response = mock.Mock()
    response.status_code = 200
    response.json.return_value = json_body
    response.raise_for_status.return_value = None
    return response


def _chat_completion(content):
    return {"choices": [{"message": {"content": content}}]}


def _patched_api_key(value="test-key-not-real"):
    import llm_provider
    return mock.patch.object(llm_provider, "OPENROUTER_API_KEY", value)


def _system_prompt(post_mock, call_index=0):
    return post_mock.call_args_list[call_index].kwargs["json"]["messages"][0]["content"]


# ----------------------------------------------------------------------
# 1. 1-day chat still works
# ----------------------------------------------------------------------

class TestOneDayChatStillWorks(MemoryIsolatedTestCase):
    def test_one_day_plan_reaches_the_chat_prompt(self):
        create_meal_plan(MealPlanRequest(**dict(BASE, athlete_id=ALICE_ID)))
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            chat.chat_reply("What am I eating today?", athlete_id=ALICE_ID)

        system_content = _system_prompt(post)
        self.assertIn("MOST RECENT MEAL PLAN", system_content)
        self.assertIn("Duration: 1 day(s)", system_content)
        self.assertNotIn("(no meal plan generated yet)", system_content)

    def test_existing_episodic_recent_plans_block_is_unaffected(self):
        """The OLDER episodic "RECENT MEAL PLANS" mechanism must still work
        exactly as before for the 1-day case - this feature is additive."""
        create_meal_plan(MealPlanRequest(**dict(BASE, athlete_id=ALICE_ID)))
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            chat.chat_reply("hi", athlete_id=ALICE_ID)

        system_content = _system_prompt(post)
        self.assertIn("RECENT MEAL PLANS", system_content)
        self.assertIn("goal: Improve football performance", system_content)


# ----------------------------------------------------------------------
# 2. 7-day plan creates usable chat context
# ----------------------------------------------------------------------

class TestSevenDayPlanUsableInChat(MemoryIsolatedTestCase):
    def test_seven_day_plan_context_has_all_days_and_protein_sources(self):
        create_meal_plan(MealPlanRequest(**dict(BASE, duration_days=7, athlete_id=ALICE_ID)))
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            chat.chat_reply("How does my week support my goal?", athlete_id=ALICE_ID)

        system_content = _system_prompt(post)
        self.assertIn("Duration: 7 day(s)", system_content)
        self.assertIn("Protein sources across this plan", system_content)
        for day_n in range(1, 8):
            self.assertIn(f"Day {day_n}:", system_content)

    def test_seven_day_plan_is_never_written_to_episodic_history(self):
        """The Phase 2 decision stands: episodic stays single-day-only."""
        create_meal_plan(MealPlanRequest(**dict(BASE, duration_days=7, athlete_id=ALICE_ID)))
        self.assertEqual(mm.list("episodic", ALICE_ID), [])


# ----------------------------------------------------------------------
# 3. Chat can distinguish Day 1 vs Day 4
# ----------------------------------------------------------------------

class TestDayDistinction(MemoryIsolatedTestCase):
    def test_day_1_and_day_4_lines_are_different(self):
        create_meal_plan(MealPlanRequest(**dict(BASE, duration_days=7, athlete_id=ALICE_ID)))
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            chat.chat_reply("What am I eating on Day 4?", athlete_id=ALICE_ID)

        system_content = _system_prompt(post)
        day1_line = re.search(r"- Day 1:.*", system_content).group()
        day4_line = re.search(r"- Day 4:.*", system_content).group()
        self.assertNotEqual(day1_line, day4_line)


# ----------------------------------------------------------------------
# 4. A new plan supersedes the old plan
# ----------------------------------------------------------------------

class TestNewPlanSupersedesOld(MemoryIsolatedTestCase):
    def test_generating_a_new_plan_replaces_the_previous_context(self):
        create_meal_plan(MealPlanRequest(**dict(BASE, duration_days=7, athlete_id=ALICE_ID)))
        create_meal_plan(MealPlanRequest(**dict(BASE, duration_days=1, weight_kg=80, athlete_id=ALICE_ID)))

        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            chat.chat_reply("hi", athlete_id=ALICE_ID)

        system_content = _system_prompt(post)
        self.assertIn("Duration: 1 day(s)", system_content)
        self.assertNotIn("Duration: 7 day(s)", system_content)

    def test_only_one_latest_plan_is_ever_stored_not_accumulated(self):
        create_meal_plan(MealPlanRequest(**dict(BASE, duration_days=7, athlete_id=ALICE_ID)))
        create_meal_plan(MealPlanRequest(**dict(BASE, duration_days=1, athlete_id=ALICE_ID)))
        create_meal_plan(MealPlanRequest(**dict(BASE, duration_days=7, athlete_id=ALICE_ID)))

        stored = mm.get("latest_plan", ALICE_ID)
        self.assertEqual(stored["duration_days"], 7)
        self.assertEqual(len(stored["days"]), 7)  # the newest 7-day plan, not a merge of all three


# ----------------------------------------------------------------------
# 5. Athlete A's plan cannot appear for Athlete B
# ----------------------------------------------------------------------

class TestAthleteIsolation(MemoryIsolatedTestCase):
    def test_bob_with_no_plan_gets_the_placeholder_not_alices_data(self):
        create_meal_plan(MealPlanRequest(**dict(
            BASE, duration_days=7, region="Chennai, Tamil Nadu", athlete_id=ALICE_ID,
        )))
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            chat.chat_reply("hi", athlete_id=BOB_ID)

        system_content = _system_prompt(post)
        self.assertIn("(no meal plan generated yet)", system_content)
        self.assertNotIn("Chennai", system_content)

    def test_switching_athletes_never_exposes_the_others_plan(self):
        create_meal_plan(MealPlanRequest(**dict(BASE, duration_days=7, athlete_id=ALICE_ID)))
        create_meal_plan(MealPlanRequest(**dict(
            BASE, duration_days=1, region="Kolkata, West Bengal", diet="Non-Vegetarian", athlete_id=BOB_ID,
        )))

        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            chat.chat_reply("hi", athlete_id=ALICE_ID, session_id="s-a")
            chat.chat_reply("hi", athlete_id=BOB_ID, session_id="s-b")

        alice_prompt = _system_prompt(post, call_index=0)
        bob_prompt = _system_prompt(post, call_index=1)
        self.assertIn("Duration: 7 day(s)", alice_prompt)
        self.assertIn("Duration: 1 day(s)", bob_prompt)
        self.assertNotIn("Kolkata", alice_prompt)
        self.assertNotIn("day(s); Goal: Improve football performance; Diet: Vegetarian; Region: Lucknow",
                          bob_prompt)


# ----------------------------------------------------------------------
# 6. No ingredients/notes are persisted
# ----------------------------------------------------------------------

class TestNoIngredientsOrNotesPersisted(MemoryIsolatedTestCase):
    def test_available_ingredients_never_appear_in_latest_plan(self):
        create_meal_plan(MealPlanRequest(**dict(
            BASE, duration_days=7, available_ingredients="rice, dal, SECRET_PANTRY_MARKER",
            athlete_id=ALICE_ID,
        )))
        stored = mm.get("latest_plan", ALICE_ID)
        self.assertNotIn("available_ingredients", stored)
        self.assertNotIn("SECRET_PANTRY_MARKER", json.dumps(stored))

    def test_notes_never_appear_in_latest_plan(self):
        create_meal_plan(MealPlanRequest(**dict(
            BASE, duration_days=7, notes="SECRET_NOTE_MARKER match tomorrow", athlete_id=ALICE_ID,
        )))
        stored = mm.get("latest_plan", ALICE_ID)
        self.assertNotIn("notes", stored)
        self.assertNotIn("SECRET_NOTE_MARKER", json.dumps(stored))

    def test_goal_training_days_intensity_are_not_written_to_profile_memory(self):
        """Explicit guard against the exact thing the task warned against:
        this feature must not start persisting goal/training_days/
        training_intensity into Profile memory just because they now also
        appear in the latest_plan summary."""
        create_meal_plan(MealPlanRequest(**dict(BASE, duration_days=7, athlete_id=ALICE_ID)))
        stored_profile = mm.get("profile", ALICE_ID) or {}
        for field in ("goal", "training_days_per_week", "training_intensity"):
            self.assertNotIn(field, stored_profile)


# ----------------------------------------------------------------------
# 7. Existing Phase 1 diet/age/allergy constraints remain intact
# ----------------------------------------------------------------------

class TestPhase1ConstraintsIntact(MemoryIsolatedTestCase):
    def test_hard_rules_still_present_alongside_the_latest_plan_block(self):
        create_meal_plan(MealPlanRequest(**dict(
            BASE, age=10, diet="Vegetarian", allergies="dairy", duration_days=7, athlete_id=ALICE_ID,
        )))
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            chat.chat_reply("What should I eat before training?", athlete_id=ALICE_ID)

        system_content = _system_prompt(post)
        self.assertIn("AGE / DEVELOPMENT RULE", system_content)
        self.assertIn("DIET RULE", system_content)
        self.assertIn("ALLERGY / AVOIDANCE RULE", system_content)
        self.assertIn("MOST RECENT MEAL PLAN", system_content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
