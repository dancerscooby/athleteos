"""
AthleteOS - Phase 1 hardening tests: chat respects the active athlete's
diet/age/allergy/avoidance/region constraints as HARD rules, not soft
suggestions, and chat retrieval is profile-aware like the meal-plan path.

Root cause this fixes (see chat.py's module docstring): a 10-year-old
Vegetarian athlete asking a routine fueling question previously got a reply
suggesting "1 boiled egg (if you're okay with eggs)" - the profile block
included diet/age facts, but only as informational text ("use when
relevant"), and chat's own RAG retrieval had no age/region top-up, so it
could surface diet-agnostic reference material (a general "eating out"
chunk that happens to list egg dishes) without anything telling the model
diet/age were non-negotiable.

These tests verify the INPUT the chat pipeline sends to the LLM (the
assembled system prompt) rather than the LLM's own output - the same
approach test_age_developmental_stage.py already uses for
meal_planner.build_prompt(), since we cannot assert on what a live model
will say without making a real API call. No live OpenRouter calls anywhere
in this file - httpx.post is mocked at the module boundary, same technique
as tests/test_chat.py.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

import chat
import memory_manager as mm

ATHLETE_ID = "99999999-9999-4999-8999-999999999999"
ALICE_ID = "11111111-1111-4111-8111-111111111111"
BOB_ID = "22222222-2222-4222-8222-222222222222"


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
# 1. 10-year-old vegetarian: hard age + diet rules reach the prompt
# ----------------------------------------------------------------------

class TestMinorVegetarianHardRules(MemoryIsolatedTestCase):
    def test_age_and_diet_hard_rules_are_in_the_prompt(self):
        mm.set("profile", ATHLETE_ID, {
            "age": 10, "diet": "Vegetarian", "region": "Chennai, Tamil Nadu", "allergies": "none",
        })
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            chat.chat_reply("What should I eat before training?", athlete_id=ATHLETE_ID)

        system_content = _system_prompt(post)
        self.assertIn("AGE / DEVELOPMENT RULE", system_content)
        self.assertIn("This athlete's age is 10.", system_content)
        self.assertIn("not a smaller adult", system_content)
        self.assertIn("DIET RULE", system_content)
        self.assertIn("no meat, fish, or eggs by default", system_content)

    def test_rules_are_marked_as_hard_constraints_not_suggestions(self):
        """The system instructions must explicitly say these rules override
        general knowledge - the exact soft-instruction gap that let the
        model hedge on diet before this fix."""
        mm.set("profile", ATHLETE_ID, {"age": 10, "diet": "Vegetarian", "allergies": "none"})
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            chat.chat_reply("hi", athlete_id=ATHLETE_ID)

        system_content = _system_prompt(post)
        self.assertIn("hard constraints", system_content.lower())


# ----------------------------------------------------------------------
# 2. Dairy allergy: no dairy recommendations
# ----------------------------------------------------------------------

class TestDairyAllergyHardRule(MemoryIsolatedTestCase):
    def test_dairy_allergy_expands_to_concrete_excluded_foods(self):
        mm.set("profile", ATHLETE_ID, {
            "age": 25, "diet": "Vegetarian", "allergies": "dairy", "region": "Lucknow, Uttar Pradesh",
        })
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            chat.chat_reply("What protein sources work for me?", athlete_id=ATHLETE_ID)

        system_content = _system_prompt(post).lower()
        self.assertIn("allergy / avoidance rule".lower(), system_content)
        for word in ["curd", "paneer", "milk", "cheese", "ghee", "buttermilk"]:
            self.assertIn(word, system_content)

    def test_no_allergies_or_avoidances_emits_no_rule_section(self):
        """A profile with nothing to exclude shouldn't get an empty/awkward
        ALLERGY / AVOIDANCE RULE heading with nothing under it."""
        mm.set("profile", ATHLETE_ID, {"age": 25, "diet": "Vegetarian", "allergies": "none"})
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            chat.chat_reply("hi", athlete_id=ATHLETE_ID)

        system_content = _system_prompt(post)
        self.assertNotIn("ALLERGY / AVOIDANCE RULE", system_content)


# ----------------------------------------------------------------------
# 3. Foods to avoid: respected
# ----------------------------------------------------------------------

class TestFoodsToAvoidHardRule(MemoryIsolatedTestCase):
    def test_foods_to_avoid_reach_the_hard_rule(self):
        mm.set("profile", ATHLETE_ID, {"age": 25, "diet": "Vegetarian", "allergies": "none"})
        mm.set("preference", ATHLETE_ID, {"foods_to_avoid": "mushrooms"})
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            chat.chat_reply("Any snack ideas?", athlete_id=ATHLETE_ID)

        system_content = _system_prompt(post).lower()
        self.assertIn("allergy / avoidance rule".lower(), system_content)
        self.assertIn("mushroom", system_content)

    def test_allergies_and_foods_to_avoid_combine_in_one_rule(self):
        mm.set("profile", ATHLETE_ID, {"age": 25, "diet": "Vegetarian", "allergies": "peanuts"})
        mm.set("preference", ATHLETE_ID, {"foods_to_avoid": "mushrooms"})
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            chat.chat_reply("Any snack ideas?", athlete_id=ATHLETE_ID)

        system_content = _system_prompt(post).lower()
        self.assertIn("peanut", system_content)
        self.assertIn("mushroom", system_content)


# ----------------------------------------------------------------------
# 4. Chennai / South Indian profile: regional context available
# ----------------------------------------------------------------------

class TestRegionalContextAvailable(MemoryIsolatedTestCase):
    def test_south_indian_region_surfaces_regional_reference_material(self):
        """Real RAG pipeline (offline FAISS + reranker, no network) - proves
        the profile-aware region top-up (rag.retrieve_profile_chunks) is
        actually wired into chat now, not just message-only retrieval."""
        mm.set("profile", ATHLETE_ID, {
            "age": 25, "diet": "Vegetarian", "region": "Chennai, Tamil Nadu", "allergies": "none",
        })
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            chat.chat_reply("What should I eat before training?", athlete_id=ATHLETE_ID)

        system_content = _system_prompt(post)
        self.assertIn("REFERENCE MATERIAL", system_content)
        self.assertIn("indian_regional_eating_patterns.md", system_content)


# ----------------------------------------------------------------------
# 5. Adult profile: adult guidance remains possible (not over-restricted)
# ----------------------------------------------------------------------

class TestAdultProfileUnaffected(MemoryIsolatedTestCase):
    def test_adult_gets_a_normal_reply_and_the_correct_age_in_the_rule(self):
        mm.set("profile", ATHLETE_ID, {
            "age": 25, "diet": "Vegetarian", "region": "Lucknow, Uttar Pradesh", "allergies": "none",
        })
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("Try dal and paneer for protein."))
            result = chat.chat_reply("What should I eat for protein?", athlete_id=ATHLETE_ID)

        self.assertEqual(result["reply"], "Try dal and paneer for protein.")
        system_content = _system_prompt(post)
        self.assertIn("This athlete's age is 25.", system_content)

    def test_no_saved_profile_yields_no_constraint_section_and_still_works(self):
        """A brand-new athlete with no memory at all gets no constraint
        block (nothing to constrain on) and chat still answers normally -
        this fix must not make chat unusable before a profile exists."""
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("General fueling advice."))
            result = chat.chat_reply("What should I eat before training?", athlete_id="brand-new-athlete")

        self.assertEqual(result["reply"], "General fueling advice.")
        system_content = _system_prompt(post)
        self.assertNotIn("AGE / DEVELOPMENT RULE", system_content)
        self.assertNotIn("DIET RULE", system_content)
        self.assertIn("no saved profile yet", system_content)


# ----------------------------------------------------------------------
# 6. Switching athletes: chat uses the newly active athlete's profile only
# ----------------------------------------------------------------------

class TestSwitchingAthletesUsesNewProfile(MemoryIsolatedTestCase):
    def test_switching_never_carries_the_previous_athletes_constraints(self):
        mm.set("profile", ALICE_ID, {
            "age": 10, "diet": "Vegetarian", "allergies": "dairy", "region": "Chennai, Tamil Nadu",
        })
        mm.set("profile", BOB_ID, {
            "age": 30, "diet": "Non-Vegetarian", "allergies": "none", "region": "Lucknow, Uttar Pradesh",
        })

        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            chat.chat_reply("hi", athlete_id=ALICE_ID, session_id="session-a")
            chat.chat_reply("hi", athlete_id=BOB_ID, session_id="session-b")

        alice_prompt = _system_prompt(post, call_index=0)
        bob_prompt = _system_prompt(post, call_index=1)

        self.assertIn("This athlete's age is 10.", alice_prompt)
        self.assertIn("ALLERGY / AVOIDANCE RULE", alice_prompt)
        self.assertNotIn("This athlete's age is 30.", alice_prompt)

        self.assertIn("This athlete's age is 30.", bob_prompt)
        self.assertNotIn("ALLERGY / AVOIDANCE RULE", bob_prompt)  # Bob has no allergies
        self.assertNotIn("This athlete's age is 10.", bob_prompt)
        # Bob's own RAG reference material may legitimately mention dairy
        # (e.g. north Indian regional patterns) - that's not a leak. What
        # must never appear is ALICE's profile facts.
        self.assertNotIn("Chennai", bob_prompt)
        self.assertNotIn("age: 10", bob_prompt)

    def test_same_athlete_id_reused_after_a_different_athlete_gets_fresh_data(self):
        """Not just isolation between two ids in one process - re-reading
        the SAME athlete_id after their stored profile changed (e.g. they
        updated diet) must reflect the new value, proving nothing is cached
        across calls."""
        mm.set("profile", ATHLETE_ID, {"age": 20, "diet": "Vegetarian", "allergies": "none"})
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            chat.chat_reply("hi", athlete_id=ATHLETE_ID, session_id="s1")

        mm.set("profile", ATHLETE_ID, {"age": 20, "diet": "Non-Vegetarian", "allergies": "none"})
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            chat.chat_reply("hi", athlete_id=ATHLETE_ID, session_id="s2")

        system_content = _system_prompt(post)
        self.assertIn("diet: Non-Vegetarian", system_content)


# ----------------------------------------------------------------------
# 7. Chat API failure: clean user-facing error, no leaked internals
# ----------------------------------------------------------------------

class TestCleanErrorOnProviderFailure(MemoryIsolatedTestCase):
    def _client(self):
        from fastapi.testclient import TestClient
        from api import app
        return TestClient(app)

    def test_network_failure_returns_a_clean_message_not_the_raw_exception(self):
        client = self._client()
        with _patched_api_key(), mock.patch(
            "httpx.post", side_effect=httpx.ConnectError("Connection refused to internal-host:9999")
        ):
            response = client.post("/chat", json={"message": "hello", "athlete_id": ATHLETE_ID})

        self.assertEqual(response.status_code, 503)
        detail = response.json()["detail"]
        self.assertNotIn("internal-host", detail)
        self.assertNotIn("Connection refused", detail)
        self.assertIn("try again", detail.lower())

    def test_missing_api_key_returns_a_clean_message_not_the_env_var_name(self):
        client = self._client()
        with _patched_api_key(None):
            response = client.post("/chat", json={"message": "hello", "athlete_id": ATHLETE_ID})

        self.assertEqual(response.status_code, 503)
        detail = response.json()["detail"]
        self.assertNotIn("OPENROUTER_API_KEY", detail)
        self.assertIn("try again", detail.lower())

    def test_malformed_provider_error_payload_is_not_echoed_to_the_athlete(self):
        client = self._client()
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(
                {"error": {"message": "internal-secret-token-abc123 rejected"}}
            )
            response = client.post("/chat", json={"message": "hello", "athlete_id": ATHLETE_ID})

        self.assertEqual(response.status_code, 503)
        detail = response.json()["detail"]
        self.assertNotIn("internal-secret-token-abc123", detail)


if __name__ == "__main__":
    unittest.main(verbosity=2)
