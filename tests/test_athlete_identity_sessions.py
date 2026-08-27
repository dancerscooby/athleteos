"""
AthleteOS - Phase 1 tests: athlete identity + chat session isolation.

Proves the two guarantees Phase 1 exists for:

  IDENTITY  - an athlete is a UUID, never a name. Two athletes with the
              SAME display name have different ids and their stored
              profile/preference/episodic memory can never mix.

  SESSIONS  - chat transcripts are session-scoped and in-memory only:
              a new session starts empty, the same athlete continues within
              one session, a different athlete gets a completely separate
              conversation, and nothing survives a "reload".

No live OpenRouter calls anywhere - httpx.post is mocked at the module
boundary, the same technique tests/test_chat.py and
tests/test_openrouter_provider.py already use.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import chat
import episodic_memory
import memory_integration as mi
import memory_manager as mm
from api import MealPlanRequest, create_meal_plan

# Two DIFFERENT athletes who happen to share a display name - the exact
# collision the whole identity design exists to prevent.
ALICE_ID = "11111111-1111-4111-8111-111111111111"
BOB_ID = "22222222-2222-4222-8222-222222222222"
SHARED_NAME = "Rahul Sharma"


class MemoryIsolatedTestCase(unittest.TestCase):
    """Redirects memory_manager's storage into a fresh temp dir per test, and
    clears session bindings so tests never leak state into each other."""

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


BASE_PROFILE = dict(
    age=22, height_cm=178, weight_kg=72, sex="Male",
    goal="Improve football performance",
    diet="Vegetarian", training_days_per_week=5, training_intensity="High",
    region="Lucknow, Uttar Pradesh", allergies="none",
    budget_per_day=None, food_preferences=None, foods_to_avoid=None, notes=None,
)


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


# ----------------------------------------------------------------------
# 1. Identity: same name, different UUIDs, no shared memory
# ----------------------------------------------------------------------

class TestSameNameDifferentIdentities(MemoryIsolatedTestCase):
    def test_two_athletes_with_the_same_name_have_different_ids(self):
        """The frontend stores {id, name} pairs; the name is a label. This
        asserts the invariant the backend depends on: identity is the id."""
        alice = {"id": ALICE_ID, "name": SHARED_NAME}
        bob = {"id": BOB_ID, "name": SHARED_NAME}
        self.assertEqual(alice["name"], bob["name"])
        self.assertNotEqual(alice["id"], bob["id"])

    def test_profile_memory_cannot_mix_between_same_name_athletes(self):
        mm.set("profile", ALICE_ID, {"weight_kg": 72, "diet": "Vegetarian"})
        mm.set("profile", BOB_ID, {"weight_kg": 95, "diet": "Non-Vegetarian"})

        self.assertEqual(mm.get("profile", ALICE_ID)["weight_kg"], 72)
        self.assertEqual(mm.get("profile", BOB_ID)["weight_kg"], 95)
        self.assertEqual(mm.get("profile", ALICE_ID)["diet"], "Vegetarian")
        self.assertEqual(mm.get("profile", BOB_ID)["diet"], "Non-Vegetarian")

    def test_preference_memory_cannot_mix_between_same_name_athletes(self):
        mm.set("preference", ALICE_ID, {"food_preferences": "paneer"})
        mm.set("preference", BOB_ID, {"food_preferences": "chicken"})
        self.assertEqual(mm.get("preference", ALICE_ID)["food_preferences"], "paneer")
        self.assertEqual(mm.get("preference", BOB_ID)["food_preferences"], "chicken")

    def test_episodic_memory_is_isolated_per_athlete(self):
        plan = {
            "confidence_score": 8,
            "day_summary": {"total_calories": 2800},
            "meals": [{"meal": "Breakfast", "ingredients": [{"item": "Poha", "quantity": "1 plate"}]}],
        }
        episodic_memory.record_episode({"goal": "alice goal"}, plan, athlete_id=ALICE_ID)

        self.assertEqual(len(mm.list("episodic", ALICE_ID)), 1)
        self.assertEqual(mm.list("episodic", BOB_ID), [])
        self.assertEqual(mm.list("episodic", ALICE_ID)[0]["goal"], "alice goal")

    def test_merge_with_memory_never_pulls_another_athletes_profile(self):
        mm.set("profile", ALICE_ID, {"weight_kg": 72, "allergies": "peanuts"})
        # Bob has nothing stored - his blanks must stay blank, not inherit Alice's.
        merged = mi.merge_with_memory(dict(BASE_PROFILE, weight_kg=None, allergies=None), BOB_ID)
        self.assertIsNone(merged["weight_kg"])
        self.assertIsNone(merged["allergies"])


# ----------------------------------------------------------------------
# 2. Identity through the REAL /meal-plan endpoint
# ----------------------------------------------------------------------

class TestMealPlanEndpointIdentity(MemoryIsolatedTestCase):
    def test_plan_writes_memory_under_the_supplied_athlete_id(self):
        request = MealPlanRequest(**dict(BASE_PROFILE, athlete_id=ALICE_ID))
        create_meal_plan(request)

        self.assertIsNotNone(mm.get("profile", ALICE_ID))
        self.assertEqual(mm.get("profile", BOB_ID), None)
        self.assertEqual(len(mm.list("episodic", ALICE_ID)), 1)
        self.assertEqual(mm.list("episodic", BOB_ID), [])

    def test_two_same_name_athletes_keep_separate_stored_profiles(self):
        create_meal_plan(MealPlanRequest(**dict(BASE_PROFILE, weight_kg=72, athlete_id=ALICE_ID)))
        create_meal_plan(MealPlanRequest(**dict(BASE_PROFILE, weight_kg=95, athlete_id=BOB_ID)))

        self.assertEqual(mm.get("profile", ALICE_ID)["weight_kg"], 72)
        self.assertEqual(mm.get("profile", BOB_ID)["weight_kg"], 95)

    def test_athlete_id_is_not_passed_into_the_planning_pipeline(self):
        """Identity selects whose memory to use; it must not leak into the
        profile the planner/prompt sees. Spies on the real orchestrator call
        while delegating to the real implementation, so behaviour is
        observed, not changed."""
        import orchestrator
        real_run_with_trace = orchestrator.run_with_trace
        captured = {}

        def spy(profile, athlete_id=mi.DEFAULT_ATHLETE_ID, **kwargs):
            captured["profile"] = profile
            captured["athlete_id"] = athlete_id
            return real_run_with_trace(profile, athlete_id, **kwargs)

        with mock.patch("api.run_with_trace", side_effect=spy):
            create_meal_plan(MealPlanRequest(**dict(BASE_PROFILE, athlete_id=ALICE_ID)))

        self.assertNotIn("athlete_id", captured["profile"])
        self.assertEqual(captured["athlete_id"], ALICE_ID)

    def test_response_schema_is_unchanged(self):
        response = create_meal_plan(MealPlanRequest(**dict(BASE_PROFILE, athlete_id=ALICE_ID)))
        self.assertEqual(
            set(response.keys()),
            {"confidence_score", "confidence_reason", "day_summary", "meals", "disclaimer"},
        )

    def test_omitting_athlete_id_still_uses_the_default_identity(self):
        """Migration safety: a caller that predates Phase 1 (CLI, old tests,
        a cached frontend) keeps reading/writing the existing 'default' memory."""
        create_meal_plan(MealPlanRequest(**BASE_PROFILE))
        self.assertIsNotNone(mm.get("profile", mi.DEFAULT_ATHLETE_ID))

    def test_malformed_athlete_id_is_rejected(self):
        """athlete_id reaches a filesystem path (memory/episodes/<id>.json),
        so a traversal-style value must never validate."""
        from pydantic import ValidationError
        for bad in ["../../etc/passwd", "not-a-uuid", "", "alice/../bob"]:
            with self.assertRaises(ValidationError):
                MealPlanRequest(**dict(BASE_PROFILE, athlete_id=bad))


# ----------------------------------------------------------------------
# 3. Chat session isolation
# ----------------------------------------------------------------------

SESSION_A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
SESSION_B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


class TestChatSessionIsolation(MemoryIsolatedTestCase):
    def test_same_athlete_same_session_continues_the_conversation(self):
        history = [
            {"role": "user", "content": "What's a good pre-match breakfast?"},
            {"role": "assistant", "content": "Poha with peanuts works well."},
        ]
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("You could also try oats."))
            chat.chat_reply("Any other options?", athlete_id=ALICE_ID,
                            history=history, session_id=SESSION_A)

        sent = post.call_args.kwargs["json"]["messages"]
        self.assertEqual(sent[1]["content"], "What's a good pre-match breakfast?")
        self.assertEqual(sent[2]["content"], "Poha with peanuts works well.")
        self.assertEqual(sent[3]["content"], "Any other options?")

    def test_a_different_athlete_cannot_reuse_another_athletes_session(self):
        """The core cross-contamination guard: Bob presenting Alice's
        session_id is refused outright, so he can never continue - or be
        answered inside - her conversation."""
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            chat.chat_reply("hello", athlete_id=ALICE_ID, session_id=SESSION_A)

            with self.assertRaises(chat.SessionOwnershipError):
                chat.chat_reply(
                    "what did the other athlete just ask?",
                    athlete_id=BOB_ID,
                    history=[{"role": "user", "content": "Alice's private message"}],
                    session_id=SESSION_A,
                )

    def test_each_athlete_in_their_own_session_works_independently(self):
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            chat.chat_reply("hi", athlete_id=ALICE_ID, session_id=SESSION_A)
            chat.chat_reply("hi", athlete_id=BOB_ID, session_id=SESSION_B)
        self.assertEqual(post.call_count, 2)

    def test_switching_athlete_starts_an_empty_conversation(self):
        """After a switch the frontend sends a NEW session and empty history -
        confirm the prompt then contains no earlier turns at all."""
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            chat.chat_reply("Alice's question", athlete_id=ALICE_ID,
                            history=[{"role": "user", "content": "Alice's earlier turn"}],
                            session_id=SESSION_A)
            chat.chat_reply("Bob's question", athlete_id=BOB_ID, history=[], session_id=SESSION_B)

        bob_messages = post.call_args.kwargs["json"]["messages"]
        self.assertEqual(len(bob_messages), 2)  # system + Bob's message only
        joined = " ".join(m["content"] for m in bob_messages)
        self.assertNotIn("Alice's earlier turn", joined)
        self.assertNotIn("Alice's question", joined)

    def test_reload_starts_a_new_session_with_empty_history(self):
        """A reload means a fresh session_id and no history from the client -
        the same athlete simply begins again with nothing carried over."""
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            chat.chat_reply("before reload", athlete_id=ALICE_ID,
                            history=[{"role": "user", "content": "old turn"}],
                            session_id=SESSION_A)
            chat.chat_reply("after reload", athlete_id=ALICE_ID, history=[], session_id=SESSION_B)

        messages = post.call_args.kwargs["json"]["messages"]
        self.assertEqual(len(messages), 2)
        self.assertNotIn("old turn", " ".join(m["content"] for m in messages))

    def test_same_athlete_may_open_several_sessions(self):
        """One athlete on two devices/tabs is fine - only cross-ATHLETE reuse
        of a session is refused."""
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            chat.chat_reply("tab one", athlete_id=ALICE_ID, session_id=SESSION_A)
            chat.chat_reply("tab two", athlete_id=ALICE_ID, session_id=SESSION_B)
        self.assertEqual(post.call_count, 2)

    def test_chat_messages_are_never_written_to_disk(self):
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("a reply worth finding"))
            chat.chat_reply("a message worth finding", athlete_id=ALICE_ID, session_id=SESSION_A)

        written = "".join(
            path.read_text(encoding="utf-8")
            for path in Path(mm.MEMORY_DIR).rglob("*.json")
        )
        self.assertNotIn("a message worth finding", written)
        self.assertNotIn("a reply worth finding", written)

    def test_session_bindings_do_not_survive_a_restart(self):
        """reset_sessions() stands in for a process restart: bindings are
        in-memory only, so the id is claimable again afterwards."""
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            chat.chat_reply("hi", athlete_id=ALICE_ID, session_id=SESSION_A)
            chat.reset_sessions()
            chat.chat_reply("hi", athlete_id=BOB_ID, session_id=SESSION_A)  # must not raise
        self.assertEqual(post.call_count, 2)

    def test_omitting_session_id_keeps_previous_behaviour(self):
        """Callers that predate Phase 1 (CLI, direct calls) still work."""
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            chat.chat_reply("hi", athlete_id=ALICE_ID)
            chat.chat_reply("hi", athlete_id=BOB_ID)
        self.assertEqual(post.call_count, 2)


# ----------------------------------------------------------------------
# 4. Chat isolation through the REAL /chat endpoint
# ----------------------------------------------------------------------

class TestChatEndpointIsolation(MemoryIsolatedTestCase):
    def _client(self):
        from fastapi.testclient import TestClient
        from api import app
        return TestClient(app)

    def test_session_reuse_by_another_athlete_returns_409(self):
        client = self._client()
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            first = client.post("/chat", json={
                "message": "hello", "athlete_id": ALICE_ID, "session_id": SESSION_A,
            })
            second = client.post("/chat", json={
                "message": "hello", "athlete_id": BOB_ID, "session_id": SESSION_A,
            })

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 409)

    def test_each_athlete_reads_only_their_own_memory(self):
        mm.set("preference", ALICE_ID, {"food_preferences": "ALICE_ONLY_MARKER"})
        mm.set("preference", BOB_ID, {"food_preferences": "BOB_ONLY_MARKER"})

        client = self._client()
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(_chat_completion("ok"))
            client.post("/chat", json={
                "message": "what do I like?", "athlete_id": BOB_ID, "session_id": SESSION_B,
            })

        system_prompt = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("BOB_ONLY_MARKER", system_prompt)
        self.assertNotIn("ALICE_ONLY_MARKER", system_prompt)

    def test_malformed_athlete_id_is_rejected_by_the_endpoint(self):
        client = self._client()
        response = client.post("/chat", json={
            "message": "hello", "athlete_id": "../../etc/passwd",
        })
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main(verbosity=2)
