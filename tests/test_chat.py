"""
AthleteOS - Tests for chat.py ("Ask AthleteOS").

No real OpenRouter HTTP requests are made anywhere in this file - httpx's
POST call is mocked at the module boundary (mock.patch("httpx.post")), same
technique as tests/test_openrouter_provider.py. The RAG retrieval test is
the one exception: it exercises the REAL retriever/FAISS/reranker pipeline
(offline, local models) to prove chat is genuinely wired into it, matching
how test_orchestrator.py already tests real RAG retrieval elsewhere.

Covers exactly what the task asked for:
    1. Normal chat (a full round trip through chat_reply()).
    2. RAG context retrieval reaches the prompt.
    3. Memory (profile/preference/episodic) is available to the prompt.
    4. Conversation history reaches the request sent to OpenRouter.
    5. Missing API key.
    6. Malformed provider response.
    7. (Verified separately: the full existing suite still passes.)

Plus light coverage of the one memory-write path (extract_preference_update)
since the task explicitly asked for careful, simple behaviour there.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import chat
import episodic_memory
import memory_manager as mm
from llm_provider import LLMProviderError


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

    def tearDown(self):
        for name, value in self._originals.items():
            setattr(mm, name, value)
        self._tmpdir.cleanup()


# A valid UUID4: api.py validates athlete_id against ATHLETE_ID_PATTERN
# (UUID or "default") since Phase 1, both to keep identity opaque and to stop
# a traversal-style value reaching memory/episodes/<athlete_id>.json.
ATHLETE_ID = "99999999-9999-4999-8999-999999999999"


def _fake_httpx_response(status_code=200, json_body=None):
    response = mock.Mock()
    response.status_code = status_code
    response.json.return_value = json_body or {}
    if status_code >= 400:
        import httpx
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=mock.Mock(), response=response
        )
    else:
        response.raise_for_status.return_value = None
    return response


def _chat_completion(content):
    return {"choices": [{"message": {"content": content}}]}


def _patched_api_key(value="test-key-not-real"):
    import llm_provider
    return mock.patch.object(llm_provider, "OPENROUTER_API_KEY", value)


# ----------------------------------------------------------------------
# 1. Normal chat
# ----------------------------------------------------------------------

class TestNormalChat(MemoryIsolatedTestCase):
    def test_chat_reply_returns_the_models_text(self):
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(
                json_body=_chat_completion("Try swapping paneer for soy chunks - similar protein, cheaper.")
            )
            result = chat.chat_reply("Can I substitute paneer for something cheaper?", athlete_id=ATHLETE_ID)

        self.assertEqual(result["reply"], "Try swapping paneer for soy chunks - similar protein, cheaper.")
        self.assertIsNone(result["memory_updated"])

    def test_chat_reply_uses_call_openrouter_not_a_new_provider(self):
        """Reuses llm_provider.call_openrouter() - the exact same function
        openrouter_provider() uses for meal plans - proving no second
        LLM-calling code path was created."""
        with _patched_api_key(), mock.patch("chat.call_openrouter", return_value="ok") as called:
            chat.chat_reply("hello", athlete_id=ATHLETE_ID)
        called.assert_called_once()

    def test_no_response_format_is_sent_for_chat(self):
        """Chat replies are free text, not the meal-plan JSON schema."""
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(json_body=_chat_completion("Sure, here's an idea."))
            chat.chat_reply("What should I eat before training?", athlete_id=ATHLETE_ID)
        self.assertNotIn("response_format", post.call_args.kwargs["json"])


# ----------------------------------------------------------------------
# 2. RAG context retrieval reaches the prompt (REAL retriever, no mocking)
# ----------------------------------------------------------------------

class TestRagContextReachesThePrompt(MemoryIsolatedTestCase):
    def test_relevant_chunks_are_retrieved_and_included(self):
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(json_body=_chat_completion("Stay hydrated!"))
            chat.chat_reply("How much water should I drink before a match?", athlete_id=ATHLETE_ID)

        sent_messages = post.call_args.kwargs["json"]["messages"]
        system_content = sent_messages[0]["content"]
        self.assertIn("REFERENCE MATERIAL", system_content)
        # The hydration guide should be the real, semantically-retrieved
        # match for a hydration question - not an empty/placeholder block.
        self.assertNotIn("(nothing relevant found)", system_content)

    def test_build_messages_calls_the_existing_profile_aware_retriever(self):
        """Phase 1 hardening: chat now retrieves via rag.retrieve_profile_chunks()
        (profile, query, ...) - the SAME profile-aware path (with age/region
        top-ups) the meal-plan pipeline uses - instead of a message-only
        retrieval. This replaces the old test that asserted on the
        message-only retriever.retrieve_relevant_chunks() call, which no
        longer exists in chat.py."""
        with mock.patch("chat.retrieve_profile_chunks", return_value=[]) as retrieve:
            chat.build_messages("some question", ATHLETE_ID, history=None)
        retrieve.assert_called_once()
        self.assertEqual(retrieve.call_args.args[1], "some question")


# ----------------------------------------------------------------------
# 3. Memory (profile/preference/episodic) is available to the prompt
# ----------------------------------------------------------------------

class TestMemoryIsAvailable(MemoryIsolatedTestCase):
    def test_profile_and_preference_reach_the_prompt(self):
        mm.set("profile", ATHLETE_ID, {"age": 22, "diet": "Vegetarian", "region": "Lucknow, Uttar Pradesh"})
        mm.set("preference", ATHLETE_ID, {"food_preferences": "paneer, dal", "budget_per_day": "Rs 300"})

        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(json_body=_chat_completion("ok"))
            chat.chat_reply("What's my usual diet?", athlete_id=ATHLETE_ID)

        system_content = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("diet: Vegetarian", system_content)
        self.assertIn("paneer, dal", system_content)
        self.assertIn("Rs 300", system_content)

    def test_recent_episodes_reach_the_prompt(self):
        fake_plan = {
            "confidence_score": 8,
            "day_summary": {"total_calories": 2800},
            "meals": [{"meal": "Breakfast", "ingredients": [{"item": "Poha", "quantity": "1 plate"}]}],
        }
        episodic_memory.record_episode(
            {"goal": "Improve football performance", "training_intensity": "High", "training_days_per_week": 5},
            fake_plan,
            athlete_id=ATHLETE_ID,
        )

        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(json_body=_chat_completion("ok"))
            chat.chat_reply("What did my last plan look like?", athlete_id=ATHLETE_ID)

        system_content = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("Improve football performance", system_content)
        self.assertIn("2800", system_content)
        self.assertIn("Poha", system_content)

    def test_missing_memory_does_not_crash_and_says_so(self):
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(json_body=_chat_completion("ok"))
            chat.chat_reply("hello", athlete_id="brand-new-athlete-never-seen-before")

        system_content = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("no saved profile yet", system_content)
        self.assertIn("no saved preferences yet", system_content)
        self.assertIn("no recent meal plans on record", system_content)


# ----------------------------------------------------------------------
# 4. Conversation history reaches the request
# ----------------------------------------------------------------------

class TestConversationHistory(MemoryIsolatedTestCase):
    def test_recent_turns_are_included_in_order(self):
        history = [
            {"role": "user", "content": "What's a good pre-match breakfast?"},
            {"role": "assistant", "content": "Poha with peanuts works well."},
        ]
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(json_body=_chat_completion("You could also try oats."))
            chat.chat_reply("Any other options?", athlete_id=ATHLETE_ID, history=history)

        sent_messages = post.call_args.kwargs["json"]["messages"]
        # system, then the 2 history turns, then the new user message
        self.assertEqual(sent_messages[1], {"role": "user", "content": "What's a good pre-match breakfast?"})
        self.assertEqual(sent_messages[2], {"role": "assistant", "content": "Poha with peanuts works well."})
        self.assertEqual(sent_messages[3], {"role": "user", "content": "Any other options?"})

    def test_history_is_capped_to_the_most_recent_turns(self):
        long_history = [{"role": "user", "content": f"message {i}"} for i in range(20)]
        messages = chat.build_messages("latest question", ATHLETE_ID, history=long_history)

        # system + capped history + new message
        self.assertEqual(len(messages), 1 + chat.MAX_HISTORY_TURNS + 1)
        self.assertEqual(messages[1]["content"], f"message {20 - chat.MAX_HISTORY_TURNS}")

    def test_no_history_still_works(self):
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(json_body=_chat_completion("Sure."))
            chat.chat_reply("hi", athlete_id=ATHLETE_ID, history=None)
        sent_messages = post.call_args.kwargs["json"]["messages"]
        self.assertEqual(len(sent_messages), 2)  # system + the new message only


# ----------------------------------------------------------------------
# 5. Missing API key
# ----------------------------------------------------------------------

class TestMissingApiKey(MemoryIsolatedTestCase):
    def test_missing_key_raises_llm_provider_error(self):
        with _patched_api_key(None), mock.patch("httpx.post") as post:
            with self.assertRaises(LLMProviderError):
                chat.chat_reply("hello", athlete_id=ATHLETE_ID)
        post.assert_not_called()

    def test_missing_key_via_the_real_endpoint_returns_503(self):
        """Phase 1 hardening: the endpoint must return a clean, generic
        message rather than leaking the internal env var name or any raw
        provider exception text - this replaces the old assertion that
        specifically expected "OPENROUTER_API_KEY" to appear in the
        athlete-facing detail, which was exactly the leak being fixed."""
        from fastapi.testclient import TestClient
        from api import app

        with _patched_api_key(None):
            client = TestClient(app)
            response = client.post("/chat", json={"message": "hello", "athlete_id": ATHLETE_ID})

        self.assertEqual(response.status_code, 503)
        detail = response.json()["detail"]
        self.assertNotIn("OPENROUTER_API_KEY", detail)
        self.assertIn("try again", detail.lower())


# ----------------------------------------------------------------------
# 6. Malformed provider response
# ----------------------------------------------------------------------

class TestMalformedProviderResponse(MemoryIsolatedTestCase):
    def test_empty_choices_raises(self):
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(json_body={"choices": []})
            with self.assertRaises(LLMProviderError):
                chat.chat_reply("hello", athlete_id=ATHLETE_ID)

    def test_api_error_payload_raises(self):
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(json_body={"error": {"message": "rate-limited"}})
            with self.assertRaises(LLMProviderError) as ctx:
                chat.chat_reply("hello", athlete_id=ATHLETE_ID)
            self.assertIn("rate-limited", str(ctx.exception))

    def test_http_error_status_raises(self):
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(status_code=500, json_body={})
            with self.assertRaises(LLMProviderError):
                chat.chat_reply("hello", athlete_id=ATHLETE_ID)

    def test_malformed_response_via_the_real_endpoint_returns_503(self):
        from fastapi.testclient import TestClient
        from api import app

        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(json_body={"choices": []})
            client = TestClient(app)
            response = client.post("/chat", json={"message": "hello", "athlete_id": ATHLETE_ID})

        self.assertEqual(response.status_code, 503)


# ----------------------------------------------------------------------
# 7. OpenRouter 429 rate-limit handling
# ----------------------------------------------------------------------
# Root cause (see llm_provider.RateLimitError's docstring): the chat feature
# always makes a real OpenRouter call against a free-tier, shared-pool
# model, which can return HTTP 429 when that shared pool is oversubscribed -
# live-reproduced with real /chat calls returning intermixed 200s and 429s.
# This is an external capacity limit, not a bug in prompt/memory/session
# handling, so a 429 still gets a specific, honest athlete-facing message.
#
# Phase D update: call_openrouter() now tries OPENROUTER_FALLBACK_MODEL once
# if OPENROUTER_MODEL's call fails for ANY reason, including a 429 - this is
# NOT the same thing the original comment here ruled out. Retrying the SAME
# oversubscribed model/pool would only add to its congestion (still never
# done - see test_a_429_tries_the_fallback_model_exactly_once below, which
# proves there is still no repeated retry of either model). Trying a
# DIFFERENT model on a different pool is a genuinely separate attempt, not
# congestion-adding, so it is now the intended behaviour.

class TestRateLimitHandling(MemoryIsolatedTestCase):
    def test_429_raises_the_specific_rate_limit_error_type(self):
        from llm_provider import RateLimitError
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(status_code=429, json_body={})
            with self.assertRaises(RateLimitError):
                chat.chat_reply("hello", athlete_id=ATHLETE_ID)

    def test_429_is_still_a_plain_llm_provider_error(self):
        """RateLimitError must be a LLMProviderError subclass so any existing
        `except LLMProviderError` handling elsewhere keeps working unchanged."""
        from llm_provider import RateLimitError
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(status_code=429, json_body={})
            with self.assertRaises(LLMProviderError):
                chat.chat_reply("hello", athlete_id=ATHLETE_ID)
        self.assertTrue(issubclass(RateLimitError, LLMProviderError))

    def test_429_via_the_real_endpoint_returns_the_specific_friendly_message(self):
        from fastapi.testclient import TestClient
        from api import app

        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(status_code=429, json_body={})
            client = TestClient(app)
            response = client.post("/chat", json={"message": "hello", "athlete_id": ATHLETE_ID})

        self.assertEqual(response.status_code, 503)
        detail = response.json()["detail"]
        self.assertIn("temporarily busy", detail)
        self.assertIn("try again", detail.lower())

    def test_non_429_provider_error_still_gets_the_generic_message(self):
        """A non-429 failure (e.g. a 500 from OpenRouter) must keep using the
        existing generic message, not the new 429-specific one - proves the
        two error paths stay distinct rather than the new branch swallowing
        everything."""
        from fastapi.testclient import TestClient
        from api import app

        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(status_code=500, json_body={})
            client = TestClient(app)
            response = client.post("/chat", json={"message": "hello", "athlete_id": ATHLETE_ID})

        self.assertEqual(response.status_code, 503)
        detail = response.json()["detail"]
        self.assertNotIn("temporarily busy", detail)
        self.assertIn("isn't available right now", detail)

    def test_missing_api_key_still_gets_the_generic_message_not_429_message(self):
        """A missing-key LLMProviderError never even reaches httpx, so it
        can't carry a 429 status - must still fall through to the existing
        generic branch, not the new rate-limit one."""
        from fastapi.testclient import TestClient
        from api import app

        with _patched_api_key(None):
            client = TestClient(app)
            response = client.post("/chat", json={"message": "hello", "athlete_id": ATHLETE_ID})

        self.assertEqual(response.status_code, 503)
        detail = response.json()["detail"]
        self.assertNotIn("temporarily busy", detail)

    def test_a_429_tries_the_fallback_model_exactly_once(self):
        """A 429 on the primary model triggers exactly one fallback attempt
        (a different model/pool) - never a retry loop, and never more than
        the one fallback call even though both calls in this test return
        429 (proving there is no repeated retry of either model)."""
        import llm_provider
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(status_code=429, json_body={})
            try:
                chat.chat_reply("hello", athlete_id=ATHLETE_ID)
            except LLMProviderError:
                pass
        self.assertEqual(post.call_count, 2)
        models_called = [call.kwargs["json"]["model"] for call in post.call_args_list]
        self.assertEqual(models_called, [llm_provider.OPENROUTER_MODEL, llm_provider.OPENROUTER_FALLBACK_MODEL])

    def test_fallback_success_returns_a_normal_reply_not_an_error(self):
        """If the primary model 429s but the fallback model succeeds, the
        athlete gets a normal reply - the 429 is fully absorbed, not
        surfaced as an error."""
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.side_effect = [
                _fake_httpx_response(status_code=429, json_body={}),
                _fake_httpx_response(json_body=_chat_completion("Fallback reply")),
            ]
            reply = chat.chat_reply("hello", athlete_id=ATHLETE_ID)
        self.assertEqual(reply["reply"], "Fallback reply")
        self.assertEqual(post.call_count, 2)


# ----------------------------------------------------------------------
# Safety-instructions presence (deterministic - it's a fixed system prompt)
# ----------------------------------------------------------------------

class TestSafetyInstructionsAlwaysPresent(MemoryIsolatedTestCase):
    def test_safety_boundary_text_is_always_in_the_prompt(self):
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(json_body=_chat_completion("ok"))
            chat.chat_reply("How can I lose 10kg fast for wrestling weigh-in?", athlete_id=ATHLETE_ID)

        system_content = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("not a doctor", system_content)
        self.assertIn("pediatrician", system_content)


# ----------------------------------------------------------------------
# Narrow, deterministic preference-saving rule
# ----------------------------------------------------------------------

class TestExtractPreferenceUpdate(unittest.TestCase):
    def test_remember_i_like_is_recognized(self):
        self.assertEqual(
            chat.extract_preference_update("Remember I like sweet potatoes"),
            ("food_preferences", "sweet potatoes"),
        )

    def test_remember_i_avoid_is_recognized(self):
        self.assertEqual(
            chat.extract_preference_update("remember that I avoid mushrooms"),
            ("foods_to_avoid", "mushrooms"),
        )

    def test_ordinary_message_mentioning_food_is_not_saved(self):
        self.assertIsNone(chat.extract_preference_update("I had paneer for lunch today, was it a good choice?"))

    def test_a_question_about_a_preference_is_not_saved(self):
        self.assertIsNone(chat.extract_preference_update("Do you remember what I like to eat?"))


class TestPreferenceUpdateEndToEnd(MemoryIsolatedTestCase):
    def test_explicit_preference_message_is_saved_and_reported(self):
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(json_body=_chat_completion("Got it, noted!"))
            result = chat.chat_reply("Remember I like sweet potatoes", athlete_id=ATHLETE_ID)

        self.assertEqual(result["memory_updated"], {"field": "food_preferences", "value": "sweet potatoes"})
        stored = mm.get("preference", ATHLETE_ID)
        self.assertEqual(stored["food_preferences"], "sweet potatoes")

    def test_preference_update_appends_to_existing_value(self):
        mm.set("preference", ATHLETE_ID, {"food_preferences": "paneer, dal"})
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(json_body=_chat_completion("Noted."))
            chat.chat_reply("Remember I like sweet potatoes", athlete_id=ATHLETE_ID)

        stored = mm.get("preference", ATHLETE_ID)
        self.assertEqual(stored["food_preferences"], "paneer, dal, sweet potatoes")

    def test_ordinary_chat_never_writes_to_memory(self):
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(json_body=_chat_completion("Sure, here's an idea."))
            chat.chat_reply("What's a good post-training snack?", athlete_id=ATHLETE_ID)

        self.assertIsNone(mm.get("preference", ATHLETE_ID))


if __name__ == "__main__":
    unittest.main(verbosity=2)
