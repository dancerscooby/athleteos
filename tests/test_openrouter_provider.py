"""
AthleteOS - Tests for llm_provider.openrouter_provider() (the second, free
LLM provider, alongside the existing mock and Claude providers).

No real OpenRouter HTTP requests are made anywhere in this file - httpx's
POST call is mocked at the module boundary (mock.patch("httpx.post")), the
same technique test_agent_context_integration.py uses for the Anthropic SDK.
Zero network, zero cost.

Covers exactly what the task asked for:
    1. The correct prompt/context reaches the provider.
    2. Structured-output (JSON) parsing of a successful response.
    3. Missing API key handling.
    4. Malformed model output handling.
    5. The mock provider remains completely unchanged.
    6. Provider switching (generate_meal_plan's "openrouter" branch).
    7. (Verified separately: the full existing suite still passes.)
"""

import json
import unittest
from unittest import mock

import llm_provider
import meal_planner
from llm_provider import LLMProviderError, build_prompt, openrouter_provider
from meal_planner import generate_meal_plan, mock_provider

BASE_PROFILE = dict(
    age=22, height_cm=178, weight_kg=72, sex="Male",
    goal="Improve football performance",
    diet="Vegetarian", training_days_per_week=5, training_intensity="High",
    region="Lucknow, Uttar Pradesh", allergies="none",
    budget_per_day="Rs 300", food_preferences=None, foods_to_avoid=None, notes=None,
)

FAKE_PLAN = {
    "confidence_score": 8.0,
    "confidence_reason": ["test"],
    "day_summary": {
        "total_calories": 2000, "total_protein_g": 100,
        "total_carbs_g": 250, "total_fats_g": 60, "notes": "test",
    },
    "meals": [],
    "disclaimer": "test disclaimer",
}


def _fake_httpx_response(status_code=200, json_body=None):
    """A minimal stand-in for httpx.Response - only what openrouter_provider reads."""
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
    """Shape of a successful OpenRouter (OpenAI-compatible) chat completion."""
    return {"choices": [{"message": {"content": content}}]}


def _patched_api_key(value="test-key-not-real"):
    return mock.patch.object(llm_provider, "OPENROUTER_API_KEY", value)


# ----------------------------------------------------------------------
# 1. The correct prompt/context reaches the provider
# ----------------------------------------------------------------------

class TestPromptAndContextReachTheProvider(unittest.TestCase):
    def test_base_prompt_reaches_openrouter_with_no_agent_context(self):
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(json_body=_chat_completion(json.dumps(FAKE_PLAN)))
            openrouter_provider(BASE_PROFILE)

        sent_body = post.call_args.kwargs["json"]
        self.assertEqual(sent_body["messages"][0]["content"], build_prompt(BASE_PROFILE))

    def test_rag_context_reaches_the_prompt(self):
        agent_context = {"rag_context": "UNIQUE_RAG_MARKER_XYZ", "tool_results": {}}
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(json_body=_chat_completion(json.dumps(FAKE_PLAN)))
            openrouter_provider(BASE_PROFILE, agent_context)

        sent_prompt = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("UNIQUE_RAG_MARKER_XYZ", sent_prompt)
        self.assertIn("Reference material", sent_prompt)

    def test_tool_results_reach_the_prompt(self):
        agent_context = {
            "rag_context": "",
            "tool_results": {
                "nutrition_tool": {"protein": 129.6, "calories": 3110, "carbs": 576, "hydration": 3.52},
            },
        }
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(json_body=_chat_completion(json.dumps(FAKE_PLAN)))
            openrouter_provider(BASE_PROFILE, agent_context)

        sent_prompt = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("Precomputed values", sent_prompt)
        self.assertIn("129.6", sent_prompt)
        self.assertIn("3110", sent_prompt)

    def test_no_rag_retrieval_happens_when_agent_context_is_none(self):
        """openrouter_provider must NEVER retrieve RAG context itself - even
        with no agent_context at all (unlike claude_provider's standalone mode)."""
        with _patched_api_key(), \
             mock.patch.object(llm_provider, "get_context_block") as get_ctx, \
             mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(json_body=_chat_completion(json.dumps(FAKE_PLAN)))
            openrouter_provider(BASE_PROFILE)  # no agent_context

        get_ctx.assert_not_called()

    def test_model_and_key_are_sent_correctly(self):
        with _patched_api_key("secret-abc"), \
             mock.patch.object(llm_provider, "OPENROUTER_MODEL", "some/free-model:free"), \
             mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(json_body=_chat_completion(json.dumps(FAKE_PLAN)))
            openrouter_provider(BASE_PROFILE)

        call = post.call_args
        self.assertEqual(call.kwargs["json"]["model"], "some/free-model:free")
        self.assertEqual(call.kwargs["headers"]["Authorization"], "Bearer secret-abc")

    def test_reasoning_effort_is_capped_to_reduce_hidden_token_usage(self):
        """gpt-oss-style reasoning models can burn the whole token budget on
        hidden reasoning and return empty content on a long prompt (live-
        verified) - reasoning.effort must be sent to cap that."""
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(json_body=_chat_completion(json.dumps(FAKE_PLAN)))
            openrouter_provider(BASE_PROFILE)

        self.assertEqual(post.call_args.kwargs["json"]["reasoning"], {"effort": llm_provider.OPENROUTER_EFFORT})


# ----------------------------------------------------------------------
# 2. Structured-output (JSON) parsing of a successful response
# ----------------------------------------------------------------------

class TestSuccessfulResponseParsing(unittest.TestCase):
    def test_valid_json_response_is_parsed_and_returned(self):
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(json_body=_chat_completion(json.dumps(FAKE_PLAN)))
            plan = openrouter_provider(BASE_PROFILE)

        self.assertEqual(plan, FAKE_PLAN)

    def test_json_object_response_format_is_requested(self):
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(json_body=_chat_completion(json.dumps(FAKE_PLAN)))
            openrouter_provider(BASE_PROFILE)

        self.assertEqual(
            post.call_args.kwargs["json"]["response_format"], {"type": "json_object"}
        )


# ----------------------------------------------------------------------
# 3. Missing API key handling
# ----------------------------------------------------------------------

class TestMissingApiKey(unittest.TestCase):
    def test_missing_key_raises_llm_provider_error(self):
        with _patched_api_key(None), mock.patch("httpx.post") as post:
            with self.assertRaises(LLMProviderError) as ctx:
                openrouter_provider(BASE_PROFILE)
        self.assertIn("OPENROUTER_API_KEY", str(ctx.exception))
        post.assert_not_called()  # never even attempts the request

    def test_missing_key_does_not_crash_the_app(self):
        """Fails gracefully - a caller can catch this and fall back to mock."""
        with _patched_api_key(None):
            try:
                openrouter_provider(BASE_PROFILE)
                self.fail("expected LLMProviderError")
            except LLMProviderError:
                pass  # exactly the graceful failure contract
        # And the mock provider is still perfectly usable right after.
        plan = mock_provider(dict(BASE_PROFILE))
        self.assertIn("day_summary", plan)


# ----------------------------------------------------------------------
# 4. Malformed model output handling
# ----------------------------------------------------------------------

class TestMalformedOutputHandling(unittest.TestCase):
    def test_non_json_text_raises_llm_provider_error(self):
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(
                json_body=_chat_completion("Sure! Here's a great meal plan for you: ...")
            )
            with self.assertRaises(LLMProviderError):
                openrouter_provider(BASE_PROFILE)

    def test_json_missing_required_fields_raises_llm_provider_error(self):
        incomplete_plan = {"confidence_score": 8.0}  # missing everything else
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(json_body=_chat_completion(json.dumps(incomplete_plan)))
            with self.assertRaises(LLMProviderError) as ctx:
                openrouter_provider(BASE_PROFILE)
        self.assertIn("missing required fields", str(ctx.exception))

    def test_empty_choices_raises_llm_provider_error(self):
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(json_body={"choices": []})
            with self.assertRaises(LLMProviderError):
                openrouter_provider(BASE_PROFILE)

    def test_api_error_payload_raises_llm_provider_error(self):
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(
                json_body={"error": {"message": "model is currently rate-limited"}}
            )
            with self.assertRaises(LLMProviderError) as ctx:
                openrouter_provider(BASE_PROFILE)
        self.assertIn("rate-limited", str(ctx.exception))

    def test_http_error_status_raises_llm_provider_error(self):
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(status_code=401, json_body={})
            with self.assertRaises(LLMProviderError):
                openrouter_provider(BASE_PROFILE)

    def test_429_raises_the_specific_rate_limit_error_subclass(self):
        """openrouter_provider() shares call_openrouter() with chat.py, so
        the meal-plan path gets the same RateLimitError distinction."""
        from llm_provider import RateLimitError
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(status_code=429, json_body={})
            with self.assertRaises(RateLimitError):
                openrouter_provider(BASE_PROFILE)
            with self.assertRaises(LLMProviderError):
                openrouter_provider(BASE_PROFILE)

    def test_network_exception_raises_llm_provider_error(self):
        import httpx
        with _patched_api_key(), mock.patch("httpx.post", side_effect=httpx.ConnectError("no network")):
            with self.assertRaises(LLMProviderError):
                openrouter_provider(BASE_PROFILE)


# ----------------------------------------------------------------------
# 5. The mock provider remains completely unchanged
# ----------------------------------------------------------------------

class TestMockProviderUnaffected(unittest.TestCase):
    def test_mock_provider_output_unchanged_by_this_feature(self):
        with mock.patch.object(meal_planner, "PROVIDER", "mock"):
            plan = generate_meal_plan(dict(BASE_PROFILE))
        direct = mock_provider(dict(BASE_PROFILE))
        self.assertEqual(plan, direct)

    def test_mock_path_never_imports_or_calls_openrouter(self):
        with mock.patch.object(meal_planner, "PROVIDER", "mock"), \
             mock.patch("llm_provider.openrouter_provider") as openrouter_mock:
            generate_meal_plan(dict(BASE_PROFILE), agent_context={"rag_context": "x", "tool_results": {}})
        openrouter_mock.assert_not_called()


# ----------------------------------------------------------------------
# 6. Provider switching
# ----------------------------------------------------------------------

class TestProviderSwitching(unittest.TestCase):
    def test_openrouter_branch_forwards_profile_and_agent_context(self):
        agent_context = {"rag_context": "ctx", "tool_results": {}}
        with mock.patch.object(meal_planner, "PROVIDER", "openrouter"), \
             mock.patch("llm_provider.openrouter_provider", return_value=FAKE_PLAN) as provider_mock:
            plan = generate_meal_plan(dict(BASE_PROFILE), agent_context=agent_context)

        provider_mock.assert_called_once_with(dict(BASE_PROFILE), agent_context)
        self.assertEqual(plan, FAKE_PLAN)

    def test_switching_between_all_three_providers(self):
        with mock.patch.object(meal_planner, "PROVIDER", "mock"):
            mock_plan = generate_meal_plan(dict(BASE_PROFILE))
        with mock.patch.object(meal_planner, "PROVIDER", "claude"), \
             mock.patch("llm_provider.claude_provider", return_value=FAKE_PLAN):
            claude_plan = generate_meal_plan(dict(BASE_PROFILE))
        with mock.patch.object(meal_planner, "PROVIDER", "openrouter"), \
             mock.patch("llm_provider.openrouter_provider", return_value=FAKE_PLAN):
            openrouter_plan = generate_meal_plan(dict(BASE_PROFILE))

        self.assertIn("day_summary", mock_plan)
        self.assertEqual(claude_plan, FAKE_PLAN)
        self.assertEqual(openrouter_plan, FAKE_PLAN)

    def test_unknown_provider_still_raises_value_error(self):
        with mock.patch.object(meal_planner, "PROVIDER", "nonexistent"):
            with self.assertRaises(ValueError):
                generate_meal_plan(dict(BASE_PROFILE))


if __name__ == "__main__":
    unittest.main(verbosity=2)
