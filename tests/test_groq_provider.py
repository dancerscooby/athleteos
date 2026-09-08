"""
AthleteOS - Tests for llm_provider.groq_provider() (Groq evaluated as a
third, free inference provider alongside mock / Claude / OpenRouter).

No real Groq HTTP requests are made anywhere in this file - httpx's POST
call is mocked at the module boundary (mock.patch("httpx.post")), exactly
the technique tests/test_openrouter_provider.py already uses. Zero network,
zero cost, and no GROQ_API_KEY needed to run these.

Covers:
    1. The provider returns a parsed plan and satisfies the shared
       provider contract (same signature and same output shape as
       mock_provider / openrouter_provider).
    2. The request payload is the OpenAI-compatible shape Groq documents,
       against the right URL, with the right auth header.
    3. Structured output really is schema-enforced: the response_format
       carries the EXISTING MEAL_PLAN_SCHEMA with strict=true, unmodified.
    4. agent_context (RAG context + tool results the orchestrator already
       computed) reaches the prompt - i.e. the existing deterministic
       agent/tool loop keeps working with this provider.
    5. Graceful failure: missing key, HTTP 429 -> RateLimitError, other
       HTTP errors, network errors, empty choices, unparseable JSON, and a
       well-formed-but-incomplete plan all raise LLMProviderError (or its
       RateLimitError subclass) rather than crashing.
    6. Provider switching: generate_meal_plan()'s new "groq" branch, and
       the fact that multi-day is deliberately NOT implemented for Groq.
    7. The existing OpenRouter provider is completely untouched by this
       addition.
"""

import json
import unittest
from unittest import mock

import llm_provider
import meal_planner
from llm_provider import LLMProviderError, RateLimitError, groq_provider

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
    """A minimal stand-in for httpx.Response - only what call_groq reads."""
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
    """Shape of a successful Groq (OpenAI-compatible) chat completion."""
    return {"choices": [{"message": {"content": content}}]}


def _patched_api_key(value="test-key-not-real"):
    return mock.patch.object(llm_provider, "GROQ_API_KEY", value)


def _call(profile=None, agent_context=None, body=None, status_code=200):
    """Run groq_provider with a mocked HTTP layer; returns (plan, post_mock)."""
    if body is None:
        body = _chat_completion(json.dumps(FAKE_PLAN))
    with _patched_api_key(), mock.patch("httpx.post") as post:
        post.return_value = _fake_httpx_response(status_code, body)
        plan = groq_provider(profile or dict(BASE_PROFILE), agent_context)
    return plan, post


# ----------------------------------------------------------------------
# 1. Provider contract
# ----------------------------------------------------------------------

class TestProviderContract(unittest.TestCase):
    def test_returns_the_parsed_plan(self):
        plan, _ = _call()
        self.assertEqual(plan, FAKE_PLAN)

    def test_output_shape_matches_the_other_providers(self):
        plan, _ = _call()
        mock_plan = meal_planner.mock_provider(dict(BASE_PROFILE))
        self.assertEqual(set(plan), set(mock_plan))

    def test_agent_context_is_optional(self):
        """Same signature as mock_provider/openrouter_provider: the second
        argument defaults to None and the call must still work."""
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(
                200, _chat_completion(json.dumps(FAKE_PLAN))
            )
            plan = groq_provider(dict(BASE_PROFILE))
        self.assertEqual(plan, FAKE_PLAN)


# ----------------------------------------------------------------------
# 2-3. Request payload and schema-enforced structured output
# ----------------------------------------------------------------------

class TestRequestPayload(unittest.TestCase):
    def _sent(self):
        _, post = _call()
        return post.call_args

    def test_posts_to_groqs_openai_compatible_endpoint(self):
        args = self._sent()
        self.assertEqual(args.args[0], llm_provider.GROQ_API_URL)
        self.assertIn("api.groq.com/openai/v1/chat/completions", llm_provider.GROQ_API_URL)

    def test_sends_the_api_key_as_a_bearer_token(self):
        args = self._sent()
        self.assertEqual(
            args.kwargs["headers"]["Authorization"], "Bearer test-key-not-real"
        )

    def test_sends_the_configured_model_and_token_budget(self):
        payload = self._sent().kwargs["json"]
        self.assertEqual(payload["model"], llm_provider.GROQ_MODEL)
        # Groq/OpenAI deprecated max_tokens for reasoning models in favour
        # of max_completion_tokens - the Groq path must use the new name.
        self.assertEqual(payload["max_completion_tokens"], llm_provider.GROQ_MAX_TOKENS)
        self.assertNotIn("max_tokens", payload)

    def test_default_model_is_the_one_being_evaluated(self):
        self.assertEqual(llm_provider.GROQ_MODEL, "openai/gpt-oss-20b")

    def test_sends_a_reasoning_effort_gpt_oss_actually_accepts(self):
        """gpt-oss on Groq accepts only low/medium/high - NOT the "none"
        the OpenRouter path uses. Sending "none" would be rejected."""
        payload = self._sent().kwargs["json"]
        self.assertEqual(payload["reasoning_effort"], llm_provider.GROQ_REASONING_EFFORT)
        self.assertIn(llm_provider.GROQ_REASONING_EFFORT, {"low", "medium", "high"})

    def test_sends_a_single_user_message_containing_the_prompt(self):
        messages = self._sent().kwargs["json"]["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("JSON", messages[0]["content"])


class TestStructuredOutput(unittest.TestCase):
    def test_uses_schema_enforced_json_not_just_json_object(self):
        _, post = _call()
        response_format = post.call_args.kwargs["json"]["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])

    def test_sends_the_existing_meal_plan_schema_unmodified(self):
        _, post = _call()
        sent = post.call_args.kwargs["json"]["response_format"]["json_schema"]["schema"]
        self.assertIs(sent, llm_provider.MEAL_PLAN_SCHEMA)

    def test_meal_plan_schema_satisfies_groq_strict_mode_requirements(self):
        """strict=true is only valid if every object lists all of its
        properties in "required" and sets additionalProperties: false. This
        asserts the existing schema already does, at every level - which is
        why it can be reused as-is instead of being modified."""
        def check(node, path="root"):
            if not isinstance(node, dict):
                return
            if node.get("type") == "object":
                self.assertEqual(
                    set(node.get("properties", {})), set(node.get("required", [])), path
                )
                self.assertIs(node.get("additionalProperties"), False, path)
                for key, value in node.get("properties", {}).items():
                    check(value, path + "." + key)
            elif node.get("type") == "array":
                check(node.get("items", {}), path + "[]")

        check(llm_provider.MEAL_PLAN_SCHEMA)


# ----------------------------------------------------------------------
# 4. agent_context reaches the prompt (the existing agent/tool loop)
# ----------------------------------------------------------------------

class TestAgentContext(unittest.TestCase):
    def test_rag_context_and_tool_results_reach_the_prompt(self):
        agent_context = {
            "rag_context": "RAG_SENTINEL_TEXT",
            "tool_results": {"nutrition_tool": {"protein": {"protein_g": 123.4}}},
        }
        _, post = _call(agent_context=agent_context)
        prompt = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("RAG_SENTINEL_TEXT", prompt)
        self.assertIn("123.4", prompt)

    def test_the_provider_never_runs_rag_or_tools_itself(self):
        """Like openrouter_provider, this must NOT do its own retrieval or
        tool calling - the orchestrator already did both before calling it.
        A second retrieval here would duplicate the app's heaviest step."""
        with mock.patch("llm_provider.get_context_block") as retrieval, \
             mock.patch("llm_provider.run_with_tools") as tool_loop:
            _call(agent_context={"rag_context": "x", "tool_results": {}})
        retrieval.assert_not_called()
        tool_loop.assert_not_called()


# ----------------------------------------------------------------------
# 5. Graceful failure - one LLMProviderError, never a crash
# ----------------------------------------------------------------------

class TestGracefulFailure(unittest.TestCase):
    def test_missing_api_key_raises_llm_provider_error(self):
        with mock.patch.object(llm_provider, "GROQ_API_KEY", None):
            with self.assertRaises(LLMProviderError) as ctx:
                groq_provider(dict(BASE_PROFILE))
        self.assertIn("GROQ_API_KEY", str(ctx.exception))

    def test_missing_api_key_makes_no_http_request(self):
        with mock.patch.object(llm_provider, "GROQ_API_KEY", None), \
             mock.patch("httpx.post") as post:
            with self.assertRaises(LLMProviderError):
                groq_provider(dict(BASE_PROFILE))
        post.assert_not_called()

    def test_http_429_raises_rate_limit_error(self):
        with self.assertRaises(RateLimitError):
            _call(status_code=429)

    def test_rate_limit_error_is_still_an_llm_provider_error(self):
        """Existing `except LLMProviderError` handling in api.py must keep
        catching Groq rate limits without any change."""
        self.assertTrue(issubclass(RateLimitError, LLMProviderError))
        with self.assertRaises(LLMProviderError):
            _call(status_code=429)

    def test_other_http_errors_raise_llm_provider_error_not_rate_limit(self):
        with self.assertRaises(LLMProviderError) as ctx:
            _call(status_code=500)
        self.assertNotIsInstance(ctx.exception, RateLimitError)

    def test_network_error_raises_llm_provider_error(self):
        import httpx
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.side_effect = httpx.ConnectError("no route to host")
            with self.assertRaises(LLMProviderError):
                groq_provider(dict(BASE_PROFILE))

    def test_api_level_error_object_raises_llm_provider_error(self):
        with self.assertRaises(LLMProviderError):
            _call(body={"error": {"message": "model_decommissioned"}})

    def test_empty_choices_raises_llm_provider_error(self):
        with self.assertRaises(LLMProviderError):
            _call(body={"choices": []})

    def test_empty_content_raises_llm_provider_error(self):
        with self.assertRaises(LLMProviderError):
            _call(body=_chat_completion(""))

    def test_unparseable_json_raises_llm_provider_error(self):
        with self.assertRaises(LLMProviderError):
            _call(body=_chat_completion("Here is your plan: {not json"))

    def test_valid_json_missing_required_fields_raises_llm_provider_error(self):
        incomplete = {k: v for k, v in FAKE_PLAN.items() if k != "meals"}
        with self.assertRaises(LLMProviderError) as ctx:
            _call(body=_chat_completion(json.dumps(incomplete)))
        self.assertIn("meals", str(ctx.exception))


# ----------------------------------------------------------------------
# 6. Provider switching through generate_meal_plan()
# ----------------------------------------------------------------------

class TestProviderSwitching(unittest.TestCase):
    def test_generate_meal_plan_routes_to_groq_when_selected(self):
        with mock.patch.object(meal_planner, "PROVIDER", "groq"), \
             mock.patch("llm_provider.groq_provider", return_value=FAKE_PLAN) as provider:
            plan = meal_planner.generate_meal_plan(dict(BASE_PROFILE))
        provider.assert_called_once()
        self.assertEqual(plan, FAKE_PLAN)

    def test_agent_context_is_forwarded_through_generate_meal_plan(self):
        agent_context = {"rag_context": "ctx", "tool_results": {}}
        with mock.patch.object(meal_planner, "PROVIDER", "groq"), \
             mock.patch("llm_provider.groq_provider", return_value=FAKE_PLAN) as provider:
            meal_planner.generate_meal_plan(dict(BASE_PROFILE), agent_context=agent_context)
        self.assertEqual(provider.call_args.args[1], agent_context)

    def test_multi_day_is_deliberately_not_implemented_for_groq(self):
        """Multi-day was explicitly out of scope for this evaluation, so it
        must fail loudly rather than silently ignoring `days`."""
        with mock.patch.object(meal_planner, "PROVIDER", "groq"):
            with self.assertRaises(NotImplementedError):
                meal_planner.generate_meal_plan(dict(BASE_PROFILE), days=7)

    def test_default_provider_is_still_not_groq(self):
        """Adding this provider must not change what runs by default."""
        self.assertNotEqual(meal_planner.PROVIDER, "groq")


# ----------------------------------------------------------------------
# 7. The existing OpenRouter provider is untouched
# ----------------------------------------------------------------------

class TestOpenRouterUnchanged(unittest.TestCase):
    def test_openrouter_config_is_unchanged(self):
        self.assertEqual(
            llm_provider.OPENROUTER_API_URL,
            "https://openrouter.ai/api/v1/chat/completions",
        )
        self.assertEqual(llm_provider.OPENROUTER_EFFORT, "none")

    def test_openrouter_provider_still_uses_json_object_not_groqs_schema(self):
        with mock.patch.object(llm_provider, "OPENROUTER_API_KEY", "test-key"), \
             mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(
                200, _chat_completion(json.dumps(FAKE_PLAN))
            )
            llm_provider.openrouter_provider(dict(BASE_PROFILE))
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertIn("max_tokens", payload)
        self.assertEqual(payload["reasoning"], {"effort": llm_provider.OPENROUTER_EFFORT})

    def test_a_groq_failure_cannot_affect_the_openrouter_path(self):
        """The two providers share no mutable state - call_groq failing
        must leave call_openrouter fully usable in the same process."""
        with mock.patch.object(llm_provider, "GROQ_API_KEY", None):
            with self.assertRaises(LLMProviderError):
                groq_provider(dict(BASE_PROFILE))
        with mock.patch.object(llm_provider, "OPENROUTER_API_KEY", "test-key"), \
             mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(
                200, _chat_completion(json.dumps(FAKE_PLAN))
            )
            plan = llm_provider.openrouter_provider(dict(BASE_PROFILE))
        self.assertEqual(plan, FAKE_PLAN)


if __name__ == "__main__":
    unittest.main(verbosity=2)
