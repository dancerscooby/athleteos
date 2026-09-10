"""
AthleteOS - Tests for provider-failure diagnostic classification.

WHY THIS EXISTS
    Both /meal-plan and /chat map every non-429 provider failure onto one
    generic 503. That is correct for the athlete, but it meant the SERVER
    LOG said only "LLM provider failed" for an upstream 502, a read
    timeout, a DNS failure, a 200 response carrying an error object, and a
    body that would not parse as JSON. Diagnosing a real production 503
    twice required querying the database to work out how far the request
    had got, because the log line carried nothing.

    LLMProviderError now optionally carries provider / category /
    status_code / detail, and api.py logs them via format_provider_error().
    This is diagnostic visibility ONLY: no retries were added, no user
    facing behaviour changed, and every one of these failures still returns
    the same 503 with the same text.

No network anywhere in this file - httpx.post is mocked throughout.

Covers:
    1. Each failure category is classified distinctly.
    2. RateLimitError behaviour is completely preserved.
    3. The user-facing 503 responses are unchanged and leak nothing.
    4. Secrets are redacted from anything logged.
    5. The log line contains the diagnostic fields.
    6. Errors raised without the new fields still format cleanly.
"""

import json
import unittest
from unittest import mock

import httpx
from fastapi.testclient import TestClient

import llm_provider
from api import app
from llm_provider import (
    CATEGORY_API_ERROR,
    CATEGORY_EMPTY_RESPONSE,
    CATEGORY_HTTP_STATUS,
    CATEGORY_INVALID_JSON,
    CATEGORY_MISSING_API_KEY,
    CATEGORY_NETWORK,
    CATEGORY_RATE_LIMIT,
    CATEGORY_SCHEMA_MISMATCH,
    CATEGORY_TIMEOUT,
    LLMProviderError,
    RateLimitError,
    format_provider_error,
    sanitize_for_log,
)

BASE_PROFILE = dict(
    age=22, height_cm=178, weight_kg=72, sex="Male",
    goal="Improve football performance",
    diet="Vegetarian", training_days_per_week=5, training_intensity="High",
    region="Lucknow, Uttar Pradesh", allergies="none",
    budget_per_day="Rs 300", food_preferences=None, foods_to_avoid=None, notes=None,
)

VALID_PLAN = {
    "confidence_score": 8.0,
    "confidence_reason": ["test"],
    "day_summary": {
        "total_calories": 2000, "total_protein_g": 100,
        "total_carbs_g": 250, "total_fats_g": 60, "notes": "test",
    },
    "meals": [],
    "disclaimer": "test disclaimer",
}


def _response(status_code=200, text="", json_body=None):
    response = mock.Mock()
    response.status_code = status_code
    response.text = text
    response.json.return_value = json_body if json_body is not None else {}
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "err", request=mock.Mock(), response=response
        )
    else:
        response.raise_for_status.return_value = None
    return response


def _completion(content):
    return {"choices": [{"message": {"content": content}}]}


def _raise_from_groq(testcase, **post_kwargs):
    """Drive call_groq with a mocked transport and return the raised error."""
    with mock.patch.object(llm_provider, "GROQ_API_KEY", "gsk_testkeynotreal0000"), \
         mock.patch("httpx.post", **post_kwargs):
        with testcase.assertRaises(LLMProviderError) as ctx:
            llm_provider.call_groq([{"role": "user", "content": "hi"}])
    return ctx.exception


def _raise_from_openrouter(testcase, **post_kwargs):
    with mock.patch.object(llm_provider, "OPENROUTER_API_KEY", "sk-or-v1-testnotreal"), \
         mock.patch.object(llm_provider, "OPENROUTER_FALLBACK_MODEL", ""), \
         mock.patch("httpx.post", **post_kwargs):
        with testcase.assertRaises(LLMProviderError) as ctx:
            llm_provider.call_openrouter([{"role": "user", "content": "hi"}])
    return ctx.exception


# ----------------------------------------------------------------------
# 1. Each category is distinguished
# ----------------------------------------------------------------------

class TestGroqFailureCategories(unittest.TestCase):
    def test_upstream_502_is_http_status_with_the_code(self):
        exc = _raise_from_groq(self, return_value=_response(502, "Bad Gateway"))
        self.assertEqual(exc.category, CATEGORY_HTTP_STATUS)
        self.assertEqual(exc.status_code, 502)
        self.assertEqual(exc.provider, "groq")

    def test_gateway_504_is_http_status_not_timeout(self):
        """A 504 from the gateway is a RESPONSE we received, not us giving
        up waiting - the two used to be indistinguishable in the log."""
        exc = _raise_from_groq(self, return_value=_response(504, "Gateway Timeout"))
        self.assertEqual(exc.category, CATEGORY_HTTP_STATUS)
        self.assertEqual(exc.status_code, 504)

    def test_401_is_http_status(self):
        exc = _raise_from_groq(self, return_value=_response(401, "Invalid API Key"))
        self.assertEqual(exc.category, CATEGORY_HTTP_STATUS)
        self.assertEqual(exc.status_code, 401)

    def test_read_timeout_is_timeout(self):
        exc = _raise_from_groq(self, side_effect=httpx.ReadTimeout("timed out"))
        self.assertEqual(exc.category, CATEGORY_TIMEOUT)
        self.assertIsNone(exc.status_code)

    def test_connect_timeout_is_timeout(self):
        exc = _raise_from_groq(self, side_effect=httpx.ConnectTimeout("timed out"))
        self.assertEqual(exc.category, CATEGORY_TIMEOUT)

    def test_connection_error_is_network_not_timeout(self):
        exc = _raise_from_groq(self, side_effect=httpx.ConnectError("no route to host"))
        self.assertEqual(exc.category, CATEGORY_NETWORK)
        self.assertEqual(exc.detail, "ConnectError")

    def test_http_200_carrying_an_error_object_is_api_error(self):
        exc = _raise_from_groq(
            self, return_value=_response(200, "", {"error": {"message": "upstream failure"}})
        )
        self.assertEqual(exc.category, CATEGORY_API_ERROR)
        self.assertEqual(exc.status_code, 200)
        self.assertIn("upstream failure", exc.detail)

    def test_no_choices_is_empty_response(self):
        exc = _raise_from_groq(self, return_value=_response(200, "", {"choices": []}))
        self.assertEqual(exc.category, CATEGORY_EMPTY_RESPONSE)

    def test_empty_content_is_empty_response_with_finish_reason(self):
        body = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
        exc = _raise_from_groq(self, return_value=_response(200, "", body))
        self.assertEqual(exc.category, CATEGORY_EMPTY_RESPONSE)
        self.assertIn("length", exc.detail)

    def test_missing_api_key_is_its_own_category(self):
        with mock.patch.object(llm_provider, "GROQ_API_KEY", None):
            with self.assertRaises(LLMProviderError) as ctx:
                llm_provider.call_groq([{"role": "user", "content": "hi"}])
        self.assertEqual(ctx.exception.category, CATEGORY_MISSING_API_KEY)

    def test_unparseable_json_is_invalid_json_with_a_length(self):
        body = _completion("Here is your plan: {not json")
        with mock.patch.object(llm_provider, "GROQ_API_KEY", "gsk_testkeynotreal0000"), \
             mock.patch("httpx.post", return_value=_response(200, "", body)):
            with self.assertRaises(LLMProviderError) as ctx:
                llm_provider.groq_provider(dict(BASE_PROFILE))
        self.assertEqual(ctx.exception.category, CATEGORY_INVALID_JSON)
        self.assertIn("response_chars=", ctx.exception.detail)

    def test_valid_json_missing_fields_is_schema_mismatch(self):
        incomplete = {k: v for k, v in VALID_PLAN.items() if k != "meals"}
        body = _completion(json.dumps(incomplete))
        with mock.patch.object(llm_provider, "GROQ_API_KEY", "gsk_testkeynotreal0000"), \
             mock.patch("httpx.post", return_value=_response(200, "", body)):
            with self.assertRaises(LLMProviderError) as ctx:
                llm_provider.groq_provider(dict(BASE_PROFILE))
        self.assertEqual(ctx.exception.category, CATEGORY_SCHEMA_MISMATCH)

    def test_every_category_produces_a_distinct_classification(self):
        """The whole point: these used to be one indistinguishable branch."""
        seen = {
            _raise_from_groq(self, return_value=_response(502, "x")).category,
            _raise_from_groq(self, side_effect=httpx.ReadTimeout("x")).category,
            _raise_from_groq(self, side_effect=httpx.ConnectError("x")).category,
            _raise_from_groq(self, return_value=_response(200, "", {"error": {"m": 1}})).category,
            _raise_from_groq(self, return_value=_response(200, "", {"choices": []})).category,
        }
        self.assertEqual(len(seen), 5, seen)


class TestOpenRouterFailureCategories(unittest.TestCase):
    def test_openrouter_502_is_classified_and_names_its_provider(self):
        exc = _raise_from_openrouter(self, return_value=_response(502, "Bad Gateway"))
        self.assertEqual(exc.provider, "openrouter")
        self.assertEqual(exc.category, CATEGORY_HTTP_STATUS)
        self.assertEqual(exc.status_code, 502)

    def test_openrouter_upstream_idle_timeout_is_timeout(self):
        exc = _raise_from_openrouter(self, side_effect=httpx.ReadTimeout("idle timeout"))
        self.assertEqual(exc.provider, "openrouter")
        self.assertEqual(exc.category, CATEGORY_TIMEOUT)

    def test_the_two_providers_are_told_apart(self):
        groq = _raise_from_groq(self, return_value=_response(502, "x"))
        openrouter = _raise_from_openrouter(self, return_value=_response(502, "x"))
        self.assertNotEqual(groq.provider, openrouter.provider)


# ----------------------------------------------------------------------
# 2. RateLimitError behaviour preserved
# ----------------------------------------------------------------------

class TestRateLimitPreserved(unittest.TestCase):
    def test_429_still_raises_rate_limit_error(self):
        exc = _raise_from_groq(self, return_value=_response(429, "too many requests"))
        self.assertIsInstance(exc, RateLimitError)
        self.assertEqual(exc.category, CATEGORY_RATE_LIMIT)
        self.assertEqual(exc.status_code, 429)

    def test_rate_limit_error_is_still_an_llm_provider_error(self):
        self.assertTrue(issubclass(RateLimitError, LLMProviderError))

    def test_429_still_gets_its_own_user_facing_message(self):
        with mock.patch("api.run_with_trace", side_effect=RateLimitError("429")):
            response = TestClient(app).post("/meal-plan", json={
                "athlete_id": "00000001-0000-4000-8000-000000000001", **BASE_PROFILE})
        self.assertEqual(response.status_code, 503)
        self.assertIn("temporarily busy", response.json()["detail"].lower())

    def test_generic_error_still_gets_the_other_message(self):
        with mock.patch("api.run_with_trace", side_effect=LLMProviderError("boom")):
            response = TestClient(app).post("/meal-plan", json={
                "athlete_id": "00000001-0000-4000-8000-000000000001", **BASE_PROFILE})
        self.assertEqual(response.status_code, 503)
        self.assertIn("isn't available", response.json()["detail"])


# ----------------------------------------------------------------------
# 3. User-facing behaviour unchanged
# ----------------------------------------------------------------------

class TestUserFacingBehaviourUnchanged(unittest.TestCase):
    def _post(self, exc):
        with mock.patch("api.run_with_trace", side_effect=exc):
            return TestClient(app).post("/meal-plan", json={
                "athlete_id": "00000001-0000-4000-8000-000000000001", **BASE_PROFILE})

    def test_every_category_still_returns_the_same_503(self):
        for category in (CATEGORY_HTTP_STATUS, CATEGORY_TIMEOUT, CATEGORY_NETWORK,
                         CATEGORY_API_ERROR, CATEGORY_EMPTY_RESPONSE, CATEGORY_INVALID_JSON):
            with self.subTest(category=category):
                response = self._post(
                    LLMProviderError("boom", provider="groq", category=category, status_code=502)
                )
                self.assertEqual(response.status_code, 503)
                self.assertIn("isn't available", response.json()["detail"])

    def test_the_category_is_never_exposed_to_the_athlete(self):
        response = self._post(LLMProviderError(
            "boom", provider="groq", category=CATEGORY_HTTP_STATUS, status_code=502,
            detail="upstream said no"))
        body = response.text
        for leak in ("groq", "http_status", "502", "upstream said no", "boom"):
            self.assertNotIn(leak, body, leak)

    def test_an_api_key_in_the_detail_never_reaches_the_response(self):
        response = self._post(LLMProviderError(
            "boom", provider="groq", category=CATEGORY_HTTP_STATUS,
            detail="Invalid API Key: gsk_realkeymaterial0000"))
        self.assertNotIn("gsk_", response.text)


# ----------------------------------------------------------------------
# 4. Secret redaction
# ----------------------------------------------------------------------

class TestSecretRedaction(unittest.TestCase):
    def test_bearer_tokens_are_redacted(self):
        self.assertNotIn("gsk_", sanitize_for_log("Authorization: Bearer gsk_abc123def456"))

    def test_each_provider_key_shape_is_redacted(self):
        for secret in ("gsk_liveKeyMaterial0000",
                       "sk-or-v1-deadbeefdeadbeef",
                       "sk-ant-api03-realkeymaterial"):
            with self.subTest(secret=secret):
                cleaned = sanitize_for_log(f"rejected key {secret} sorry")
                self.assertNotIn(secret, cleaned)
                self.assertIn("[REDACTED]", cleaned)

    def test_ordinary_diagnostic_text_is_left_alone(self):
        text = "502 Bad Gateway from upstream after 30s"
        self.assertEqual(sanitize_for_log(text), text)

    def test_long_details_are_truncated(self):
        cleaned = sanitize_for_log("x" * 5000)
        self.assertLess(len(cleaned), 400)
        self.assertIn("truncated", cleaned)

    def test_detail_is_sanitised_when_the_error_is_constructed(self):
        exc = LLMProviderError("boom", detail="key gsk_realkeymaterial0000 rejected")
        self.assertNotIn("gsk_realkeymaterial0000", exc.detail)

    def test_a_key_leaked_in_a_response_body_is_redacted_in_the_log_line(self):
        exc = _raise_from_groq(
            self, return_value=_response(401, "Invalid API Key: gsk_realkeymaterial0000")
        )
        self.assertNotIn("gsk_realkeymaterial0000", format_provider_error(exc))
        self.assertIn("[REDACTED]", format_provider_error(exc))


# ----------------------------------------------------------------------
# 5. The log line itself
# ----------------------------------------------------------------------

class TestLogLine(unittest.TestCase):
    def test_log_line_carries_provider_category_and_status(self):
        exc = _raise_from_groq(self, return_value=_response(502, "Bad Gateway"))
        line = format_provider_error(exc)
        self.assertIn("provider=groq", line)
        self.assertIn("category=http_status", line)
        self.assertIn("status=502", line)

    def test_status_is_omitted_when_there_was_no_response(self):
        exc = _raise_from_groq(self, side_effect=httpx.ConnectError("no route"))
        self.assertNotIn("status=", format_provider_error(exc))

    def test_api_logs_the_diagnostic_line_on_a_generic_failure(self):
        exc = LLMProviderError("boom", provider="groq",
                               category=CATEGORY_HTTP_STATUS, status_code=502)
        with mock.patch("api.run_with_trace", side_effect=exc), \
             self.assertLogs("athleteos.api", level="ERROR") as logs:
            TestClient(app).post("/meal-plan", json={
                "athlete_id": "00000001-0000-4000-8000-000000000001", **BASE_PROFILE})
        output = "\n".join(logs.output)
        self.assertIn("provider=groq", output)
        self.assertIn("category=http_status", output)
        self.assertIn("status=502", output)

    def test_api_logs_the_diagnostic_line_on_a_rate_limit(self):
        exc = RateLimitError("429", provider="groq",
                             category=CATEGORY_RATE_LIMIT, status_code=429)
        with mock.patch("api.run_with_trace", side_effect=exc), \
             self.assertLogs("athleteos.api", level="WARNING") as logs:
            TestClient(app).post("/meal-plan", json={
                "athlete_id": "00000001-0000-4000-8000-000000000001", **BASE_PROFILE})
        self.assertIn("category=rate_limit", "\n".join(logs.output))


# ----------------------------------------------------------------------
# 6. Backwards compatibility
# ----------------------------------------------------------------------

class TestBackwardsCompatible(unittest.TestCase):
    def test_an_error_without_the_new_fields_still_constructs(self):
        exc = LLMProviderError("just a message")
        self.assertEqual(str(exc), "just a message")
        self.assertIsNone(exc.provider)
        self.assertIsNone(exc.category)

    def test_an_unclassified_error_formats_as_unknown(self):
        line = format_provider_error(LLMProviderError("just a message"))
        self.assertIn("provider=unknown", line)
        self.assertIn("category=unknown", line)

    def test_a_plain_exception_does_not_break_the_formatter(self):
        line = format_provider_error(RuntimeError("something else"))
        self.assertIn("provider=unknown", line)

    def test_log_context_returns_the_fields(self):
        exc = LLMProviderError("boom", provider="groq",
                               category=CATEGORY_TIMEOUT, status_code=None)
        self.assertEqual(exc.log_context()["provider"], "groq")
        self.assertEqual(exc.log_context()["category"], CATEGORY_TIMEOUT)


# ----------------------------------------------------------------------
# 7. Success path untouched
# ----------------------------------------------------------------------

class TestSuccessPathUnchanged(unittest.TestCase):
    def test_a_successful_groq_call_still_returns_the_plan(self):
        body = _completion(json.dumps(VALID_PLAN))
        with mock.patch.object(llm_provider, "GROQ_API_KEY", "gsk_testkeynotreal0000"), \
             mock.patch("httpx.post", return_value=_response(200, "", body)):
            plan = llm_provider.groq_provider(dict(BASE_PROFILE))
        self.assertEqual(plan, VALID_PLAN)

    def test_a_successful_openrouter_call_still_returns_the_plan(self):
        body = _completion(json.dumps(VALID_PLAN))
        with mock.patch.object(llm_provider, "OPENROUTER_API_KEY", "sk-or-v1-testnotreal"), \
             mock.patch("httpx.post", return_value=_response(200, "", body)):
            plan = llm_provider.openrouter_provider(dict(BASE_PROFILE))
        self.assertEqual(plan, VALID_PLAN)


if __name__ == "__main__":
    unittest.main(verbosity=2)
