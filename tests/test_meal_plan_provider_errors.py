"""
AthleteOS - Tests for /meal-plan's provider-failure handling.

Before this, an LLM provider failure during generation escaped
create_meal_plan() uncaught, so FastAPI returned a plain-text 500
"Internal Server Error". That was opaque twice over: the athlete saw
nothing useful, and the body carried no information to debug with either
(diagnosing one such production failure required querying the database to
work out how far the request had got). /chat already converted the same
exceptions into a clean 503 JSON response; this brings /meal-plan in line.

No real OpenRouter HTTP requests are made anywhere in this file - the
failure is injected at api.run_with_trace, the boundary where generation
happens. Zero network, zero cost.

Covers:
    1. A generic LLMProviderError returns 503 with a JSON `detail`.
    2. A RateLimitError returns 503 with its own, more specific message.
    3. The raw exception text never reaches the athlete-facing detail.
    4. The body is real JSON (the actual regression: it used to be plain
       text, which the frontend could not parse).
    5. A failed generation writes NO memory - the try/except wraps only
       generation, so the write-backs below it still never run on failure.
    6. The success path is completely unchanged (still 200 + the plan).
    7. The multi-day (duration_days=7) path gets the same 503 treatment.
"""

import unittest
from unittest import mock

from fastapi.testclient import TestClient

from api import app
from llm_provider import LLMProviderError, RateLimitError

ATHLETE_ID = "00000001-0000-4000-8000-000000000001"

BASE_REQUEST = {
    "athlete_id": ATHLETE_ID,
    "age": 22, "height_cm": 178, "weight_kg": 72, "sex": "Male",
    "goal": "Improve football performance",
    "diet": "Vegetarian", "training_days_per_week": 5, "training_intensity": "High",
    "region": "Lucknow, Uttar Pradesh", "allergies": "none",
}

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


def _post(payload=None):
    client = TestClient(app)
    return client.post("/meal-plan", json=payload or BASE_REQUEST)


class TestProviderFailureReturns503(unittest.TestCase):
    def test_generic_provider_error_returns_503_json(self):
        with mock.patch("api.run_with_trace", side_effect=LLMProviderError("boom")):
            response = _post()

        self.assertEqual(response.status_code, 503)
        detail = response.json()["detail"]
        self.assertIn("try again", detail.lower())

    def test_rate_limit_error_returns_503_with_its_own_message(self):
        with mock.patch("api.run_with_trace", side_effect=RateLimitError("429 from pool")):
            response = _post()

        self.assertEqual(response.status_code, 503)
        detail = response.json()["detail"]
        self.assertIn("temporarily busy", detail.lower())

    def test_raw_exception_text_is_never_leaked_to_the_athlete(self):
        secret = "OPENROUTER_API_KEY=sk-or-v1-should-never-be-shown"
        with mock.patch("api.run_with_trace", side_effect=LLMProviderError(secret)):
            response = _post()

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("sk-or-v1", response.text)
        self.assertNotIn("OPENROUTER_API_KEY", response.text)

    def test_error_body_is_json_not_plain_text(self):
        """The actual regression being fixed: the old unhandled 500 returned
        the plain-text body "Internal Server Error", which the frontend
        could not parse as JSON."""
        with mock.patch("api.run_with_trace", side_effect=LLMProviderError("boom")):
            response = _post()

        self.assertEqual(response.headers["content-type"].split(";")[0], "application/json")
        self.assertIsInstance(response.json(), dict)
        self.assertIn("detail", response.json())
        self.assertNotEqual(response.text.strip(), "Internal Server Error")

    def test_multi_day_request_gets_the_same_503_treatment(self):
        with mock.patch("api.run_with_trace", side_effect=LLMProviderError("boom")):
            response = _post({**BASE_REQUEST, "duration_days": 7})

        self.assertEqual(response.status_code, 503)
        self.assertIn("detail", response.json())


class TestFailedGenerationWritesNoMemory(unittest.TestCase):
    def test_no_memory_write_backs_run_when_generation_fails(self):
        """The try/except wraps ONLY run_with_trace, so the write-backs after
        it must still never execute on a failure - the same ordering the
        production incident was diagnosed with (zero profile/latest_plan rows
        for a request that 500'd)."""
        with mock.patch("api.run_with_trace", side_effect=LLMProviderError("boom")), \
             mock.patch("api.update_memory_from_profile") as update_memory, \
             mock.patch("api.record_episode") as record_episode, \
             mock.patch("api.record_latest_plan_summary") as record_latest:
            response = _post()

        self.assertEqual(response.status_code, 503)
        update_memory.assert_not_called()
        record_episode.assert_not_called()
        record_latest.assert_not_called()


class TestSuccessPathUnchanged(unittest.TestCase):
    def test_a_successful_generation_still_returns_200_and_the_plan(self):
        trace = {"plan": FAKE_PLAN, "merged_profile": dict(BASE_REQUEST)}
        with mock.patch("api.run_with_trace", return_value=trace), \
             mock.patch("api.update_memory_from_profile"), \
             mock.patch("api.record_episode"), \
             mock.patch("api.record_latest_plan_summary"):
            response = _post()

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["confidence_score"], FAKE_PLAN["confidence_score"])
        self.assertEqual(body["disclaimer"], FAKE_PLAN["disclaimer"])

    def test_a_successful_generation_still_runs_the_memory_write_backs(self):
        trace = {"plan": FAKE_PLAN, "merged_profile": dict(BASE_REQUEST)}
        with mock.patch("api.run_with_trace", return_value=trace), \
             mock.patch("api.update_memory_from_profile") as update_memory, \
             mock.patch("api.record_episode") as record_episode, \
             mock.patch("api.record_latest_plan_summary") as record_latest:
            response = _post()

        self.assertEqual(response.status_code, 200)
        update_memory.assert_called_once()
        record_episode.assert_called_once()      # duration_days defaults to 1
        record_latest.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
