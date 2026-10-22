"""
AthleteOS - Tests for the generalised multi-day batching in llm_provider.py
(shared by OpenRouter and Groq), plus the deploy-related API/Postgres fixes
that shipped with it.

No real HTTP or database traffic: httpx.post and psycopg are mocked, the
same technique test_openrouter_multi_day.py uses. Zero network, zero cost.

Covers:
    1. _batch_ranges() splits days evenly into contiguous batches.
    2. Per-batch max_tokens keeps the measured 4-day/3-day budgets.
    3. OpenRouter: a truncated/invalid JSON batch from the primary model is
       retried on the fallback model instead of failing the plan.
    4. OpenRouter: wrong day count -> fallback; mis-numbered days are
       renumbered; the avoid-list accumulates across all earlier batches.
    5. Groq: batches use a strict batch json_schema; a tokens-per-minute 429
       is waited out via Retry-After, but a long Retry-After is not.
    6. API: NotImplementedError -> JSON 422 (was a plain-text 500); "/"
       serves the app and "/health" is the JSON health check.
    7. Postgres: a dropped connection is reopened and the query retried once.
"""

import json
import sys
import types
import unittest
from unittest import mock

import httpx
from fastapi.testclient import TestClient

import llm_provider
import postgres_memory_backend
from api import app
from llm_provider import (
    LLMProviderError,
    RateLimitError,
    _batch_ranges,
    _multi_day_max_tokens,
    groq_provider_multi_day,
    openrouter_provider_multi_day,
)

BASE_PROFILE = dict(
    age=22, height_cm=178, weight_kg=72, sex="Male",
    goal="Improve football performance",
    diet="Vegetarian", training_days_per_week=5, training_intensity="High",
    region="Lucknow, Uttar Pradesh", allergies="none",
    budget_per_day="Rs 300", food_preferences=None, foods_to_avoid=None, notes=None,
)

BASE_REQUEST = {
    "athlete_id": "00000001-0000-4000-8000-000000000001",
    "age": 22, "height_cm": 178, "weight_kg": 72, "sex": "Male",
    "goal": "Improve football performance",
    "diet": "Vegetarian", "training_days_per_week": 5, "training_intensity": "High",
    "region": "Lucknow, Uttar Pradesh", "allergies": "none",
}


def _response(status_code=200, json_body=None, headers=None):
    response = mock.Mock()
    response.status_code = status_code
    response.headers = headers or {}
    response.text = json.dumps(json_body or {})
    response.json.return_value = json_body or {}
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=mock.Mock(), response=response
        )
    else:
        response.raise_for_status.return_value = None
    return response


def _completion(content):
    return _response(json_body={"choices": [{"message": {"content": content}}]})


def _batch(day_start, day_end, dish_prefix, numbered_from=None):
    """A minimal valid batch for days day_start..day_end. `numbered_from`
    lets a test fake a model that numbers its days wrongly."""
    first = day_start if numbered_from is None else numbered_from
    days = []
    for offset in range(day_end - day_start + 1):
        item = f"{dish_prefix}-{day_start + offset}"
        days.append({
            "day": first + offset,
            "day_summary": {
                "total_calories": 2000, "total_protein_g": 100,
                "total_carbs_g": 250, "total_fats_g": 60, "notes": "test",
            },
            "meals": [{
                "meal": "Breakfast",
                "ingredients": [{"item": item, "quantity": "1 serving"}],
                "calories": 500, "protein_g": 25, "carbs_g": 60, "fats_g": 15,
                "reasoning": "test",
                "resources": [{"type": "youtube", "title": "t", "url": "http://x", "item": item}],
            }],
        })
    return {
        "confidence_score": 7.0, "confidence_reason": ["test"],
        "days": days, "disclaimer": "batch disclaimer",
    }


def _sent_json(post, call_index):
    return post.call_args_list[call_index].kwargs["json"]


# ----------------------------------------------------------------------
# 1/2. Batch splitting and budgets
# ----------------------------------------------------------------------

class TestBatchRanges(unittest.TestCase):
    def test_seven_days_at_four_keeps_the_measured_four_plus_three_split(self):
        self.assertEqual(_batch_ranges(7, 4), [(1, 4), (5, 7)])

    def test_seven_days_at_two(self):
        self.assertEqual(_batch_ranges(7, 2), [(1, 2), (3, 4), (5, 6), (7, 7)])

    def test_splits_evenly_rather_than_leaving_a_tiny_last_batch(self):
        self.assertEqual(_batch_ranges(9, 4), [(1, 3), (4, 6), (7, 9)])

    def test_fewer_days_than_batch_size_is_one_batch(self):
        self.assertEqual(_batch_ranges(3, 4), [(1, 3)])

    def test_every_day_covered_exactly_once(self):
        for days in range(2, 29):
            for size in (1, 2, 3, 4):
                ranges = _batch_ranges(days, size)
                covered = [d for start, end in ranges for d in range(start, end + 1)]
                self.assertEqual(covered, list(range(1, days + 1)))
                self.assertTrue(all(end - start + 1 <= size for start, end in ranges))

    def test_budgets_match_the_measured_prototype_values(self):
        self.assertEqual(_multi_day_max_tokens(4), 6500)
        self.assertEqual(_multi_day_max_tokens(3), 5000)


# ----------------------------------------------------------------------
# 3/4. OpenRouter recovery and merging
# ----------------------------------------------------------------------

class TestOpenrouterMultiDayRecovery(unittest.TestCase):
    def setUp(self):
        patches = [
            mock.patch.object(llm_provider, "OPENROUTER_API_KEY", "test-key-not-real"),
            mock.patch.object(llm_provider, "OPENROUTER_MODEL", "primary-model"),
            mock.patch.object(llm_provider, "OPENROUTER_FALLBACK_MODEL", "fallback-model"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_truncated_primary_batch_is_retried_on_fallback_model(self):
        with mock.patch("httpx.post") as post:
            post.side_effect = [
                _completion('{"confidence_score": 7, "days": [{"day": 1'),  # cut off mid-JSON
                _completion(json.dumps(_batch(1, 4, "a"))),
                _completion(json.dumps(_batch(5, 7, "b"))),
            ]
            plan = openrouter_provider_multi_day(BASE_PROFILE, 7)

        self.assertEqual(len(plan["days"]), 7)
        self.assertEqual(_sent_json(post, 0)["model"], "primary-model")
        self.assertEqual(_sent_json(post, 1)["model"], "fallback-model")
        self.assertEqual(_sent_json(post, 2)["model"], "primary-model")

    def test_wrong_day_count_is_treated_as_a_failed_batch(self):
        with mock.patch("httpx.post") as post:
            post.side_effect = [
                _completion(json.dumps(_batch(1, 2, "short"))),  # asked for 4, got 2
                _completion(json.dumps(_batch(1, 4, "a"))),
                _completion(json.dumps(_batch(5, 7, "b"))),
            ]
            plan = openrouter_provider_multi_day(BASE_PROFILE, 7)
        self.assertEqual(post.call_count, 3)
        self.assertEqual([d["day"] for d in plan["days"]], [1, 2, 3, 4, 5, 6, 7])

    def test_misnumbered_days_are_renumbered(self):
        with mock.patch("httpx.post") as post:
            post.side_effect = [
                _completion(json.dumps(_batch(1, 4, "a"))),
                _completion(json.dumps(_batch(5, 7, "b", numbered_from=1))),
            ]
            plan = openrouter_provider_multi_day(BASE_PROFILE, 7)
        self.assertEqual([d["day"] for d in plan["days"]], [1, 2, 3, 4, 5, 6, 7])

    def test_avoid_list_accumulates_across_every_earlier_batch(self):
        with mock.patch.object(llm_provider, "OPENROUTER_MULTI_DAY_BATCH_DAYS", 2), \
             mock.patch("httpx.post") as post:
            post.side_effect = [
                _completion(json.dumps(_batch(1, 2, "first"))),
                _completion(json.dumps(_batch(3, 4, "second"))),
            ]
            openrouter_provider_multi_day(BASE_PROFILE, 4)
        second_prompt = _sent_json(post, 1)["messages"][0]["content"]
        self.assertIn("first-1", second_prompt)
        self.assertIn("first-2", second_prompt)

    def test_rate_limit_on_both_models_still_raises_rate_limit_error(self):
        with mock.patch("httpx.post", return_value=_response(status_code=429)):
            with self.assertRaises(RateLimitError):
                openrouter_provider_multi_day(BASE_PROFILE, 7)

    def test_timeout_is_configurable_and_no_longer_60s(self):
        with mock.patch("httpx.post") as post:
            post.side_effect = [
                _completion(json.dumps(_batch(1, 4, "a"))),
                _completion(json.dumps(_batch(5, 7, "b"))),
            ]
            openrouter_provider_multi_day(BASE_PROFILE, 7)
        self.assertEqual(post.call_args_list[0].kwargs["timeout"], llm_provider.OPENROUTER_TIMEOUT)
        self.assertGreater(llm_provider.OPENROUTER_TIMEOUT, 60)


# ----------------------------------------------------------------------
# 5. Groq multi-day
# ----------------------------------------------------------------------

class TestGroqMultiDay(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(llm_provider, "GROQ_API_KEY", "test-key-not-real")
        p.start()
        self.addCleanup(p.stop)

    def test_seven_days_in_four_schema_enforced_batches(self):
        batches = [(1, 2), (3, 4), (5, 6), (7, 7)]
        with mock.patch("httpx.post") as post:
            post.side_effect = [_completion(json.dumps(_batch(s, e, "g"))) for s, e in batches]
            plan = groq_provider_multi_day(BASE_PROFILE, 7)

        self.assertEqual(post.call_count, 4)
        self.assertEqual([d["day"] for d in plan["days"]], list(range(1, 8)))
        response_format = _sent_json(post, 0)["response_format"]
        self.assertEqual(response_format["json_schema"]["name"], "meal_plan_batch")
        self.assertTrue(response_format["json_schema"]["strict"])

    def test_tokens_per_minute_429_is_waited_out(self):
        with mock.patch("time.sleep") as sleep, mock.patch("httpx.post") as post:
            post.side_effect = [
                _response(status_code=429, headers={"retry-after": "12"}),
                _completion(json.dumps(_batch(1, 2, "g"))),
            ]
            plan = groq_provider_multi_day(BASE_PROFILE, 2)
        sleep.assert_called_once_with(12.0)
        self.assertEqual(len(plan["days"]), 2)

    def test_long_retry_after_is_not_waited_out(self):
        with mock.patch("time.sleep") as sleep, mock.patch("httpx.post") as post:
            post.return_value = _response(status_code=429, headers={"retry-after": "3600"})
            with self.assertRaises(RateLimitError):
                groq_provider_multi_day(BASE_PROFILE, 2)
        sleep.assert_not_called()
        self.assertEqual(post.call_count, 1)

    def test_429_without_retry_after_is_not_retried(self):
        with mock.patch("time.sleep") as sleep, mock.patch("httpx.post") as post:
            post.return_value = _response(status_code=429)
            with self.assertRaises(RateLimitError):
                groq_provider_multi_day(BASE_PROFILE, 2)
        sleep.assert_not_called()


# ----------------------------------------------------------------------
# 6. API
# ----------------------------------------------------------------------

class TestApiDeployFixes(unittest.TestCase):
    client = TestClient(app)

    def test_unsupported_duration_is_a_json_422_not_a_plain_500(self):
        with mock.patch("api.run_with_trace", side_effect=NotImplementedError("no multi-day for claude")):
            response = self.client.post("/meal-plan", json={**BASE_REQUEST, "duration_days": 7})
        self.assertEqual(response.status_code, 422)
        self.assertIn("1-day plan", response.json()["detail"])
        self.assertNotIn("claude", response.json()["detail"])

    def test_root_serves_the_app(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])

    def test_app_path_still_serves_the_app(self):
        self.assertIn("text/html", self.client.get("/app").headers["content-type"])

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.json(), {"status": "ok", "service": "AthleteOS API"})


# ----------------------------------------------------------------------
# 7. Postgres reconnect
# ----------------------------------------------------------------------

class TestPostgresReconnect(unittest.TestCase):
    def setUp(self):
        self.fake_psycopg = types.ModuleType("psycopg")

        class OperationalError(Exception):
            pass

        self.fake_psycopg.OperationalError = OperationalError
        self.fake_psycopg.connect = mock.Mock()
        patches = [
            mock.patch.dict(sys.modules, {"psycopg": self.fake_psycopg}),
            mock.patch.object(postgres_memory_backend, "DATABASE_URL", "postgresql://fake"),
            mock.patch.object(postgres_memory_backend, "_conn", None),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _connection(self, cursor):
        conn = mock.MagicMock()
        conn.closed = False
        conn.cursor.return_value.__enter__.return_value = cursor
        return conn

    def test_dead_connection_is_reopened_and_query_retried(self):
        dead_cursor = mock.Mock()
        dead_cursor.execute.side_effect = self.fake_psycopg.OperationalError("server closed the connection")
        live_cursor = mock.Mock()
        live_cursor.fetchone.return_value = ({"age": 22},)
        self.fake_psycopg.connect.side_effect = [self._connection(dead_cursor), self._connection(live_cursor)]

        self.assertEqual(postgres_memory_backend.get("profile", "default"), {"age": 22})
        self.assertEqual(self.fake_psycopg.connect.call_count, 2)

    def test_connects_with_pooler_safe_options(self):
        cursor = mock.Mock()
        cursor.fetchone.return_value = None
        self.fake_psycopg.connect.return_value = self._connection(cursor)

        postgres_memory_backend.get("profile", "default")
        kwargs = self.fake_psycopg.connect.call_args.kwargs
        self.assertIsNone(kwargs["prepare_threshold"])
        self.assertTrue(kwargs["autocommit"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
