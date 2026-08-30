"""
AthleteOS - Tests for llm_provider.openrouter_provider_multi_day() (the 4+3
batched multi-day OpenRouter provider) and its wiring into
meal_planner.generate_meal_plan().

No real OpenRouter HTTP requests are made anywhere in this file - httpx's
POST call is mocked at the module boundary, the same technique
test_openrouter_provider.py already uses. Zero network, zero cost.

Covers:
    1. _dish_names_used() extracts the right set from a batch's days.
    2. _validate_batch_shape() rejects malformed/incomplete batches.
    3. Happy path: two batches merge into duration_days=7, 7 days total,
       confidence/disclaimer taken from batch 1, batch 2's prompt receives
       batch 1's dish names as an avoid-list.
    4. A malformed/failed batch 1 raises LLMProviderError and batch 2 is
       never called.
    5. A malformed/failed batch 2 raises LLMProviderError even though
       batch 1 succeeded (no partial-week result).
    6. days != 7 raises NotImplementedError without making any HTTP call.
    7. generate_meal_plan()'s PROVIDER="openrouter" dispatch for days=7
       calls openrouter_provider_multi_day(); for an unsupported days count
       it raises NotImplementedError.
    8. The existing single-day openrouter_provider() path is unaffected
       (verified by the full existing test_openrouter_provider.py suite
       still passing unchanged, not re-tested here).
"""

import json
import unittest
from unittest import mock

import llm_provider
import meal_planner
from llm_provider import (
    LLMProviderError,
    _dish_names_used,
    _validate_batch_shape,
    openrouter_provider_multi_day,
)
from meal_planner import generate_meal_plan

BASE_PROFILE = dict(
    age=22, height_cm=178, weight_kg=72, sex="Male",
    goal="Improve football performance",
    diet="Vegetarian", training_days_per_week=5, training_intensity="High",
    region="Lucknow, Uttar Pradesh", allergies="none",
    budget_per_day="Rs 300", food_preferences=None, foods_to_avoid=None, notes=None,
)


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
    return mock.patch.object(llm_provider, "OPENROUTER_API_KEY", value)


def _batch(day_start, day_end, dish_prefix):
    """A minimal, valid batch response for days day_start..day_end, each
    day's single meal using a distinct dish name (so cross-batch overlap is
    easy to assert on)."""
    days = []
    for day in range(day_start, day_end + 1):
        item = f"{dish_prefix}-{day}"
        days.append({
            "day": day,
            "day_summary": {
                "total_calories": 2000, "total_protein_g": 100,
                "total_carbs_g": 250, "total_fats_g": 60, "notes": "test",
            },
            "meals": [{
                "meal": "Breakfast",
                "ingredients": [{"item": item, "quantity": "1 serving"}],
                "calories": 500, "protein_g": 25, "carbs_g": 60, "fats_g": 15,
                "reasoning": "test",
                "resources": [{"type": "youtube", "title": "test", "url": "http://x", "item": item}],
            }],
        })
    return {
        "confidence_score": 7.0,
        "confidence_reason": ["test"],
        "days": days,
        "disclaimer": "batch disclaimer",
    }


# ----------------------------------------------------------------------
# 1. _dish_names_used()
# ----------------------------------------------------------------------

class TestDishNamesUsed(unittest.TestCase):
    def test_collects_every_ingredient_item_across_days_and_meals(self):
        batch = _batch(1, 3, "dish")
        names = _dish_names_used(batch["days"])
        self.assertEqual(names, {"dish-1", "dish-2", "dish-3"})

    def test_empty_days_returns_empty_set(self):
        self.assertEqual(_dish_names_used([]), set())


# ----------------------------------------------------------------------
# 2. _validate_batch_shape()
# ----------------------------------------------------------------------

class TestValidateBatchShape(unittest.TestCase):
    def test_valid_batch_raises_nothing(self):
        _validate_batch_shape(_batch(1, 2, "dish"))  # should not raise

    def test_missing_top_level_field_raises(self):
        batch = _batch(1, 2, "dish")
        del batch["disclaimer"]
        with self.assertRaises(LLMProviderError):
            _validate_batch_shape(batch)

    def test_empty_days_list_raises(self):
        batch = _batch(1, 2, "dish")
        batch["days"] = []
        with self.assertRaises(LLMProviderError):
            _validate_batch_shape(batch)

    def test_day_missing_required_field_raises(self):
        batch = _batch(1, 2, "dish")
        del batch["days"][0]["meals"]
        with self.assertRaises(LLMProviderError):
            _validate_batch_shape(batch)


# ----------------------------------------------------------------------
# 3-5. openrouter_provider_multi_day() happy path and failure propagation
# ----------------------------------------------------------------------

class TestOpenrouterProviderMultiDay(unittest.TestCase):
    def test_happy_path_merges_two_batches_into_seven_days(self):
        batch1 = _batch(1, 4, "batch1dish")
        batch2 = _batch(5, 7, "batch2dish")
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.side_effect = [
                _fake_httpx_response(json_body=_chat_completion(json.dumps(batch1))),
                _fake_httpx_response(json_body=_chat_completion(json.dumps(batch2))),
            ]
            result = openrouter_provider_multi_day(BASE_PROFILE, 7)

        self.assertEqual(post.call_count, 2)
        self.assertEqual(result["duration_days"], 7)
        self.assertEqual(len(result["days"]), 7)
        self.assertEqual([d["day"] for d in result["days"]], [1, 2, 3, 4, 5, 6, 7])
        # confidence/disclaimer come from batch 1 (profile-level, not per-batch)
        self.assertEqual(result["confidence_score"], batch1["confidence_score"])
        self.assertEqual(result["disclaimer"], batch1["disclaimer"])

    def test_batch_two_prompt_receives_batch_one_dish_names_as_avoid_list(self):
        batch1 = _batch(1, 4, "batch1dish")
        batch2 = _batch(5, 7, "batch2dish")
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.side_effect = [
                _fake_httpx_response(json_body=_chat_completion(json.dumps(batch1))),
                _fake_httpx_response(json_body=_chat_completion(json.dumps(batch2))),
            ]
            openrouter_provider_multi_day(BASE_PROFILE, 7)

        second_call_prompt = post.call_args_list[1].kwargs["json"]["messages"][0]["content"]
        self.assertIn("AVOID REPEATING THESE DISHES", second_call_prompt)
        for day in range(1, 5):
            self.assertIn(f"batch1dish-{day}", second_call_prompt)

    def test_batch_one_failure_raises_and_never_calls_batch_two(self):
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_httpx_response(json_body=_chat_completion("not valid json"))
            with self.assertRaises(LLMProviderError):
                openrouter_provider_multi_day(BASE_PROFILE, 7)

        self.assertEqual(post.call_count, 1)

    def test_batch_two_failure_raises_even_though_batch_one_succeeded(self):
        batch1 = _batch(1, 4, "batch1dish")
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.side_effect = [
                _fake_httpx_response(json_body=_chat_completion(json.dumps(batch1))),
                _fake_httpx_response(json_body=_chat_completion("not valid json")),
            ]
            with self.assertRaises(LLMProviderError):
                openrouter_provider_multi_day(BASE_PROFILE, 7)

        self.assertEqual(post.call_count, 2)

    def test_unsupported_days_raises_without_any_http_call(self):
        with _patched_api_key(), mock.patch("httpx.post") as post:
            with self.assertRaises(NotImplementedError):
                openrouter_provider_multi_day(BASE_PROFILE, 3)
        post.assert_not_called()

    def test_batches_use_larger_max_tokens_than_the_single_day_default(self):
        batch1 = _batch(1, 4, "batch1dish")
        batch2 = _batch(5, 7, "batch2dish")
        with _patched_api_key(), mock.patch("httpx.post") as post:
            post.side_effect = [
                _fake_httpx_response(json_body=_chat_completion(json.dumps(batch1))),
                _fake_httpx_response(json_body=_chat_completion(json.dumps(batch2))),
            ]
            openrouter_provider_multi_day(BASE_PROFILE, 7)

        batch1_max_tokens = post.call_args_list[0].kwargs["json"]["max_tokens"]
        batch2_max_tokens = post.call_args_list[1].kwargs["json"]["max_tokens"]
        self.assertGreater(batch1_max_tokens, llm_provider.OPENROUTER_MAX_TOKENS)
        self.assertGreater(batch2_max_tokens, llm_provider.OPENROUTER_MAX_TOKENS)


# ----------------------------------------------------------------------
# 6. generate_meal_plan() dispatch
# ----------------------------------------------------------------------

class TestGenerateMealPlanMultiDayDispatch(unittest.TestCase):
    def test_openrouter_seven_day_dispatches_to_multi_day_provider(self):
        with mock.patch.object(meal_planner, "PROVIDER", "openrouter"), \
             mock.patch("llm_provider.openrouter_provider_multi_day", return_value={"fake": "plan"}) as provider_mock:
            plan = generate_meal_plan(dict(BASE_PROFILE), agent_context=None, days=7)

        provider_mock.assert_called_once_with(dict(BASE_PROFILE), 7, None)
        self.assertEqual(plan, {"fake": "plan"})

    def test_openrouter_unsupported_day_count_raises_not_implemented(self):
        with mock.patch.object(meal_planner, "PROVIDER", "openrouter"):
            with self.assertRaises(NotImplementedError):
                generate_meal_plan(dict(BASE_PROFILE), agent_context=None, days=3)

    def test_mock_seven_day_still_uses_mock_multi_day_unchanged(self):
        # Guards against the new openrouter branch accidentally shadowing
        # the existing, working mock multi-day path.
        with mock.patch.object(meal_planner, "PROVIDER", "mock"):
            plan = generate_meal_plan(dict(BASE_PROFILE), agent_context=None, days=7)
        self.assertEqual(plan["duration_days"], 7)
        self.assertEqual(len(plan["days"]), 7)

    def test_claude_multi_day_still_raises_not_implemented(self):
        with mock.patch.object(meal_planner, "PROVIDER", "claude"):
            with self.assertRaises(NotImplementedError):
                generate_meal_plan(dict(BASE_PROFILE), agent_context=None, days=7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
