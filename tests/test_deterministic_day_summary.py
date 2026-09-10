"""
AthleteOS - Tests for the shared deterministic day_summary post-processing.

THE BUG BEING FIXED
    day_summary is meant to be a SUMMARY of the day's meals - that is what
    meal_planner._day_summary() computes for mock_provider, by summing
    them. On the LLM providers nothing computed it and nothing checked it
    (there is no validator anywhere on these fields), so the model wrote it
    freehand. Measured against two real Groq production plans, it copied
    the nutrition_tool TARGETS into day_summary while its per-meal numbers
    went their own way:

        run A: protein 130 reported vs 160 in the meals (+23.1%)
               carbs   576 reported vs 435 in the meals (-24.5%)
        run B: protein 130 reported vs 105 in the meals (-19.2%)
               carbs   580 reported vs 410 in the meals (-29.3%)

    So the athlete read a correct-looking daily protein figure while the
    food actually listed delivered a quarter less. Given the identical tool
    numbers, mock_provider's deterministic path is exact on every macro -
    this is arithmetic the application already knows how to do.

    Note the fix corrects the TOTALS ONLY. Meal-level values are left
    exactly as generated - a plan whose food misses the tool target now
    shows it missing the target rather than reporting the target back.

No network and no LLM calls anywhere in this file - the post-processing
function is pure, and the provider-level tests use mock_provider or a
patched provider.

Covers:
    1. An inconsistent LLM-style summary is corrected (the regression).
    2. The summary always equals the sum of the meals.
    3. Meal-level nutrition and every other field are untouched.
    4. mock_provider output is unchanged (idempotent no-op).
    5. Both the 1-day and multi-day shapes, every day independently.
    6. It runs for every provider, via generate_meal_plan().
"""

import copy
import unittest
from unittest import mock

import meal_planner
from meal_planner import (
    apply_deterministic_day_summary,
    generate_meal_plan,
    mock_provider,
    mock_provider_multi_day,
)

BASE_PROFILE = dict(
    age=22, height_cm=178, weight_kg=72, sex="Male",
    goal="Improve football performance",
    diet="Vegetarian", training_days_per_week=5, training_intensity="High",
    region="Lucknow, Uttar Pradesh", allergies="none",
    budget_per_day="Rs 300", food_preferences=None, foods_to_avoid=None, notes=None,
)

TOTALS = ("total_calories", "total_protein_g", "total_carbs_g", "total_fats_g")
MEAL_KEYS = ("calories", "protein_g", "carbs_g", "fats_g")

# The real numbers from the two Groq production plans: the model's meals,
# and the day_summary it reported alongside them.
PRODUCTION_RUN_A_MEALS = [
    {"calories": 688, "protein_g": 29, "carbs_g": 95, "fats_g": 18},
    {"calories": 1105, "protein_g": 58, "carbs_g": 143, "fats_g": 33},
    {"calories": 545, "protein_g": 26, "carbs_g": 72, "fats_g": 17},
    {"calories": 912, "protein_g": 47, "carbs_g": 125, "fats_g": 34},
]
PRODUCTION_RUN_A_REPORTED = {
    "total_calories": 3110, "total_protein_g": 130,
    "total_carbs_g": 576, "total_fats_g": 102,
    "notes": "This plan meets the pre-calculated daily targets.",
}


def _meal(index, macros):
    """A meal carrying the given macros plus the usual non-macro fields."""
    return {
        "meal": ["Breakfast", "Lunch", "Snack", "Dinner"][index],
        "ingredients": [{"item": f"Item {index}", "quantity": "1 bowl"}],
        "reasoning": "test",
        "resources": [],
        **macros,
    }


def _plan(meals_macros, reported_summary):
    return {
        "confidence_score": 8.0,
        "confidence_reason": ["test"],
        "day_summary": dict(reported_summary),
        "meals": [_meal(i, m) for i, m in enumerate(meals_macros)],
        "disclaimer": "test disclaimer",
    }


def _assert_summary_equals_meals(testcase, day):
    for total_key, meal_key in zip(TOTALS, MEAL_KEYS):
        testcase.assertEqual(
            day["day_summary"][total_key],
            sum(m[meal_key] for m in day["meals"]),
            total_key,
        )


# ----------------------------------------------------------------------
# 1. The regression: an inconsistent LLM summary gets corrected
# ----------------------------------------------------------------------

class TestInconsistentSummaryIsCorrected(unittest.TestCase):
    def test_production_run_a_summary_is_replaced_with_the_real_sums(self):
        """The exact payload that exposed this: reported 130g protein and
        576g carbs while the meals held 160g and 435g."""
        plan = _plan(PRODUCTION_RUN_A_MEALS, PRODUCTION_RUN_A_REPORTED)
        apply_deterministic_day_summary(plan)
        summary = plan["day_summary"]
        self.assertEqual(summary["total_calories"], 3250)   # was 3110
        self.assertEqual(summary["total_protein_g"], 160)   # was 130
        self.assertEqual(summary["total_carbs_g"], 435)     # was 576
        self.assertEqual(summary["total_fats_g"], 102)      # already correct

    def test_an_overstated_summary_is_corrected_downwards(self):
        plan = _plan(
            [{"calories": 100, "protein_g": 10, "carbs_g": 20, "fats_g": 5}] * 4,
            {"total_calories": 9999, "total_protein_g": 999,
             "total_carbs_g": 999, "total_fats_g": 999, "notes": "wrong"},
        )
        apply_deterministic_day_summary(plan)
        self.assertEqual(plan["day_summary"]["total_calories"], 400)
        self.assertEqual(plan["day_summary"]["total_protein_g"], 40)

    def test_an_understated_summary_is_corrected_upwards(self):
        plan = _plan(
            [{"calories": 500, "protein_g": 40, "carbs_g": 60, "fats_g": 15}] * 4,
            {"total_calories": 1, "total_protein_g": 1,
             "total_carbs_g": 1, "total_fats_g": 1, "notes": "wrong"},
        )
        apply_deterministic_day_summary(plan)
        self.assertEqual(plan["day_summary"]["total_calories"], 2000)
        self.assertEqual(plan["day_summary"]["total_protein_g"], 160)

    def test_a_summary_copying_the_tool_target_no_longer_survives(self):
        """day_summary must describe the MEALS, not restate the target -
        that conflation is what hid the discrepancy in production."""
        tool_target = {"total_calories": 3110, "total_protein_g": 130,
                       "total_carbs_g": 576, "total_fats_g": 88, "notes": "t"}
        plan = _plan(PRODUCTION_RUN_A_MEALS, tool_target)
        apply_deterministic_day_summary(plan)
        self.assertNotEqual(plan["day_summary"]["total_protein_g"], 130)
        _assert_summary_equals_meals(self, plan)


# ----------------------------------------------------------------------
# 2. The invariant: summary == sum of meals
# ----------------------------------------------------------------------

class TestSummaryEqualsSumOfMeals(unittest.TestCase):
    def test_invariant_holds_after_post_processing(self):
        plan = _plan(PRODUCTION_RUN_A_MEALS, PRODUCTION_RUN_A_REPORTED)
        apply_deterministic_day_summary(plan)
        _assert_summary_equals_meals(self, plan)

    def test_invariant_holds_through_generate_meal_plan_for_mock(self):
        with mock.patch.object(meal_planner, "PROVIDER", "mock"):
            plan = generate_meal_plan(dict(BASE_PROFILE))
        _assert_summary_equals_meals(self, plan)

    def test_a_plan_with_no_meals_totals_to_zero(self):
        plan = _plan([], {"total_calories": 500, "total_protein_g": 5,
                          "total_carbs_g": 5, "total_fats_g": 5, "notes": "n"})
        apply_deterministic_day_summary(plan)
        for key in TOTALS:
            self.assertEqual(plan["day_summary"][key], 0)

    def test_a_missing_meal_macro_counts_as_zero_rather_than_raising(self):
        plan = _plan(PRODUCTION_RUN_A_MEALS, PRODUCTION_RUN_A_REPORTED)
        del plan["meals"][0]["protein_g"]
        apply_deterministic_day_summary(plan)
        self.assertEqual(plan["day_summary"]["total_protein_g"], 160 - 29)

    def test_a_day_without_a_day_summary_is_left_alone(self):
        plan = _plan(PRODUCTION_RUN_A_MEALS, PRODUCTION_RUN_A_REPORTED)
        del plan["day_summary"]
        apply_deterministic_day_summary(plan)
        self.assertNotIn("day_summary", plan)

    def test_applying_twice_changes_nothing(self):
        plan = _plan(PRODUCTION_RUN_A_MEALS, PRODUCTION_RUN_A_REPORTED)
        once = copy.deepcopy(apply_deterministic_day_summary(plan))
        twice = apply_deterministic_day_summary(copy.deepcopy(once))
        self.assertEqual(once, twice)

    def test_the_plan_object_is_returned_for_chaining(self):
        plan = _plan(PRODUCTION_RUN_A_MEALS, PRODUCTION_RUN_A_REPORTED)
        self.assertIs(apply_deterministic_day_summary(plan), plan)


# ----------------------------------------------------------------------
# 3. Meal values and everything else are untouched
# ----------------------------------------------------------------------

class TestNothingElseChanges(unittest.TestCase):
    def test_meal_nutrition_values_are_never_rescaled_or_modified(self):
        """Explicitly out of scope: the meals stay exactly as generated."""
        plan = _plan(PRODUCTION_RUN_A_MEALS, PRODUCTION_RUN_A_REPORTED)
        before = copy.deepcopy(plan["meals"])
        apply_deterministic_day_summary(plan)
        self.assertEqual(plan["meals"], before)

    def test_day_summary_notes_are_left_unchanged(self):
        plan = _plan(PRODUCTION_RUN_A_MEALS, PRODUCTION_RUN_A_REPORTED)
        apply_deterministic_day_summary(plan)
        self.assertEqual(
            plan["day_summary"]["notes"], PRODUCTION_RUN_A_REPORTED["notes"]
        )

    def test_day_summary_field_set_is_unchanged(self):
        plan = _plan(PRODUCTION_RUN_A_MEALS, PRODUCTION_RUN_A_REPORTED)
        apply_deterministic_day_summary(plan)
        self.assertEqual(set(plan["day_summary"]), set(PRODUCTION_RUN_A_REPORTED))

    def test_the_rest_of_the_plan_is_untouched(self):
        plan = _plan(PRODUCTION_RUN_A_MEALS, PRODUCTION_RUN_A_REPORTED)
        before = copy.deepcopy(plan)
        apply_deterministic_day_summary(plan)
        for key in ("confidence_score", "confidence_reason", "disclaimer"):
            self.assertEqual(plan[key], before[key])

    def test_recipe_resources_are_not_disturbed(self):
        """The two post-processing steps must not interfere with each other."""
        with mock.patch.object(meal_planner, "PROVIDER", "mock"):
            plan = generate_meal_plan(dict(BASE_PROFILE))
        for meal in plan["meals"]:
            for resource in meal.get("resources", []):
                self.assertTrue(
                    resource["url"].startswith(
                        "https://www.youtube.com/results?search_query="
                    )
                )


# ----------------------------------------------------------------------
# 4. mock_provider is unchanged / idempotent
# ----------------------------------------------------------------------

class TestMockProviderUnchanged(unittest.TestCase):
    def test_raw_mock_provider_output_needs_no_correction(self):
        """mock_provider is called directly, bypassing generate_meal_plan's
        post-processing, to prove _day_summary() was already consistent."""
        plan = mock_provider(dict(BASE_PROFILE))
        before = copy.deepcopy(plan)
        apply_deterministic_day_summary(plan)
        self.assertEqual(plan, before)

    def test_mock_multi_day_output_needs_no_correction(self):
        plan = mock_provider_multi_day(dict(BASE_PROFILE), 7)
        before = copy.deepcopy(plan)
        apply_deterministic_day_summary(plan)
        self.assertEqual(plan, before)

    def test_mock_stays_consistent_when_the_nutrition_tool_supplies_targets(self):
        """The production scenario, on the deterministic path: identical
        tool numbers, and the totals still equal the meals exactly."""
        agent_context = {
            "rag_context": "",
            "tool_results": {"nutrition_tool": {
                "protein": 129.6, "calories": 3110, "carbs": 576, "hydration": 3.52,
            }},
        }
        plan = mock_provider(dict(BASE_PROFILE), agent_context)
        before = copy.deepcopy(plan)
        apply_deterministic_day_summary(plan)
        self.assertEqual(plan, before)
        _assert_summary_equals_meals(self, plan)


# ----------------------------------------------------------------------
# 5. Multi-day shape, every day independently
# ----------------------------------------------------------------------

class TestMultiDayShape(unittest.TestCase):
    def _multi_day_plan(self):
        return {
            "duration_days": 3,
            "confidence_score": 8.0,
            "confidence_reason": ["test"],
            "days": [
                {
                    "day": n,
                    "day_summary": dict(PRODUCTION_RUN_A_REPORTED),
                    "meals": [_meal(i, m) for i, m in enumerate(PRODUCTION_RUN_A_MEALS)],
                }
                for n in (1, 2, 3)
            ],
            "disclaimer": "test disclaimer",
        }

    def test_every_day_is_corrected(self):
        plan = self._multi_day_plan()
        apply_deterministic_day_summary(plan)
        for day in plan["days"]:
            self.assertEqual(day["day_summary"]["total_protein_g"], 160)
            _assert_summary_equals_meals(self, day)

    def test_days_are_corrected_independently(self):
        plan = self._multi_day_plan()
        plan["days"][1]["meals"][0]["protein_g"] = 100  # 29 -> 100
        apply_deterministic_day_summary(plan)
        self.assertEqual(plan["days"][0]["day_summary"]["total_protein_g"], 160)
        self.assertEqual(plan["days"][1]["day_summary"]["total_protein_g"], 231)
        self.assertEqual(plan["days"][2]["day_summary"]["total_protein_g"], 160)

    def test_day_order_and_numbering_are_untouched(self):
        plan = self._multi_day_plan()
        apply_deterministic_day_summary(plan)
        self.assertEqual([d["day"] for d in plan["days"]], [1, 2, 3])
        self.assertEqual(plan["duration_days"], 3)


# ----------------------------------------------------------------------
# 6. Applied for every provider, through generate_meal_plan()
# ----------------------------------------------------------------------

class TestAppliedForEveryProvider(unittest.TestCase):
    def _inconsistent_plan(self):
        return _plan(PRODUCTION_RUN_A_MEALS, PRODUCTION_RUN_A_REPORTED)

    def _assert_corrected(self, plan):
        self.assertEqual(plan["day_summary"]["total_protein_g"], 160)
        _assert_summary_equals_meals(self, plan)

    def test_claude_provider_output_is_corrected(self):
        with mock.patch.object(meal_planner, "PROVIDER", "claude"), \
             mock.patch("llm_provider.claude_provider", return_value=self._inconsistent_plan()):
            self._assert_corrected(generate_meal_plan(dict(BASE_PROFILE)))

    def test_openrouter_provider_output_is_corrected(self):
        with mock.patch.object(meal_planner, "PROVIDER", "openrouter"), \
             mock.patch("llm_provider.openrouter_provider", return_value=self._inconsistent_plan()):
            self._assert_corrected(generate_meal_plan(dict(BASE_PROFILE)))

    def test_groq_provider_output_is_corrected(self):
        with mock.patch.object(meal_planner, "PROVIDER", "groq"), \
             mock.patch("llm_provider.groq_provider", return_value=self._inconsistent_plan()):
            self._assert_corrected(generate_meal_plan(dict(BASE_PROFILE)))

    def test_openrouter_multi_day_output_is_corrected(self):
        multi = {
            "duration_days": 7, "confidence_score": 8.0, "confidence_reason": ["t"],
            "days": [{
                "day": 1,
                "day_summary": dict(PRODUCTION_RUN_A_REPORTED),
                "meals": [_meal(i, m) for i, m in enumerate(PRODUCTION_RUN_A_MEALS)],
            }],
            "disclaimer": "test disclaimer",
        }
        with mock.patch.object(meal_planner, "PROVIDER", "openrouter"), \
             mock.patch("llm_provider.openrouter_provider_multi_day", return_value=multi):
            plan = generate_meal_plan(dict(BASE_PROFILE), days=7)
        self._assert_corrected(plan["days"][0])

    def test_mock_provider_through_the_endpoint_is_unaffected(self):
        with mock.patch.object(meal_planner, "PROVIDER", "mock"):
            plan = generate_meal_plan(dict(BASE_PROFILE))
        _assert_summary_equals_meals(self, plan)
        self.assertIn("notes", plan["day_summary"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
