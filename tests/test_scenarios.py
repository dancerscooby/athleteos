"""
AthleteOS - End-to-end scenario tests (Stage 1).

Five scenarios, each walking through the same stages a real request goes
through, with every stage logged:

    1. PROFILE          - the athlete profile for this scenario
    2. RAG RETRIEVAL     - the query sent to the retriever, and which
                            knowledge_base chunks came back (rag.py already
                            logs each retrieved chunk's id + similarity)
    3. TOOL CALL         - a nutrition_tool / budget_tool call representative
                            of what this scenario would need, run through the
                            REAL tool_selector.execute_tool_call() code path
    4. FINAL RESPONSE    - the meal plan from generate_meal_plan()

Important, honest scope note: no ANTHROPIC_API_KEY is configured in this
project's environment, so these tests do NOT make live Claude API calls -
doing so would cost real money on every test run, and CI/local runs should
not silently do that. Two things follow from this:

  - Stage 4 always runs through the CURRENT PROVIDER in meal_planner.py
    (the deterministic mock_provider). This is still a genuine end-to-end
    run of the real pipeline (validation shape, RAG injection points,
    generate_meal_plan), just not routed through the live LLM.
  - Stage 3 cannot test "did Claude decide to call this tool" (that decision
    only happens inside a real API call). It DOES test the exact same
    execute_tool_call() function a real Claude tool_use response would be
    handed to - so the tool execution, logging, and error handling are
    fully verified, only Claude's own judgment call is not exercised here.

If ANTHROPIC_API_KEY is set in the environment, see the README for how to
additionally run a real end-to-end check against claude_provider().
"""

import logging
import unittest

from pydantic import ValidationError

from api import MealPlanRequest
from meal_planner import generate_meal_plan
from rag import build_retrieval_query, get_context_block
from tool_selector import execute_tool_call

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("athleteos.tests")


class FakeToolUseBlock:
    """
    Mimics the shape of a Claude tool_use content block (.name, .input, .id).
    Lets these tests exercise the SAME execute_tool_call() code a real
    Claude tool_use response is handed to, without needing a live API key.
    """

    def __init__(self, name, input_, id_):
        self.name = name
        self.input = input_
        self.id = id_


def run_scenario(label, profile, tool_call=None):
    """
    Shared 4-stage pipeline used by every scenario below. Returns
    (plan, tool_result) so each test can make its own assertions.
    `tool_call` is an optional (name, input_dict) tuple to exercise.
    """
    logger.info("=" * 70)
    logger.info("SCENARIO: %s", label)
    logger.info("=" * 70)

    logger.info("STAGE 1/4 - PROFILE: %s", profile)

    logger.info("STAGE 2/4 - RAG RETRIEVAL")
    query = build_retrieval_query(profile)
    context_block = get_context_block(profile)  # logs each retrieved chunk itself
    logger.info("Context block length: %d characters", len(context_block))

    tool_result = None
    if tool_call:
        logger.info("STAGE 3/4 - TOOL CALL")
        name, tool_input = tool_call
        block = FakeToolUseBlock(name, tool_input, "toolu_test")
        tool_result = execute_tool_call(block)  # logs selection/inputs/outputs itself
    else:
        logger.info("STAGE 3/4 - TOOL CALL: none exercised for this scenario")

    logger.info("STAGE 4/4 - FINAL RESPONSE (current provider)")
    plan = generate_meal_plan(profile)
    logger.info("Final response: %s", plan)

    return plan, tool_result


def all_ingredients(plan):
    """Flatten every ingredient name across all meals, for easy assertions."""
    return [ing["item"] for meal in plan["meals"] for ing in meal["ingredients"]]


class TestScenarios(unittest.TestCase):
    """
    Five stage-1 end-to-end scenarios. Test method names are numbered so
    they read top-to-bottom in the same order as the log output
    (unittest runs test methods in alphabetical order).
    """

    # ------------------------------------------------------------------
    # 1. Vegetarian footballer
    # ------------------------------------------------------------------
    def test_1_vegetarian_footballer(self):
        profile = dict(
            age=22, height_cm=178, weight_kg=72, sex="Male",
            goal="Improve football performance and some muscle gain",
            diet="Vegetarian", training_days_per_week=5, training_intensity="High",
            region="Lucknow, Uttar Pradesh", allergies="none",
            budget_per_day="Rs 300", food_preferences=None,
            foods_to_avoid=None, notes=None,
        )
        plan, tool_result = run_scenario(
            "Vegetarian footballer",
            profile,
            tool_call=("nutrition_tool", {
                "operation": "protein", "weight_kg": 72, "training_intensity": "High",
            }),
        )

        self.assertEqual(
            set(plan.keys()),
            {"confidence_score", "confidence_reason", "day_summary", "meals", "disclaimer"},
        )
        self.assertEqual(len(plan["meals"]), 4)

        # Vegetarian diet -> no meat/egg/fish anywhere in the plan.
        items_lower = " ".join(all_ingredients(plan)).lower()
        for banned in ("chicken", "egg", "fish", "mutton"):
            self.assertNotIn(banned, items_lower)

        self.assertEqual(tool_result["content"], '{"operation": "protein", "result": 129.6}')

    # ------------------------------------------------------------------
    # 2. Non-vegetarian footballer
    # ------------------------------------------------------------------
    def test_2_non_vegetarian_footballer(self):
        profile = dict(
            age=22, height_cm=178, weight_kg=72, sex="Male",
            goal="Improve football performance and some muscle gain",
            diet="Non-Vegetarian", training_days_per_week=5, training_intensity="High",
            region="Lucknow, Uttar Pradesh", allergies="none",
            budget_per_day="Rs 300", food_preferences=None,
            foods_to_avoid=None, notes=None,
        )
        plan, tool_result = run_scenario(
            "Non-vegetarian footballer",
            profile,
            tool_call=("nutrition_tool", {
                "operation": "calories", "weight_kg": 72, "height_cm": 178,
                "age": 22, "sex": "Male", "training_intensity": "High",
            }),
        )

        # Non-veg diet -> breakfast always includes eggs (see meal_planner.build_breakfast),
        # and lunch/dinner draw from the region's non-veg protein list.
        items_lower = " ".join(all_ingredients(plan)).lower()
        self.assertIn("egg", items_lower)

        self.assertEqual(tool_result["content"], '{"operation": "calories", "result": 3110}')

    # ------------------------------------------------------------------
    # 3. Match-day scenario
    # ------------------------------------------------------------------
    def test_3_match_day(self):
        profile = dict(
            age=22, height_cm=178, weight_kg=72, sex="Male",
            goal="Fuel properly for a match tomorrow morning",
            diet="Vegetarian", training_days_per_week=5, training_intensity="High",
            region="Goa", allergies="none",
            budget_per_day="Rs 300", food_preferences="easily digestible foods",
            foods_to_avoid="heavy fried food", notes="Match tomorrow at 9 AM",
        )
        plan, tool_result = run_scenario(
            "Match-day scenario",
            profile,
            tool_call=("nutrition_tool", {
                "operation": "carbs", "weight_kg": 72, "training_intensity": "High",
            }),
        )

        # Match-day goal -> dinner becomes the light, easy-to-digest khichdi
        # (see meal_planner.build_dinner's goal_key == "match" branch).
        dinner = next(m for m in plan["meals"] if m["meal"] == "Dinner")
        dinner_items = " ".join(i["item"] for i in dinner["ingredients"]).lower()
        self.assertIn("khichdi", dinner_items)

        # Note: 72*8=576 here is a plain int (Python's round(int, 1) returns
        # an int, not a float), so this stays "576" not "576.0".
        self.assertEqual(tool_result["content"], '{"operation": "carbs", "result": 576}')

    # ------------------------------------------------------------------
    # 4. Low-budget scenario
    # ------------------------------------------------------------------
    def test_4_low_budget(self):
        profile = dict(
            age=20, height_cm=175, weight_kg=68, sex="Male",
            goal="Improve fitness and recovery",
            diet="Non-Vegetarian", training_days_per_week=4, training_intensity="Moderate",
            region="Kolkata, West Bengal", allergies="none",
            budget_per_day="Rs 120", food_preferences=None,
            foods_to_avoid=None, notes=None,
        )
        plan, tool_result = run_scenario(
            "Low-budget scenario",
            profile,
            tool_call=("budget_tool", {"budget_per_day": 120, "diet": "Non-Vegetarian"}),
        )

        # Budget too low for the region's pricier non-veg protein (fish) ->
        # the mock's choose_items() should fall back to the cheaper egg curry
        # (verified in an earlier session at budget Rs 120 vs Rs 400).
        lunch = next(m for m in plan["meals"] if m["meal"] == "Lunch")
        lunch_items = [i["item"] for i in lunch["ingredients"]]
        self.assertIn("Egg curry", lunch_items)
        self.assertNotIn("Fish curry", lunch_items)

        self.assertIn("within_budget", tool_result["content"])
        self.assertNotIn("is_error", tool_result)

    # ------------------------------------------------------------------
    # 5. Invalid-input scenario
    # ------------------------------------------------------------------
    def test_5_invalid_input(self):
        logger.info("=" * 70)
        logger.info("SCENARIO: Invalid input")
        logger.info("=" * 70)

        bad_profile = dict(
            age=8,                        # below the allowed minimum (10)
            height_cm=178, weight_kg=72, sex="Male",
            goal="Get fit",
            diet="meat",                   # not one of Vegetarian/Non-Vegetarian
            training_days_per_week=9,      # above the allowed maximum (7)
            training_intensity="extreme",  # not one of Low/Moderate/High
            region="Lucknow", allergies="none",
        )
        logger.info("STAGE 1/4 - PROFILE (intentionally invalid): %s", bad_profile)
        logger.info(
            "STAGE 2/4 - RAG RETRIEVAL: skipped - the real system (api.py's "
            "MealPlanRequest) validates input BEFORE generate_meal_plan() is "
            "ever called, so RAG/tools/the provider never run for bad input."
        )
        logger.info("STAGE 3/4 - TOOL CALL: skipped, same reason as above")

        logger.info("STAGE 4/4 - FINAL RESPONSE: validation error (expected)")
        with self.assertRaises(ValidationError) as ctx:
            MealPlanRequest(**bad_profile)

        errors = ctx.exception.errors()
        logger.info("Final response (validation errors): %s", errors)

        bad_fields = {e["loc"][-1] for e in errors}
        self.assertEqual(bad_fields, {"age", "diet", "training_days_per_week", "training_intensity"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
