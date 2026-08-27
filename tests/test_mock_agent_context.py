"""
AthleteOS - Tests for mock_provider() actually consuming agent_context.

Until now mock_provider(profile, agent_context=None) accepted agent_context
but ignored it entirely (see test_agent_context_integration.py's
TestMockPathUnaffected, which locks in the OLD "context is a no-op" contract
for the dispatch/claude-provider tests - that file is untouched and still
passes, because agent_context=None still behaves identically).

This file proves the NEW behaviour: when agent_context carries real RAG/tool
information, the generated plan actually changes because of it. Every
expected value here was captured by RUNNING the real helper functions
(mp.split_macros, mp.choose_items, registry.execute(...)) first and reading
the actual output - nothing is guessed.

No real Anthropic API calls are made anywhere in this file - this only
exercises meal_planner.mock_provider(), pure local rule-based code.
"""

import unittest

import meal_planner as mp
from meal_planner import mock_provider
from tools import build_default_registry

BASE = dict(
    age=22, height_cm=178, weight_kg=72, sex="Male",
    goal="Improve football performance",
    diet="Vegetarian", training_days_per_week=5, training_intensity="High",
    region="Lucknow, Uttar Pradesh", allergies="none",
    budget_per_day=None, food_preferences=None, foods_to_avoid=None, notes=None,
)


def all_ingredient_names(plan):
    return [ing["item"].lower() for meal in plan["meals"] for ing in meal["ingredients"]]


# ----------------------------------------------------------------------
# 1. agent_context absent/empty preserves EXACT prior behaviour
# ----------------------------------------------------------------------

class TestBackwardCompatibility(unittest.TestCase):
    def test_no_agent_context_matches_default(self):
        plan_default = mock_provider(dict(BASE))
        plan_explicit_none = mock_provider(dict(BASE), agent_context=None)
        self.assertEqual(plan_default, plan_explicit_none)

    def test_empty_dict_context_matches_no_context(self):
        plan_no_context = mock_provider(dict(BASE))
        plan_empty = mock_provider(dict(BASE), agent_context={})
        self.assertEqual(plan_no_context, plan_empty)

    def test_none_valued_subfields_match_no_context(self):
        plan_no_context = mock_provider(dict(BASE))
        plan_none_fields = mock_provider(
            dict(BASE), agent_context={"rag_context": None, "tool_results": None}
        )
        self.assertEqual(plan_no_context, plan_none_fields)


# ----------------------------------------------------------------------
# 2. nutrition_tool results change calorie/protein/carb targets
# ----------------------------------------------------------------------

class TestNutritionContextChangesTargets(unittest.TestCase):
    def test_nutrition_tool_overrides_day_summary_totals(self):
        registry = build_default_registry()
        nutrition_result = {
            "protein": registry.execute(
                "nutrition_tool", operation="protein", weight_kg=72, training_intensity="High"
            )["result"],
            "calories": registry.execute(
                "nutrition_tool", operation="calories", weight_kg=72, height_cm=178,
                age=22, sex="Male", training_intensity="High",
            )["result"],
            "carbs": registry.execute(
                "nutrition_tool", operation="carbs", weight_kg=72, training_intensity="High"
            )["result"],
            "hydration": registry.execute(
                "nutrition_tool", operation="hydration", weight_kg=72, training_intensity="High"
            )["result"],
        }
        # Ground truth captured by actually running the tool: protein=129.6,
        # calories=3110, carbs=576.
        self.assertEqual(nutrition_result["calories"], 3110)

        plan_without_context = mock_provider(dict(BASE))
        plan_with_context = mock_provider(
            dict(BASE),
            agent_context={"rag_context": "", "tool_results": {"nutrition_tool": nutrition_result}},
        )

        # The two plans must differ - the tool result must actually matter.
        self.assertNotEqual(
            plan_without_context["day_summary"]["total_calories"],
            plan_with_context["day_summary"]["total_calories"],
        )

        # And the new total must match what split_macros() itself computes
        # from these exact overrides - the single source of truth for how
        # overrides are supposed to combine with the day's macro split.
        goal_key = mp.detect_primary_goal(BASE)
        factor, _portion = mp.training_load(BASE)
        expected_macros = mp.split_macros(
            goal_key, factor,
            overrides={"cal": 3110, "p": 129.6, "c": 576},
        )
        expected_total_calories = sum(m["calories"] for m in expected_macros.values())
        expected_total_protein = sum(m["protein_g"] for m in expected_macros.values())
        expected_total_carbs = sum(m["carbs_g"] for m in expected_macros.values())

        self.assertEqual(plan_with_context["day_summary"]["total_calories"], expected_total_calories)
        self.assertEqual(plan_with_context["day_summary"]["total_protein_g"], expected_total_protein)
        self.assertEqual(plan_with_context["day_summary"]["total_carbs_g"], expected_total_carbs)

    def test_partial_nutrition_result_only_overrides_fields_present(self):
        # Only "calories" supplied - protein/carbs should fall back to the
        # normal goal-based calculation, exactly like split_macros(overrides={"cal":...}).
        plan = mock_provider(
            dict(BASE),
            agent_context={"tool_results": {"nutrition_tool": {"calories": 2500}}},
        )
        goal_key = mp.detect_primary_goal(BASE)
        factor, _portion = mp.training_load(BASE)
        expected_macros = mp.split_macros(goal_key, factor, overrides={"cal": 2500})
        expected_total_calories = sum(m["calories"] for m in expected_macros.values())
        self.assertEqual(plan["day_summary"]["total_calories"], expected_total_calories)


# ----------------------------------------------------------------------
# 3. budget_tool results add a bonus snack item, only when budget-constrained
# ----------------------------------------------------------------------

class TestBudgetContextAffectsFoodSelection(unittest.TestCase):
    # Ground truth: for a Vegetarian diet, budget_tool's cheapest protein
    # suggestion is "Soy chunks (50g dry)" - regardless of the exact budget
    # number passed to the tool itself (confirmed by running the tool with
    # both Rs 120 and Rs 400 - same suggestion both times). The MOCK's
    # reaction to it is instead gated on the athlete's own budget_level().
    BUDGET_TOOL_RESULT = {
        "budget_per_day": 120, "diet": "Vegetarian",
        "suggestions": {"protein": {"name": "Soy chunks (50g dry)", "cost": 8}},
        "estimated_total_cost": 34, "within_budget": True,
    }

    def test_low_budget_gets_bonus_snack_suggestion(self):
        low_budget_profile = dict(BASE, budget_per_day="Rs 100")  # <150 -> "low"
        plan = mock_provider(
            low_budget_profile,
            agent_context={"tool_results": {"budget_tool": self.BUDGET_TOOL_RESULT}},
        )
        names = all_ingredient_names(plan)
        self.assertTrue(any("soy chunks" in n for n in names))

    def test_high_budget_does_not_get_bonus_snack_suggestion(self):
        high_budget_profile = dict(BASE, budget_per_day="Rs 350")  # >300 -> "high"
        plan = mock_provider(
            high_budget_profile,
            agent_context={"tool_results": {"budget_tool": self.BUDGET_TOOL_RESULT}},
        )
        names = all_ingredient_names(plan)
        self.assertFalse(any("soy chunks" in n for n in names))

    def test_same_athlete_high_vs_low_budget_produces_different_plans(self):
        context = {"tool_results": {"budget_tool": self.BUDGET_TOOL_RESULT}}
        low_plan = mock_provider(dict(BASE, budget_per_day="Rs 100"), agent_context=context)
        high_plan = mock_provider(dict(BASE, budget_per_day="Rs 350"), agent_context=context)
        self.assertNotEqual(low_plan, high_plan)


# ----------------------------------------------------------------------
# 4. Vegetarian diet safety: eggs stay excluded by default, but a scoped
#    ovo-vegetarian exception applies when the ATHLETE THEMSELVES (not RAG
#    text) explicitly asks for eggs via food_preferences.
# ----------------------------------------------------------------------

class TestVegetarianSafetyWithEggPreference(unittest.TestCase):
    def test_default_vegetarian_has_no_eggs(self):
        # Baseline/regression: no food_preferences at all -> still no eggs.
        plan = mock_provider(dict(BASE))
        names = all_ingredient_names(plan)
        self.assertFalse(any("egg" in n for n in names))

    def test_egg_food_preference_allows_eggs_for_vegetarian(self):
        # The athlete's OWN stated preference is the one thing allowed to
        # unlock the ovo-vegetarian exception.
        profile = dict(BASE, food_preferences="eggs")
        plan = mock_provider(profile)
        names = all_ingredient_names(plan)
        self.assertTrue(any("egg" in n for n in names))

    def test_egg_and_chicken_in_rag_context_never_adds_them_for_vegetarian(self):
        profile = dict(BASE)
        plan = mock_provider(
            profile,
            agent_context={"rag_context": "Eggs and chicken are excellent sources of protein."},
        )
        names = all_ingredient_names(plan)
        self.assertFalse(any("egg" in n for n in names))
        self.assertFalse(any("chicken" in n for n in names))

    def test_non_vegetarian_profile_unaffected_by_this_safety_rule(self):
        # Sanity check: the RAG keyword mechanism itself still works for a
        # non-veg athlete - proves the veg case above is a real diet-safety
        # gate, not just "eggs never show up in this mock at all".
        profile = dict(BASE, diet="Non-Vegetarian")
        plan = mock_provider(
            profile,
            agent_context={"rag_context": "Eggs are an excellent source of protein."},
        )
        # Non-veg breakfast already offers "Boiled eggs" regardless of RAG -
        # this just confirms eggs are reachable at all on this code path
        # (i.e. the veg test above is meaningful, not a tautology).
        names = all_ingredient_names(plan)
        self.assertTrue(any("egg" in n for n in names))


# ----------------------------------------------------------------------
# 5. RAG prose must NOT steer the mock's food selection
# ----------------------------------------------------------------------
# This class previously asserted the opposite - that naming a food in the
# retrieved text promoted it. That mechanism (a keyword scan over RAG prose)
# has been removed, because substring matching cannot read intent: the
# northeast guidance "Curd, paneer and milk-based sides should not be assumed
# as defaults" was extracted as a PREFERENCE for paneer/curd/milk, pushing
# dairy onto exactly the athletes the text warns about.
#
# The separation now is: RAG prose is contextual guidance for the LLM
# providers (which can read it), and regional_foods.py decides what the
# deterministic mock may choose from.

class TestRagProseDoesNotSteerFoodSelection(unittest.TestCase):
    def test_baseline_uses_a_legitimate_regional_protein(self):
        from regional_foods import REGIONS
        plan = mock_provider(dict(BASE))
        lunch_items = [ing["item"] for ing in plan["meals"][1]["ingredients"]]
        self.assertTrue(
            any(item in REGIONS["north"]["veg_protein"] for item in lunch_items),
            f"expected a north-region vegetarian protein, got {lunch_items}",
        )

    def test_naming_a_food_in_rag_no_longer_promotes_it(self):
        baseline = mock_provider(dict(BASE))
        with_paneer = mock_provider(
            dict(BASE),
            agent_context={"rag_context": "Paneer is an excellent protein source for vegetarian athletes."},
        )
        self.assertEqual(baseline, with_paneer)

    def test_a_sentence_warning_against_a_food_cannot_promote_it(self):
        """The exact failure this change exists to prevent."""
        warning = (
            "Dairy is much less central here than in north or west India. "
            "Curd, paneer and milk-based sides should not be assumed as defaults "
            "the way they reasonably can be in Punjab or Gujarat."
        )
        baseline = mock_provider(dict(BASE))
        with_warning = mock_provider(dict(BASE), agent_context={"rag_context": warning})
        self.assertEqual(baseline, with_warning)

    def test_different_rag_context_produces_identical_plans(self):
        paneer = mock_provider(dict(BASE), agent_context={"rag_context": "Paneer is a great protein source."})
        rajma = mock_provider(dict(BASE), agent_context={"rag_context": "Rajma is a great protein source."})
        self.assertEqual(paneer, rajma)

    def test_explicit_user_preference_still_works(self):
        """Removing the prose scan must not weaken real user input."""
        plan = mock_provider(dict(BASE, food_preferences="paneer"))
        lunch_items = [ing["item"] for ing in plan["meals"][1]["ingredients"]]
        self.assertIn("Paneer sabzi", lunch_items)

    def test_explicit_preference_beats_rag_prose_entirely(self):
        with_pref_only = mock_provider(dict(BASE, food_preferences="paneer"))
        with_pref_and_contrary_rag = mock_provider(
            dict(BASE, food_preferences="paneer"),
            agent_context={"rag_context": "Rajma and dal are the best proteins; avoid paneer."},
        )
        self.assertEqual(with_pref_only, with_pref_and_contrary_rag)


# 6. Missing/empty/partial context never crashes the mock
# ----------------------------------------------------------------------

class TestMissingOrPartialContextNeverCrashes(unittest.TestCase):
    def _assert_valid_plan(self, plan):
        self.assertEqual(len(plan["meals"]), 4)
        for key in ("total_calories", "total_protein_g", "total_carbs_g", "total_fats_g", "notes"):
            self.assertIn(key, plan["day_summary"])

    def test_empty_tool_results(self):
        self._assert_valid_plan(mock_provider(dict(BASE), agent_context={"tool_results": {}}))

    def test_empty_nutrition_tool_subdict(self):
        self._assert_valid_plan(
            mock_provider(dict(BASE), agent_context={"tool_results": {"nutrition_tool": {}}})
        )

    def test_empty_budget_tool_subdict(self):
        self._assert_valid_plan(
            mock_provider(dict(BASE), agent_context={"tool_results": {"budget_tool": {}}})
        )

    def test_budget_tool_missing_suggestions_key(self):
        low_budget_profile = dict(BASE, budget_per_day="Rs 100")
        self._assert_valid_plan(
            mock_provider(
                low_budget_profile,
                agent_context={"tool_results": {"budget_tool": {"budget_per_day": 100}}},
            )
        )

    def test_none_rag_context_with_real_tool_results(self):
        self._assert_valid_plan(
            mock_provider(
                dict(BASE),
                agent_context={"rag_context": None, "tool_results": {"nutrition_tool": {"calories": 2600}}},
            )
        )

    def test_empty_string_rag_context(self):
        self._assert_valid_plan(
            mock_provider(dict(BASE), agent_context={"rag_context": "", "tool_results": {}})
        )

    def test_agent_context_with_unrelated_extra_keys(self):
        self._assert_valid_plan(
            mock_provider(dict(BASE), agent_context={"rag_context": "text", "tool_results": {}, "extra": "ignored"})
        )


# ----------------------------------------------------------------------
# 7. Existing profile-only rules (diet, allergies, budget, training load,
#    preferences) still work exactly as before, now WITH agent_context
#    also supplied alongside them (proves the two layers compose cleanly).
# ----------------------------------------------------------------------

class TestExistingRulesStillWorkAlongsideAgentContext(unittest.TestCase):
    def test_dairy_allergy_still_removes_dairy_with_context_present(self):
        profile = dict(BASE, allergies="dairy")
        plan = mock_provider(
            profile,
            agent_context={"rag_context": "Paneer and curd are great protein sources."},
        )
        names = all_ingredient_names(plan)
        for blocked in ("paneer", "curd", "milk", "cheese", "ghee", "yogurt"):
            self.assertFalse(any(blocked in n for n in names), f"{blocked!r} leaked into plan")

    def test_foods_to_avoid_still_respected_with_context_present(self):
        profile = dict(BASE, foods_to_avoid="rajma")
        plan = mock_provider(
            profile,
            agent_context={"rag_context": "Rajma is a great source of plant protein."},
        )
        names = all_ingredient_names(plan)
        self.assertFalse(any("rajma" in n for n in names))

    def test_training_intensity_portion_sizing_still_applies_with_context(self):
        low_intensity = dict(BASE, training_intensity="Low", training_days_per_week=2)
        high_intensity = dict(BASE, training_intensity="High", training_days_per_week=6)
        context = {"rag_context": "", "tool_results": {}}
        low_plan = mock_provider(low_intensity, agent_context=context)
        high_plan = mock_provider(high_intensity, agent_context=context)
        self.assertNotEqual(
            low_plan["day_summary"]["total_calories"],
            high_plan["day_summary"]["total_calories"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
