"""
AthleteOS - Phase 4 (steps 1-2): optional home ingredients.

Two pieces, tested here:

  SELECTION  meal_planner.ingredient_tokens() feeds the athlete's home
             ingredients into the existing `prefer` list, so owned foods are
             promoted - but only ever within what the athlete is already
             allowed to eat.

  TOOL       tools/ingredients_tool.py reports which ingredients are usable,
             which are excluded and why, which food groups they cover and
             which are missing; the orchestrator selects it only when
             ingredients were actually supplied.

The feature is optional end to end: with no ingredients, every plan must be
byte-identical to one produced before the feature existed.

Everything here is offline and deterministic - no network, no API calls.
"""

import tempfile
import unittest
from pathlib import Path

import meal_planner as mp
import memory_manager as mm
import orchestrator as orch
from tools import build_default_registry
from tools.ingredients_tool import assess_ingredients, parse_ingredients

BASE = dict(
    age=22, height_cm=178, weight_kg=72, sex="Male",
    goal="Improve football performance",
    diet="Vegetarian", training_days_per_week=5, training_intensity="High",
    region="Lucknow, Uttar Pradesh", allergies="none",
    budget_per_day=None, food_preferences=None, foods_to_avoid=None,
    notes=None, available_ingredients=None,
)


def items(profile):
    plan = mp.mock_provider(dict(profile))
    return [ing["item"] for meal in plan["meals"] for ing in meal["ingredients"]]


def joined(profile):
    return " ".join(items(profile)).lower()


# ----------------------------------------------------------------------
# 1. The feature is optional
# ----------------------------------------------------------------------

class TestIngredientsAreOptional(unittest.TestCase):
    def test_no_ingredients_matches_the_field_being_absent_entirely(self):
        without_field = {k: v for k, v in BASE.items() if k != "available_ingredients"}
        self.assertEqual(mp.mock_provider(dict(BASE)), mp.mock_provider(without_field))

    def test_empty_and_none_and_none_string_all_mean_no_ingredients(self):
        baseline = mp.mock_provider(dict(BASE))
        for value in [None, "", "   ", "none"]:
            with self.subTest(value=value):
                self.assertEqual(mp.mock_provider(dict(BASE, available_ingredients=value)), baseline)

    def test_ingredient_tokens_parses_like_the_other_token_helpers(self):
        self.assertEqual(
            mp.ingredient_tokens({"available_ingredients": "Rice, Dal and Tomatoes"}),
            ["rice", "dal", "tomatoes"],
        )
        self.assertEqual(mp.ingredient_tokens({}), [])
        self.assertEqual(mp.ingredient_tokens({"available_ingredients": "none"}), [])


# ----------------------------------------------------------------------
# 2. Ingredients influence selection where a candidate actually matches
# ----------------------------------------------------------------------

class TestIngredientsInfluenceSelection(unittest.TestCase):
    def test_an_owned_food_is_promoted_when_a_candidate_matches(self):
        baseline = joined(BASE)
        self.assertNotIn("paneer", baseline)  # not chosen on its own
        self.assertIn("paneer", joined(dict(BASE, available_ingredients="paneer")))

    def test_a_second_owned_food_is_also_promoted(self):
        self.assertIn("rajma", joined(dict(BASE, available_ingredients="rajma")))

    def test_an_unmatched_ingredient_changes_nothing_and_claims_nothing(self):
        """"If an ingredient isn't represented in the structured food
        candidates, don't pretend it was used." A food the candidate lists
        have never heard of must leave the plan untouched."""
        baseline = mp.mock_provider(dict(BASE))
        with_odd = mp.mock_provider(dict(BASE, available_ingredients="quinoa, kale, glitter"))
        self.assertEqual(baseline, with_odd)

    def test_ingredients_do_not_add_ingredients_to_the_plan_text(self):
        """Selection is biased, but nothing is inserted verbatim - the plan
        only ever contains real candidate foods."""
        plan = mp.mock_provider(dict(BASE, available_ingredients="glitter"))
        self.assertNotIn("glitter", " ".join(
            ing["item"] for m in plan["meals"] for ing in m["ingredients"]).lower())


# ----------------------------------------------------------------------
# 3. Existing rules always win
# ----------------------------------------------------------------------

class TestExistingRulesOverrideIngredients(unittest.TestCase):
    def test_allergy_beats_a_provided_ingredient(self):
        text = joined(dict(BASE, allergies="dairy", available_ingredients="paneer, curd, milk"))
        for blocked in ("paneer", "curd", "milk", "cheese", "ghee"):
            self.assertNotIn(blocked, text)

    def test_foods_to_avoid_beats_a_provided_ingredient(self):
        text = joined(dict(BASE, foods_to_avoid="rajma", available_ingredients="rajma"))
        self.assertNotIn("rajma", text)

    def test_vegetarian_diet_beats_provided_non_vegetarian_ingredients(self):
        text = joined(dict(BASE, available_ingredients="chicken, fish, mutton, prawns"))
        for meat in ("chicken", "fish", "mutton", "prawn"):
            self.assertNotIn(meat, text)

    def test_ingredients_do_not_unlock_the_vegetarian_egg_exception(self):
        """Having eggs at home is not the same as a vegetarian athlete
        asking for eggs - they may be for someone else in the household.
        Only food_preferences may unlock that exception."""
        via_ingredients = joined(dict(BASE, available_ingredients="eggs"))
        self.assertNotIn("egg", via_ingredients)
        # ...while the existing, explicit route still works.
        via_preference = joined(dict(BASE, food_preferences="eggs"))
        self.assertIn("egg", via_preference)

    def test_minor_portion_rules_are_unaffected_by_ingredients(self):
        child = dict(BASE, age=10, weight_kg=32, height_cm=145, goal="muscle growth",
                     available_ingredients="paneer, rajma")
        adult = dict(child, age=25, weight_kg=72, height_cm=178)
        child_plan = mp.mock_provider(child)
        adult_plan = mp.mock_provider(adult)
        self.assertIn("Growth-appropriate", child_plan["day_summary"]["notes"])
        self.assertNotIn("Growth-appropriate", adult_plan["day_summary"]["notes"])

    def test_region_still_governs_the_candidate_pool(self):
        """An ingredient can only promote a food the region actually offers;
        it cannot import another region's dish."""
        plan_items = items(dict(BASE, region="Chennai, Tamil Nadu",
                                available_ingredients="rajma"))
        from regional_foods import REGIONS
        south = set(REGIONS["south"]["veg_protein"])
        self.assertFalse(any(i == "Rajma" for i in plan_items))
        self.assertTrue(any(i in south for i in plan_items))


# ----------------------------------------------------------------------
# 4. Determinism
# ----------------------------------------------------------------------

class TestDeterminism(unittest.TestCase):
    def test_same_profile_and_ingredients_reproduce_the_same_plan(self):
        profile = dict(BASE, available_ingredients="paneer, rajma, curd")
        self.assertEqual(mp.mock_provider(dict(profile)), mp.mock_provider(dict(profile)))

    def test_ingredients_do_not_change_the_variety_seed(self):
        """Editing the pantry must not reshuffle the whole plan - the same
        rule already applied to allergies and preferences."""
        self.assertEqual(
            mp.variety_seed(dict(BASE)),
            mp.variety_seed(dict(BASE, available_ingredients="paneer, rajma, curd")),
        )

    def test_ingredient_order_does_not_matter_to_the_seed(self):
        self.assertEqual(
            mp.variety_seed(dict(BASE, available_ingredients="a, b")),
            mp.variety_seed(dict(BASE, available_ingredients="b, a")),
        )


# ----------------------------------------------------------------------
# 5. The ingredients tool itself
# ----------------------------------------------------------------------

class TestIngredientsTool(unittest.TestCase):
    def test_parse_handles_text_lists_and_nothing(self):
        self.assertEqual(parse_ingredients("rice, dal and curd"), ["rice", "dal", "curd"])
        self.assertEqual(parse_ingredients(["Rice", " Dal "]), ["rice", "dal"])
        self.assertEqual(parse_ingredients(None), [])
        self.assertEqual(parse_ingredients("none"), [])

    def test_usable_ingredients_are_grouped_by_food_group(self):
        result = assess_ingredients("rice, dal, tomatoes, banana, peanuts", diet="Vegetarian")
        self.assertIn("rice", result["covered_groups"]["carbohydrate"])
        self.assertIn("dal", result["covered_groups"]["protein"])
        self.assertIn("tomatoes", result["covered_groups"]["vegetable"])
        self.assertIn("banana", result["covered_groups"]["fruit"])
        self.assertIn("peanuts", result["covered_groups"]["fat"])
        self.assertEqual(result["missing_groups"], [])

    def test_missing_groups_are_reported(self):
        result = assess_ingredients("rice", diet="Vegetarian")
        self.assertEqual(result["covered_groups"], {"carbohydrate": ["rice"]})
        for group in ("protein", "fat", "fruit", "vegetable"):
            self.assertIn(group, result["missing_groups"])

    def test_non_vegetarian_ingredients_are_excluded_for_a_vegetarian(self):
        result = assess_ingredients("rice, chicken, eggs", diet="Vegetarian")
        self.assertEqual(result["usable"], ["rice"])
        excluded = {e["ingredient"] for e in result["excluded"]}
        self.assertEqual(excluded, {"chicken", "eggs"})
        for entry in result["excluded"]:
            self.assertIn("Vegetarian", entry["reason"])

    def test_non_vegetarian_athlete_may_use_them(self):
        result = assess_ingredients("rice, chicken", diet="Non-Vegetarian")
        self.assertIn("chicken", result["usable"])

    def test_avoid_words_exclude_an_ingredient_with_a_reason(self):
        result = assess_ingredients("rice, paneer", diet="Vegetarian", avoid=["paneer"])
        self.assertEqual(result["usable"], ["rice"])
        self.assertEqual(result["excluded"][0]["ingredient"], "paneer")
        self.assertIn("allergy", result["excluded"][0]["reason"])

    def test_unrecognised_ingredients_are_reported_not_silently_counted(self):
        result = assess_ingredients("glitter", diet="Vegetarian")
        self.assertEqual(result["unrecognised"], ["glitter"])
        self.assertEqual(result["covered_groups"], {})

    def test_tool_runs_through_the_registry(self):
        registry = build_default_registry()
        self.assertIn("ingredients_tool", [t["name"] for t in registry.list_tools()])
        result = registry.execute("ingredients_tool", ingredients="rice, dal", diet="Vegetarian")
        self.assertEqual(result["usable"], ["rice", "dal"])

    def test_existing_tools_still_registered(self):
        names = [t["name"] for t in build_default_registry().list_tools()]
        self.assertIn("nutrition_tool", names)
        self.assertIn("budget_tool", names)


# ----------------------------------------------------------------------
# 6. Orchestrator selection
# ----------------------------------------------------------------------

class MemoryIsolatedTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        tmp = Path(self._tmpdir.name)
        self._orig = {n: getattr(mm, n) for n in
                      ["MEMORY_DIR", "PROFILES_FILE", "PREFERENCES_FILE", "LATEST_PLAN_FILE",
                       "RECIPES_FILE", "EPISODES_DIR", "CACHE_DIR", "UPDATABLE_CATEGORIES"]}
        mm.MEMORY_DIR = tmp
        mm.PROFILES_FILE = tmp / "profiles.json"
        mm.PREFERENCES_FILE = tmp / "preferences.json"
        mm.LATEST_PLAN_FILE = tmp / "latest_plans.json"
        mm.RECIPES_FILE = tmp / "recipes.json"
        mm.EPISODES_DIR = tmp / "episodes"
        mm.CACHE_DIR = tmp / "cache"
        mm.UPDATABLE_CATEGORIES = {"profile": mm.PROFILES_FILE,
                                   "preference": mm.PREFERENCES_FILE,
                                   "latest_plan": mm.LATEST_PLAN_FILE,
                                   "recipes": mm.RECIPES_FILE}

    def tearDown(self):
        for name, value in self._orig.items():
            setattr(mm, name, value)
        self._tmpdir.cleanup()


def _decision(decisions, tool):
    return next(d for d in decisions if d["tool"] == tool)


class TestOrchestratorSelection(unittest.TestCase):
    def test_tool_is_not_selected_without_ingredients(self):
        for value in [None, "", "   "]:
            with self.subTest(value=value):
                decisions = orch.decide_tools(dict(BASE, available_ingredients=value))
                self.assertFalse(_decision(decisions, "ingredients_tool")["selected"])

    def test_tool_is_selected_when_ingredients_are_supplied(self):
        decisions = orch.decide_tools(dict(BASE, available_ingredients="rice, dal"))
        self.assertTrue(_decision(decisions, "ingredients_tool")["selected"])

    def test_every_decision_still_carries_a_reason(self):
        for value in [None, "rice, dal"]:
            for decision in orch.decide_tools(dict(BASE, available_ingredients=value)):
                self.assertTrue(decision["reason"])

    def test_existing_tool_decisions_are_unaffected(self):
        decisions = orch.decide_tools(dict(BASE, training_intensity="High", budget_per_day="Rs 300"))
        self.assertTrue(_decision(decisions, "nutrition_tool")["selected"])
        self.assertTrue(_decision(decisions, "budget_tool")["selected"])


class TestOrchestratorEndToEnd(MemoryIsolatedTestCase):
    def test_ingredients_tool_result_appears_in_the_trace(self):
        trace = orch.run_with_trace(
            dict(BASE, available_ingredients="rice, dal, paneer"),
            athlete_id="ingredients-test",
        )
        self.assertIn("ingredients_tool", trace["tool_results"])
        result = trace["tool_results"]["ingredients_tool"]
        self.assertIn("rice", result["usable"])
        self.assertIn("missing_groups", result)

    def test_no_ingredients_means_the_tool_never_runs(self):
        trace = orch.run_with_trace(dict(BASE), athlete_id="no-ingredients-test")
        self.assertNotIn("ingredients_tool", trace["tool_results"])

    def test_allergies_reach_the_tool_through_the_orchestrator(self):
        """The runner passes meal_planner.avoid_keywords(), so the tool
        inherits the dairy expansion rather than reimplementing it."""
        trace = orch.run_with_trace(
            dict(BASE, allergies="dairy", available_ingredients="rice, paneer, curd"),
            athlete_id="allergy-ingredients-test",
        )
        result = trace["tool_results"]["ingredients_tool"]
        excluded = {e["ingredient"] for e in result["excluded"]}
        self.assertEqual(excluded, {"paneer", "curd"})
        self.assertEqual(result["usable"], ["rice"])

    def test_ingredients_are_not_written_to_memory(self):
        orch.run_with_trace(
            dict(BASE, available_ingredients="rice, dal"), athlete_id="memory-test")
        stored = (mm.get("preference", "memory-test") or {})
        stored_profile = (mm.get("profile", "memory-test") or {})
        self.assertNotIn("available_ingredients", stored)
        self.assertNotIn("available_ingredients", stored_profile)


# ----------------------------------------------------------------------
# 7. Step 3 - the assessment reaches the LLM prompt
# ----------------------------------------------------------------------
# No live OpenRouter calls: httpx.post is mocked at the module boundary,
# the same technique tests/test_openrouter_provider.py already uses.

from unittest import mock  # noqa: E402  (kept beside the tests that need it)

FAKE_PLAN = {
    "confidence_score": 8.0,
    "confidence_reason": ["test"],
    "day_summary": {"total_calories": 2000, "total_protein_g": 100,
                    "total_carbs_g": 250, "total_fats_g": 60, "notes": "test"},
    "meals": [],
    "disclaimer": "test disclaimer",
}


def _fake_response(body):
    response = mock.Mock()
    response.status_code = 200
    response.json.return_value = body
    response.raise_for_status.return_value = None
    return response


def _completion(content):
    return {"choices": [{"message": {"content": content}}]}


def _patched_key(value="test-key-not-real"):
    import llm_provider
    return mock.patch.object(llm_provider, "OPENROUTER_API_KEY", value)


def _prompt_sent_for(profile, tool_results):
    """Run openrouter_provider against a mocked HTTP boundary and return the
    exact prompt text that would have been sent."""
    import json as _json
    from llm_provider import openrouter_provider
    with _patched_key(), mock.patch("httpx.post") as post:
        post.return_value = _fake_response(_completion(_json.dumps(FAKE_PLAN)))
        openrouter_provider(profile, {"rag_context": "", "tool_results": tool_results})
    return post.call_args.kwargs["json"]["messages"][0]["content"]


def _assessment(profile, avoid=None):
    return assess_ingredients(
        profile.get("available_ingredients"),
        diet=profile.get("diet", "Vegetarian"),
        avoid=avoid,
    )


class TestIngredientContextReachesTheLLM(unittest.TestCase):
    def test_usable_ingredients_reach_the_prompt(self):
        profile = dict(BASE, available_ingredients="rice, dal, tomatoes")
        prompt = _prompt_sent_for(profile, {"ingredients_tool": _assessment(profile)})
        self.assertIn("HOME INGREDIENTS", prompt)
        for food in ("rice", "dal", "tomatoes"):
            self.assertIn(food, prompt)

    def test_the_section_is_clearly_labelled(self):
        profile = dict(BASE, available_ingredients="rice")
        prompt = _prompt_sent_for(profile, {"ingredients_tool": _assessment(profile)})
        self.assertIn("HOME INGREDIENTS the athlete says they already have", prompt)
        self.assertIn("Usable (already checked against their diet and allergies)", prompt)

    def test_the_rule_tells_the_model_not_to_force_every_ingredient_in(self):
        profile = dict(BASE, available_ingredients="rice, dal")
        prompt = _prompt_sent_for(profile, {"ingredients_tool": _assessment(profile)})
        self.assertIn("AVAILABLE INGREDIENTS RULE", prompt)
        self.assertIn("Do NOT force every listed ingredient", prompt)

    def test_no_ingredients_means_no_ingredient_context_anywhere(self):
        prompt = _prompt_sent_for(dict(BASE), {})
        self.assertNotIn("HOME INGREDIENTS", prompt)
        self.assertNotIn("AVAILABLE INGREDIENTS RULE", prompt)

    def test_existing_provider_behaviour_is_intact(self):
        """A request with no ingredients must produce the same prompt it did
        before this step, and other tool results must still render."""
        nutrition = {"protein": 129.6, "calories": 3110, "carbs": 576, "hydration": 3.52}
        prompt = _prompt_sent_for(dict(BASE), {"nutrition_tool": nutrition})
        self.assertIn("Precomputed values", prompt)
        self.assertIn("129.6", prompt)
        self.assertIn("3110", prompt)
        self.assertNotIn("HOME INGREDIENTS", prompt)

    def test_the_provider_still_returns_a_parsed_plan(self):
        profile = dict(BASE, available_ingredients="rice, dal")
        import json as _json
        from llm_provider import openrouter_provider
        with _patched_key(), mock.patch("httpx.post") as post:
            post.return_value = _fake_response(_completion(_json.dumps(FAKE_PLAN)))
            plan = openrouter_provider(
                profile,
                {"rag_context": "", "tool_results": {"ingredients_tool": _assessment(profile)}},
            )
        self.assertEqual(plan, FAKE_PLAN)


class TestRestrictionsRemainAuthoritativeInThePrompt(unittest.TestCase):
    def test_allergy_excluded_ingredients_are_marked_must_not_be_used(self):
        profile = dict(BASE, allergies="dairy", available_ingredients="rice, paneer, curd")
        assessment = _assessment(profile, avoid=mp.avoid_keywords(profile))
        prompt = _prompt_sent_for(profile, {"ingredients_tool": assessment})
        self.assertIn("MUST NOT be used", prompt)
        self.assertIn("paneer", prompt)
        self.assertIn("curd", prompt)
        # ...and they must not be presented as usable.
        usable_line = next(l for l in prompt.splitlines() if "* Usable" in l)
        self.assertNotIn("paneer", usable_line)
        self.assertNotIn("curd", usable_line)

    def test_non_vegetarian_ingredients_are_marked_must_not_be_used(self):
        profile = dict(BASE, available_ingredients="rice, chicken")
        prompt = _prompt_sent_for(profile, {"ingredients_tool": _assessment(profile)})
        self.assertIn("MUST NOT be used", prompt)
        usable_line = next(l for l in prompt.splitlines() if "* Usable" in l)
        self.assertNotIn("chicken", usable_line)

    def test_the_rule_states_restrictions_outrank_availability(self):
        profile = dict(BASE, available_ingredients="rice")
        prompt = _prompt_sent_for(profile, {"ingredients_tool": _assessment(profile)})
        self.assertIn("Those rules outrank availability every time", prompt)
        # The diet and age rules are still present alongside it.
        self.assertIn("DIET RULE", prompt)
        self.assertIn("AGE / DEVELOPMENT RULE", prompt)

    def test_unrecognised_ingredients_are_not_presented_as_usable(self):
        profile = dict(BASE, available_ingredients="rice, glitter")
        prompt = _prompt_sent_for(profile, {"ingredients_tool": _assessment(profile)})
        usable_line = next(l for l in prompt.splitlines() if "* Usable" in l)
        self.assertIn("rice", usable_line)
        self.assertNotIn("glitter", usable_line)
        self.assertIn("Could not be verified", prompt)

    def test_missing_food_groups_are_surfaced_for_the_model(self):
        profile = dict(BASE, available_ingredients="rice")
        prompt = _prompt_sent_for(profile, {"ingredients_tool": _assessment(profile)})
        self.assertIn("Food groups NOT covered", prompt)
        for group in ("protein", "fat", "fruit", "vegetable"):
            self.assertIn(group, prompt)
        self.assertIn("name a small number of realistic, affordable additions", prompt)


# ----------------------------------------------------------------------
# 8. Step 4 - the field travels correctly through the real HTTP endpoint
# ----------------------------------------------------------------------
# The frontend is plain HTML/JS with no JS test harness in this project, so
# the browser behaviours (clear-on-switch, empty-on-reload) are verified by
# driving the real page. What IS testable here is the contract the page
# relies on: the endpoint accepts the field, routes it to the tool, and
# never persists it.

class TestMealPlanEndpointAcceptsIngredients(MemoryIsolatedTestCase):
    def _request(self, **overrides):
        from api import MealPlanRequest
        payload = {k: v for k, v in BASE.items()}
        payload.update(overrides)
        return MealPlanRequest(**payload)

    def test_the_field_is_optional_in_the_request_model(self):
        self.assertIsNone(self._request().available_ingredients)

    def test_the_field_is_accepted_and_carried(self):
        request = self._request(available_ingredients="rice, dal, tomatoes")
        self.assertEqual(request.available_ingredients, "rice, dal, tomatoes")

    def test_endpoint_runs_the_tool_when_ingredients_are_supplied(self):
        from api import create_meal_plan
        captured = {}
        real = orch.run_with_trace

        def spy(profile, athlete_id=None, **kwargs):
            trace = real(profile, athlete_id, **kwargs) if athlete_id else real(profile, **kwargs)
            captured["trace"] = trace
            return trace

        with mock.patch("api.run_with_trace", side_effect=spy):
            create_meal_plan(self._request(available_ingredients="rice, dal"))
        self.assertIn("ingredients_tool", captured["trace"]["tool_results"])

    def test_endpoint_skips_the_tool_when_none_supplied(self):
        from api import create_meal_plan
        captured = {}
        real = orch.run_with_trace

        def spy(profile, athlete_id=None, **kwargs):
            trace = real(profile, athlete_id, **kwargs) if athlete_id else real(profile, **kwargs)
            captured["trace"] = trace
            return trace

        with mock.patch("api.run_with_trace", side_effect=spy):
            create_meal_plan(self._request())
        self.assertNotIn("ingredients_tool", captured["trace"]["tool_results"])

    def test_response_schema_is_unchanged(self):
        from api import create_meal_plan
        response = create_meal_plan(self._request(available_ingredients="rice, dal"))
        self.assertEqual(
            set(response.keys()),
            {"confidence_score", "confidence_reason", "day_summary", "meals", "disclaimer"},
        )

    def test_endpoint_never_persists_ingredients_to_memory(self):
        from api import create_meal_plan
        create_meal_plan(self._request(
            athlete_id="44444444-4444-4444-8444-444444444444",
            available_ingredients="rice, dal, paneer",
        ))
        for category in ("profile", "preference"):
            stored = mm.get(category, "44444444-4444-4444-8444-444444444444") or {}
            self.assertNotIn("available_ingredients", stored)

    def test_a_second_request_without_ingredients_does_not_inherit_them(self):
        """Ingredients are ephemeral: having sent them once must not leave
        them applying to the next request for the same athlete."""
        from api import create_meal_plan
        athlete = "55555555-5555-4555-8555-555555555555"
        create_meal_plan(self._request(athlete_id=athlete, available_ingredients="paneer"))

        captured = {}
        real = orch.run_with_trace

        def spy(profile, athlete_id=None, **kwargs):
            trace = real(profile, athlete_id, **kwargs) if athlete_id else real(profile, **kwargs)
            captured["trace"] = trace
            return trace

        with mock.patch("api.run_with_trace", side_effect=spy):
            create_meal_plan(self._request(athlete_id=athlete))

        self.assertIsNone(captured["trace"]["merged_profile"].get("available_ingredients"))
        self.assertNotIn("ingredients_tool", captured["trace"]["tool_results"])


class TestIngredientsStayEphemeralOnTheLLMPath(MemoryIsolatedTestCase):
    def test_running_the_llm_path_never_persists_ingredients(self):
        profile = dict(BASE, available_ingredients="rice, dal")
        _prompt_sent_for(profile, {"ingredients_tool": _assessment(profile)})
        for category in ("profile", "preference"):
            stored = mm.get(category, "default") or {}
            self.assertNotIn("available_ingredients", stored)


if __name__ == "__main__":
    unittest.main(verbosity=2)
