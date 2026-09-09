"""
AthleteOS - Tests for the shared deterministic recipe-URL post-processing.

THE BUG BEING FIXED
    Every LLM provider is asked for a "resources" list containing a "url",
    and nothing in the prompt, MEAL_PLAN_SCHEMA, or api.Resource constrains
    what that string may be - so models fabricate plausible-looking video
    pages that are dead. Confirmed in production against BOTH providers
    (7/7 fabricated links in a Groq plan, 9/9 in an earlier OpenRouter
    plan, every one resolving to "Video unavailable"), so this is a
    prompt/schema-level defect, not a provider-specific one.

    mock_provider was never affected: it builds resources from
    recipe_links.get_recipe_resource(), whose URLs are YouTube SEARCH
    queries rather than specific pages. The fix gives every provider that
    same guarantee by rebuilding the URL after generation.

    Secondary, same root cause: the frontend looks resources up by EXACT
    ingredient string, so a model writing "Chole" for an ingredient it
    listed as "Chole (chickpea curry)" silently lost the link. The same
    post-processing step repairs that attribution.

No network and no LLM calls anywhere in this file - the post-processing
function is pure, and the provider-level tests use mock_provider or a
patched provider.

Covers:
    1. A fabricated URL is replaced with the deterministic search URL.
    2. Parenthetical ingredient names resolve and produce a real URL.
    3. mock_provider output is unchanged (idempotent no-op).
    4. Both the 1-day and multi-day shapes are covered.
    5. The other resource fields (type/title) and the rest of the plan are
       untouched, and no resource is added or removed.
    6. It runs for every provider, via generate_meal_plan().
"""

import copy
import json
import unittest
from unittest import mock

import meal_planner
from meal_planner import (
    apply_deterministic_recipe_urls,
    generate_meal_plan,
    mock_provider,
    mock_provider_multi_day,
)
from recipe_links import _search_url

BASE_PROFILE = dict(
    age=22, height_cm=178, weight_kg=72, sex="Male",
    goal="Improve football performance",
    diet="Vegetarian", training_days_per_week=5, training_intensity="High",
    region="Lucknow, Uttar Pradesh", allergies="none",
    budget_per_day="Rs 300", food_preferences=None, foods_to_avoid=None, notes=None,
)

# A fabricated URL of exactly the shape observed in production.
FAKE_URL = "https://youtu.be/abcd1234"


def _plan_with_resources(ingredient_items, resource_items):
    """A minimal 1-day plan carrying fabricated resource URLs."""
    return {
        "confidence_score": 8.0,
        "confidence_reason": ["test"],
        "day_summary": {
            "total_calories": 2000, "total_protein_g": 100,
            "total_carbs_g": 250, "total_fats_g": 60, "notes": "test",
        },
        "meals": [{
            "meal": "Lunch",
            "ingredients": [{"item": i, "quantity": "1 bowl"} for i in ingredient_items],
            "calories": 700, "protein_g": 30, "carbs_g": 90, "fats_g": 20,
            "reasoning": "test",
            "resources": [
                {"type": "youtube", "title": f"How to make {i}", "url": FAKE_URL, "item": i}
                for i in resource_items
            ],
        }],
        "disclaimer": "test disclaimer",
    }


def _all_resources(plan):
    days = plan.get("days") or [plan]
    return [r for day in days for meal in day["meals"] for r in meal.get("resources", [])]


# ----------------------------------------------------------------------
# 1. Fabricated URLs are replaced
# ----------------------------------------------------------------------

class TestFabricatedUrlIsReplaced(unittest.TestCase):
    def test_fake_youtube_url_becomes_the_deterministic_search_url(self):
        plan = _plan_with_resources(["Chole"], ["Chole"])
        apply_deterministic_recipe_urls(plan)
        url = plan["meals"][0]["resources"][0]["url"]
        self.assertEqual(url, "https://www.youtube.com/results?search_query=chole+recipe")
        self.assertNotEqual(url, FAKE_URL)

    def test_no_fabricated_specific_video_url_can_survive(self):
        """Every shape of invented link the models actually produced."""
        for fake in (
            "https://youtu.be/abcd1234",
            "https://www.youtube.com/watch?v=6G0iZ7b6Z6I",
            "https://example.com/not-a-recipe",
            "",
        ):
            with self.subTest(fake=fake):
                plan = _plan_with_resources(["Rajma"], ["Rajma"])
                plan["meals"][0]["resources"][0]["url"] = fake
                apply_deterministic_recipe_urls(plan)
                self.assertEqual(
                    plan["meals"][0]["resources"][0]["url"], _search_url("Rajma")
                )

    def test_every_url_is_a_youtube_search_never_a_specific_page(self):
        plan = _plan_with_resources(
            ["Chole", "Rajma", "Paneer sabzi"], ["Chole", "Rajma", "Paneer sabzi"]
        )
        apply_deterministic_recipe_urls(plan)
        for resource in _all_resources(plan):
            self.assertTrue(
                resource["url"].startswith("https://www.youtube.com/results?search_query="),
                resource["url"],
            )
            self.assertNotIn("/watch?v=", resource["url"])
            self.assertNotIn("youtu.be/", resource["url"])


# ----------------------------------------------------------------------
# 2. Parenthetical ingredient names
# ----------------------------------------------------------------------

class TestParentheticalIngredientNames(unittest.TestCase):
    def test_resource_resolves_to_the_parenthetical_ingredient(self):
        """The exact production case: model listed the ingredient as
        "Chole (chickpea curry)" but named the resource "Chole", so the
        frontend's exact-string lookup dropped the link."""
        plan = _plan_with_resources(["Chole (chickpea curry)"], ["Chole"])
        apply_deterministic_recipe_urls(plan)
        resource = plan["meals"][0]["resources"][0]
        self.assertEqual(resource["item"], "Chole (chickpea curry)")
        self.assertEqual(resource["url"], _search_url("Chole (chickpea curry)"))

    def test_mid_string_parenthetical_also_resolves(self):
        """The second production case: "Palak (spinach) sabzi" vs
        "Palak sabzi" - the aside is in the middle, not trailing."""
        plan = _plan_with_resources(["Palak (spinach) sabzi"], ["Palak sabzi"])
        apply_deterministic_recipe_urls(plan)
        resource = plan["meals"][0]["resources"][0]
        self.assertEqual(resource["item"], "Palak (spinach) sabzi")

    def test_resolved_item_matches_an_ingredient_exactly(self):
        """What the frontend actually needs: resource["item"] must be an
        exact member of that meal's ingredient item strings."""
        plan = _plan_with_resources(
            ["Chole (chickpea curry)", "Palak (spinach) sabzi", "Moong dal"],
            ["Chole", "Palak sabzi", "Moong dal"],
        )
        apply_deterministic_recipe_urls(plan)
        meal = plan["meals"][0]
        items = {i["item"] for i in meal["ingredients"]}
        for resource in meal["resources"]:
            self.assertIn(resource["item"], items)

    def test_a_resource_naming_no_ingredient_is_left_attributed_as_is(self):
        """Nothing is invented to force a match - but the URL is still
        made real rather than left fabricated."""
        plan = _plan_with_resources(["Moong dal"], ["Something Not In This Meal"])
        apply_deterministic_recipe_urls(plan)
        resource = plan["meals"][0]["resources"][0]
        self.assertEqual(resource["item"], "Something Not In This Meal")
        self.assertEqual(resource["url"], _search_url("Something Not In This Meal"))

    def test_case_differences_resolve_too(self):
        plan = _plan_with_resources(["Moong Dal"], ["moong dal"])
        apply_deterministic_recipe_urls(plan)
        self.assertEqual(plan["meals"][0]["resources"][0]["item"], "Moong Dal")


# ----------------------------------------------------------------------
# 3. mock_provider is unchanged / idempotent
# ----------------------------------------------------------------------

class TestMockProviderUnchanged(unittest.TestCase):
    def test_post_processing_is_a_no_op_for_mock_output(self):
        with mock.patch.object(meal_planner, "PROVIDER", "mock"):
            plan = generate_meal_plan(dict(BASE_PROFILE))
        before = copy.deepcopy(plan)
        apply_deterministic_recipe_urls(plan)
        self.assertEqual(plan, before)

    def test_raw_mock_provider_output_is_already_correct(self):
        """mock_provider is called directly here, bypassing
        generate_meal_plan's post-processing, to prove the mock path was
        never broken and needs no repair."""
        plan = mock_provider(dict(BASE_PROFILE))
        before = copy.deepcopy(plan)
        apply_deterministic_recipe_urls(plan)
        self.assertEqual(plan, before)

    def test_applying_twice_changes_nothing(self):
        plan = _plan_with_resources(["Chole (chickpea curry)"], ["Chole"])
        once = copy.deepcopy(apply_deterministic_recipe_urls(plan))
        twice = apply_deterministic_recipe_urls(copy.deepcopy(once))
        self.assertEqual(once, twice)

    def test_mock_resources_still_point_at_real_search_urls(self):
        with mock.patch.object(meal_planner, "PROVIDER", "mock"):
            plan = generate_meal_plan(dict(BASE_PROFILE))
        resources = _all_resources(plan)
        self.assertGreater(len(resources), 0, "expected the mock plan to carry resources")
        for resource in resources:
            self.assertTrue(
                resource["url"].startswith("https://www.youtube.com/results?search_query=")
            )


# ----------------------------------------------------------------------
# 4. 1-day and multi-day shapes
# ----------------------------------------------------------------------

class TestBothPlanShapes(unittest.TestCase):
    def test_single_day_shape_is_covered(self):
        plan = _plan_with_resources(["Chole"], ["Chole"])
        apply_deterministic_recipe_urls(plan)
        self.assertEqual(plan["meals"][0]["resources"][0]["url"], _search_url("Chole"))

    def test_multi_day_shape_is_covered_on_every_day(self):
        day = _plan_with_resources(["Chole (chickpea curry)"], ["Chole"])["meals"]
        plan = {
            "duration_days": 3,
            "confidence_score": 8.0,
            "confidence_reason": ["test"],
            "days": [
                {"day": n, "day_summary": {"total_calories": 2000}, "meals": copy.deepcopy(day)}
                for n in (1, 2, 3)
            ],
            "disclaimer": "test disclaimer",
        }
        apply_deterministic_recipe_urls(plan)
        resources = _all_resources(plan)
        self.assertEqual(len(resources), 3)
        for resource in resources:
            self.assertEqual(resource["url"], _search_url("Chole (chickpea curry)"))
            self.assertEqual(resource["item"], "Chole (chickpea curry)")

    def test_mock_multi_day_output_is_unchanged(self):
        plan = mock_provider_multi_day(dict(BASE_PROFILE), 7)
        before = copy.deepcopy(plan)
        apply_deterministic_recipe_urls(plan)
        self.assertEqual(plan, before)

    def test_a_meal_with_no_resources_is_handled(self):
        plan = _plan_with_resources(["Rice"], [])
        apply_deterministic_recipe_urls(plan)
        self.assertEqual(plan["meals"][0]["resources"], [])


# ----------------------------------------------------------------------
# 5. Nothing else is touched
# ----------------------------------------------------------------------

class TestNothingElseChanges(unittest.TestCase):
    def test_type_and_title_are_preserved(self):
        plan = _plan_with_resources(["Chole"], ["Chole"])
        plan["meals"][0]["resources"][0]["title"] = "Model's own title"
        apply_deterministic_recipe_urls(plan)
        resource = plan["meals"][0]["resources"][0]
        self.assertEqual(resource["type"], "youtube")
        self.assertEqual(resource["title"], "Model's own title")

    def test_resource_field_set_is_unchanged(self):
        plan = _plan_with_resources(["Chole"], ["Chole"])
        apply_deterministic_recipe_urls(plan)
        self.assertEqual(
            set(plan["meals"][0]["resources"][0]), {"type", "title", "url", "item"}
        )

    def test_no_resource_is_added_or_removed(self):
        plan = _plan_with_resources(["Chole", "Rice", "Moong dal"], ["Chole", "Moong dal"])
        apply_deterministic_recipe_urls(plan)
        self.assertEqual(len(plan["meals"][0]["resources"]), 2)

    def test_ingredients_macros_and_summary_are_untouched(self):
        plan = _plan_with_resources(["Chole (chickpea curry)"], ["Chole"])
        before = copy.deepcopy(plan)
        apply_deterministic_recipe_urls(plan)
        meal, meal_before = plan["meals"][0], before["meals"][0]
        self.assertEqual(meal["ingredients"], meal_before["ingredients"])
        for key in ("calories", "protein_g", "carbs_g", "fats_g", "reasoning", "meal"):
            self.assertEqual(meal[key], meal_before[key])
        self.assertEqual(plan["day_summary"], before["day_summary"])
        self.assertEqual(plan["confidence_score"], before["confidence_score"])
        self.assertEqual(plan["disclaimer"], before["disclaimer"])

    def test_the_plan_object_is_returned_for_chaining(self):
        plan = _plan_with_resources(["Chole"], ["Chole"])
        self.assertIs(apply_deterministic_recipe_urls(plan), plan)


# ----------------------------------------------------------------------
# 6. Applied for every provider, through generate_meal_plan()
# ----------------------------------------------------------------------

class TestAppliedForEveryProvider(unittest.TestCase):
    def _fake_llm_plan(self):
        return _plan_with_resources(["Chole (chickpea curry)"], ["Chole"])

    def test_claude_provider_output_is_post_processed(self):
        with mock.patch.object(meal_planner, "PROVIDER", "claude"), \
             mock.patch("llm_provider.claude_provider", return_value=self._fake_llm_plan()):
            plan = generate_meal_plan(dict(BASE_PROFILE))
        self.assertEqual(plan["meals"][0]["resources"][0]["url"],
                         _search_url("Chole (chickpea curry)"))

    def test_openrouter_provider_output_is_post_processed(self):
        with mock.patch.object(meal_planner, "PROVIDER", "openrouter"), \
             mock.patch("llm_provider.openrouter_provider", return_value=self._fake_llm_plan()):
            plan = generate_meal_plan(dict(BASE_PROFILE))
        self.assertEqual(plan["meals"][0]["resources"][0]["url"],
                         _search_url("Chole (chickpea curry)"))

    def test_groq_provider_output_is_post_processed(self):
        with mock.patch.object(meal_planner, "PROVIDER", "groq"), \
             mock.patch("llm_provider.groq_provider", return_value=self._fake_llm_plan()):
            plan = generate_meal_plan(dict(BASE_PROFILE))
        self.assertEqual(plan["meals"][0]["resources"][0]["url"],
                         _search_url("Chole (chickpea curry)"))

    def test_openrouter_multi_day_output_is_post_processed(self):
        multi = {
            "duration_days": 7, "confidence_score": 8.0, "confidence_reason": ["t"],
            "days": [{"day": 1, "day_summary": {}, "meals": self._fake_llm_plan()["meals"]}],
            "disclaimer": "test disclaimer",
        }
        with mock.patch.object(meal_planner, "PROVIDER", "openrouter"), \
             mock.patch("llm_provider.openrouter_provider_multi_day", return_value=multi):
            plan = generate_meal_plan(dict(BASE_PROFILE), days=7)
        self.assertEqual(plan["days"][0]["meals"][0]["resources"][0]["url"],
                         _search_url("Chole (chickpea curry)"))

    def test_an_unknown_provider_still_raises_rather_than_post_processing(self):
        with mock.patch.object(meal_planner, "PROVIDER", "nonsense"):
            with self.assertRaises(ValueError):
                generate_meal_plan(dict(BASE_PROFILE))


# ----------------------------------------------------------------------
# Regression guard: the real production response that exposed this
# ----------------------------------------------------------------------

class TestRealProductionResponse(unittest.TestCase):
    """Rebuilt from the actual Groq production plan whose 7 links were all
    dead, to prove the fix repairs that exact payload."""

    PRODUCTION_ITEMS = [
        ("Oats", "Oats"),
        ("Soy chunks", "Soy chunks"),
        ("Chole (chickpea curry)", "Chole"),
        ("Whole-wheat roti", "Whole-wheat roti"),
        ("Roasted chickpeas", "Roasted chickpeas"),
        ("Moong dal", "Moong dal"),
        ("Palak (spinach) sabzi", "Palak sabzi"),
    ]

    def test_all_seven_links_become_real_and_correctly_attributed(self):
        ingredients = [i for i, _ in self.PRODUCTION_ITEMS]
        resources = [r for _, r in self.PRODUCTION_ITEMS]
        plan = _plan_with_resources(ingredients, resources)
        apply_deterministic_recipe_urls(plan)

        meal = plan["meals"][0]
        items = {i["item"] for i in meal["ingredients"]}
        self.assertEqual(len(meal["resources"]), 7)
        for resource in meal["resources"]:
            self.assertIn(resource["item"], items)          # frontend will render it
            self.assertNotEqual(resource["url"], FAKE_URL)  # no longer fabricated
            self.assertTrue(
                resource["url"].startswith("https://www.youtube.com/results?search_query=")
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
