"""
AthleteOS - Stage 2 tests: per-food nutrition data (first pass).

Covers exactly what Stage 2 shipped: ONE sourced entry (Banana, IFCT 2017,
per-100g, not rescaled to the app's "1" quantity), everything else
deliberately left uncovered per the Stage 1 audit's strict portion-matching
rules. Verifies:
    1. The sourced entry (Banana) returns the exact documented values.
    2. Everything else - including items that were investigated in Stage 1
       but explicitly NOT implemented (Boiled eggs, Rice, Curd, and every
       composite dish) - returns None, not a guess.
    3. meal_planner._attach_nutrition() only adds "nutrition" where sourced,
       and never touches meal-level macros.
    4. Existing meal/day nutrition totals are completely unchanged by this
       feature (byte-for-byte equal macros with the registry patched empty).
    5. 1-day and 7-day generation both still work with the new optional field.
    6. The API's Ingredient/NutritionInfo schema accepts the new shape.

Everything here is offline and deterministic - no network, no API calls.
"""

import unittest
from unittest import mock

import meal_planner as mp
from nutrition_data import NUTRITION, get_nutrition_info

BASE = dict(
    age=22, height_cm=178, weight_kg=72, sex="Male",
    goal="Improve football performance",
    diet="Vegetarian", training_days_per_week=5, training_intensity="High",
    region="Lucknow, Uttar Pradesh", allergies="none",
    budget_per_day=None, food_preferences=None, foods_to_avoid=None, notes=None,
)


# ----------------------------------------------------------------------
# 1. The one sourced entry
# ----------------------------------------------------------------------

class TestBananaIsSourcedCorrectly(unittest.TestCase):
    def test_banana_returns_the_documented_ifct_values(self):
        info = get_nutrition_info("Banana")
        self.assertIsNotNone(info)
        self.assertEqual(info["kcal"], 105)
        self.assertEqual(info["protein_g"], 1.23)
        self.assertEqual(info["carbs_g"], 23.63)
        self.assertEqual(info["fat_g"], 0.33)
        self.assertEqual(info["source"], "IFCT 2017 (ICMR-National Institute of Nutrition)")
        self.assertIn("E012", info["source_ref"])

    def test_banana_basis_explicitly_says_per_100g_not_rescaled(self):
        """The whole point of Stage 1's strict rule: never imply the "1"
        quantity was converted to a real serving - the basis text must say
        so plainly."""
        info = get_nutrition_info("Banana")
        self.assertIn("100g", info["basis"])
        self.assertIn("not", info["basis"].lower())

    def test_registry_contains_exactly_one_entry(self):
        """Guards against silently expanding coverage without a matching
        Stage 1 sourcing pass - this pass approved exactly one food."""
        self.assertEqual(list(NUTRITION.keys()), ["Banana"])


# ----------------------------------------------------------------------
# 2. Everything else stays uncovered - including items investigated but
#    explicitly rejected in Stage 1 (no fabricated fallback values)
# ----------------------------------------------------------------------

class TestEverythingElseStaysUncovered(unittest.TestCase):
    def test_boiled_eggs_uncovered(self):
        """Rejected in Stage 1: no authoritative single-egg gram weight was
        established without stacking unsourced assumptions."""
        self.assertIsNone(get_nutrition_info("Boiled eggs"))

    def test_rice_uncovered(self):
        """Explicitly deferred: the app's "N servings" label has no verified
        serving-to-gram mapping, even though a USDA value was found for
        cooked rice in the abstract."""
        self.assertIsNone(get_nutrition_info("Rice"))

    def test_curd_uncovered(self):
        """No IFCT or USDA entry was ever found for curd at all."""
        self.assertIsNone(get_nutrition_info("Curd"))

    def test_composite_dishes_uncovered(self):
        for dish in ("Egg curry", "Chicken curry", "Chana masala", "Kadhi",
                     "Vegetable upma", "Soya chunk curry", "Sambar (toor dal)",
                     "Dal", "Moong dal", "Rajma"):
            self.assertIsNone(get_nutrition_info(dish), msg=f"{dish} should be uncovered")

    def test_unknown_item_returns_none(self):
        self.assertIsNone(get_nutrition_info("Not a real food at all"))


# ----------------------------------------------------------------------
# 3. meal_planner wiring - additive only
# ----------------------------------------------------------------------

class TestNutritionAttachment(unittest.TestCase):
    def test_attach_nutrition_adds_field_only_where_sourced(self):
        ingredients = [
            {"item": "Banana", "quantity": "1"},
            {"item": "Curd", "quantity": "1 small bowl"},
        ]
        result = mp._attach_nutrition(list(ingredients))
        by_item = {i["item"]: i for i in result}
        self.assertIn("nutrition", by_item["Banana"])
        self.assertNotIn("nutrition", by_item["Curd"])

    def test_attach_nutrition_does_not_alter_item_or_quantity(self):
        ingredients = [{"item": "Banana", "quantity": "1"}]
        result = mp._attach_nutrition(list(ingredients))
        self.assertEqual(result[0]["item"], "Banana")
        self.assertEqual(result[0]["quantity"], "1")


# ----------------------------------------------------------------------
# 4. Existing meal/day totals are completely unchanged by this feature
# ----------------------------------------------------------------------

class TestExistingTotalsUnchanged(unittest.TestCase):
    def test_meal_and_day_macros_identical_with_nutrition_registry_emptied(self):
        """The strongest possible proof this feature is additive: patch the
        registry to return nothing for everything, regenerate the SAME
        deterministic plan, and confirm every meal's calories/protein_g/
        carbs_g/fats_g and the day_summary are byte-for-byte identical."""
        with_nutrition = mp.mock_provider(dict(BASE))
        with mock.patch("meal_planner.get_nutrition_info", return_value=None):
            without_nutrition = mp.mock_provider(dict(BASE))

        self.assertEqual(with_nutrition["day_summary"], without_nutrition["day_summary"])
        for m1, m2 in zip(with_nutrition["meals"], without_nutrition["meals"]):
            for key in ("calories", "protein_g", "carbs_g", "fats_g"):
                self.assertEqual(m1[key], m2[key], msg=f"{m1['meal']}.{key} changed")

    def test_ingredients_are_identical_apart_from_the_nutrition_key(self):
        with_nutrition = mp.mock_provider(dict(BASE))
        with mock.patch("meal_planner.get_nutrition_info", return_value=None):
            without_nutrition = mp.mock_provider(dict(BASE))

        for m1, m2 in zip(with_nutrition["meals"], without_nutrition["meals"]):
            for i1, i2 in zip(m1["ingredients"], m2["ingredients"]):
                self.assertEqual(i1["item"], i2["item"])
                self.assertEqual(i1["quantity"], i2["quantity"])
                stripped = {k: v for k, v in i1.items() if k != "nutrition"}
                self.assertEqual(stripped, i2)


# ----------------------------------------------------------------------
# 5. 1-day and 7-day generation still work
# ----------------------------------------------------------------------

class TestGenerationStillWorks(unittest.TestCase):
    def test_one_day_plan_generates_and_resources_still_work(self):
        plan = mp.mock_provider(dict(BASE))
        self.assertEqual(len(plan["meals"]), 4)
        # Recipe links (Step 4C) must keep working alongside nutrition data.
        for meal in plan["meals"]:
            self.assertIn("resources", meal)

    def test_seven_day_plan_generates(self):
        plan = mp.mock_provider_multi_day(dict(BASE), days=7)
        self.assertEqual(len(plan["days"]), 7)

    def test_banana_shows_nutrition_when_it_appears_in_a_real_plan(self):
        """End-to-end: whenever a real generated plan includes "Banana" as
        an ingredient, it must carry the sourced nutrition field."""
        found_banana_with_nutrition = False
        profile = dict(BASE)
        for i in range(10):
            profile["notes"] = f"probe-{i}"
            plan = mp.mock_provider(dict(profile))
            for meal in plan["meals"]:
                for ing in meal["ingredients"]:
                    if ing["item"] == "Banana":
                        self.assertIn("nutrition", ing)
                        found_banana_with_nutrition = True
        self.assertTrue(found_banana_with_nutrition, "expected Banana to appear at least once across 10 rotations")


# ----------------------------------------------------------------------
# 6. API schema compatibility
# ----------------------------------------------------------------------

class TestApiSchemaCompatibility(unittest.TestCase):
    def test_ingredient_model_accepts_nutrition_field(self):
        from api import Ingredient, NutritionInfo
        info = get_nutrition_info("Banana")
        ing = Ingredient(item="Banana", quantity="1", nutrition=NutritionInfo(**info))
        self.assertEqual(ing.nutrition.kcal, 105)

    def test_ingredient_model_still_works_without_nutrition(self):
        from api import Ingredient
        ing = Ingredient(item="Curd", quantity="1 small bowl")
        self.assertIsNone(ing.nutrition)

    def test_real_endpoint_response_validates_against_the_schema(self):
        """Generate a real plan through the actual /meal-plan-shaped response
        model to prove the optional field doesn't break existing validation."""
        from api import MealPlanResponse
        plan = mp.mock_provider(dict(BASE))
        validated = MealPlanResponse(**plan)
        self.assertEqual(len(validated.meals), 4)


if __name__ == "__main__":
    unittest.main()
