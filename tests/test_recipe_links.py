"""
AthleteOS - Phase 4C tests: recipe links.

Verifies the recipe-link registry (recipe_links.py) and its wiring into
meal_planner.py: complex dishes get exactly one attributed recipe resource,
simple dishes get none, multiple complex dishes in one meal each get their
own correctly-attributed resource, no dish produces a duplicate resource,
and 1-day / 7-day generation both still succeed with the new resource shape.

Everything here is offline and deterministic - no network, no API calls.
"""

import unittest

import meal_planner as mp
from recipe_links import COMPLEX_DISHES, get_recipe_resource

BASE = dict(
    age=22, height_cm=178, weight_kg=72, sex="Male",
    goal="Improve football performance",
    diet="Vegetarian", training_days_per_week=5, training_intensity="High",
    region="Chennai, Tamil Nadu", allergies="none",
    budget_per_day=None, food_preferences=None, foods_to_avoid=None, notes=None,
)


def all_resources(plan):
    return [r for meal in plan["meals"] for r in meal["resources"]]


# ----------------------------------------------------------------------
# get_recipe_resource() - unit level
# ----------------------------------------------------------------------

class TestGetRecipeResource(unittest.TestCase):
    def test_simple_food_returns_none(self):
        self.assertIsNone(get_recipe_resource("Rice"))
        self.assertIsNone(get_recipe_resource("Curd"))
        self.assertIsNone(get_recipe_resource("Poha"))

    def test_complex_food_returns_a_resource(self):
        resource = get_recipe_resource("Rajma")
        self.assertIsNotNone(resource)
        self.assertEqual(resource["item"], "Rajma")
        self.assertEqual(resource["type"], "youtube")
        self.assertIn("Rajma", resource["title"])
        self.assertTrue(resource["url"].startswith("https://www.youtube.com/results?"))

    def test_approved_exceptions_are_simple(self):
        # User-approved overrides: everyday quick prep, not worth a link.
        for dish in ("Poha", "Poha with peanuts", "Poha with sev",
                     "Vegetable upma", "Semiya upma", "Ragi mudde"):
            self.assertNotIn(dish, COMPLEX_DISHES, msg=dish)


# ----------------------------------------------------------------------
# meal_planner wiring - _resources_for_ingredients / _meal
# ----------------------------------------------------------------------

class TestMealResourceAttachment(unittest.TestCase):
    def test_meal_with_only_simple_ingredients_has_no_resources(self):
        ingredients = [{"item": "Rice", "quantity": "1 bowl"},
                       {"item": "Curd", "quantity": "1 bowl"}]
        resources = mp._resources_for_ingredients(ingredients)
        self.assertEqual(resources, [])

    def test_meal_with_one_complex_ingredient_gets_one_resource(self):
        ingredients = [{"item": "Rice", "quantity": "1 bowl"},
                       {"item": "Rajma", "quantity": "1 bowl"}]
        resources = mp._resources_for_ingredients(ingredients)
        self.assertEqual(len(resources), 1)
        self.assertEqual(resources[0]["item"], "Rajma")

    def test_meal_with_multiple_complex_dishes_gets_multiple_resources(self):
        ingredients = [
            {"item": "Rice", "quantity": "1 bowl"},
            {"item": "Rajma", "quantity": "1 bowl"},
            {"item": "Paneer sabzi", "quantity": "1 bowl"},
            {"item": "Curd", "quantity": "1 bowl"},
        ]
        resources = mp._resources_for_ingredients(ingredients)
        items = {r["item"] for r in resources}
        self.assertEqual(items, {"Rajma", "Paneer sabzi"})

    def test_no_cross_dish_contamination(self):
        ingredients = [{"item": "Rajma", "quantity": "1 bowl"},
                       {"item": "Chole", "quantity": "1 bowl"}]
        resources = mp._resources_for_ingredients(ingredients)
        by_item = {r["item"]: r for r in resources}
        self.assertIn("Rajma", by_item["Rajma"]["title"])
        self.assertNotIn("Chole", by_item["Rajma"]["title"])
        self.assertIn("Chole", by_item["Chole"]["title"])
        self.assertNotIn("Rajma", by_item["Chole"]["title"])

    def test_repeated_complex_dish_produces_only_one_resource(self):
        ingredients = [{"item": "Rajma", "quantity": "1 bowl"},
                       {"item": "Rajma", "quantity": "1 bowl"}]
        resources = mp._resources_for_ingredients(ingredients)
        self.assertEqual(len(resources), 1)


# ----------------------------------------------------------------------
# End-to-end: real generated plans
# ----------------------------------------------------------------------

class TestEndToEndPlanGeneration(unittest.TestCase):
    def test_one_day_plan_still_generates_with_new_resource_shape(self):
        plan = mp.mock_provider(dict(BASE))
        self.assertEqual(len(plan["meals"]), 4)
        for meal in plan["meals"]:
            self.assertIn("resources", meal)
            for resource in meal["resources"]:
                self.assertIn("item", resource)
                self.assertIn("url", resource)
                self.assertIn("title", resource)
                self.assertIn("type", resource)

    def test_seven_day_plan_still_generates_with_new_resource_shape(self):
        plan = mp.mock_provider_multi_day(dict(BASE), days=7)
        self.assertEqual(len(plan["days"]), 7)
        for day in plan["days"]:
            for meal in day["meals"]:
                self.assertIn("resources", meal)
                for resource in meal["resources"]:
                    self.assertIn("item", resource)

    def test_no_duplicate_resource_for_the_same_dish_within_a_meal(self):
        plan = mp.mock_provider(dict(BASE))
        for meal in plan["meals"]:
            items = [r["item"] for r in meal["resources"]]
            self.assertEqual(len(items), len(set(items)))

    def test_resource_item_names_match_an_actual_meal_ingredient(self):
        plan = mp.mock_provider(dict(BASE))
        for meal in plan["meals"]:
            ingredient_names = {i["item"] for i in meal["ingredients"]}
            for resource in meal["resources"]:
                self.assertIn(resource["item"], ingredient_names)

    def test_south_region_lunch_can_produce_two_complex_dish_links(self):
        # South Tamil Nadu lunch draws its protein from a pool containing
        # "Sambar (toor dal)" and its veg from a pool containing "Avial" -
        # both classified complex, so on at least one rotation a lunch
        # should carry links for both a protein dish and a veg dish.
        found_two_or_more = False
        profile = dict(BASE)
        for day_index in range(10):
            profile["notes"] = f"variety-probe-{day_index}"
            plan = mp.mock_provider(dict(profile))
            lunch = next(m for m in plan["meals"] if m["meal"] == "Lunch")
            if len(lunch["resources"]) >= 2:
                found_two_or_more = True
                break
        self.assertTrue(found_two_or_more,
                         "expected at least one rotation to surface 2+ complex "
                         "dishes (e.g. Sambar + Avial) in the same lunch")


# ----------------------------------------------------------------------
# Sabzi/prepared-vegetable-dish coverage expansion (Step 4C follow-up)
# ----------------------------------------------------------------------
# The audit found the registry too narrow - plain "X sabzi"/"X fry"/
# "X poriyal" dishes were excluded even though they have real technique.
# Approved additions (~28 dishes) plus the 3 borderline calls: "Palak" and
# "Pumpkin with herbs" stay Simple, "Stir-fried greens" becomes Complex.

NEWLY_ADDED_COMPLEX_DISHES = [
    "Beans poriyal", "Cabbage thoran", "Bhindi fry", "Carrot poriyal",
    "Keerai masiyal",
    "Aloo-gobi", "Palak sabzi", "Mixed veg", "Bhindi masala",
    "Baingan bharta", "Lauki sabzi", "Tinda sabzi",
    "Mixed veg shaak", "Bhindi sabzi", "Cabbage sabzi", "Gawar sabzi",
    "Dudhi (lauki) sabzi", "Tendli sabzi",
    "Aloo-potol sabzi", "Mixed vegetable ghonto", "Begun bhaja",
    "Lau (bottle gourd) ghonto", "Bandhakopir torkari",
    "Aloo-baingan sabzi", "Chech bhaji (greens)",
    "Lai xaak (mustard greens)", "Squash curry", "Stir-fried greens",
]


class TestSabziCoverageExpansion(unittest.TestCase):
    def test_all_newly_approved_sabzi_dishes_are_complex(self):
        for dish in NEWLY_ADDED_COMPLEX_DISHES:
            resource = get_recipe_resource(dish)
            self.assertIsNotNone(resource, msg=f"{dish} should now have a recipe resource")
            self.assertEqual(resource["item"], dish)

    def test_at_least_28_sabzi_dishes_were_added(self):
        self.assertGreaterEqual(len(NEWLY_ADDED_COMPLEX_DISHES), 28)
        self.assertTrue(set(NEWLY_ADDED_COMPLEX_DISHES).issubset(COMPLEX_DISHES))

    def test_borderline_items_excluded_from_this_pass_stay_simple(self):
        # "Palak" (East) - bare ingredient name, read as plain spinach, not
        # a named preparation like North's "Palak sabzi".
        self.assertIsNone(get_recipe_resource("Palak"))
        # "Pumpkin with herbs" (Northeast) - descriptive, not a named
        # technique dish (contrast "Squash curry", approved as Complex).
        self.assertIsNone(get_recipe_resource("Pumpkin with herbs"))

    def test_stir_fried_greens_is_complex_per_the_approved_decision(self):
        resource = get_recipe_resource("Stir-fried greens")
        self.assertIsNotNone(resource)
        self.assertEqual(resource["item"], "Stir-fried greens")

    def test_still_basic_boiled_items_stay_simple(self):
        self.assertIsNone(get_recipe_resource("Boiled seasonal vegetables"))
        self.assertIsNone(get_recipe_resource("Boiled beans with herbs"))

    def test_every_newly_added_dish_actually_exists_in_regional_foods(self):
        """Guards against a typo'd registry key that would silently never
        match anything real - every added string must be an exact dish
        name that appears in at least one region's veg pool."""
        from regional_foods import REGIONS
        all_veg_dishes = set()
        for data in REGIONS.values():
            all_veg_dishes.update(data["veg"])
        for dish in NEWLY_ADDED_COMPLEX_DISHES:
            self.assertIn(dish, all_veg_dishes, msg=f"{dish} not found in any region's veg pool")


if __name__ == "__main__":
    unittest.main()
