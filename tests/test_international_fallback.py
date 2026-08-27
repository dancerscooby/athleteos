"""
AthleteOS - Phase 2B tests: international/general fallback region.

Before this phase, an athlete outside India (or with blank/unrecognised
region text) silently got North Indian food - DEFAULT_REGION was "north".
That's not a location->cuisine mapping, it's a wrong guess. Phase 2B adds a
genuine 7th pool, "international" (globally common, not derived from any one
national cuisine - see regional_foods.py's own note on this), and makes it
the actual default.

Nothing about WHICH nutrition targets an athlete gets changes here - see
TestLocationOnlyAffectsCuisine below, which is the direct test of the
"location must never imply genetics/physiology" requirement: two athletes
with identical body stats/training load get identical calorie/protein/carb/
fat targets regardless of region, only the foods those targets are
expressed in differ.

Everything here is offline and deterministic - no network, no random.choice().
"""

import unittest

import meal_planner as mp
from regional_foods import DEFAULT_REGION, REGIONS, detect_region

BASE = dict(
    age=22, height_cm=178, weight_kg=72, sex="Male",
    goal="Improve football performance",
    diet="Vegetarian", training_days_per_week=5, training_intensity="High",
    region="Lucknow, Uttar Pradesh", allergies="none",
    budget_per_day=None, food_preferences=None, foods_to_avoid=None, notes=None,
)

INDIAN_REGION_CITY = {
    region: data["keywords"][0] for region, data in REGIONS.items() if data["keywords"]
}


def all_items(plan):
    return [ing["item"] for meal in plan["meals"] for ing in meal["ingredients"]]


# ----------------------------------------------------------------------
# 1. Indian regional detection still works, completely unaffected
# ----------------------------------------------------------------------

class TestIndianDetectionUnaffected(unittest.TestCase):
    def test_every_indian_region_still_detects_correctly(self):
        for region, city in INDIAN_REGION_CITY.items():
            with self.subTest(region=region):
                self.assertEqual(detect_region({"region": city}), region)

    def test_full_indian_addresses_still_detect_correctly(self):
        cases = {
            "Lucknow, Uttar Pradesh": "north",
            "Chennai, Tamil Nadu": "south",
            "Kolkata, West Bengal": "east",
            "Pune, Maharashtra": "west",
            "Indore, Madhya Pradesh": "central",
            "Guwahati, Assam": "northeast",
        }
        for address, expected in cases.items():
            with self.subTest(address=address):
                self.assertEqual(detect_region({"region": address}), expected)


# ----------------------------------------------------------------------
# 2. Non-Indian locations resolve to "international"
# ----------------------------------------------------------------------

class TestNonIndianLocationsResolveToInternational(unittest.TestCase):
    def test_common_non_indian_cities_resolve_to_international(self):
        for city in ["London, UK", "New York, USA", "Toronto, Canada",
                     "Sydney, Australia", "Berlin, Germany", "Tokyo, Japan",
                     "Lagos, Nigeria", "Sao Paulo, Brazil"]:
            with self.subTest(city=city):
                self.assertEqual(detect_region({"region": city}), "international")

    def test_international_is_the_default_region_constant(self):
        """DEFAULT_REGION itself must be "international" - the previous
        default ("north") is exactly the bug this phase fixes."""
        self.assertEqual(DEFAULT_REGION, "international")
        self.assertNotEqual(DEFAULT_REGION, "north")


# ----------------------------------------------------------------------
# 3. Unknown/blank locations safely fall back to "international"
# ----------------------------------------------------------------------

class TestUnknownOrBlankLocationFallsBackToInternational(unittest.TestCase):
    def test_nonsense_location_falls_back(self):
        self.assertEqual(detect_region({"region": "Atlantis"}), "international")

    def test_empty_string_falls_back(self):
        self.assertEqual(detect_region({"region": ""}), "international")

    def test_missing_region_key_falls_back(self):
        self.assertEqual(detect_region({}), "international")

    def test_none_region_falls_back(self):
        self.assertEqual(detect_region({"region": None}), "international")

    def test_whitespace_only_region_falls_back(self):
        self.assertEqual(detect_region({"region": "   "}), "international")


# ----------------------------------------------------------------------
# 4. International meals actually use the international pool
# ----------------------------------------------------------------------

class TestInternationalMealsUseTheInternationalPool(unittest.TestCase):
    def test_generated_plan_only_uses_international_and_generic_items(self):
        """Nothing Indian-regional should leak into an international plan -
        mirrors test_food_variety_regions.py's own per-region isolation
        test, applied to the new pool."""
        pool = REGIONS["international"]
        own = set(
            pool["breakfast"] + pool["breakfast_sides"] + pool["snack"]
            + pool["carbs"] + pool["veg_protein"] + pool["veg"]
        )
        # Items the builders can add regardless of region (see
        # test_food_variety_regions.py's identical list).
        generic = {"Banana", "Dates (5-6)", "Roasted chana", "Curd",
                   "Khichdi (rice + moong dal)", "Vegetable upma",
                   "Boiled eggs", "Moong dal", "Rice"}
        plan = mp.mock_provider(dict(BASE, region="London, UK"))
        for item in all_items(plan):
            self.assertTrue(
                item in own or item in generic,
                f"unexpected non-international item {item!r} in an international plan",
            )

    def test_at_least_one_real_international_dish_appears(self):
        """Not just "nothing Indian leaked in" - the pool must actually be
        reachable and used, not merely present but never selected."""
        pool_dishes = set(REGIONS["international"]["breakfast"])
        plan = mp.mock_provider(dict(BASE, region="London, UK"))
        breakfast_main = plan["meals"][0]["ingredients"][0]["item"]
        self.assertIn(breakfast_main, pool_dishes)

    def test_blank_region_also_reaches_the_international_pool(self):
        plan = mp.mock_provider(dict(BASE, region=""))
        pool_dishes = set(REGIONS["international"]["breakfast"])
        breakfast_main = plan["meals"][0]["ingredients"][0]["item"]
        self.assertIn(breakfast_main, pool_dishes)

    def test_no_dairy_side_added_automatically_for_international(self):
        """dairy_side=None for "international" (see regional_foods.py's
        note - a generic international meal pattern doesn't auto-add a
        dairy side to lunch/dinner the way Indian regions do)."""
        self.assertIsNone(REGIONS["international"]["dairy_side"])

    def test_multi_day_international_plan_generates(self):
        plan = mp.mock_provider_multi_day(dict(BASE, region="London, UK"), days=7)
        self.assertEqual(len(plan["days"]), 7)
        pool = REGIONS["international"]
        own = set(pool["breakfast"] + pool["breakfast_sides"] + pool["snack"]
                  + pool["carbs"] + pool["veg_protein"] + pool["veg"])
        generic = {"Banana", "Dates (5-6)", "Roasted chana", "Curd",
                   "Khichdi (rice + moong dal)", "Vegetable upma",
                   "Boiled eggs", "Moong dal", "Rice"}
        for day in plan["days"]:
            for item in [ing["item"] for meal in day["meals"] for ing in meal["ingredients"]]:
                self.assertTrue(item in own or item in generic)


# ----------------------------------------------------------------------
# 5. Existing diet/allergy/age/goal rules still work for international
# ----------------------------------------------------------------------

class TestSafetyRulesStillWorkForInternational(unittest.TestCase):
    def test_allergy_respected_for_international(self):
        plan = mp.mock_provider(dict(BASE, region="London, UK", allergies="dairy"))
        names = [i.lower() for i in all_items(plan)]
        for blocked in ("milk", "curd", "paneer", "cheese", "ghee", "yogurt"):
            self.assertFalse(any(blocked in n for n in names), f"{blocked} leaked past a dairy allergy")

    def test_vegetarian_diet_respected_for_international(self):
        plan = mp.mock_provider(dict(BASE, region="London, UK", diet="Vegetarian"))
        names = [i.lower() for i in all_items(plan)]
        for meat in ("chicken", "fish", "salmon", "turkey", "beef"):
            self.assertFalse(any(meat in n for n in names), f"{meat} leaked into a vegetarian plan")

    def test_foods_to_avoid_respected_for_international(self):
        plan = mp.mock_provider(dict(BASE, region="London, UK", foods_to_avoid="lentil soup"))
        names = [i.lower() for i in all_items(plan)]
        self.assertFalse(any("lentil soup" in n for n in names))

    def test_minor_portion_cap_holds_for_international(self):
        minor_profile = dict(BASE, region="London, UK", age=15,
                             training_days_per_week=7, training_intensity="High")
        plan = mp.mock_provider(minor_profile)
        lunch_carb_qty = plan["meals"][1]["ingredients"][0]["quantity"]
        serving_count = int(lunch_carb_qty.split()[0])
        self.assertLessEqual(serving_count, 3)

    def test_goal_still_drives_macros_for_international(self):
        """Location must never change nutrition targets - only which foods
        express them. A muscle-gain goal must still raise calories/protein
        over maintenance, identically to an Indian-region athlete."""
        maintenance = mp.mock_provider(dict(BASE, region="London, UK", goal="Maintain fitness"))
        muscle_gain = mp.mock_provider(dict(BASE, region="London, UK", goal="Muscle gain and strength"))
        self.assertGreater(
            muscle_gain["day_summary"]["total_calories"],
            maintenance["day_summary"]["total_calories"],
        )


# ----------------------------------------------------------------------
# Location influences cuisine only - never genetics/physiology/requirements
# ----------------------------------------------------------------------

class TestLocationOnlyAffectsCuisine(unittest.TestCase):
    def test_identical_stats_give_identical_nutrition_targets_regardless_of_region(self):
        """The exact guarantee regional_foods.py's own docstring makes:
        two athletes with the same body stats/training load get the SAME
        nutrition targets no matter where they live - only the foods
        expressing those targets differ."""
        indian_athlete = mp.mock_provider(dict(BASE, region="Chennai, Tamil Nadu"))
        international_athlete = mp.mock_provider(dict(BASE, region="London, UK"))
        self.assertEqual(
            indian_athlete["day_summary"]["total_calories"],
            international_athlete["day_summary"]["total_calories"],
        )
        self.assertEqual(
            indian_athlete["day_summary"]["total_protein_g"],
            international_athlete["day_summary"]["total_protein_g"],
        )
        self.assertEqual(
            indian_athlete["day_summary"]["total_carbs_g"],
            international_athlete["day_summary"]["total_carbs_g"],
        )
        self.assertEqual(
            indian_athlete["day_summary"]["total_fats_g"],
            international_athlete["day_summary"]["total_fats_g"],
        )
        # Only the actual food differs.
        self.assertNotEqual(all_items(indian_athlete), all_items(international_athlete))


if __name__ == "__main__":
    unittest.main(verbosity=2)
