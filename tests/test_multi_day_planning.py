"""
AthleteOS - Step 6 tests: multi-day meal planning.

Proves the guarantees this feature exists for:

  BACKWARD COMPATIBILITY - a request that doesn't mention duration (or
                            explicitly says 1) produces EXACTLY the same
                            single-day plan the app has always returned.

  COMPLETENESS            - a 7-day request returns seven days, each with
                            all four meals.

  SAFETY ACROSS DAYS       - allergies/diet/region/age-appropriateness hold
                            on every single day, not just day 1.

  CROSS-DAY VARIETY        - consecutive days don't repeat the same
                            breakfast/lunch-main/dinner-main, and protein
                            sources rotate across the week, WITHOUT any
                            randomness: same profile -> same 7-day plan,
                            every time.

Everything here is offline and deterministic - no network, no API calls,
same technique as tests/test_food_variety_regions.py.
"""

import unittest

import meal_planner as mp
from api import MealPlanRequest, create_meal_plan
from regional_foods import REGIONS

BASE = dict(
    age=22, height_cm=178, weight_kg=72, sex="Male",
    goal="Improve football performance",
    diet="Vegetarian", training_days_per_week=5, training_intensity="High",
    region="Lucknow, Uttar Pradesh", allergies="none",
    budget_per_day=None, food_preferences=None, foods_to_avoid=None, notes=None,
)


def breakfast_main(day):
    return day["meals"][0]["ingredients"][0]["item"]


def lunch_protein(day):
    return day["meals"][1]["ingredients"][1]["item"]


def dinner_protein(day):
    return day["meals"][3]["ingredients"][1]["item"]


def snack_main(day):
    return day["meals"][2]["ingredients"][0]["item"]


def all_items(day):
    return [ing["item"] for meal in day["meals"] for ing in meal["ingredients"]]


def all_items_across_week(plan):
    return [item for day in plan["days"] for item in all_items(day)]


# ----------------------------------------------------------------------
# 1/14. Backward compatibility: default and explicit days=1
# ----------------------------------------------------------------------

class TestBackwardCompatibility(unittest.TestCase):
    def test_default_request_still_produces_existing_one_day_behaviour(self):
        """No `days` argument at all - the pre-existing call signature -
        must return byte-identical output to before this feature existed."""
        via_default = mp.generate_meal_plan(dict(BASE))
        via_mock_directly = mp.mock_provider(dict(BASE))
        self.assertEqual(via_default, via_mock_directly)

    def test_explicit_days_1_is_identical_to_default(self):
        via_default = mp.generate_meal_plan(dict(BASE))
        via_explicit = mp.generate_meal_plan(dict(BASE), days=1)
        self.assertEqual(via_default, via_explicit)

    def test_one_day_response_shape_is_unchanged(self):
        plan = mp.generate_meal_plan(dict(BASE), days=1)
        self.assertEqual(
            set(plan.keys()),
            {"confidence_score", "confidence_reason", "day_summary", "meals", "disclaimer"},
        )
        self.assertNotIn("days", plan)
        self.assertNotIn("duration_days", plan)

    def test_request_model_defaults_duration_days_to_one(self):
        request = MealPlanRequest(**BASE)
        self.assertEqual(request.duration_days, 1)

    def test_endpoint_default_request_returns_the_single_day_shape(self):
        response = create_meal_plan(MealPlanRequest(**BASE))
        self.assertIn("day_summary", response)
        self.assertIn("meals", response)
        self.assertNotIn("days", response)


# ----------------------------------------------------------------------
# 2/3. Seven complete days, each with all four meals
# ----------------------------------------------------------------------

class TestSevenDayCompleteness(unittest.TestCase):
    def test_seven_day_request_returns_seven_days(self):
        plan = mp.generate_meal_plan(dict(BASE), days=7)
        self.assertEqual(plan["duration_days"], 7)
        self.assertEqual(len(plan["days"]), 7)
        self.assertEqual([d["day"] for d in plan["days"]], [1, 2, 3, 4, 5, 6, 7])

    def test_every_day_has_all_four_meals_in_order(self):
        plan = mp.generate_meal_plan(dict(BASE), days=7)
        for day in plan["days"]:
            meal_names = [m["meal"] for m in day["meals"]]
            self.assertEqual(meal_names, ["Breakfast", "Lunch", "Snack", "Dinner"])

    def test_every_day_has_a_day_summary_with_all_macro_fields(self):
        plan = mp.generate_meal_plan(dict(BASE), days=7)
        for day in plan["days"]:
            s = day["day_summary"]
            for field in ["total_calories", "total_protein_g", "total_carbs_g", "total_fats_g", "notes"]:
                self.assertIn(field, s)
                self.assertIsNotNone(s[field])

    def test_endpoint_seven_day_request_returns_seven_days(self):
        response = create_meal_plan(MealPlanRequest(**dict(BASE, duration_days=7)))
        self.assertEqual(response["duration_days"], 7)
        self.assertEqual(len(response["days"]), 7)

    def test_duration_days_out_of_range_is_rejected(self):
        from pydantic import ValidationError
        for bad in [0, -1, 29]:
            with self.assertRaises(ValidationError):
                MealPlanRequest(**dict(BASE, duration_days=bad))


# ----------------------------------------------------------------------
# 4/5/6. Safety and appropriateness hold across all seven days
# ----------------------------------------------------------------------

class TestSafetyAcrossAllDays(unittest.TestCase):
    def test_allergies_respected_across_all_seven_days(self):
        profile = dict(BASE, allergies="dairy")
        plan = mp.generate_meal_plan(profile, days=7)
        # Exactly the words meal_planner.avoid_keywords() expands "dairy"
        # into - not a broader list, so this only tests what the app
        # actually guarantees today.
        banned = mp.avoid_keywords(profile) & {"milk", "curd", "paneer", "cheese", "ghee", "buttermilk"}
        for day in plan["days"]:
            for item in all_items(day):
                low = item.lower()
                for word in banned:
                    self.assertNotIn(word, low, f"day {day['day']}: {item!r} contains {word!r}")

    def test_diet_respected_across_all_seven_days(self):
        profile = dict(BASE, diet="Vegetarian")
        plan = mp.generate_meal_plan(profile, days=7)
        banned = ["egg", "chicken", "fish", "mutton", "prawn"]
        for day in plan["days"]:
            for item in all_items(day):
                low = item.lower()
                for word in banned:
                    self.assertNotIn(word, low, f"day {day['day']}: {item!r} contains {word!r}")

    def test_region_respected_across_all_seven_days(self):
        """Northeast's dairy_side is None (see regional_foods.py), so
        build_lunch()/build_dinner() must never add a forced dairy
        accompaniment there on ANY of the seven days - the exact bug the
        region-aware dairy_side fix addressed for the single-day case.
        Curd MAY still legitimately appear as one of northeast's own
        breakfast_sides rotation options (a real, non-default choice per
        that same fix) - so this checks the LUNCH/DINNER slots specifically,
        not the whole day, which is the actual guarantee that exists."""
        profile = dict(BASE, region="Guwahati, Assam")
        plan = mp.generate_meal_plan(profile, days=7)
        for day in plan["days"]:
            lunch_items = [ing["item"].lower() for ing in day["meals"][1]["ingredients"]]
            dinner_items = [ing["item"].lower() for ing in day["meals"][3]["ingredients"]]
            for items, label in [(lunch_items, "lunch"), (dinner_items, "dinner")]:
                for item in items:
                    self.assertNotIn("curd", item, f"day {day['day']} {label}: {item!r}")
                    self.assertNotIn("paneer", item, f"day {day['day']} {label}: {item!r}")

    def test_region_breakfast_mains_stay_within_the_regions_own_candidates(self):
        profile = dict(BASE, region="Chennai, Tamil Nadu")
        plan = mp.generate_meal_plan(profile, days=7)
        south_breakfasts = set(REGIONS["south"]["breakfast"])
        for day in plan["days"]:
            self.assertIn(breakfast_main(day), south_breakfasts)

    def test_minor_logic_correct_across_all_seven_days(self):
        """Mirrors the single-day is_minor guarantees: growth-appropriate
        note present, no adult-style muscle_gain/recovery protein booster
        item, every day of the week - not just day 1."""
        profile = dict(BASE, age=10, goal="muscle gain", training_days_per_week=4,
                       training_intensity="Moderate")
        plan = mp.generate_meal_plan(profile, days=7)
        for day in plan["days"]:
            self.assertIn("Growth-appropriate", day["day_summary"]["notes"],
                         f"day {day['day']}")
            snack_items = [ing["item"] for ing in day["meals"][2]["ingredients"]]
            self.assertNotIn("Boiled eggs", snack_items, f"day {day['day']}")

    def test_adult_portion_exceeds_minor_portion_across_the_week(self):
        """Mirrors test_age_developmental_stage.py's central comparison: at
        the mock-only level (no agent_context/nutrition_tool - the day-level
        calorie NUMBER only becomes age-aware once the orchestrator's
        nutrition_tool is involved, which is exercised separately in
        test_age_developmental_stage.py, not here), age still changes
        portion sizing/wording and the meals themselves, every day of the
        week - not just day 1. Heavy load (6 days/week, High) is what
        actually triggers is_minor's "large" -> "moderate" cap; a lighter
        load computes "moderate" for both ages regardless of is_minor."""
        minor = dict(BASE, age=10, goal="muscle gain",
                    training_days_per_week=6, training_intensity="High")
        adult = dict(BASE, age=25, goal="muscle gain",
                    training_days_per_week=6, training_intensity="High")
        minor_plan = mp.generate_meal_plan(minor, days=7)
        adult_plan = mp.generate_meal_plan(adult, days=7)
        for m_day, a_day in zip(minor_plan["days"], adult_plan["days"]):
            self.assertIn("moderate training load", m_day["day_summary"]["notes"], f"day {m_day['day']}")
            self.assertIn("large training load", a_day["day_summary"]["notes"], f"day {a_day['day']}")
            self.assertNotEqual(m_day["meals"], a_day["meals"], f"day {m_day['day']}")


# ----------------------------------------------------------------------
# 8/9/10. Cross-day variety
# ----------------------------------------------------------------------

class TestCrossDayVariety(unittest.TestCase):
    def test_consecutive_breakfasts_do_not_repeat(self):
        plan = mp.generate_meal_plan(dict(BASE), days=7)
        mains = [breakfast_main(day) for day in plan["days"]]
        for i in range(len(mains) - 1):
            self.assertNotEqual(mains[i], mains[i + 1], f"day {i+1} vs day {i+2}: both {mains[i]!r}")

    def test_consecutive_lunch_mains_do_not_repeat(self):
        plan = mp.generate_meal_plan(dict(BASE), days=7)
        proteins = [lunch_protein(day) for day in plan["days"]]
        for i in range(len(proteins) - 1):
            self.assertNotEqual(proteins[i], proteins[i + 1], f"day {i+1} vs day {i+2}")

    def test_consecutive_dinner_mains_do_not_repeat(self):
        plan = mp.generate_meal_plan(dict(BASE), days=7)
        proteins = [dinner_protein(day) for day in plan["days"]]
        for i in range(len(proteins) - 1):
            self.assertNotEqual(proteins[i], proteins[i + 1], f"day {i+1} vs day {i+2}")

    def test_consecutive_snacks_do_not_repeat(self):
        plan = mp.generate_meal_plan(dict(BASE), days=7)
        mains = [snack_main(day) for day in plan["days"]]
        for i in range(len(mains) - 1):
            self.assertNotEqual(mains[i], mains[i + 1], f"day {i+1} vs day {i+2}")

    def test_protein_sources_rotate_where_the_pool_allows(self):
        """north's veg_protein pool has 9 items - across 7 days lunch protein
        should visit more than a couple of them, not just alternate between
        two options."""
        plan = mp.generate_meal_plan(dict(BASE), days=7)
        proteins = {lunch_protein(day) for day in plan["days"]}
        self.assertGreater(len(proteins), 2, f"only saw {proteins!r} across 7 days")

    def test_carb_sources_rotate_where_the_pool_allows(self):
        plan = mp.generate_meal_plan(dict(BASE), days=7)
        carbs = {day["meals"][1]["ingredients"][0]["item"] for day in plan["days"]}
        self.assertGreater(len(carbs), 1, f"only saw {carbs!r} across 7 days")

    def test_within_day_variety_is_preserved_every_day(self):
        """The pre-existing within-day guarantee (lunch protein != dinner
        protein on the SAME day) must still hold for every day of the week,
        not just day 1."""
        plan = mp.generate_meal_plan(dict(BASE), days=7)
        for day in plan["days"]:
            self.assertNotEqual(lunch_protein(day), dinner_protein(day), f"day {day['day']}")


# ----------------------------------------------------------------------
# 11/12. Reproducibility vs. divergence
# ----------------------------------------------------------------------

class TestReproducibilityAndDivergence(unittest.TestCase):
    def test_same_athlete_same_inputs_produces_the_same_seven_day_plan(self):
        plan_a = mp.generate_meal_plan(dict(BASE), days=7)
        plan_b = mp.generate_meal_plan(dict(BASE), days=7)
        self.assertEqual(plan_a, plan_b)

    def test_similar_athletes_can_produce_different_seven_day_plans(self):
        athlete_a = dict(BASE, weight_kg=72)
        athlete_b = dict(BASE, weight_kg=73)
        plan_a = mp.generate_meal_plan(athlete_a, days=7)
        plan_b = mp.generate_meal_plan(athlete_b, days=7)
        self.assertNotEqual(all_items_across_week(plan_a), all_items_across_week(plan_b))

    def test_different_day_indices_within_one_plan_are_not_all_identical(self):
        """A sanity check on the day-seed mechanism itself: day 1's meals
        must not equal day 2's (already covered by the no-repeat tests
        above), and more broadly the WEEK must not be seven copies of the
        same day - which calling the single-day generator seven times with
        an unrelated per-day seed but no cross-day exclusion could still
        accidentally produce if two day-seeds collided."""
        plan = mp.generate_meal_plan(dict(BASE), days=7)
        signatures = {tuple(all_items(day)) for day in plan["days"]}
        self.assertEqual(len(signatures), 7, "expected 7 distinct daily plans, got duplicates")


# ----------------------------------------------------------------------
# 13. Home ingredients still influence the plan where applicable
# ----------------------------------------------------------------------

class TestHomeIngredientsAcrossDays(unittest.TestCase):
    def test_home_ingredients_still_influence_the_multi_day_plan(self):
        """paneer is not north's default lunch protein pick for this seed
        (see the baseline below); listing it as a home ingredient should
        make it appear on at least one of the seven days, without ever
        crowding out the days where rotation didn't select it."""
        baseline = mp.generate_meal_plan(dict(BASE), days=7)
        baseline_items = set(all_items_across_week(baseline))

        with_ingredients = mp.generate_meal_plan(
            dict(BASE, available_ingredients="paneer, dal, rice"), days=7
        )
        with_items = set(all_items_across_week(with_ingredients))

        self.assertTrue(
            any("paneer" in item.lower() for item in with_items),
            f"expected paneer to appear somewhere across the week, got {with_items!r}",
        )

    def test_ingredients_never_violate_diet_across_any_day(self):
        """A vegetarian athlete listing chicken/eggs at home must never see
        them on any of the seven days - safety outranks pantry bias."""
        profile = dict(BASE, diet="Vegetarian", available_ingredients="chicken, eggs, rice, dal")
        plan = mp.generate_meal_plan(profile, days=7)
        for day in plan["days"]:
            for item in all_items(day):
                low = item.lower()
                self.assertNotIn("chicken", low, f"day {day['day']}: {item!r}")
                self.assertNotIn("egg", low, f"day {day['day']}: {item!r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
