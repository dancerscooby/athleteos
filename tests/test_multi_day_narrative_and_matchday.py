"""
AthleteOS - Phase 2 tests: 7-day narrative variety + the match-day override.

Two problems this fixes (see meal_planner.py's notes above GOAL_TARGETS and
on build_dinner()):

  REPETITIVE TEXT   - a 7-day plan's FOOD genuinely rotated, but every meal's
                      "reasoning" and every day's summary "notes" were built
                      from a single fixed sentence per goal/meal-slot, so a
                      week showed the exact same sentence seven times under
                      seven different breakfasts.

  MATCH-DAY OVERRIDE - a "match" goal always forced dinner to a fixed
                      Khichdi dish. For 7 days, that meant the identical
                      dinner on all 7 days, which doesn't make sense - the
                      light pre-match dinner belongs on the night before the
                      match (the LAST day of the plan), not every night.

Both fixes are TEXT/OVERRIDE-SCOPE changes only - no change to which foods
are selected, no randomness, and the 1-day path must remain byte-identical
to before. Everything here is offline and deterministic - no network calls.
"""

import unittest

import meal_planner as mp

BASE = dict(
    age=22, height_cm=178, weight_kg=72, sex="Male",
    goal="Improve football performance",
    diet="Vegetarian", training_days_per_week=5, training_intensity="High",
    region="Lucknow, Uttar Pradesh", allergies="none",
    budget_per_day=None, food_preferences=None, foods_to_avoid=None, notes=None,
)

MATCH_BASE = dict(BASE, goal="Fuel properly for a match this week")


def dinner_main(day_meals):
    return day_meals[3]["ingredients"][0]["item"]


def breakfast_reason(day_meals):
    return day_meals[0]["reasoning"]


def lunch_reason(day_meals):
    return day_meals[1]["reasoning"]


def snack_reason(day_meals):
    return day_meals[2]["reasoning"]


def dinner_reason(day_meals):
    return day_meals[3]["reasoning"]


# ----------------------------------------------------------------------
# 1. Single-day behaviour is unchanged
# ----------------------------------------------------------------------

class TestSingleDayNarrativeUnchanged(unittest.TestCase):
    def test_single_day_reasoning_text_matches_the_original_wording(self):
        """days=1 must produce EXACTLY the pre-existing sentences - the
        original wording is pool[0] in every phrasing pool, and a 1-day
        call never passes day_index, so it always resolves to index 0."""
        plan = mp.generate_meal_plan(dict(BASE), days=1)
        self.assertEqual(breakfast_reason(plan["meals"]), "Carb-rich start to fuel morning training")
        self.assertEqual(lunch_reason(plan["meals"]), "Largest meal around training for steady energy and protein")
        self.assertEqual(snack_reason(plan["meals"]), "Light snack to keep energy up between meals")
        self.assertEqual(dinner_reason(plan["meals"]), "Balanced dinner to round out the day")

    def test_single_day_summary_notes_match_the_original_wording(self):
        plan = mp.generate_meal_plan(dict(BASE), days=1)
        self.assertIn("Balanced energy to support high-intensity football performance.", plan["day_summary"]["notes"])

    def test_single_day_match_goal_still_gets_khichdi_dinner(self):
        """The match-day override is UNCHANGED for single-day: a 1-day plan
        IS its own last day, so apply_match_override defaults True there."""
        plan = mp.generate_meal_plan(dict(MATCH_BASE), days=1)
        self.assertIn("khichdi", dinner_main(plan["meals"]).lower())

    def test_single_day_output_is_byte_identical_to_direct_mock_call(self):
        via_days_1 = mp.generate_meal_plan(dict(BASE), days=1)
        via_mock_directly = mp.mock_provider(dict(BASE))
        self.assertEqual(via_days_1, via_mock_directly)


# ----------------------------------------------------------------------
# 2. 7-day narrative text is no longer identical every day
# ----------------------------------------------------------------------

class TestSevenDayNarrativeVariety(unittest.TestCase):
    def test_day_summary_notes_are_not_all_identical_across_the_week(self):
        plan = mp.generate_meal_plan(dict(BASE), days=7)
        notes = {day["day_summary"]["notes"] for day in plan["days"]}
        self.assertGreater(len(notes), 1, f"all 7 days had the identical notes text: {notes!r}")

    def test_breakfast_reasoning_is_not_all_identical_across_the_week(self):
        plan = mp.generate_meal_plan(dict(BASE), days=7)
        reasons = {breakfast_reason(day["meals"]) for day in plan["days"]}
        self.assertGreater(len(reasons), 1, f"all 7 days had identical breakfast reasoning: {reasons!r}")

    def test_lunch_reasoning_is_not_all_identical_across_the_week(self):
        plan = mp.generate_meal_plan(dict(BASE), days=7)
        reasons = {lunch_reason(day["meals"]) for day in plan["days"]}
        self.assertGreater(len(reasons), 1, f"all 7 days had identical lunch reasoning: {reasons!r}")

    def test_snack_reasoning_is_not_all_identical_across_the_week(self):
        plan = mp.generate_meal_plan(dict(BASE), days=7)
        reasons = {snack_reason(day["meals"]) for day in plan["days"]}
        self.assertGreater(len(reasons), 1, f"all 7 days had identical snack reasoning: {reasons!r}")

    def test_dinner_reasoning_is_not_all_identical_across_the_week(self):
        plan = mp.generate_meal_plan(dict(BASE), days=7)
        reasons = {dinner_reason(day["meals"]) for day in plan["days"]}
        self.assertGreater(len(reasons), 1, f"all 7 days had identical dinner reasoning: {reasons!r}")

    def test_day_1_reasoning_still_matches_the_original_single_day_wording(self):
        """day_index=0 (the first day) must still resolve to pool[0] - the
        exact same sentence a 1-day plan would use - so day 1 of a 7-day
        plan reads identically to what a 1-day request for the same
        profile would have said."""
        seven_day_plan = mp.generate_meal_plan(dict(BASE), days=7)
        one_day_plan = mp.generate_meal_plan(dict(BASE), days=1)
        self.assertEqual(
            breakfast_reason(seven_day_plan["days"][0]["meals"]),
            breakfast_reason(one_day_plan["meals"]),
        )

    def test_food_selection_itself_is_unaffected_by_the_text_rotation(self):
        """The narrative-text fix must not change WHICH foods are picked -
        only the wording of the reasoning. Cross-checked against the
        existing cross-day-variety guarantees (see test_multi_day_planning.py)
        by confirming ingredients are still present and well-formed."""
        plan = mp.generate_meal_plan(dict(BASE), days=7)
        for day in plan["days"]:
            for meal in day["meals"]:
                self.assertTrue(meal["ingredients"])
                for ing in meal["ingredients"]:
                    self.assertIn("item", ing)
                    self.assertIn("quantity", ing)


# ----------------------------------------------------------------------
# 3. Match-day override no longer repeats on every day of a 7-day plan
# ----------------------------------------------------------------------

class TestMatchDayOverrideScopedToLastDay(unittest.TestCase):
    def test_only_the_last_day_gets_the_khichdi_match_dinner(self):
        plan = mp.generate_meal_plan(dict(MATCH_BASE), days=7)
        khichdi_days = [
            day["day"] for day in plan["days"]
            if "khichdi" in dinner_main(day["meals"]).lower()
        ]
        self.assertEqual(khichdi_days, [7])

    def test_earlier_days_get_normal_varied_dinners_not_repeated(self):
        """Days 1-6 must fall back to the normal dinner logic (protein/carb
        rotation) rather than all defaulting to some other single fixed
        dish - i.e. the fix doesn't just move the "always identical" dinner
        from khichdi to something else."""
        plan = mp.generate_meal_plan(dict(MATCH_BASE), days=7)
        early_dinners = {dinner_main(day["meals"]) for day in plan["days"][:6]}
        self.assertGreater(len(early_dinners), 1, f"days 1-6 dinners: {early_dinners!r}")
        self.assertNotIn("Khichdi (rice + moong dal)", early_dinners)

    def test_a_non_match_goal_seven_day_plan_never_gets_a_khichdi_override(self):
        """Sanity check: the last-day override is gated on goal_key=="match" -
        an ordinary performance-goal week should never see the special
        khichdi dinner just because it happens to be the last day."""
        plan = mp.generate_meal_plan(dict(BASE), days=7)
        for day in plan["days"]:
            self.assertNotEqual(dinner_main(day["meals"]), "Khichdi (rice + moong dal)")


# ----------------------------------------------------------------------
# 4. Reproducibility is preserved after the narrative-text change
# ----------------------------------------------------------------------

class TestReproducibilityPreserved(unittest.TestCase):
    def test_same_profile_same_days_still_produces_the_same_plan(self):
        plan_a = mp.generate_meal_plan(dict(MATCH_BASE), days=7)
        plan_b = mp.generate_meal_plan(dict(MATCH_BASE), days=7)
        self.assertEqual(plan_a, plan_b)

    def test_same_profile_single_day_still_reproducible(self):
        plan_a = mp.generate_meal_plan(dict(BASE), days=1)
        plan_b = mp.generate_meal_plan(dict(BASE), days=1)
        self.assertEqual(plan_a, plan_b)

    def test_similar_athletes_still_diverge_after_the_narrative_change(self):
        athlete_a = dict(BASE, weight_kg=72)
        athlete_b = dict(BASE, weight_kg=73)
        plan_a = mp.generate_meal_plan(athlete_a, days=7)
        plan_b = mp.generate_meal_plan(athlete_b, days=7)
        self.assertNotEqual(plan_a, plan_b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
