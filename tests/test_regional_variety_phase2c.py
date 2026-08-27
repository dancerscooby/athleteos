"""
AthleteOS - Phase 2C tests: regional food variety additions + duplicate fixes.

Covers exactly what the approved Phase 2C audit implemented:

  1. Duplicate/over-reuse fixes: West's "Dhokla" and "Buttermilk (chaas)"
     narrowed from 2-3 pools to 1-2; Northeast's "Boiled black gram"
     narrowed from 3 pools to 2.
  2. Central: 2 new nonveg_protein dishes (Mutton curry, Fish curry) and 2
     new veg dishes (Bharwa baingan, Bhutte ki sabzi) - the additions
     approved in the earlier Central audit, implemented now.
  3. One new carb per region (except Northeast, left unchanged per the
     approved plan): North (Jowar roti), South (Idiyappam), East (Panta
     bhat), West (Nachni bhakri), Central (Bafla).
  4. Two new breakfast_sides: South (Coconut chutney), North (Pickled
     vegetables).

Nothing here touches nutrition_data.py, RAG, or the Phase 2A rotation
mechanism itself (generation_index/recent-item exclusion) - this file only
proves the EXPANDED pools still work correctly under that unchanged
mechanism.

Everything here is offline and deterministic - no network, no random.choice().
"""

import unittest

import meal_planner as mp
from regional_foods import REGIONS

BASE = dict(
    age=22, height_cm=178, weight_kg=72, sex="Male",
    goal="Improve football performance",
    diet="Vegetarian", training_days_per_week=5, training_intensity="High",
    region="Lucknow, Uttar Pradesh", allergies="none",
    budget_per_day=None, food_preferences=None, foods_to_avoid=None, notes=None,
)

REGION_CITY = {
    region: data["keywords"][0] for region, data in REGIONS.items() if data["keywords"]
}


def all_items(plan):
    return [ing["item"] for meal in plan["meals"] for ing in meal["ingredients"]]


# ----------------------------------------------------------------------
# 1. New foods are present in the correct region/category
# ----------------------------------------------------------------------

class TestNewFoodsPresentInCorrectPlace(unittest.TestCase):
    def test_central_new_nonveg_protein_dishes(self):
        for dish in ("Mutton curry", "Fish curry"):
            self.assertIn(dish, REGIONS["central"]["nonveg_protein"])

    def test_central_new_veg_dishes(self):
        for dish in ("Bharwa baingan (stuffed eggplant)", "Bhutte ki sabzi (corn kernel sabzi)"):
            self.assertIn(dish, REGIONS["central"]["veg"])

    def test_new_carb_per_region(self):
        expected = {
            "north": "Jowar roti",
            "south": "Idiyappam",
            "east": "Panta bhat (fermented rice)",
            "west": "Nachni bhakri (ragi)",
            "central": "Bafla (wheat dumpling)",
        }
        for region, dish in expected.items():
            with self.subTest(region=region):
                self.assertIn(dish, REGIONS[region]["carbs"])

    def test_northeast_carbs_deliberately_left_unchanged(self):
        """Approved plan: no clearly appropriate carb was identified for
        Northeast, so nothing was added - same 4 items as before Phase 2C."""
        self.assertEqual(
            REGIONS["northeast"]["carbs"],
            ["Rice", "Sticky rice", "Rice flakes", "Whole-wheat roti"],
        )

    def test_new_breakfast_sides(self):
        self.assertIn("Coconut chutney", REGIONS["south"]["breakfast_sides"])
        self.assertIn("Pickled vegetables (achaar)", REGIONS["north"]["breakfast_sides"])


# ----------------------------------------------------------------------
# Duplicate/over-reuse fixes
# ----------------------------------------------------------------------

class TestDuplicationFixes(unittest.TestCase):
    def test_dhokla_no_longer_duplicated_in_west(self):
        """Was in both breakfast and snack - now only breakfast."""
        self.assertIn("Dhokla", REGIONS["west"]["breakfast"])
        self.assertNotIn("Dhokla", REGIONS["west"]["snack"])

    def test_west_buttermilk_narrowed_from_three_pools_to_two(self):
        """Was in breakfast_sides, snack, AND dairy_side - snack removed."""
        self.assertIn("Buttermilk (chaas)", REGIONS["west"]["breakfast_sides"])
        self.assertEqual(REGIONS["west"]["dairy_side"], "Buttermilk (chaas)")
        self.assertNotIn("Buttermilk (chaas)", REGIONS["west"]["snack"])

    def test_northeast_boiled_black_gram_narrowed_from_three_pools_to_two(self):
        """Was in breakfast_sides, snack, AND veg_protein - snack removed."""
        self.assertIn("Boiled black gram", REGIONS["northeast"]["breakfast_sides"])
        self.assertIn("Boiled black gram", REGIONS["northeast"]["veg_protein"])
        self.assertNotIn("Boiled black gram", REGIONS["northeast"]["snack"])

    def test_northeast_curd_placement_reviewed_and_left_as_is(self):
        """Curd is already last in breakfast_sides (not the default pick
        under rotation) - consistent with the region's own "not the
        default" comment, so no change was made here."""
        self.assertEqual(REGIONS["northeast"]["breakfast_sides"][-1], "Curd")


# ----------------------------------------------------------------------
# 2. Regional selection can actually reach the new foods
# ----------------------------------------------------------------------

class TestNewFoodsAreReachable(unittest.TestCase):
    def test_central_mutton_and_fish_curry_reachable_via_lunch_dinner(self):
        profile = dict(BASE, region=REGION_CITY["central"], diet="Non-Vegetarian")
        plan = mp.mock_provider_multi_day(dict(profile), days=7)
        proteins = {
            i["item"]
            for day in plan["days"]
            for meal in day["meals"] if meal["meal"] in ("Lunch", "Dinner")
            for i in meal["ingredients"]
        }
        self.assertTrue({"Mutton curry", "Fish curry"} & proteins,
                         f"expected Mutton curry or Fish curry to appear across a 7-day plan, got {proteins}")

    def test_central_new_veg_dishes_reachable(self):
        profile = dict(BASE, region=REGION_CITY["central"])
        plan = mp.mock_provider_multi_day(dict(profile), days=7)
        veg_items = {
            i["item"]
            for day in plan["days"]
            for meal in day["meals"] if meal["meal"] in ("Lunch", "Dinner")
            for i in meal["ingredients"]
        }
        new_veg = {"Bharwa baingan (stuffed eggplant)", "Bhutte ki sabzi (corn kernel sabzi)"}
        self.assertTrue(new_veg & veg_items,
                         f"expected at least one new veg dish across a 7-day plan, got {veg_items}")

    def test_new_carbs_reachable_per_region(self):
        expected = {
            "north": "Jowar roti",
            "south": "Idiyappam",
            "east": "Panta bhat (fermented rice)",
            "west": "Nachni bhakri (ragi)",
            "central": "Bafla (wheat dumpling)",
        }
        for region, dish in expected.items():
            with self.subTest(region=region):
                profile = dict(BASE, region=REGION_CITY[region])
                plan = mp.mock_provider_multi_day(dict(profile), days=7)
                carbs_seen = {
                    day["meals"][1]["ingredients"][0]["item"] for day in plan["days"]
                } | {
                    day["meals"][3]["ingredients"][0]["item"] for day in plan["days"]
                }
                self.assertIn(dish, carbs_seen, f"{region}: {carbs_seen}")

    def test_new_breakfast_sides_reachable(self):
        cases = {"south": "Coconut chutney", "north": "Pickled vegetables (achaar)"}
        for region, side in cases.items():
            with self.subTest(region=region):
                profile = dict(BASE, region=REGION_CITY[region])
                plan = mp.mock_provider_multi_day(dict(profile), days=14)
                sides_seen = {
                    i["item"] for day in plan["days"] for i in day["meals"][0]["ingredients"]
                }
                self.assertIn(side, sides_seen, f"{region}: {sides_seen}")


# ----------------------------------------------------------------------
# 3. No eggs/meat/fish leak into vegetarian/diet-unaware pools (re-verified
#    across ALL 6 Indian regions after the Phase 2C additions)
# ----------------------------------------------------------------------

class TestNoMeatLeaksIntoVegetarianPools(unittest.TestCase):
    MEAT_WORDS = ("egg", "chicken", "mutton", "fish", "prawn", "pork", "meat")

    def test_breakfast_breakfast_sides_snack_veg_protein_are_meat_free(self):
        diet_unaware_or_veg_only = ("breakfast", "breakfast_sides", "snack", "veg_protein")
        for region, data in REGIONS.items():
            for category in diet_unaware_or_veg_only:
                for dish in data[category]:
                    with self.subTest(region=region, category=category, dish=dish):
                        low = dish.lower()
                        for word in self.MEAT_WORDS:
                            self.assertNotIn(word, low, f"{region}.{category} contains {dish!r}")

    def test_vegetarian_diet_still_respected_end_to_end_for_every_indian_region(self):
        for region, city in REGION_CITY.items():
            with self.subTest(region=region):
                plan = mp.mock_provider(dict(BASE, region=city, diet="Vegetarian"))
                names = [i.lower() for i in all_items(plan)]
                for meat in self.MEAT_WORDS:
                    self.assertFalse(any(meat in n for n in names), f"{region}: {meat} leaked in")


# ----------------------------------------------------------------------
# 4. Allergies/avoid lists still override the new foods
# ----------------------------------------------------------------------

class TestSafetyRulesOverrideNewFoods(unittest.TestCase):
    def test_foods_to_avoid_blocks_mutton_curry(self):
        profile = dict(BASE, region=REGION_CITY["central"], diet="Non-Vegetarian",
                       foods_to_avoid="mutton")
        for gen in range(6):
            plan = mp.mock_provider(dict(profile), agent_context={
                "variety_context": {"generation_index": gen, "recent_proteins": [], "recent_carbs": []}
            })
            names = [i.lower() for i in all_items(plan)]
            self.assertFalse(any("mutton" in n for n in names), f"generation_index={gen}")

    def test_foods_to_avoid_blocks_idiyappam(self):
        profile = dict(BASE, region=REGION_CITY["south"], foods_to_avoid="idiyappam")
        for gen in range(6):
            plan = mp.mock_provider(dict(profile), agent_context={
                "variety_context": {"generation_index": gen, "recent_proteins": [], "recent_carbs": []}
            })
            names = [i.lower() for i in all_items(plan)]
            self.assertFalse(any("idiyappam" in n for n in names), f"generation_index={gen}")

    def test_dairy_allergy_still_blocks_curd_alongside_new_coconut_chutney(self):
        """New "Coconut chutney" contains no dairy, so it should NOT be
        blocked by a dairy allergy - only actual dairy items should be."""
        profile = dict(BASE, region=REGION_CITY["south"], allergies="dairy")
        plan = mp.mock_provider(dict(profile))
        names = [i.lower() for i in all_items(plan)]
        for blocked in ("curd", "buttermilk", "paneer"):
            self.assertFalse(any(blocked in n for n in names))


# ----------------------------------------------------------------------
# 5. Recent-meal rotation (Phase 2A) still works with the expanded pools
# ----------------------------------------------------------------------

class TestPhase2ARotationStillWorksWithExpandedPools(unittest.TestCase):
    def test_generation_index_still_visits_every_carb_position(self):
        """Central's carbs pool grew from 4 to 5 items (Bafla added) - the
        SAME round-robin mechanism Phase 2A relies on must still cover every
        position across that many generations."""
        region_data = REGIONS["central"]
        pool_size = len(region_data["carbs"])
        profile = dict(BASE, region=REGION_CITY["central"])

        seen = set()
        for gen in range(pool_size):
            plan = mp.mock_provider(dict(profile), agent_context={
                "variety_context": {"generation_index": gen, "recent_proteins": [], "recent_carbs": []}
            })
            seen.add(plan["meals"][1]["ingredients"][0]["item"])
        self.assertEqual(len(seen), pool_size, f"expected all {pool_size} carbs visited, got {seen}")

    def test_recent_carb_exclusion_still_works_with_the_new_carb_present(self):
        profile = dict(BASE, region=REGION_CITY["central"])
        baseline = mp.mock_provider(dict(profile), agent_context={
            "variety_context": {"generation_index": 0, "recent_proteins": [], "recent_carbs": []}
        })
        recently_served = baseline["meals"][1]["ingredients"][0]["item"]

        deprioritized = mp.mock_provider(dict(profile), agent_context={
            "variety_context": {"generation_index": 0, "recent_proteins": [], "recent_carbs": [recently_served]}
        })
        self.assertNotEqual(deprioritized["meals"][1]["ingredients"][0]["item"], recently_served)

    def test_generation_index_visits_every_central_nonveg_protein_position(self):
        region_data = REGIONS["central"]
        pool_size = len(region_data["nonveg_protein"])
        profile = dict(BASE, region=REGION_CITY["central"], diet="Non-Vegetarian")

        seen = set()
        for gen in range(pool_size):
            plan = mp.mock_provider(dict(profile), agent_context={
                "variety_context": {"generation_index": gen, "recent_proteins": [], "recent_carbs": []}
            })
            seen.add(plan["meals"][1]["ingredients"][1]["item"])
        self.assertEqual(len(seen), pool_size, f"expected all {pool_size} positions visited, got {seen}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
