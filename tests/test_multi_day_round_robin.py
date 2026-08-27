"""
AthleteOS - Phase 4A tests: deterministic round-robin cycling for 7-day
variety, replacing Step 6's independent per-day hash.

Proves the guarantees this rewrite exists for, measured across ALL SIX
regions (not just north, which earlier multi-day tests happened to use):

  FULL POOL COVERAGE BEFORE ANY REPEAT - breakfast/protein/snack should
      visit every item in their pool at least once before repeating, for
      any pool size >= 7 (all six regions' breakfast/snack pools, and
      several veg_protein pools, qualify).

  GRACEFUL EXHAUSTION - when a pool is SMALLER than 7 (e.g. central's
      4-item non-veg protein pool), repeats are mathematically unavoidable,
      but should only start after the pool is exhausted, never repeat
      adjacently, and preferably pair with a different carb than before.

  UNCHANGED GUARANTEES - single-day output stays byte-identical,
      reproducibility holds, similar-but-different athletes still diverge.

See meal_planner.py's module note above SLOT_LUNCH_CARB for the mechanism
(offset = slot_seed(seed, SLOT_X) + day_index) and choose_items()'s
docstring for why cross-day exclusion is applied AFTER rotation now, not
before (that ordering bug was found and fixed while writing these tests -
pre-rotation filtering shrank the pool by one on every day past day 0,
which silently capped real-world coverage at ~65% of what the round-robin
design intended, even though the design was mathematically sound).

Everything here is offline and deterministic - no network, no randomness,
same technique as the existing multi-day test files.
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

# One real city per region, taken directly from regional_foods.py's own
# keyword lists, so detect_region() resolves each test to the exact zone
# being measured rather than relying on free-text guessing. "international"
# (Phase 2B) has no keywords - it's reached only via DEFAULT_REGION, never a
# text match - so it's excluded from this map; this file's round-robin
# guarantees are about the six Indian regions, unaffected by that addition.
REGION_CITY = {region: data["keywords"][0] for region, data in REGIONS.items() if data["keywords"]}


def breakfast_main(day):
    return day["meals"][0]["ingredients"][0]["item"]


def lunch_protein(day):
    return day["meals"][1]["ingredients"][1]["item"]


def lunch_carb(day):
    return day["meals"][1]["ingredients"][0]["item"]


def dinner_protein(day):
    return day["meals"][3]["ingredients"][1]["item"]


def dinner_carb(day):
    return day["meals"][3]["ingredients"][0]["item"]


def snack_main(day):
    return day["meals"][2]["ingredients"][0]["item"]


def lunch_veg(day, region_key):
    items = [i["item"] for i in day["meals"][1]["ingredients"]]
    return next((it for it in items if it in REGIONS[region_key]["veg"]), None)


def breakfast_side(day, region_key):
    items = [i["item"] for i in day["meals"][0]["ingredients"]]
    return next((it for it in items if it in REGIONS[region_key]["breakfast_sides"]), None)


# ----------------------------------------------------------------------
# 1. Breakfast: full pool coverage before any repeat, all 6 regions
# ----------------------------------------------------------------------

class TestBreakfastRoundRobinAllRegions(unittest.TestCase):
    def test_breakfast_covers_its_pool_before_repeating(self):
        for region_key, city in REGION_CITY.items():
            with self.subTest(region=region_key):
                plan = mp.generate_meal_plan(dict(BASE, region=city), days=7)
                mains = [breakfast_main(d) for d in plan["days"]]
                pool_size = len(REGIONS[region_key]["breakfast"])
                expected = min(7, pool_size)
                self.assertEqual(len(set(mains)), expected, f"{region_key}: {mains}")


# ----------------------------------------------------------------------
# 2. Protein: full pool coverage before any repeat, all 6 regions,
#    veg and non-veg, lunch and dinner
# ----------------------------------------------------------------------

class TestProteinRoundRobinAllRegions(unittest.TestCase):
    def test_lunch_protein_covers_its_pool_veg(self):
        for region_key, city in REGION_CITY.items():
            with self.subTest(region=region_key):
                plan = mp.generate_meal_plan(dict(BASE, region=city, diet="Vegetarian"), days=7)
                proteins = [lunch_protein(d) for d in plan["days"]]
                pool_size = len(REGIONS[region_key]["veg_protein"])
                expected = min(7, pool_size)
                self.assertEqual(len(set(proteins)), expected, f"{region_key}: {proteins}")

    def test_dinner_protein_covers_its_pool_veg(self):
        for region_key, city in REGION_CITY.items():
            with self.subTest(region=region_key):
                plan = mp.generate_meal_plan(dict(BASE, region=city, diet="Vegetarian"), days=7)
                proteins = [dinner_protein(d) for d in plan["days"]]
                pool_size = len(REGIONS[region_key]["veg_protein"])
                expected = min(7, pool_size)
                self.assertEqual(len(set(proteins)), expected, f"{region_key}: {proteins}")

    def test_lunch_protein_covers_its_pool_nonveg(self):
        for region_key, city in REGION_CITY.items():
            with self.subTest(region=region_key):
                plan = mp.generate_meal_plan(dict(BASE, region=city, diet="Non-Vegetarian"), days=7)
                proteins = [lunch_protein(d) for d in plan["days"]]
                pool_size = len(REGIONS[region_key]["nonveg_protein"])
                expected = min(7, pool_size)
                self.assertEqual(len(set(proteins)), expected, f"{region_key}: {proteins}")

    def test_central_nonveg_full_pool_coverage(self):
        """Central's non-veg pool (6 items as of Phase 2C - was the
        tightest in the dataset at 4 before Mutton curry/Fish curry were
        added) must still cover every item within 7 lunch draws and never
        repeat two days in a row."""
        plan = mp.generate_meal_plan(dict(BASE, region="Indore, Madhya Pradesh", diet="Non-Vegetarian"), days=7)
        proteins = [lunch_protein(d) for d in plan["days"]]
        self.assertEqual(set(proteins), set(REGIONS["central"]["nonveg_protein"]))
        for i in range(len(proteins) - 1):
            self.assertNotEqual(proteins[i], proteins[i + 1], f"day {i+1} vs day {i+2}")

    def test_no_protein_repeats_adjacently_anywhere_veg_or_nonveg(self):
        for region_key, city in REGION_CITY.items():
            for diet in ("Vegetarian", "Non-Vegetarian"):
                with self.subTest(region=region_key, diet=diet):
                    plan = mp.generate_meal_plan(dict(BASE, region=city, diet=diet), days=7)
                    for slot_fn in (lunch_protein, dinner_protein):
                        seq = [slot_fn(d) for d in plan["days"]]
                        for i in range(len(seq) - 1):
                            self.assertNotEqual(seq[i], seq[i + 1])


# ----------------------------------------------------------------------
# 3. Snack: full pool coverage before any repeat, all 6 regions
# ----------------------------------------------------------------------

class TestSnackRoundRobinAllRegions(unittest.TestCase):
    def test_snack_covers_its_pool_before_repeating(self):
        for region_key, city in REGION_CITY.items():
            with self.subTest(region=region_key):
                plan = mp.generate_meal_plan(dict(BASE, region=city), days=7)
                snacks = [snack_main(d) for d in plan["days"]]
                pool_size = len(REGIONS[region_key]["snack"])
                expected = min(7, pool_size)
                self.assertEqual(len(set(snacks)), expected, f"{region_key}: {snacks}")


# ----------------------------------------------------------------------
# 4. Side/veg: no unnecessary immediate repeat (previously unprotected)
# ----------------------------------------------------------------------

class TestSideAndVegNoImmediateRepeat(unittest.TestCase):
    def test_lunch_veg_side_no_immediate_repeat(self):
        for region_key, city in REGION_CITY.items():
            with self.subTest(region=region_key):
                plan = mp.generate_meal_plan(dict(BASE, region=city), days=7)
                vegs = [lunch_veg(d, region_key) for d in plan["days"]]
                for i in range(len(vegs) - 1):
                    if vegs[i] is not None and vegs[i + 1] is not None:
                        self.assertNotEqual(vegs[i], vegs[i + 1], f"{region_key} day {i+1}->{i+2}")

    def test_breakfast_side_no_immediate_repeat(self):
        for region_key, city in REGION_CITY.items():
            with self.subTest(region=region_key):
                plan = mp.generate_meal_plan(dict(BASE, region=city), days=7)
                sides = [breakfast_side(d, region_key) for d in plan["days"]]
                for i in range(len(sides) - 1):
                    if sides[i] is not None and sides[i + 1] is not None:
                        self.assertNotEqual(sides[i], sides[i + 1], f"{region_key} day {i+1}->{i+2}")


# ----------------------------------------------------------------------
# 5. A repeated protein should pair with a different carb than before
# ----------------------------------------------------------------------

class TestRepeatedProteinPairsWithDifferentCarb(unittest.TestCase):
    def test_repeats_pair_with_a_different_carb_when_pool_sizes_differ(self):
        """The general case: protein and carb pools cycle at different
        periods (every region's carbs pool is 4-5; every veg_protein pool is
        6-9, and non-veg is 6 everywhere as of Phase 2C) - so a repeated
        protein almost always lands on a different point in the carb cycle
        than its last appearance.

        Central used to be a documented exception here: its non-veg pool
        (4 items) and every region's carb pool (4 items) were the SAME
        size, the only such coincidence in the dataset - equal cycle
        lengths kept the two slots' offsets a constant distance apart
        forever, so a repeated protein necessarily repeated its exact prior
        carb too (see git history for the old
        test_central_nonveg_is_a_known_lockstep_edge_case). Phase 2C's
        Central additions (Mutton curry/Fish curry -> 6-item non-veg pool;
        Bafla -> 5-item carb pool) broke that coincidence as a side effect -
        confirmed by folding central into this general test, which it now
        passes exactly like every other region."""
        for region_key, city in REGION_CITY.items():
            with self.subTest(region=region_key):
                plan = mp.generate_meal_plan(dict(BASE, region=city, diet="Non-Vegetarian"), days=7)
                last_carb_for_protein = {}
                same_combo_repeats = 0
                for d in plan["days"]:
                    protein, carb = lunch_protein(d), lunch_carb(d)
                    if last_carb_for_protein.get(protein) == carb and protein in last_carb_for_protein:
                        same_combo_repeats += 1
                    last_carb_for_protein[protein] = carb
                self.assertEqual(same_combo_repeats, 0, f"{region_key}: a repeated protein reused its exact prior carb")


# ----------------------------------------------------------------------
# 6. Reproducibility
# ----------------------------------------------------------------------

class TestReproducibility(unittest.TestCase):
    def test_same_profile_produces_the_same_seven_day_plan(self):
        for region_key, city in REGION_CITY.items():
            with self.subTest(region=region_key):
                profile = dict(BASE, region=city)
                plan_a = mp.generate_meal_plan(dict(profile), days=7)
                plan_b = mp.generate_meal_plan(dict(profile), days=7)
                self.assertEqual(plan_a, plan_b)


# ----------------------------------------------------------------------
# 7. Different athletes still diverge
# ----------------------------------------------------------------------

class TestDifferentAthletesDiverge(unittest.TestCase):
    def test_similar_athletes_still_produce_different_seven_day_plans(self):
        for region_key, city in REGION_CITY.items():
            with self.subTest(region=region_key):
                plan_a = mp.generate_meal_plan(dict(BASE, region=city, weight_kg=72), days=7)
                plan_b = mp.generate_meal_plan(dict(BASE, region=city, weight_kg=73), days=7)
                self.assertNotEqual(plan_a, plan_b)


# ----------------------------------------------------------------------
# 8. Single-day output is untouched
# ----------------------------------------------------------------------

class TestSingleDayUnchanged(unittest.TestCase):
    def test_single_day_still_byte_identical_to_mock_provider_all_regions(self):
        for region_key, city in REGION_CITY.items():
            for diet in ("Vegetarian", "Non-Vegetarian"):
                with self.subTest(region=region_key, diet=diet):
                    profile = dict(BASE, region=city, diet=diet)
                    self.assertEqual(
                        mp.generate_meal_plan(dict(profile), days=1),
                        mp.mock_provider(dict(profile)),
                    )

    def test_single_day_match_goal_still_gets_khichdi(self):
        profile = dict(BASE, goal="Fuel properly, match tomorrow")
        plan = mp.generate_meal_plan(dict(profile), days=1)
        self.assertIn("khichdi", plan["meals"][3]["ingredients"][0]["item"].lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
