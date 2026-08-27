"""
AthleteOS - Phase 2A tests: memory-aware meal variety/rotation.

Phase 2A closes a write-only loop: episodic_memory.py already recorded, on
every generation, a compact history (episodic entries, and "latest_plan"'s
deduplicated protein_sources/carb_sources) - but nothing ever read it back
into the NEXT generation. Two things were added, both by REUSING existing
mechanisms rather than inventing new ones:

  1. generation_index - orchestrator.py counts this athlete's past episodic
     entries (memory_manager.list("episodic", ...)) and hands the count to
     meal_planner as agent_context["variety_context"]["generation_index"].
     meal_planner then uses it exactly like a multi-day plan's day_index
     (offset = slot_seed(seed, SLOT) + day_index) - the SAME round-robin
     cycle math Phase 4A already proved covers every pool position once
     before any repeat, just continued across separate requests instead of
     restarting at 0 every time.

  2. recent_proteins/recent_carbs - orchestrator.py reads this athlete's
     "latest_plan" (memory_manager.get(...)) and hands its protein_sources/
     carb_sources to meal_planner, which folds them into the SAME `exclude`
     lists _build_day() already builds from `prev` for cross-day variety -
     not a new exclusion mechanism, a wider input to the existing one.

Two test groups:
  - Fast, meal_planner-level unit tests (no memory, no RAG): construct
    agent_context["variety_context"] by hand and prove the mechanism itself.
  - Slower, end-to-end tests via the real create_meal_plan() endpoint (real
    orchestrator + real memory + real RAG, memory-isolated to a temp dir):
    prove the actual wiring - repeated requests for one athlete differ,
    two athletes never affect each other, first-ever request is unchanged.

Everything here is offline and deterministic - no network, no random.choice(),
no wall-clock-derived values anywhere in the mechanism under test.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import memory_manager as mm
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

# "international" (Phase 2B) has no keywords - it's reached only via
# DEFAULT_REGION, never a text match - so it's excluded from this map.
REGION_CITY = {region: data["keywords"][0] for region, data in REGIONS.items() if data["keywords"]}


def lunch_protein(plan):
    return plan["meals"][1]["ingredients"][1]["item"]


def lunch_carb(plan):
    return plan["meals"][1]["ingredients"][0]["item"]


def dinner_protein(plan):
    return plan["meals"][3]["ingredients"][1]["item"]


def dinner_carb(plan):
    return plan["meals"][3]["ingredients"][0]["item"]


def _with_generation_index(n, recent_proteins=None, recent_carbs=None):
    return {"variety_context": {
        "generation_index": n,
        "recent_proteins": recent_proteins or [],
        "recent_carbs": recent_carbs or [],
    }}


# ----------------------------------------------------------------------
# 1. First-generation compatibility (generation_index=0 / no variety_context)
# ----------------------------------------------------------------------

class TestFirstGenerationUnchanged(unittest.TestCase):
    def test_no_agent_context_matches_generation_index_zero(self):
        """agent_context=None (every pre-Phase-2A caller) must produce the
        exact same plan as an explicit generation_index=0 with no recent
        items - proving generation_index=0 is a true no-op, not just "close
        enough"."""
        without_context = mp.mock_provider(dict(BASE))
        with_explicit_zero = mp.mock_provider(dict(BASE), agent_context=_with_generation_index(0))
        self.assertEqual(without_context, with_explicit_zero)

    def test_empty_agent_context_also_matches(self):
        """An agent_context with tool_results/rag_context but no
        variety_context key at all (e.g. an older caller) must behave
        identically too."""
        without_context = mp.mock_provider(dict(BASE))
        with_partial = mp.mock_provider(dict(BASE), agent_context={"rag_context": "x", "tool_results": {}})
        self.assertEqual(without_context, with_partial)

    def test_multi_day_generation_index_zero_matches_pre_phase2a(self):
        without_context = mp.mock_provider_multi_day(dict(BASE), days=7)
        with_explicit_zero = mp.mock_provider_multi_day(dict(BASE), days=7, agent_context=_with_generation_index(0))
        self.assertEqual(without_context, with_explicit_zero)


# ----------------------------------------------------------------------
# 2. Repeated generations for the same athlete CAN differ
# ----------------------------------------------------------------------

class TestRepeatedGenerationsCanDiffer(unittest.TestCase):
    def test_generation_index_visits_every_position_before_repeating(self):
        """Same guarantee Phase 4A proved for day_index, now proved for
        generation_index: cycling it across a pool's own size must visit
        every distinct value at least once before any repeat - not just
        "sometimes different"."""
        region_data = REGIONS["south"]
        pool_size = len(region_data["veg_protein"])
        profile = dict(BASE, region=REGION_CITY["south"])

        seen = set()
        for gen in range(pool_size):
            plan = mp.mock_provider(dict(profile), agent_context=_with_generation_index(gen))
            seen.add(lunch_protein(plan))
        self.assertEqual(len(seen), pool_size, f"expected all {pool_size} positions visited, got {seen}")

    def test_a_specific_repeated_request_differs_from_the_first(self):
        plan_0 = mp.mock_provider(dict(BASE), agent_context=_with_generation_index(0))
        plan_1 = mp.mock_provider(dict(BASE), agent_context=_with_generation_index(1))
        self.assertNotEqual(plan_0["meals"], plan_1["meals"])


# ----------------------------------------------------------------------
# 3. Finite pools eventually repeat HONESTLY (not infinite variety)
# ----------------------------------------------------------------------

class TestFinitePoolsRepeatHonestly(unittest.TestCase):
    def test_cycling_past_pool_size_repeats_a_prior_position(self):
        region_data = REGIONS["central"]  # a genuinely small nonveg_protein pool
        pool_size = len(region_data["nonveg_protein"])
        profile = dict(BASE, region=REGION_CITY["central"], diet="Non-Vegetarian")

        picks = [
            dinner_protein(mp.mock_provider(dict(profile), agent_context=_with_generation_index(gen)))
            for gen in range(pool_size + 1)
        ]
        # pool_size+1 requests from a pool of exactly pool_size items MUST
        # repeat at least once - this test would fail if the mechanism
        # somehow claimed unlimited variety from a finite candidate list.
        self.assertLess(len(set(picks)), len(picks))


# ----------------------------------------------------------------------
# 4. Recently-served protein/carb sources are deprioritized
# ----------------------------------------------------------------------

class TestRecentItemsDeprioritized(unittest.TestCase):
    def test_recent_protein_is_excluded_when_pool_allows_it(self):
        region_data = REGIONS["south"]
        self.assertGreater(len(region_data["veg_protein"]), 1)  # precondition for a meaningful test
        profile = dict(BASE, region=REGION_CITY["south"])

        baseline = mp.mock_provider(dict(profile), agent_context=_with_generation_index(0))
        recently_served = lunch_protein(baseline)

        deprioritized = mp.mock_provider(
            dict(profile),
            agent_context=_with_generation_index(0, recent_proteins=[recently_served]),
        )
        self.assertNotEqual(lunch_protein(deprioritized), recently_served)

    def test_recent_carb_is_excluded_when_pool_allows_it(self):
        region_data = REGIONS["south"]
        self.assertGreater(len(region_data["carbs"]), 1)
        profile = dict(BASE, region=REGION_CITY["south"])

        baseline = mp.mock_provider(dict(profile), agent_context=_with_generation_index(0))
        recently_served = lunch_carb(baseline)

        deprioritized = mp.mock_provider(
            dict(profile),
            agent_context=_with_generation_index(0, recent_carbs=[recently_served]),
        )
        self.assertNotEqual(lunch_carb(deprioritized), recently_served)

    def test_excluding_every_item_still_produces_a_meal(self):
        """choose_items()'s existing "never let exclusion empty the list
        entirely" fallback must still apply - deprioritization, not a hard
        block that could break generation."""
        region_data = REGIONS["central"]
        profile = dict(BASE, region=REGION_CITY["central"], diet="Non-Vegetarian")
        all_nonveg = region_data["nonveg_protein"]

        plan = mp.mock_provider(
            dict(profile),
            agent_context=_with_generation_index(0, recent_proteins=list(all_nonveg)),
        )
        self.assertIn(dinner_protein(plan), all_nonveg)  # still a valid, real dish - not empty/crashed


# ----------------------------------------------------------------------
# 5. Safety rules always win over variety
# ----------------------------------------------------------------------

class TestSafetyRulesAlwaysWinOverVariety(unittest.TestCase):
    def test_allergy_never_appears_across_any_generation_index(self):
        profile = dict(BASE, allergies="dairy")
        for gen in range(10):
            plan = mp.mock_provider(dict(profile), agent_context=_with_generation_index(gen))
            all_items = " ".join(i["item"].lower() for m in plan["meals"] for i in m["ingredients"])
            for bad in ("curd", "paneer", "milk", "ghee", "buttermilk"):
                self.assertNotIn(bad, all_items, msg=f"generation_index={gen} leaked {bad!r}")

    def test_diet_never_violated_across_any_generation_index(self):
        profile = dict(BASE, diet="Vegetarian")
        nonveg_words = ("chicken", "egg", "fish", "prawn", "mutton", "goat", "pork", "duck")
        for gen in range(10):
            plan = mp.mock_provider(dict(profile), agent_context=_with_generation_index(gen))
            all_items = " ".join(i["item"].lower() for m in plan["meals"] for i in m["ingredients"])
            for bad in nonveg_words:
                self.assertNotIn(bad, all_items, msg=f"generation_index={gen} leaked {bad!r}")

    def test_foods_to_avoid_never_appears_across_any_generation_index(self):
        profile = dict(BASE, foods_to_avoid="rajma")
        for gen in range(10):
            plan = mp.mock_provider(dict(profile), agent_context=_with_generation_index(gen))
            all_items = " ".join(i["item"].lower() for m in plan["meals"] for i in m["ingredients"])
            self.assertNotIn("rajma", all_items, msg=f"generation_index={gen} leaked rajma")

    def test_explicit_preference_still_honoured_across_generation_index(self):
        profile = dict(BASE, food_preferences="paneer")
        for gen in range(5):
            plan = mp.mock_provider(dict(profile), agent_context=_with_generation_index(gen))
            all_items = [i["item"] for m in plan["meals"] for i in m["ingredients"]]
            self.assertTrue(
                any("paneer" in item.lower() for item in all_items),
                msg=f"generation_index={gen} dropped the explicit paneer preference",
            )


# ----------------------------------------------------------------------
# 6. Age/developmental-stage behaviour remains intact
# ----------------------------------------------------------------------

class TestAgeDevelopmentalBehaviourIntact(unittest.TestCase):
    def test_minor_portion_cap_holds_across_generation_index(self):
        """A minor's portion is capped at "moderate" regardless of training
        load (meal_planner.mock_provider()'s is_minor logic) - this must
        still hold no matter which generation_index the rotation lands on.
        Checked via the carb serving count, which _carb_servings() ties
        directly to portion tier: capped at 3 servings for a minor even
        when training_days_per_week=7/High would give an adult 4 (or 5 with
        a muscle_gain goal)."""
        minor_profile = dict(BASE, age=15, training_days_per_week=7, training_intensity="High")
        for gen in range(6):
            plan = mp.mock_provider(dict(minor_profile), agent_context=_with_generation_index(gen))
            lunch_carb_qty = plan["meals"][1]["ingredients"][0]["quantity"]
            serving_count = int(lunch_carb_qty.split()[0])
            self.assertLessEqual(serving_count, 3, msg=f"generation_index={gen} exceeded the minor portion cap")

    def test_adult_behaviour_unaffected_by_generation_index(self):
        adult_profile = dict(BASE, age=30, training_days_per_week=7, training_intensity="High")
        for gen in range(6):
            plan = mp.mock_provider(dict(adult_profile), agent_context=_with_generation_index(gen))
            self.assertEqual(len(plan["meals"]), 4)


# ----------------------------------------------------------------------
# 7. Existing multi-day variety still works, now generation-index-aware
# ----------------------------------------------------------------------

class TestMultiDayVarietyStillWorks(unittest.TestCase):
    def test_seven_day_plan_still_covers_full_pool_before_repeating(self):
        """The Phase 4A guarantee (full pool coverage before any repeat
        within one multi-day call) must survive Phase 2A unchanged when
        generation_index=0 (the default, e.g. a brand new athlete)."""
        region_data = REGIONS["south"]
        profile = dict(BASE, region=REGION_CITY["south"])
        plan = mp.mock_provider_multi_day(dict(profile), days=7)
        proteins = {lunch_protein(day) for day in plan["days"]}
        self.assertGreaterEqual(len(proteins), min(7, len(region_data["veg_protein"])))

    def test_generation_index_shifts_the_whole_week_forward(self):
        """A nonzero generation_index (an athlete requesting ANOTHER 7-day
        plan) must continue the rotation, not restart it - day 1 of the new
        week should differ from day 1 of the first week when the pool is
        larger than 1."""
        profile = dict(BASE, region=REGION_CITY["south"])
        week_1 = mp.mock_provider_multi_day(dict(profile), days=7)
        week_2 = mp.mock_provider_multi_day(dict(profile), days=7, agent_context=_with_generation_index(7))
        self.assertNotEqual(lunch_protein(week_1["days"][0]), lunch_protein(week_2["days"][0]))

    def test_day_labels_and_match_override_unaffected_by_generation_index(self):
        """"Day" labels must stay 1..N regardless of generation_index (it
        shifts the rotation, not the reported day numbers)."""
        profile = dict(BASE, region=REGION_CITY["south"])
        plan = mp.mock_provider_multi_day(dict(profile), days=7, agent_context=_with_generation_index(20))
        self.assertEqual([d["day"] for d in plan["days"]], [1, 2, 3, 4, 5, 6, 7])


# ----------------------------------------------------------------------
# 8. End-to-end: the real create_meal_plan() endpoint, real orchestrator,
#    real memory (isolated to a temp dir), real RAG
# ----------------------------------------------------------------------

class MemoryIsolatedTestCase(unittest.TestCase):
    """Redirects memory_manager's storage into a fresh temp dir per test -
    same isolation technique as every other memory-touching test suite."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmpdir.name)
        self._originals = {
            "MEMORY_DIR": mm.MEMORY_DIR,
            "PROFILES_FILE": mm.PROFILES_FILE,
            "PREFERENCES_FILE": mm.PREFERENCES_FILE,
            "LATEST_PLAN_FILE": mm.LATEST_PLAN_FILE,
            "RECIPES_FILE": mm.RECIPES_FILE,
            "EPISODES_DIR": mm.EPISODES_DIR,
            "CACHE_DIR": mm.CACHE_DIR,
            "UPDATABLE_CATEGORIES": mm.UPDATABLE_CATEGORIES,
        }
        mm.MEMORY_DIR = tmp_path
        mm.PROFILES_FILE = tmp_path / "profiles.json"
        mm.PREFERENCES_FILE = tmp_path / "preferences.json"
        mm.LATEST_PLAN_FILE = tmp_path / "latest_plans.json"
        mm.RECIPES_FILE = tmp_path / "recipes.json"
        mm.EPISODES_DIR = tmp_path / "episodes"
        mm.CACHE_DIR = tmp_path / "cache"
        mm.UPDATABLE_CATEGORIES = {
            "profile": mm.PROFILES_FILE,
            "preference": mm.PREFERENCES_FILE,
            "latest_plan": mm.LATEST_PLAN_FILE,
            "recipes": mm.RECIPES_FILE,
        }

    def tearDown(self):
        for name, value in self._originals.items():
            setattr(mm, name, value)
        self._tmpdir.cleanup()


ATHLETE_A = "11111111-1111-4111-8111-111111111111"
ATHLETE_B = "22222222-2222-4222-8222-222222222222"


class TestEndToEndMemoryAwareVariety(MemoryIsolatedTestCase):
    def test_first_ever_request_is_unaffected_by_phase_2a(self):
        """A brand-new athlete's very first request must produce exactly
        the plan generate_meal_plan() would with generation_index=0 and no
        recent items - i.e. Phase 2A must be a true no-op on request one."""
        request = MealPlanRequest(**dict(BASE, region="Lucknow, Uttar Pradesh", athlete_id=ATHLETE_A))
        response = create_meal_plan(request)
        # Compare against a profile built from the request's OWN validated
        # model_dump() (matching test_api_orchestrator_integration.py's
        # established pattern) rather than the raw BASE dict - Pydantic
        # coerces e.g. height_cm=178 (int) to 178.0 (float), which alone
        # changes variety_seed()'s hash input and would falsely look like a
        # Phase 2A behaviour change if compared against the raw dict.
        profile = {k: v for k, v in request.model_dump().items() if k != "athlete_id"}
        direct = mp.mock_provider(profile, agent_context=_with_generation_index(0))
        self.assertEqual(
            [m["ingredients"] for m in response["meals"]],
            [m["ingredients"] for m in direct["meals"]],
        )

    def test_repeated_requests_for_the_same_athlete_can_differ(self):
        request = MealPlanRequest(**dict(BASE, region="Lucknow, Uttar Pradesh", athlete_id=ATHLETE_A))
        first = create_meal_plan(request)
        second = create_meal_plan(request)
        self.assertNotEqual(
            [m["ingredients"] for m in first["meals"]],
            [m["ingredients"] for m in second["meals"]],
        )

    def test_recently_served_protein_is_deprioritized_on_the_next_request(self):
        request = MealPlanRequest(**dict(BASE, region="Lucknow, Uttar Pradesh", athlete_id=ATHLETE_A))
        first = create_meal_plan(request)
        first_lunch_protein = first["meals"][1]["ingredients"][1]["item"]

        second = create_meal_plan(request)
        second_lunch_protein = second["meals"][1]["ingredients"][1]["item"]

        self.assertNotEqual(first_lunch_protein, second_lunch_protein)

    def test_two_athletes_have_independent_generation_histories(self):
        """Athlete A generating several plans must never influence athlete
        B's first plan - B's history is empty regardless of how many times
        A has generated."""
        request_a = MealPlanRequest(**dict(BASE, region="Lucknow, Uttar Pradesh", athlete_id=ATHLETE_A))
        for _ in range(3):
            create_meal_plan(request_a)  # build up A's history

        request_b = MealPlanRequest(**dict(BASE, region="Lucknow, Uttar Pradesh", athlete_id=ATHLETE_B))
        b_response = create_meal_plan(request_b)
        # See test_first_ever_request_is_unaffected_by_phase_2a for why the
        # comparison profile is derived from model_dump(), not the raw dict.
        b_profile = {k: v for k, v in request_b.model_dump().items() if k != "athlete_id"}
        b_direct = mp.mock_provider(b_profile, agent_context=_with_generation_index(0))
        self.assertEqual(
            [m["ingredients"] for m in b_response["meals"]],
            [m["ingredients"] for m in b_direct["meals"]],
        )

    def test_one_athletes_meals_cannot_leak_into_anothers_exclusion(self):
        """Even after A has a rich history, B's variety_context (read via
        orchestrator.run_with_trace) must be scoped to B's own athlete_id -
        checked directly against the real trace, not inferred from output."""
        import orchestrator

        request_a = MealPlanRequest(**dict(BASE, region="Lucknow, Uttar Pradesh", athlete_id=ATHLETE_A))
        for _ in range(3):
            create_meal_plan(request_a)

        trace_b = orchestrator.run_with_trace(
            dict(BASE, region="Lucknow, Uttar Pradesh"),
            athlete_id=ATHLETE_B,
        )
        self.assertEqual(trace_b["variety_context"]["generation_index"], 0)
        self.assertEqual(trace_b["variety_context"]["recent_proteins"], [])
        self.assertEqual(trace_b["variety_context"]["recent_carbs"], [])

    def test_generation_index_grows_with_episodic_history(self):
        import orchestrator

        request = MealPlanRequest(**dict(BASE, region="Lucknow, Uttar Pradesh", athlete_id=ATHLETE_A))
        create_meal_plan(request)
        create_meal_plan(request)

        trace = orchestrator.run_with_trace(
            dict(BASE, region="Lucknow, Uttar Pradesh"),
            athlete_id=ATHLETE_A,
        )
        self.assertEqual(trace["variety_context"]["generation_index"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
