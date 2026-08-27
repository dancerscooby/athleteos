"""
AthleteOS - Tests for the age/developmental-stage audit fixes.

Covers all three targeted fixes from the audit (see the audit's own
findings for context):
    1. rag.build_retrieval_query() includes age -> a minor's query actually
       differs from an adult's, and real retrieval surfaces
       youth_athlete_nutrition.md for a minor.
    2. meal_planner.build_prompt() has an AGE / DEVELOPMENT RULE section
       that (a) is always present and (b) correctly reflects this athlete's
       age.
    3. mock_provider() actually changes FOOD SELECTION (portion sizing,
       serving counts, muscle-gain boosters) for a minor - not just the
       calorie number, which is what the audit found was already working
       (via nutrition_tool) while everything else silently wasn't.

The central proof requested: run the SAME profile (weight/height/goal/
training) at age 10 vs age 25 and show the plans actually differ in more
than just calories - and that adult behaviour (age >= 18, or age absent)
is unchanged from before these fixes.

RAG retrieval tests use the REAL retriever/FAISS/reranker pipeline
(offline, local models, no network) - same approach as the existing
test_orchestrator.py RAG test. No live OpenRouter/API calls anywhere.
"""

import tempfile
import unittest
from pathlib import Path

import meal_planner as mp
import memory_manager as mm
from orchestrator import run_with_trace
from rag import _developmental_stage_phrase, build_retrieval_query


class MemoryIsolatedTestCase(unittest.TestCase):
    """Redirects memory_manager's storage into a fresh temp dir per test -
    same isolation technique used by the other memory-touching test suites."""

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


CHILD = dict(
    age=10, height_cm=145, weight_kg=32, sex="Male", goal="muscle growth",
    diet="Vegetarian", training_days_per_week=6, training_intensity="High",
    region="Lucknow, Uttar Pradesh", allergies="none", budget_per_day=None,
    food_preferences=None, foods_to_avoid=None, notes=None,
)


def adult_version(profile=CHILD, age=25):
    return dict(profile, age=age)


# ----------------------------------------------------------------------
# 1a. build_retrieval_query() includes age
# ----------------------------------------------------------------------

class TestRetrievalQueryIncludesAge(unittest.TestCase):
    def test_child_and_adult_produce_different_queries(self):
        child_query = build_retrieval_query(CHILD)
        adult_query = build_retrieval_query(adult_version())
        self.assertNotEqual(child_query, adult_query)
        self.assertIn("10 year old child athlete", child_query)
        self.assertNotIn("year old", adult_query)

    def test_adolescent_gets_a_different_phrase_than_child(self):
        self.assertEqual(_developmental_stage_phrase(10), "10 year old child athlete")
        self.assertEqual(_developmental_stage_phrase(15), "15 year old adolescent athlete")

    def test_adult_and_missing_age_both_produce_no_phrase(self):
        self.assertEqual(_developmental_stage_phrase(25), "")
        self.assertEqual(_developmental_stage_phrase(18), "")
        self.assertEqual(_developmental_stage_phrase(None), "")

    def test_age_contributes_nothing_to_an_adult_query(self):
        """Age must add nothing for an adult.

        Asserted by comparing the adult query against the same profile with
        no age at all, rather than against a hard-coded whole-query string:
        region is also part of the query now (a later, separate fix), and
        that is not this test's concern."""
        profile = adult_version()
        with_adult_age = build_retrieval_query(profile)
        without_age = build_retrieval_query(dict(profile, age=None))
        self.assertEqual(with_adult_age, without_age)
        self.assertNotIn("year old", with_adult_age)


# ----------------------------------------------------------------------
# 1b. Real retrieval surfaces the youth document for a minor (offline)
# ----------------------------------------------------------------------

class TestRealRetrievalForAMinor(MemoryIsolatedTestCase):
    def test_child_profile_retrieves_the_youth_document(self):
        trace = run_with_trace(CHILD, athlete_id="child-retrieval-test")
        sources = [c["source"] for c in trace["rag_chunks"]]
        self.assertIn("youth_athlete_nutrition.md", sources)

    def test_adult_profile_with_same_details_does_not_prioritize_it(self):
        trace = run_with_trace(adult_version(), athlete_id="adult-retrieval-test")
        sources = [c["source"] for c in trace["rag_chunks"]]
        # The adult query never mentions age/child/adolescent, so the top
        # chunk should not be the youth document (it may still appear lower
        # if the corpus is small, but must not be the single best match).
        self.assertNotEqual(sources[0], "youth_athlete_nutrition.md")

    def test_child_and_adult_retrieve_different_chunks(self):
        child_trace = run_with_trace(CHILD, athlete_id="child-diff-test")
        adult_trace = run_with_trace(adult_version(), athlete_id="adult-diff-test")
        child_ids = [c["id"] for c in child_trace["rag_chunks"]]
        adult_ids = [c["id"] for c in adult_trace["rag_chunks"]]
        self.assertNotEqual(child_ids, adult_ids)


# ----------------------------------------------------------------------
# 2. build_prompt() has an AGE / DEVELOPMENT RULE section
# ----------------------------------------------------------------------

class TestPromptAgeRule(unittest.TestCase):
    def test_age_rule_section_is_present(self):
        prompt = mp.build_prompt(CHILD)
        self.assertIn("AGE / DEVELOPMENT RULE", prompt)

    def test_age_rule_reflects_this_athletes_actual_age(self):
        child_prompt = mp.build_prompt(CHILD)
        adult_prompt = mp.build_prompt(adult_version())
        self.assertIn("This athlete's age is 10.", child_prompt)
        self.assertIn("This athlete's age is 25.", adult_prompt)

    def test_age_rule_mentions_not_a_smaller_adult_and_professional_involvement(self):
        prompt = mp.build_prompt(CHILD)
        self.assertIn("not a smaller adult", prompt)
        self.assertIn("pediatrician", prompt)

    def test_age_rule_does_not_invent_specific_numbers(self):
        """The rule must defer to reference material / precomputed values,
        not state its own calorie/protein figures for a minor."""
        prompt = mp.build_prompt(CHILD)
        rule_section = prompt.split("AGE / DEVELOPMENT RULE")[1].split("TASK")[0]
        self.assertIn("depends on", rule_section)
        self.assertNotRegex(rule_section, r"\d+\s*(kcal|calories|g protein)")

    def test_present_and_present_regardless_of_age_value(self):
        """The section itself is always emitted (existing DIET RULE
        precedent) - only its content is age-conditional, matching how the
        LLM is expected to self-gate on the stated age."""
        for age in (10, 17, 18, 25, 60):
            prompt = mp.build_prompt(dict(CHILD, age=age))
            self.assertIn("AGE / DEVELOPMENT RULE", prompt)


# ----------------------------------------------------------------------
# 3. mock_provider() food selection actually changes for a minor
# ----------------------------------------------------------------------

class TestMockProviderAgeAwareFoodSelection(unittest.TestCase):
    def test_portion_capped_at_moderate_for_a_minor_with_heavy_training_load(self):
        """CHILD's training load (6 days/week, High) would compute "large"
        via training_load() alone - confirm that, then confirm the minor's
        plan is capped at "moderate" while an identical adult reaches "large"."""
        factor, uncapped_portion = mp.training_load(CHILD)
        self.assertEqual(uncapped_portion, "large")  # sanity: this profile WOULD be "large" if not a minor

        child_plan = mp.mock_provider(dict(CHILD))
        adult_plan = mp.mock_provider(adult_version())

        self.assertIn("moderate training load", child_plan["day_summary"]["notes"])
        self.assertIn("large training load", adult_plan["day_summary"]["notes"])

    def test_child_breakfast_uses_smaller_plate_than_adult(self):
        child_plan = mp.mock_provider(dict(CHILD))
        adult_plan = mp.mock_provider(adult_version())
        child_qty = child_plan["meals"][0]["ingredients"][0]["quantity"]
        adult_qty = adult_plan["meals"][0]["ingredients"][0]["quantity"]
        self.assertEqual(child_qty, "1 plate")
        self.assertEqual(adult_qty, "1 large plate")

    def test_child_gets_fewer_carb_servings_at_lunch_and_dinner(self):
        child_plan = mp.mock_provider(dict(CHILD))
        adult_plan = mp.mock_provider(adult_version())
        for meal_index in (1, 3):  # Lunch, Dinner
            child_carb = child_plan["meals"][meal_index]["ingredients"][0]["quantity"]
            adult_carb = adult_plan["meals"][meal_index]["ingredients"][0]["quantity"]
            self.assertNotEqual(child_carb, adult_carb)

    def test_muscle_gain_snack_booster_skipped_for_a_minor(self):
        """Use Non-Vegetarian so the booster ("Boiled eggs") is a distinct
        item from the region's default snack picks, making its
        presence/absence unambiguous (no dedup with an already-picked item)."""
        child_nonveg = dict(CHILD, diet="Non-Vegetarian")
        adult_nonveg = adult_version(child_nonveg)

        child_plan = mp.mock_provider(child_nonveg)
        adult_plan = mp.mock_provider(adult_nonveg)

        child_snack_items = [i["item"] for i in child_plan["meals"][2]["ingredients"]]
        adult_snack_items = [i["item"] for i in adult_plan["meals"][2]["ingredients"]]

        self.assertNotIn("Boiled eggs", child_snack_items)
        self.assertIn("Boiled eggs", adult_snack_items)

    def test_carb_servings_helper_skips_muscle_gain_bump_for_a_minor(self):
        self.assertEqual(mp._carb_servings("moderate", "muscle_gain", is_minor=False), "4 servings")
        self.assertEqual(mp._carb_servings("moderate", "muscle_gain", is_minor=True), "3 servings")

    def test_day_summary_notes_say_growth_appropriate_for_a_minor(self):
        child_plan = mp.mock_provider(dict(CHILD))
        adult_plan = mp.mock_provider(adult_version())
        self.assertIn("Growth-appropriate portions for a young athlete", child_plan["day_summary"]["notes"])
        self.assertNotIn("Growth-appropriate", adult_plan["day_summary"]["notes"])


# ----------------------------------------------------------------------
# The central comparison: age changes more than calories
# ----------------------------------------------------------------------

class TestChildVsAdultFullComparison(MemoryIsolatedTestCase):
    def test_child_and_adult_plans_differ_beyond_calories(self):
        """Full pipeline (orchestrator -> RAG + nutrition_tool -> mock),
        matching how a real request actually flows - calling mock_provider()
        directly with no agent_context wouldn't exercise nutrition_tool at
        all, since that's where age-aware calories actually come from."""
        child_trace = run_with_trace(CHILD, athlete_id="child-full-compare")
        adult_trace = run_with_trace(adult_version(), athlete_id="adult-full-compare")
        child_plan, adult_plan = child_trace["plan"], adult_trace["plan"]

        self.assertNotEqual(
            child_plan["day_summary"]["total_calories"],
            adult_plan["day_summary"]["total_calories"],
        )
        # The actual point of this audit fix: MEALS themselves (not just the
        # calorie number) must differ.
        self.assertNotEqual(child_plan["meals"], adult_plan["meals"])
        # And RAG context itself must have differed too (see the retrieval
        # tests above) - the whole flow, not just one stage.
        self.assertNotEqual(child_trace["rag_chunks"], adult_trace["rag_chunks"])

    def test_mock_provider_alone_still_changes_food_selection_without_agent_context(self):
        """Even WITHOUT the orchestrator/nutrition_tool (agent_context=None,
        e.g. the CLI path), age must still change portion sizing/servings -
        this is mock_provider's own fix, independent of the tool-based
        calorie fix."""
        child_plan = mp.mock_provider(dict(CHILD))
        adult_plan = mp.mock_provider(adult_version())
        self.assertNotEqual(child_plan["meals"], adult_plan["meals"])

    def test_diet_and_allergy_rules_still_apply_for_a_minor(self):
        """These fixes must not weaken existing safety rules - a minor with
        a dairy allergy must still get no dairy, exactly like an adult."""
        child_allergic = dict(CHILD, allergies="dairy")
        plan = mp.mock_provider(child_allergic)
        names = [i["item"].lower() for m in plan["meals"] for i in m["ingredients"]]
        for blocked in ("paneer", "curd", "milk", "cheese", "ghee"):
            self.assertFalse(any(blocked in n for n in names), f"{blocked!r} leaked into a minor's plan")

    def test_vegetarian_rule_still_applies_for_a_minor(self):
        plan = mp.mock_provider(dict(CHILD))  # Vegetarian, no explicit egg preference
        names = [i["item"].lower() for m in plan["meals"] for i in m["ingredients"]]
        self.assertFalse(any("egg" in n for n in names))


# ----------------------------------------------------------------------
# Adult behaviour is unchanged
# ----------------------------------------------------------------------

class TestAdultBehaviourUnchanged(unittest.TestCase):
    def test_missing_age_is_not_treated_as_a_minor(self):
        """age >= 18 and age absent (None) both mean is_minor=False, so
        neither gets the child-specific treatment.

        This asserts the is_minor guarantee itself rather than whole-plan
        equality: since Phase 2, age feeds the per-profile variety rotation,
        so a 25-year-old and an age-unknown athlete legitimately eat
        different (equally adult-sized) foods. What must stay identical is
        every minor-specific behaviour."""
        with_adult_age = mp.mock_provider(adult_version())
        without_age = mp.mock_provider(dict(adult_version(), age=None))

        for plan in (with_adult_age, without_age):
            self.assertNotIn("Growth-appropriate", plan["day_summary"]["notes"])
            self.assertIn("large training load", plan["day_summary"]["notes"])
            # Adult carb servings include the muscle_gain bulking bump.
            self.assertEqual(plan["meals"][1]["ingredients"][0]["quantity"], "5 servings")
            self.assertEqual(plan["meals"][0]["ingredients"][0]["quantity"], "1 large plate")

    def test_exactly_18_is_treated_as_an_adult(self):
        plan = mp.mock_provider(adult_version(age=18))
        self.assertNotIn("Growth-appropriate", plan["day_summary"]["notes"])

    def test_adult_muscle_gain_booster_and_full_portions_still_work(self):
        """Regression guard: the existing adult-oriented behaviour this
        audit explicitly said must be preserved."""
        plan = mp.mock_provider(adult_version(dict(CHILD, diet="Non-Vegetarian")))
        snack_items = [i["item"] for i in plan["meals"][2]["ingredients"]]
        self.assertIn("Boiled eggs", snack_items)
        self.assertEqual(plan["meals"][0]["ingredients"][0]["quantity"], "1 large plate")


if __name__ == "__main__":
    unittest.main(verbosity=2)
