"""
AthleteOS - Tests for orchestrator.py (Agent v1, rule-based orchestration).

Two groups:

  - Fast, pure-logic tests (no model loading, no I/O): decide_tools() and
    the budget-parsing helper, tested directly as ordinary functions.

  - End-to-end tests (slower - load the local embedding/reranker models):
    run_with_trace() / orchestrate() against real profiles, covering all
    four tool-selection combinations explicitly required:
        - Nutrition Tool only
        - Budget Tool only
        - both
        - neither
    plus memory isolation, RAG retrieval, and contract compliance.

Every test that touches memory runs against a temporary memory/ directory -
never the project's real one - using the same isolation technique as the
other memory test suites.

Run with:
    python3 -m unittest tests.test_orchestrator -v
"""

import tempfile
import unittest
from pathlib import Path

import memory_manager as mm
import orchestrator as orch
from tools import build_default_registry


class MemoryIsolatedTestCase(unittest.TestCase):
    """Redirects memory_manager's storage into a fresh temp dir per test."""

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


BASE_PROFILE = dict(
    age=22, height_cm=178, weight_kg=72, sex="Male",
    goal="Improve football performance",
    diet="Vegetarian", training_days_per_week=5, training_intensity="High",
    region="Lucknow, Uttar Pradesh", allergies="none",
    budget_per_day=None, food_preferences=None, foods_to_avoid=None, notes=None,
)


# ----------------------------------------------------------------------
# Fast, pure-logic tests - no I/O, no model loading
# ----------------------------------------------------------------------

class TestDecideTools(unittest.TestCase):
    def test_high_intensity_selects_nutrition_tool(self):
        decisions = orch.decide_tools(dict(BASE_PROFILE, training_intensity="High"))
        nutrition = next(d for d in decisions if d["tool"] == "nutrition_tool")
        self.assertTrue(nutrition["selected"])
        self.assertIn("training_intensity", nutrition["reason"])

    def test_moderate_intensity_selects_nutrition_tool(self):
        decisions = orch.decide_tools(dict(BASE_PROFILE, training_intensity="Moderate"))
        nutrition = next(d for d in decisions if d["tool"] == "nutrition_tool")
        self.assertTrue(nutrition["selected"])

    def test_low_intensity_skips_nutrition_tool(self):
        decisions = orch.decide_tools(dict(BASE_PROFILE, training_intensity="Low"))
        nutrition = next(d for d in decisions if d["tool"] == "nutrition_tool")
        self.assertFalse(nutrition["selected"])
        self.assertIn("reason", nutrition)  # a reason is given even when skipped

    def test_budget_provided_selects_budget_tool(self):
        decisions = orch.decide_tools(dict(BASE_PROFILE, budget_per_day="Rs 300"))
        budget = next(d for d in decisions if d["tool"] == "budget_tool")
        self.assertTrue(budget["selected"])

    def test_no_budget_skips_budget_tool(self):
        decisions = orch.decide_tools(dict(BASE_PROFILE, budget_per_day=None))
        budget = next(d for d in decisions if d["tool"] == "budget_tool")
        self.assertFalse(budget["selected"])

    def test_every_decision_has_a_reason_regardless_of_outcome(self):
        decisions = orch.decide_tools(dict(BASE_PROFILE, training_intensity="Low", budget_per_day=None))
        for decision in decisions:
            self.assertTrue(decision["reason"])  # non-empty explanation either way


class TestParseBudgetAmount(unittest.TestCase):
    def test_parses_rupee_prefixed_string(self):
        self.assertEqual(orch._parse_budget_amount("Rs 300"), 300)

    def test_parses_plain_number_string(self):
        self.assertEqual(orch._parse_budget_amount("120"), 120)

    def test_none_input_returns_none(self):
        self.assertIsNone(orch._parse_budget_amount(None))

    def test_unparseable_string_returns_none(self):
        self.assertIsNone(orch._parse_budget_amount("unknown"))


class TestToolRunnersInIsolation(unittest.TestCase):
    """These exercise the real ToolRegistry (fast - the tools do simple
    arithmetic, no model loading) without going through the full pipeline."""

    def setUp(self):
        self.registry = build_default_registry()

    def test_run_nutrition_tool_computes_all_four_operations(self):
        results = orch._run_nutrition_tool(self.registry, BASE_PROFILE)
        self.assertEqual(set(results), {"protein", "calories", "hydration", "carbs"})
        self.assertEqual(results["protein"], 129.6)  # 72kg * 1.8 (High) - matches tools/nutrition_tool.py

    def test_run_budget_tool_returns_none_without_a_budget(self):
        result = orch._run_budget_tool(self.registry, dict(BASE_PROFILE, budget_per_day=None))
        self.assertIsNone(result)

    def test_run_budget_tool_returns_suggestions_with_a_budget(self):
        result = orch._run_budget_tool(self.registry, dict(BASE_PROFILE, budget_per_day="Rs 300"))
        self.assertIn("suggestions", result)
        self.assertIn("within_budget", result)


# ----------------------------------------------------------------------
# End-to-end tests - the full pipeline, all four required combinations
# ----------------------------------------------------------------------

class TestOrchestrateEndToEnd(MemoryIsolatedTestCase):
    def test_nutrition_only_required(self):
        profile = dict(BASE_PROFILE, training_intensity="High", budget_per_day=None)
        trace = orch.run_with_trace(profile)
        self.assertIn("nutrition_tool", trace["tool_results"])
        self.assertNotIn("budget_tool", trace["tool_results"])

    def test_budget_only_required(self):
        profile = dict(BASE_PROFILE, training_intensity="Low", budget_per_day="Rs 200")
        trace = orch.run_with_trace(profile)
        self.assertNotIn("nutrition_tool", trace["tool_results"])
        self.assertIn("budget_tool", trace["tool_results"])

    def test_both_required(self):
        profile = dict(BASE_PROFILE, training_intensity="High", budget_per_day="Rs 300")
        trace = orch.run_with_trace(profile)
        self.assertIn("nutrition_tool", trace["tool_results"])
        self.assertIn("budget_tool", trace["tool_results"])

    def test_neither_required(self):
        profile = dict(BASE_PROFILE, training_intensity="Low", budget_per_day=None)
        trace = orch.run_with_trace(profile)
        self.assertEqual(trace["tool_results"], {})

    def test_rag_retrieval_returns_relevant_chunks(self):
        profile = dict(BASE_PROFILE, goal="cheap vegetarian protein sources in India")
        trace = orch.run_with_trace(profile)
        self.assertGreater(len(trace["rag_chunks"]), 0)
        sources = {c["source"] for c in trace["rag_chunks"]}
        self.assertIn("indian_protein_sources.md", sources)

    def test_memory_used_reflects_stored_memory(self):
        mm.set("profile", orch.DEFAULT_ATHLETE_ID, {"region": "Kolkata, West Bengal"})
        trace = orch.run_with_trace(dict(BASE_PROFILE, region=None))
        self.assertEqual(trace["memory_used"]["profile"]["region"], "Kolkata, West Bengal")
        self.assertEqual(trace["merged_profile"]["region"], "Kolkata, West Bengal")

    def test_memory_used_is_empty_for_a_new_athlete(self):
        trace = orch.run_with_trace(dict(BASE_PROFILE))
        self.assertIsNone(trace["memory_used"]["profile"])
        self.assertIsNone(trace["memory_used"]["preference"])

    def test_orchestrator_never_writes_memory(self):
        # The orchestrator only READS memory (Steps 1-2) - it must not
        # persist anything, unlike api.py's update_memory_from_profile /
        # record_episode (out of scope here, see module docstring).
        orch.run_with_trace(dict(BASE_PROFILE))
        self.assertIsNone(mm.get("profile", orch.DEFAULT_ATHLETE_ID))
        self.assertIsNone(mm.get("preference", orch.DEFAULT_ATHLETE_ID))
        self.assertEqual(mm.list("episodic", orch.DEFAULT_ATHLETE_ID), [])

    def test_orchestrate_returns_only_the_plan_matching_the_response_contract(self):
        plan = orch.orchestrate(dict(BASE_PROFILE))
        self.assertEqual(
            set(plan.keys()),
            {"confidence_score", "confidence_reason", "day_summary", "meals", "disclaimer"},
        )

    def test_plan_is_still_generated_correctly(self):
        plan = orch.orchestrate(dict(BASE_PROFILE))
        self.assertEqual(len(plan["meals"]), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
