"""
AthleteOS - Integration tests for the wired-up Agent v1 request flow.

api.py's create_meal_plan() is now coordinated by orchestrator.run_with_trace()
instead of calling merge_with_memory()/generate_meal_plan() directly. These
tests exercise the REAL endpoint function (called directly - no HTTP server
needed, same technique as the earlier memory integration tests) and verify:

  1. A request requiring Nutrition Tool actually runs it.
  2. A request requiring Budget Tool actually runs it.
  3. A request requiring both runs both.
  4. A request requiring neither runs neither (and generation is unaffected).
  5. Memory is loaded and merged correctly through the real endpoint.
  6. Episodic memory is recorded only after a SUCCESSFUL generation.
  7. The response schema is unchanged.
  8. Memory operations run EXACTLY ONCE per request - not duplicated between
     api.py and the orchestrator.

Because api.py's response contract intentionally does NOT expose the
orchestrator's internal trace (tool decisions, RAG chunks, merged profile),
several tests use unittest.mock.patch with `side_effect`/`wraps` pointing at
the REAL implementation - this lets a test observe what happened internally
without changing behaviour or reimplementing any logic itself.

Every test runs against a temporary memory/ directory - never the project's
real one - using the same isolation technique as the other memory suites.

Run with:
    python3 -m unittest tests.test_api_orchestrator_integration -v
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import episodic_memory
import memory_integration as mi
import memory_manager as mm
import orchestrator
from api import MealPlanRequest, create_meal_plan
from meal_planner import generate_meal_plan


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


def _call_and_capture_trace(request):
    """
    Call the REAL create_meal_plan() endpoint function while capturing the
    orchestrator's internal trace. api.py never exposes this by design (to
    preserve the response contract), so we spy on api.run_with_trace,
    delegating every call to the real implementation - behaviour is
    completely unchanged, we're only observing it.
    """
    captured = {}

    def spy(*args, **kwargs):
        trace = orchestrator.run_with_trace(*args, **kwargs)
        captured["trace"] = trace
        return trace

    with mock.patch("api.run_with_trace", side_effect=spy):
        response = create_meal_plan(request)
    return response, captured["trace"]


# ----------------------------------------------------------------------
# 1-4. Tool selection through the real endpoint
# ----------------------------------------------------------------------

class TestToolSelectionViaRealEndpoint(MemoryIsolatedTestCase):
    def test_1_nutrition_tool_required(self):
        request = MealPlanRequest(**dict(BASE_PROFILE, training_intensity="High", budget_per_day=None))
        response, trace = _call_and_capture_trace(request)
        self.assertIn("nutrition_tool", trace["tool_results"])
        self.assertNotIn("budget_tool", trace["tool_results"])

    def test_2_budget_tool_required(self):
        request = MealPlanRequest(**dict(BASE_PROFILE, training_intensity="Low", budget_per_day="Rs 200"))
        response, trace = _call_and_capture_trace(request)
        self.assertNotIn("nutrition_tool", trace["tool_results"])
        self.assertIn("budget_tool", trace["tool_results"])

    def test_3_both_tools_required(self):
        request = MealPlanRequest(**dict(BASE_PROFILE, training_intensity="High", budget_per_day="Rs 300"))
        response, trace = _call_and_capture_trace(request)
        self.assertIn("nutrition_tool", trace["tool_results"])
        self.assertIn("budget_tool", trace["tool_results"])

    def test_4_neither_tool_required(self):
        request = MealPlanRequest(**dict(BASE_PROFILE, training_intensity="Low", budget_per_day=None))
        response, trace = _call_and_capture_trace(request)
        self.assertEqual(trace["tool_results"], {})
        # With no tools selected, the endpoint must add nothing of its own on
        # top of what the orchestrator supplies - calling the planner directly
        # with the SAME agent_context has to reproduce the plan exactly.
        #
        # (This used to compare against generate_meal_plan(profile) with no
        # agent_context at all. That only matched while the retrieved RAG text
        # had no food in common with the region's short candidate lists; since
        # Phase 2 those lists are long enough that RAG genuinely influences
        # food choice, so the no-context call is legitimately a different
        # request, not the same one.)
        profile = {k: v for k, v in request.model_dump().items() if k != "athlete_id"}
        direct_plan = generate_meal_plan(
            profile,
            agent_context={"rag_context": trace["rag_context"], "tool_results": trace["tool_results"]},
        )
        self.assertEqual(response, direct_plan)


# ----------------------------------------------------------------------
# 5. Memory loaded correctly through the real endpoint
# ----------------------------------------------------------------------

class TestMemoryLoadedViaRealEndpoint(MemoryIsolatedTestCase):
    def test_5_memory_loaded_and_merged_correctly(self):
        mm.set("preference", mi.DEFAULT_ATHLETE_ID, {"budget_per_day": "Rs 300"})
        request = MealPlanRequest(**dict(BASE_PROFILE, budget_per_day=None))
        response, trace = _call_and_capture_trace(request)
        self.assertEqual(trace["memory_used"]["preference"]["budget_per_day"], "Rs 300")
        self.assertEqual(trace["merged_profile"]["budget_per_day"], "Rs 300")
        # And it actually reached the tool: budget_tool should have run,
        # since the merged profile now has a budget even though the
        # request itself didn't send one.
        self.assertIn("budget_tool", trace["tool_results"])


# ----------------------------------------------------------------------
# 6. Episodic memory recorded only after success
# ----------------------------------------------------------------------

class TestEpisodicOnlyAfterSuccess(MemoryIsolatedTestCase):
    def test_6a_episode_recorded_after_successful_generation(self):
        create_meal_plan(MealPlanRequest(**BASE_PROFILE))
        entries = mm.list("episodic", mi.DEFAULT_ATHLETE_ID)
        self.assertEqual(len(entries), 1)

    def test_6b_no_episode_recorded_when_generation_fails(self):
        with mock.patch("orchestrator.generate_meal_plan", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                create_meal_plan(MealPlanRequest(**BASE_PROFILE))
        self.assertEqual(mm.list("episodic", mi.DEFAULT_ATHLETE_ID), [])
        # Profile/Preference memory must also stay unwritten - the failure
        # happened before either write-back step ever ran.
        self.assertIsNone(mm.get("profile", mi.DEFAULT_ATHLETE_ID))


# ----------------------------------------------------------------------
# 7. Response schema unchanged
# ----------------------------------------------------------------------

class TestResponseContractUnchanged(MemoryIsolatedTestCase):
    def test_7_response_schema_is_unchanged(self):
        response = create_meal_plan(MealPlanRequest(**BASE_PROFILE))
        self.assertEqual(
            set(response.keys()),
            {"confidence_score", "confidence_reason", "day_summary", "meals", "disclaimer"},
        )


# ----------------------------------------------------------------------
# Not accidentally executed twice
# ----------------------------------------------------------------------

class TestNoDuplicateExecution(MemoryIsolatedTestCase):
    def test_merge_with_memory_runs_exactly_once_per_request(self):
        with mock.patch("orchestrator.merge_with_memory", wraps=mi.merge_with_memory) as spy:
            create_meal_plan(MealPlanRequest(**BASE_PROFILE))
        self.assertEqual(spy.call_count, 1)

    def test_update_memory_from_profile_runs_exactly_once_per_request(self):
        with mock.patch("api.update_memory_from_profile", wraps=mi.update_memory_from_profile) as spy:
            create_meal_plan(MealPlanRequest(**BASE_PROFILE))
        self.assertEqual(spy.call_count, 1)

    def test_record_episode_runs_exactly_once_per_request(self):
        with mock.patch("api.record_episode", wraps=episodic_memory.record_episode) as spy:
            create_meal_plan(MealPlanRequest(**BASE_PROFILE))
        self.assertEqual(spy.call_count, 1)

    def test_generate_meal_plan_runs_exactly_once_per_request(self):
        with mock.patch("orchestrator.generate_meal_plan", wraps=generate_meal_plan) as spy:
            create_meal_plan(MealPlanRequest(**BASE_PROFILE))
        self.assertEqual(spy.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
