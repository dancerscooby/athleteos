"""
AthleteOS - Tests for the agent_context integration (Option A).

Covers exactly the 7 requirements:
    1. agent_context=None preserves existing behaviour.
    2. Agent-provided RAG context reaches the Claude provider's prompt.
    3. Agent-provided tool results reach the Claude provider's prompt.
    4. No duplicate RAG retrieval when agent_context is supplied.
    5. No duplicate tool-calling loop when agent_context is supplied.
    6. The mock path is completely unaffected by agent_context.
    7. (Verified separately, at the end of the report - the full existing
       suite still passes.)

No real Anthropic API calls are made anywhere in this file - no API key,
no network, no cost. The 'anthropic' package only needs to be IMPORTABLE
(claude_provider does `import anthropic` locally); its `Anthropic` client
class is mocked out entirely, so these tests exercise our own dispatch
logic (which call gets made, what the prompt contains) without touching
the real SDK's network layer at all.
"""

import json
import unittest
from unittest import mock

import llm_provider
import meal_planner
from meal_planner import build_prompt, generate_meal_plan, mock_provider


BASE_PROFILE = dict(
    age=22, height_cm=178, weight_kg=72, sex="Male",
    goal="Improve football performance",
    diet="Vegetarian", training_days_per_week=5, training_intensity="High",
    region="Lucknow, Uttar Pradesh", allergies="none",
    budget_per_day="Rs 300", food_preferences=None, foods_to_avoid=None, notes=None,
)

# A minimal, schema-valid plan for the mocked Claude response - only the
# top-level required keys matter to claude_provider()'s own parsing logic.
FAKE_PLAN = {
    "confidence_score": 8.0,
    "confidence_reason": ["test"],
    "day_summary": {
        "total_calories": 2000, "total_protein_g": 100,
        "total_carbs_g": 250, "total_fats_g": 60, "notes": "test",
    },
    "meals": [],
    "disclaimer": "test disclaimer",
}


class FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeResponse:
    """Mimics the shape of an Anthropic Message enough for claude_provider
    to parse it: .stop_reason and .content (a list of blocks with .type/.text)."""

    def __init__(self, plan=FAKE_PLAN, stop_reason="end_turn"):
        self.stop_reason = stop_reason
        self.content = [FakeTextBlock(json.dumps(plan))]


def _patched_api_key():
    """A dummy, non-empty API key so claude_provider() doesn't short-circuit
    on the 'no key configured' check before reaching the code under test."""
    return mock.patch.object(llm_provider, "LLM_API_KEY", "test-key-not-real")


# ----------------------------------------------------------------------
# 1. agent_context=None preserves existing behaviour
# ----------------------------------------------------------------------

class TestAgentContextNonePreservesBehaviour(unittest.TestCase):
    def test_legacy_path_calls_get_context_block_and_run_with_tools(self):
        with _patched_api_key(), \
             mock.patch.object(llm_provider, "get_context_block", return_value="STORED KNOWLEDGE") as get_ctx, \
             mock.patch.object(llm_provider, "run_with_tools", return_value=FakeResponse()) as run_tools, \
             mock.patch("anthropic.Anthropic") as anthropic_cls:

            plan = llm_provider.claude_provider(BASE_PROFILE)  # no agent_context at all

        get_ctx.assert_called_once_with(BASE_PROFILE)
        run_tools.assert_called_once()
        self.assertEqual(plan, FAKE_PLAN)

    def test_legacy_prompt_format_is_byte_identical_to_before(self):
        """The exact string this integration must not change for existing callers."""
        with _patched_api_key(), \
             mock.patch.object(llm_provider, "get_context_block", return_value="STORED KNOWLEDGE") as _, \
             mock.patch.object(llm_provider, "run_with_tools", return_value=FakeResponse()) as run_tools, \
             mock.patch("anthropic.Anthropic"):

            llm_provider.claude_provider(BASE_PROFILE)

        sent_prompt = run_tools.call_args.kwargs["messages"][0]["content"]
        expected = (
            "Reference material (use if relevant to this athlete; rely on your "
            "own knowledge for anything it doesn't cover):\n\n"
            "STORED KNOWLEDGE\n\n" + build_prompt(BASE_PROFILE)
        )
        self.assertEqual(sent_prompt, expected)

    def test_no_context_block_falls_back_to_bare_prompt(self):
        with _patched_api_key(), \
             mock.patch.object(llm_provider, "get_context_block", return_value=""), \
             mock.patch.object(llm_provider, "run_with_tools", return_value=FakeResponse()) as run_tools, \
             mock.patch("anthropic.Anthropic"):

            llm_provider.claude_provider(BASE_PROFILE)

        sent_prompt = run_tools.call_args.kwargs["messages"][0]["content"]
        self.assertEqual(sent_prompt, build_prompt(BASE_PROFILE))


# ----------------------------------------------------------------------
# 2 & 3. Agent-provided RAG context / tool results reach the prompt
# ----------------------------------------------------------------------

class TestAgentContextReachesThePrompt(unittest.TestCase):
    def test_rag_context_reaches_the_prompt(self):
        agent_context = {"rag_context": "UNIQUE_RAG_MARKER_XYZ", "tool_results": {}}
        with _patched_api_key(), mock.patch("anthropic.Anthropic") as anthropic_cls:
            client = anthropic_cls.return_value
            client.messages.create.return_value = FakeResponse()

            llm_provider.claude_provider(BASE_PROFILE, agent_context)

        sent_prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("UNIQUE_RAG_MARKER_XYZ", sent_prompt)
        self.assertIn("Reference material", sent_prompt)

    def test_tool_results_reach_the_prompt(self):
        agent_context = {
            "rag_context": "",
            "tool_results": {
                "nutrition_tool": {"protein": 129.6, "calories": 3110, "carbs": 576, "hydration": 3.52},
            },
        }
        with _patched_api_key(), mock.patch("anthropic.Anthropic") as anthropic_cls:
            client = anthropic_cls.return_value
            client.messages.create.return_value = FakeResponse()

            llm_provider.claude_provider(BASE_PROFILE, agent_context)

        sent_prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("Precomputed values", sent_prompt)
        self.assertIn("129.6", sent_prompt)
        self.assertIn("3110", sent_prompt)

    def test_budget_tool_results_reach_the_prompt(self):
        agent_context = {
            "rag_context": "",
            "tool_results": {
                "budget_tool": {
                    "budget_per_day": 300, "diet": "Vegetarian",
                    "suggestions": {"protein": {"name": "Dal", "cost": 10}},
                    "estimated_total_cost": 34, "within_budget": True,
                },
            },
        }
        with _patched_api_key(), mock.patch("anthropic.Anthropic") as anthropic_cls:
            client = anthropic_cls.return_value
            client.messages.create.return_value = FakeResponse()

            llm_provider.claude_provider(BASE_PROFILE, agent_context)

        sent_prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("Dal", sent_prompt)
        self.assertIn("Rs 10", sent_prompt)


# ----------------------------------------------------------------------
# 4. No duplicate RAG retrieval when agent_context is supplied
# ----------------------------------------------------------------------

class TestNoDuplicateRagRetrieval(unittest.TestCase):
    def test_get_context_block_is_never_called_when_agent_context_given(self):
        agent_context = {"rag_context": "already retrieved", "tool_results": {}}
        with _patched_api_key(), \
             mock.patch.object(llm_provider, "get_context_block") as get_ctx, \
             mock.patch("anthropic.Anthropic") as anthropic_cls:
            anthropic_cls.return_value.messages.create.return_value = FakeResponse()

            llm_provider.claude_provider(BASE_PROFILE, agent_context)

        get_ctx.assert_not_called()


# ----------------------------------------------------------------------
# 5. No duplicate tool-calling loop when agent_context is supplied
# ----------------------------------------------------------------------

class TestNoDuplicateToolCalling(unittest.TestCase):
    def test_run_with_tools_is_never_called_when_agent_context_given(self):
        agent_context = {"rag_context": "", "tool_results": {"nutrition_tool": {"protein": 100}}}
        with _patched_api_key(), \
             mock.patch.object(llm_provider, "run_with_tools") as run_tools, \
             mock.patch("anthropic.Anthropic") as anthropic_cls:
            client = anthropic_cls.return_value
            client.messages.create.return_value = FakeResponse()

            llm_provider.claude_provider(BASE_PROFILE, agent_context)

        run_tools.assert_not_called()
        # A single-shot call must still have happened - proving this isn't
        # just "nothing ran", but a deliberate no-tools request.
        client.messages.create.assert_called_once()

    def test_no_tools_parameter_is_offered_to_claude(self):
        agent_context = {"rag_context": "", "tool_results": {}}
        with _patched_api_key(), mock.patch("anthropic.Anthropic") as anthropic_cls:
            client = anthropic_cls.return_value
            client.messages.create.return_value = FakeResponse()

            llm_provider.claude_provider(BASE_PROFILE, agent_context)

        self.assertNotIn("tools", client.messages.create.call_args.kwargs)


# ----------------------------------------------------------------------
# 6. The mock path is completely unaffected by agent_context
# ----------------------------------------------------------------------

class TestMockPathUnaffected(unittest.TestCase):
    def test_mock_provider_ignores_agent_context_entirely(self):
        with mock.patch.object(meal_planner, "PROVIDER", "mock"):
            without_context = generate_meal_plan(dict(BASE_PROFILE))
            with_context = generate_meal_plan(
                dict(BASE_PROFILE),
                agent_context={"rag_context": "irrelevant", "tool_results": {"x": 1}},
            )
        self.assertEqual(without_context, with_context)

    def test_mock_path_never_touches_llm_provider_at_all(self):
        with mock.patch.object(meal_planner, "PROVIDER", "mock"), \
             mock.patch("llm_provider.claude_provider") as claude_mock:
            generate_meal_plan(dict(BASE_PROFILE), agent_context={"rag_context": "x", "tool_results": {}})
        claude_mock.assert_not_called()

    def test_mock_provider_output_matches_direct_call(self):
        direct = mock_provider(dict(BASE_PROFILE))
        via_dispatch = generate_meal_plan(dict(BASE_PROFILE))
        self.assertEqual(direct, via_dispatch)


# ----------------------------------------------------------------------
# generate_meal_plan's new "claude" dispatch branch (end-to-end wiring,
# still fully mocked - no real API calls)
# ----------------------------------------------------------------------

class TestGenerateMealPlanClaudeDispatch(unittest.TestCase):
    def test_claude_branch_forwards_agent_context(self):
        agent_context = {"rag_context": "ctx", "tool_results": {}}
        with mock.patch.object(meal_planner, "PROVIDER", "claude"), \
             mock.patch("llm_provider.claude_provider", return_value=FAKE_PLAN) as claude_mock:
            plan = generate_meal_plan(dict(BASE_PROFILE), agent_context=agent_context)

        claude_mock.assert_called_once_with(dict(BASE_PROFILE), agent_context)
        self.assertEqual(plan, FAKE_PLAN)


if __name__ == "__main__":
    unittest.main(verbosity=2)
