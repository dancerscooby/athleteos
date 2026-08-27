# AthleteOS Tests

## What's here

Stage 1 end-to-end tests for five scenarios:

1. **Vegetarian footballer**
2. **Non-vegetarian footballer**
3. **Match-day scenario**
4. **Low-budget scenario**
5. **Invalid-input scenario**

Each test (except the invalid-input one) walks through the same 4 stages a real request goes through, and logs every stage:

1. **Profile** — the athlete's input.
2. **RAG retrieval** — the search query sent to the retriever, and which `knowledge_base/` chunks came back (with similarity scores).
3. **Tool call** — a `nutrition_tool` or `budget_tool` call appropriate to that scenario, run through the real tool-execution code, logging the tool name, inputs, and outputs.
4. **Final response** — the full meal plan returned by `generate_meal_plan()`.

The invalid-input test is different on purpose: it checks that bad data (age too low, an invalid diet value, etc.) is rejected by the API's validation layer *before* any of the above stages run — matching how the real app behaves.

## Important: no live Claude API calls

These tests do **not** call the real Claude API, even if you have `ANTHROPIC_API_KEY` set. Reasons:

- Running tests shouldn't cost real money every time, and definitely shouldn't do so silently.
- `generate_meal_plan()` currently uses `meal_planner.PROVIDER = "mock"`, so Stage 4 in every test runs through the same deterministic mock provider the rest of the app uses by default. This is still a genuine end-to-end run of the real pipeline (validation shape, RAG injection point, response schema) — just not routed through the live LLM.
- Stage 3 (tool call) cannot test *whether Claude decides* to call a tool — that decision only happens inside a real API call. What it **does** test is the exact same `tool_selector.execute_tool_call()` function a real Claude tool request would be handed to, so the tool execution, logging, and error handling are fully verified. Only Claude's own judgment call is untested here.

If you want to verify the real LLM path (costs a small amount of money per call), set `ANTHROPIC_API_KEY` and call `llm_provider.claude_provider(profile)` directly in a Python shell — that is not part of this automated suite.

## How to run

From the project root (`AthleteOS/`):

```bash
python3 -m unittest tests.test_scenarios -v
```

The first run will download the local embedding model (`all-MiniLM-L6-v2`) if it isn't cached yet, so it may take a few extra seconds. Every stage's log line prints to the console as the tests run (timestamped, via Python's standard `logging` module) — you don't need any extra flags to see them.

### Requirements

The retrieval stage needs `sentence-transformers` (see the project's `requirements.txt`) and a populated `embeddings.json` — both should already be set up from the RAG stage. If `embeddings.json` is missing or stale, regenerate it first:

```bash
python3 embeddings.py
```

## What to expect

All 5 tests pass. A typical run looks like:

```
test_1_vegetarian_footballer ... ok
test_2_non_vegetarian_footballer ... ok
test_3_match_day ... ok
test_4_low_budget ... ok
test_5_invalid_input ... ok

Ran 5 tests in ~7s

OK
```

## If a test fails

- **A `nutrition_tool` / `budget_tool` assertion fails** — the formulas in `tools/nutrition_tool.py` or `tools/budget_tool.py` changed. Check what the tool now returns and decide whether the test's expected value or the tool's formula is the one that's wrong.
- **A diet/ingredient assertion fails** (e.g. "no meat in vegetarian plan") — something in `meal_planner.py`'s rule-based mock changed. This would be a real regression worth investigating, since diet correctness is a core promise of the app.
- **A RAG-related failure** (an import error, or `get_context_block` raising) — usually means `sentence-transformers` isn't installed, or `embeddings.json` doesn't exist yet. Run `python3 embeddings.py` and retry.
- **The invalid-input test fails** — either the validation rules in `api.py`'s `MealPlanRequest` changed, or the set of fields flagged as bad no longer matches `{"age", "diet", "training_days_per_week", "training_intensity"}`. Check what actually failed validation and update the test or the model accordingly.

One real example from developing this suite: `test_3_match_day` initially failed because the test assumed `nutrition_tool`'s carbs calculation would return a float (`576.0`), but Python's `round(576, 1)` on a whole-number result returns a plain `int` (`576`) — so the JSON output was `576` not `576.0`. That was a wrong assumption in the test, not a bug in the app; the fix was updating the expected string to match the real (correct) output.
