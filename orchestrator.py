"""
AthleteOS - Agent v1: rule-based orchestrator.

Coordinates the existing components (Profile/Preference Memory, the RAG
pipeline, the tool registry, and the meal-generation provider) using
DETERMINISTIC RULES, not another LLM call. Nothing here decides anything by
"thinking" - every decision is an explicit if/else with a logged reason,
which is the whole point: this is a rule-based orchestrator making itself
auditable, not an autonomous agent pretending to reason.

Reuses every existing component - nothing it imports had its own logic
rewritten for this file, though generate_meal_plan()/claude_provider() each
gained one new OPTIONAL parameter (agent_context) so this orchestrator can
hand them the RAG/tool results it already computed, instead of the provider
redoing that work (see meal_planner.py and llm_provider.py):
    memory_integration.merge_with_memory   (Stage 1, unmodified)
    memory_manager.get                     (unmodified)
    rag.build_retrieval_query,
    rag.format_context_block               (unmodified)
    retriever.retrieve_relevant_chunks     (unmodified - FAISS + reranking)
    tools.build_default_registry           (unmodified)
    meal_planner.generate_meal_plan        (extended with optional agent_context;
                                             behaviour for existing callers unchanged)

Standalone module, NOT wired into api.py yet - none of the required/allowed
changes mentioned the API layer or the live request path, so nothing about
api.py, the request/response schemas, or generate_meal_plan()'s behaviour
changes because this file exists. It is independently callable and testable
on its own, the same way llm_provider.py and memory_manager.py were each
introduced before a later, separate integration step.

Two entry points:
    orchestrate(profile, athlete_id)
        Contract-compliant: returns ONLY the plan dict, matching
        MealPlanResponse exactly - safe to wire into api.py later with zero
        response-shape changes.

    run_with_trace(profile, athlete_id)
        Returns the plan PLUS every intermediate decision (memory loaded,
        RAG chunks, tool decisions, tool results) - used by tests and for
        transparency; not something the public API contract exposes.

Deliberately out of scope (see the Stage 2 requirements list): writing
memory back (update_memory_from_profile / record_episode). This orchestrator
only READS memory as input context for one generation; persisting results
remains the API layer's job if/when this gets wired in.
"""

import logging

import memory_manager
from memory_integration import DEFAULT_ATHLETE_ID, merge_with_memory
from meal_planner import avoid_keywords, generate_meal_plan
from rag import build_retrieval_query, format_context_block, retrieve_profile_chunks
from retriever import retrieve_relevant_chunks
from tools import build_default_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("athleteos.orchestrator")

RAG_TOP_K = 3

# Training intensities precise-enough to be worth computing exact nutrition
# targets for, rather than relying on generic guidance alone.
NUTRITION_TRIGGER_INTENSITIES = {"Moderate", "High"}

NUTRITION_OPERATIONS = ["protein", "calories", "hydration", "carbs"]


# ----------------------------------------------------------------------
# Step 4: deterministic tool-selection rules
# ----------------------------------------------------------------------

def decide_tools(profile):
    """
    Deterministic, explainable decisions about which tools this profile
    needs. Returns a list of {"tool", "selected", "reason"} - one entry per
    known tool, including tools that were NOT selected, so the reasoning is
    visible either way (this is what "explicit decisions, not pretending to
    be autonomous" means in practice).
    """
    decisions = []

    intensity = profile.get("training_intensity")
    nutrition_needed = intensity in NUTRITION_TRIGGER_INTENSITIES
    decisions.append({
        "tool": "nutrition_tool",
        "selected": nutrition_needed,
        "reason": (
            f"training_intensity={intensity!r} is precise-training-load enough "
            "to warrant exact computed nutrition targets."
            if nutrition_needed else
            f"training_intensity={intensity!r} does not warrant precise "
            "computed targets - generic guidance is sufficient."
        ),
    })

    budget_value = profile.get("budget_per_day")
    budget_needed = budget_value is not None
    decisions.append({
        "tool": "budget_tool",
        "selected": budget_needed,
        "reason": (
            f"budget_per_day={budget_value!r} was provided - suggesting "
            "affordable foods within it."
            if budget_needed else
            "budget_per_day was not provided - nothing for the budget tool "
            "to work with."
        ),
    })

    # Home ingredients are entirely optional (Phase 4): with nothing supplied
    # this tool never runs and the request is indistinguishable from one made
    # before the feature existed.
    ingredients_value = profile.get("available_ingredients")
    ingredients_needed = bool(ingredients_value and str(ingredients_value).strip())
    decisions.append({
        "tool": "ingredients_tool",
        "selected": ingredients_needed,
        "reason": (
            "available_ingredients were provided - checking which are usable "
            "and which food groups are still missing."
            if ingredients_needed else
            "no available_ingredients were provided - nothing for the "
            "ingredients tool to assess."
        ),
    })

    return decisions


# ----------------------------------------------------------------------
# Step 5: running the selected tools via the existing ToolRegistry
# ----------------------------------------------------------------------

def _parse_budget_amount(budget_per_day):
    """Extract the leading integer from a budget string like 'Rs 300' -> 300."""
    if budget_per_day is None:
        return None
    digits = "".join(c if c.isdigit() else " " for c in str(budget_per_day)).split()
    return int(digits[0]) if digits else None


def _run_nutrition_tool(registry, profile):
    """Run every nutrition_tool operation relevant to a daily plan."""
    shared_kwargs = {
        "weight_kg": profile.get("weight_kg"),
        "training_intensity": profile.get("training_intensity"),
    }
    calorie_kwargs = {
        **shared_kwargs,
        "height_cm": profile.get("height_cm"),
        "age": profile.get("age"),
        "sex": profile.get("sex"),
    }
    kwargs_by_operation = {
        "protein": shared_kwargs,
        "calories": calorie_kwargs,
        "hydration": shared_kwargs,
        "carbs": shared_kwargs,
    }

    results = {}
    for operation in NUTRITION_OPERATIONS:
        outcome = registry.execute(
            "nutrition_tool", operation=operation, **kwargs_by_operation[operation]
        )
        results[operation] = outcome["result"]
    return results


def _run_budget_tool(registry, profile):
    amount = _parse_budget_amount(profile.get("budget_per_day"))
    if amount is None:
        return None
    return registry.execute(
        "budget_tool", budget_per_day=amount, diet=profile.get("diet", "Vegetarian")
    )


def _run_ingredients_tool(registry, profile):
    """
    Assess the athlete's home ingredients.

    The allergy/foods-to-avoid words are computed here with
    meal_planner.avoid_keywords() - the app's single source of truth for
    that expansion (dairy -> milk/curd/paneer/..., singular/plural) - and
    passed in, so tools/ stays free of any dependency on the application and
    the two never drift apart.
    """
    return registry.execute(
        "ingredients_tool",
        ingredients=profile.get("available_ingredients"),
        diet=profile.get("diet", "Vegetarian"),
        avoid=sorted(avoid_keywords(profile)),
    )


TOOL_RUNNERS = {
    "nutrition_tool": _run_nutrition_tool,
    "budget_tool": _run_budget_tool,
    "ingredients_tool": _run_ingredients_tool,
}


# ----------------------------------------------------------------------
# The orchestration pipeline
# ----------------------------------------------------------------------

def run_with_trace(profile, athlete_id=DEFAULT_ATHLETE_ID, days=1):
    """
    Run the full pipeline and return EVERYTHING - the plan plus every
    intermediate decision - for transparency and testing. See orchestrate()
    for the contract-compliant, plan-only entry point.

    `days` (optional, default 1 - multi-day planning) passes straight
    through to generate_meal_plan(); every other step (memory, RAG, tool
    decisions) is profile-level and unaffected by how many days of meals get
    generated from its output.
    """
    logger.info("=" * 70)
    logger.info("ORCHESTRATOR: starting run for athlete_id=%r", athlete_id)
    logger.info("=" * 70)

    # --- Steps 1-2: load Profile/Preference memory and merge with request ---
    logger.info("STEP 1/5 - Loading Profile/Preference memory")
    stored_profile = memory_manager.get("profile", athlete_id)
    stored_preference = memory_manager.get("preference", athlete_id)
    logger.info("  profile memory:    %s", stored_profile or "(none stored yet)")
    logger.info("  preference memory: %s", stored_preference or "(none stored yet)")

    merged_profile = merge_with_memory(profile, athlete_id)  # logs its own merge details
    logger.info("STEP 1/5 complete - request merged with stored memory")

    # --- Phase 2A: read this athlete's EXISTING generation history for
    # memory-aware meal variety - memory_manager.list()/.get() only, no new
    # category, no memory_manager.py change (see meal_planner.mock_provider()
    # and _build_day() for how this is consumed) ---
    logger.info("STEP 1b - Loading generation history for memory-aware variety")
    episodes = memory_manager.list("episodic", athlete_id, limit=None)
    generation_index = len(episodes)
    latest_plan = memory_manager.get("latest_plan", athlete_id)
    recent_proteins = sorted((latest_plan or {}).get("protein_sources") or [])
    recent_carbs = sorted((latest_plan or {}).get("carb_sources") or [])
    logger.info(
        "  generation_index=%d (from %d past episode(s)), recent_proteins=%s, recent_carbs=%s",
        generation_index, len(episodes), recent_proteins, recent_carbs,
    )
    variety_context = {
        "generation_index": generation_index,
        "recent_proteins": recent_proteins,
        "recent_carbs": recent_carbs,
    }

    # --- Personal Recipes: fetch this athlete's saved recipes (RAW, not
    # filtered here) - memory_manager.get() on the "recipes" category, the
    # same update-in-place read every other memory category already uses.
    # This orchestrator does NOT decide which recipes are safe to offer -
    # that is meal_planner.classify_personal_recipes()'s job, called
    # independently by whichever provider actually needs the answer (mock,
    # Claude, OpenRouter, or chat.py), so no one ever has to trust that
    # THIS function filtered correctly. Never stored in RAG - see
    # personal_recipes.py's module docstring. ---
    personal_recipes = memory_manager.get("recipes", athlete_id) or []
    logger.info("  saved recipes: %d", len(personal_recipes))

    # --- Step 3: retrieve relevant knowledge through the existing RAG pipeline ---
    logger.info("STEP 2/5 - Retrieving relevant knowledge via RAG")
    query = build_retrieval_query(merged_profile)
    # retrieve_profile_chunks() is the normal search plus a small
    # region-specific top-up (see rag.py) - still the same retriever, just
    # queried a second time so the athlete's own region actually surfaces.
    rag_chunks = retrieve_profile_chunks(merged_profile, query, top_k=RAG_TOP_K)
    rag_context = format_context_block(rag_chunks)
    logger.info(
        "STEP 2/5 complete - retrieved %d chunks: %s",
        len(rag_chunks), [c["id"] for c in rag_chunks],
    )

    # --- Step 4: decide which tools are needed ---
    logger.info("STEP 3/5 - Deciding which tools are needed")
    tool_decisions = decide_tools(merged_profile)
    for decision in tool_decisions:
        logger.info(
            "  decision: %-14s -> %-8s (%s)",
            decision["tool"],
            "SELECTED" if decision["selected"] else "skipped",
            decision["reason"],
        )

    # --- Step 5: execute selected tools via the existing ToolRegistry ---
    logger.info("STEP 4/5 - Executing selected tools")
    registry = build_default_registry()
    tool_results = {}
    for decision in tool_decisions:
        if not decision["selected"]:
            continue
        runner = TOOL_RUNNERS[decision["tool"]]
        tool_results[decision["tool"]] = runner(registry, merged_profile)  # logs per-call itself
    logger.info(
        "STEP 4/5 complete - tools run: %s",
        list(tool_results) or "(none)",
    )

    # --- Step 6: pass control to the provider abstraction, along with the
    # RAG context and tool results already computed above, so the "claude"
    # provider (once enabled) does not redo either piece of work itself ---
    logger.info("STEP 5/5 - Passing control to generate_meal_plan() (provider abstraction)")
    agent_context = {
        "rag_context": rag_context,
        "tool_results": tool_results,
        "variety_context": variety_context,
        "personal_recipes": personal_recipes,
    }
    plan = generate_meal_plan(merged_profile, agent_context=agent_context, days=days)
    logger.info("STEP 5/5 complete - plan generated")

    logger.info("ORCHESTRATOR: run complete for athlete_id=%r", athlete_id)

    return {
        "memory_used": {"profile": stored_profile, "preference": stored_preference},
        "merged_profile": merged_profile,
        "rag_query": query,
        "rag_chunks": rag_chunks,
        "rag_context": rag_context,
        "tool_decisions": tool_decisions,
        "tool_results": tool_results,
        "variety_context": variety_context,
        "personal_recipes": personal_recipes,
        "plan": plan,
    }


def orchestrate(profile, athlete_id=DEFAULT_ATHLETE_ID, days=1):
    """
    Contract-compliant entry point (Step 7): returns ONLY the plan dict,
    identical in shape to what generate_meal_plan() / the API's
    MealPlanResponse already return. Use run_with_trace() when you need to
    inspect the intermediate decisions.
    """
    return run_with_trace(profile, athlete_id, days)["plan"]
