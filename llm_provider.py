"""
AthleteOS - LLM providers (Step 4; agent_context integration added later;
openrouter_provider added after that).

Two more implementations of the same provider contract as mock_provider:

    provider(profile: dict, agent_context: dict | None = None) -> meal_plan (dict)

Both build the prompt with meal_planner.build_prompt() (no duplicated business
logic) and return the exact same JSON shape the mock returns and the API's
MealPlanResponse describes:

  - claude_provider()     asks Claude for a meal plan using STRUCTURED OUTPUT
                          (a JSON schema the model must fill). Has its own
                          standalone RAG+tool-calling mode when called with
                          no agent_context (see below).
  - openrouter_provider()  asks a free OpenRouter model for a meal plan via
                          prompt-only JSON instructions (most free models
                          don't support Claude-style structured output).
                          ALWAYS relies on agent_context for RAG/tool
                          information - it never retrieves or calls tools
                          itself, even when agent_context is None.

Design goals shared by both:
- Same profile in, same JSON structure out (drop-in for mock_provider).
- Fail GRACEFULLY: any problem (no API key, SDK/package missing, network/API
  error, a safety refusal, or malformed output) raises one clear
  LLMProviderError instead of crashing with a raw traceback - so the mock
  provider stays usable even if a provider is misconfigured.
- Configuration lives in one place (the constants below), driven by env vars.
  Neither provider ever hard-codes an API key.

agent_context (Option A integration): orchestrator.py already runs RAG
retrieval and the deterministic tools before calling generate_meal_plan(),
so a provider does not need to redo that work. When agent_context is
supplied - {"rag_context": str, "tool_results": dict} - both providers inject
it into the prompt as clearly labelled reference sections (see
_compose_prompt()) instead of retrieving/calling anything themselves.

claude_provider() ALSO supports a standalone/legacy mode: when agent_context
is None (e.g. calling it directly, outside the orchestrator), it does its own
RAG retrieval and offers Claude the full tool-calling loop - behaviour
unchanged from before agent_context existed. openrouter_provider() has no
such mode (see its own docstring for why).
"""

import json
import logging
import os

from dotenv import load_dotenv

from meal_planner import build_prompt, classify_personal_recipes
from rag import get_context_block
from tool_selector import run_with_tools

# Loads variables from a local .env file (see .env in the project root) into
# the environment, e.g. OPENROUTER_API_KEY / ANTHROPIC_API_KEY - so a
# developer never has to export them by hand. Only affects THIS process's
# environment; never overwrites a variable already set for real (default
# override=False), and does nothing if .env doesn't exist.
load_dotenv()

logger = logging.getLogger("athleteos.llm_provider")


# ----------------------------------------------------------------------
# Configuration (all overridable via environment variables)
# ----------------------------------------------------------------------
# Model default is Claude Opus 5. It is the most capable option but also the
# priciest; set ATHLETEOS_LLM_MODEL=claude-sonnet-5 or claude-haiku-4-5 for a
# cheaper run. The API key is read by the SDK from ANTHROPIC_API_KEY.
LLM_MODEL = os.environ.get("ATHLETEOS_LLM_MODEL", "claude-opus-5")
LLM_MAX_TOKENS = int(os.environ.get("ATHLETEOS_LLM_MAX_TOKENS", "3000"))
LLM_EFFORT = os.environ.get("ATHLETEOS_LLM_EFFORT", "low")  # low keeps cost/latency down
LLM_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# OpenRouter (openrouter_provider) - a second, free-tier provider. The model
# ID is env-overridable because OpenRouter's free lineup rotates; verify the
# current one at https://openrouter.ai/models?max_price=0 before relying on
# the default. Confirmed free and working (live-tested, not guessed) as of
# 2026-08-11: openai/gpt-oss-20b:free.
#
# gpt-oss-20b is a REASONING model - it spends hidden "reasoning" tokens
# (counted against max_tokens) before writing its actual answer, and a long
# prompt (e.g. with RAG context + tool results attached) can make it reason
# so much it runs out of budget and returns NO content at all. Capping
# reasoning effort to "low" (OpenRouter's `reasoning.effort` field) fixes
# this - live-tested at ~90 reasoning tokens vs ~8000 (truncated, empty)
# without the cap, for the same full-context prompt.
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")
# Tried once, automatically, only if OPENROUTER_MODEL's call fails (see
# call_openrouter() below) - the same free-tier-congestion failure mode
# this file already documents on RateLimitError. Set to "" to disable
# fallback entirely and only ever try OPENROUTER_MODEL.
OPENROUTER_FALLBACK_MODEL = os.environ.get("OPENROUTER_FALLBACK_MODEL", "openai/gpt-oss-20b:free")
OPENROUTER_MAX_TOKENS = int(os.environ.get("ATHLETEOS_OPENROUTER_MAX_TOKENS", "3000"))
OPENROUTER_EFFORT = os.environ.get("ATHLETEOS_OPENROUTER_EFFORT", "low")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")


class LLMProviderError(Exception):
    """Raised for any failure while getting a plan from an LLM provider (Claude or OpenRouter)."""


class RateLimitError(LLMProviderError):
    """
    Raised specifically for an HTTP 429 from the OpenRouter API - a subclass
    of LLMProviderError so any existing `except LLMProviderError` handling
    keeps working unchanged, while callers that want to tell a rate limit
    apart from other provider failures (see chat.py's api.py handler) can
    catch this more specific type first. Root cause: `openai/gpt-oss-20b:free`
    sits behind OpenRouter's shared, globally-oversubscribed free-tier pool -
    this is an external capacity limit, not a bug in our prompt construction,
    memory, or session handling (confirmed via live reproduction: repeated
    real /chat calls returned intermixed 200s and 429s with no fixed pattern,
    consistent with shared-pool congestion rather than a per-session cap)."""


# The exact JSON shape we want back - mirrors the API's MealPlanResponse so a
# validated request/response contract stays consistent across providers.
MEAL_PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "confidence_score": {"type": "number"},
        "confidence_reason": {"type": "array", "items": {"type": "string"}},
        "day_summary": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "total_calories": {"type": "integer"},
                "total_protein_g": {"type": "integer"},
                "total_carbs_g": {"type": "integer"},
                "total_fats_g": {"type": "integer"},
                "notes": {"type": "string"},
            },
            "required": [
                "total_calories", "total_protein_g", "total_carbs_g",
                "total_fats_g", "notes",
            ],
        },
        "meals": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "meal": {"type": "string"},
                    "ingredients": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "item": {"type": "string"},
                                "quantity": {"type": "string"},
                            },
                            "required": ["item", "quantity"],
                        },
                    },
                    "calories": {"type": "integer"},
                    "protein_g": {"type": "integer"},
                    "carbs_g": {"type": "integer"},
                    "fats_g": {"type": "integer"},
                    "reasoning": {"type": "string"},
                    "resources": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "type": {"type": "string"},
                                "title": {"type": "string"},
                                "url": {"type": "string"},
                            },
                            "required": ["type", "title", "url"],
                        },
                    },
                },
                "required": [
                    "meal", "ingredients", "calories", "protein_g",
                    "carbs_g", "fats_g", "reasoning", "resources",
                ],
            },
        },
        "disclaimer": {"type": "string"},
    },
    "required": [
        "confidence_score", "confidence_reason", "day_summary",
        "meals", "disclaimer",
    ],
}


def is_configured():
    """True if this provider could actually run (SDK installed + API key set)."""
    if not LLM_API_KEY:
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _format_tool_results(tool_results):
    """
    Render the orchestrator's tool_results dict (see orchestrator.py) as a
    short, labelled block of exact numbers for the prompt - not the raw
    dict, just the values a nutritionist would actually want to read.
    """
    lines = []

    nutrition = tool_results.get("nutrition_tool")
    if nutrition:
        lines.append(
            "- Nutrition targets: "
            f"protein {nutrition.get('protein')} g/day, "
            f"calories {nutrition.get('calories')} kcal/day, "
            f"carbohydrate {nutrition.get('carbs')} g/day, "
            f"hydration {nutrition.get('hydration')} L/day"
        )

    budget = tool_results.get("budget_tool")
    if budget:
        suggestions = budget.get("suggestions") or {}
        picks = ", ".join(
            f"{category}: {info['name']} (Rs {info['cost']})"
            for category, info in suggestions.items()
        )
        lines.append(
            f"- Budget-friendly food suggestions (target Rs {budget.get('budget_per_day')}/day, "
            f"estimated total Rs {budget.get('estimated_total_cost')}): {picks}"
        )

    ingredients = tool_results.get("ingredients_tool")
    if ingredients:
        lines.append(_format_ingredients(ingredients))

    return "\n".join(lines) if lines else "(no precomputed tool results)"


def _format_ingredients(ingredients):
    """
    Render ingredients_tool's assessment as a clearly labelled block.

    The three lists are kept separate on purpose. "Usable" has already been
    checked against the athlete's diet and allergy/avoid words, "must not be
    used" records what was rejected AND why, and "could not be verified" is
    the honest bucket for anything the tool could not place in a food group -
    typed by the athlete, but never to be treated as a confirmed ingredient.
    Collapsing these into one list would let the model quietly promote an
    excluded or unverified item into a meal.
    """
    lines = ["- HOME INGREDIENTS the athlete says they already have:"]

    unrecognised = ingredients.get("unrecognised") or []
    # An unrecognised item passed the diet/allergy checks but was never
    # confirmed to be a usable food, so it must not be presented as usable -
    # it gets its own line below. Listing it in both places would invite the
    # model to just take it from the first one.
    verified = [i for i in (ingredients.get("usable") or []) if i not in unrecognised]
    lines.append(
        f"    * Usable (already checked against their diet and allergies): "
        f"{', '.join(verified) if verified else '(none)'}"
    )

    excluded = ingredients.get("excluded") or []
    if excluded:
        detail = "; ".join(f"{e['ingredient']} - {e['reason']}" for e in excluded)
        lines.append(f"    * MUST NOT be used: {detail}")

    if unrecognised:
        lines.append(
            "    * Could not be verified as usable foods - do not build meals around "
            f"these or assume they are suitable: {', '.join(unrecognised)}"
        )

    missing = ingredients.get("missing_groups") or []
    if missing:
        lines.append(
            "    * Food groups NOT covered by what they have: "
            f"{', '.join(missing)} (suggest a few additions if needed)"
        )

    return "\n".join(lines)


def _format_recipes(recipes, profile):
    """
    Render an athlete's saved personal recipes as a clearly labelled block -
    mirrors _format_ingredients()'s usable/"MUST NOT be used" split for the
    same reason: an excluded recipe must say why, never be silently dropped,
    so the model can't quietly promote a recipe that conflicts with an
    allergy or diet rule.

    Classification is done HERE, independently, via
    meal_planner.classify_personal_recipes() - the SAME function
    mock_provider() and chat.py call - so the LLM path is never the only
    place recipe safety is decided; it just gets told the same answer.
    """
    classification = classify_personal_recipes(recipes, profile)
    lines = ["- PERSONAL RECIPES this athlete has saved (their own homemade/family recipes):"]

    usable = classification["usable"]
    if usable:
        described = "; ".join(
            f"{r['name']} ({', '.join(r.get('ingredients') or [])})" for r in usable
        )
        lines.append(f"    * May be offered (already checked against their diet and allergies): {described}")
    else:
        lines.append("    * May be offered: (none currently safe for this profile)")

    excluded = classification["excluded"]
    if excluded:
        detail = "; ".join(f"{e['name']} - {e['reason']}" for e in excluded)
        lines.append(f"    * MUST NOT be used: {detail}")

    lines.append(
        "    * These are a personalization signal, like a stated food preference - "
        "never force one into every meal, and never let a saved recipe override the "
        "diet/allergy/age rules above. Those rules outrank a saved recipe every time."
    )
    return "\n".join(lines)


def _compose_prompt(base_prompt, rag_context, tool_results, recipes_context=None):
    """
    Combine the base profile prompt with optional RAG/tool-result/personal-
    recipe reference sections. Shared by every provider that injects Agent
    v1's precomputed context instead of retrieving/calling anything itself.
    """
    sections = []
    if rag_context:
        sections.append(
            "Reference material (use if relevant to this athlete; rely on your "
            "own knowledge for anything it doesn't cover):\n\n"
            f"{rag_context}"
        )
    if tool_results:
        sections.append(
            "Precomputed values (already calculated for this athlete by "
            "deterministic tools - use these exact numbers where relevant "
            "instead of estimating them yourself):\n\n"
            f"{_format_tool_results(tool_results)}"
        )
    if recipes_context:
        sections.append(recipes_context)
    sections.append(base_prompt)
    return "\n\n".join(sections)


def _validate_plan_shape(plan):
    """Raise LLMProviderError if a parsed model response is missing required keys."""
    missing = [k for k in MEAL_PLAN_SCHEMA["required"] if k not in plan]
    if missing:
        raise LLMProviderError(f"Model response missing required fields: {missing}")


def claude_provider(profile, agent_context=None):
    """
    Get a meal plan from Claude. Same signature (plus one optional
    parameter) and same return shape as mock_provider(profile). Raises
    LLMProviderError on any failure. See the module docstring for how
    agent_context changes behaviour.
    """
    if not LLM_API_KEY:
        raise LLMProviderError(
            "No API key configured. Set ANTHROPIC_API_KEY to use the LLM provider."
        )

    try:
        import anthropic
    except ImportError as exc:
        raise LLMProviderError(
            "The 'anthropic' package is not installed. Run: python3 -m pip install anthropic"
        ) from exc

    base_prompt = build_prompt(profile)  # reuse existing prompt logic - no duplication

    if agent_context is not None:
        # Agent v1 (orchestrator.py) already retrieved knowledge and ran
        # tools - reuse its results instead of doing this work a second
        # time. rag_context/tool_results default to "nothing found" shapes
        # if the orchestrator's own retrieval/tools came back empty.
        rag_context = agent_context.get("rag_context") or ""
        tool_results = agent_context.get("tool_results") or {}
        personal_recipes = agent_context.get("personal_recipes") or []
        logger.info(
            "agent_context supplied: reusing precomputed RAG context (%d chars) and "
            "tool results %s - skipping a second RAG retrieval and disabling "
            "Claude's own tool-calling loop for this request",
            len(rag_context), list(tool_results),
        )
    else:
        # Standalone/legacy path (no orchestrator involved): do our own RAG
        # retrieval, exactly as before this integration.
        rag_context = get_context_block(profile)
        tool_results = None
        personal_recipes = []

    recipes_context = _format_recipes(personal_recipes, profile) if personal_recipes else None
    prompt = _compose_prompt(base_prompt, rag_context, tool_results, recipes_context)

    logger.info("Final prompt length: %d characters", len(prompt))

    try:
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        request_kwargs = dict(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            output_config={
                "effort": LLM_EFFORT,
                "format": {"type": "json_schema", "schema": MEAL_PLAN_SCHEMA},
            },
            messages=[{"role": "user", "content": prompt}],
        )
        if agent_context is not None:
            # Agent v1 already decided which tools to run - Claude is not
            # offered any tools this turn, so a tool-calling loop is not
            # just discouraged, it is structurally impossible (no tools=
            # means the API has nothing to call tool_use on).
            response = client.messages.create(**request_kwargs)
        else:
            # Standalone/legacy path: Claude MAY call nutrition_tool /
            # budget_tool before answering (its choice - see
            # tool_selector.run_with_tools). Structured output (the JSON
            # schema above) still governs its final answer either way.
            response = run_with_tools(client=client, **request_kwargs)
    except Exception as exc:  # network error, auth error, bad request, etc.
        raise LLMProviderError(f"LLM request failed: {exc}") from exc

    # A safety classifier may decline the request (HTTP 200, no usable content).
    if response.stop_reason == "refusal":
        raise LLMProviderError("The model declined to answer this request.")

    # With structured output the first text block is guaranteed-valid JSON.
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise LLMProviderError("The model returned no text content.")

    try:
        plan = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMProviderError(f"Could not parse the model's JSON: {exc}") from exc

    # Cheap sanity check that the shape matches our contract.
    _validate_plan_shape(plan)

    return plan


# ----------------------------------------------------------------------
# OpenRouter provider (free-tier, no structured-output guarantee)
# ----------------------------------------------------------------------

def _call_openrouter_once(messages, response_format, model):
    """
    One OpenRouter chat-completion HTTP call against a specific `model` -
    the actual request/response handling, unchanged from what
    call_openrouter() used to do inline. Raises LLMProviderError (or its
    RateLimitError subclass for HTTP 429) on any failure.
    """
    try:
        import httpx
    except ImportError as exc:
        raise LLMProviderError(
            "The 'httpx' package is not installed. Run: python3 -m pip install httpx"
        ) from exc

    payload = {
        "model": model,
        "max_tokens": OPENROUTER_MAX_TOKENS,
        "reasoning": {"effort": OPENROUTER_EFFORT},
        "messages": messages,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    try:
        response = httpx.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            raise RateLimitError(f"OpenRouter rate limit hit: {exc}") from exc
        raise LLMProviderError(f"OpenRouter request failed: {exc}") from exc
    except httpx.HTTPError as exc:
        raise LLMProviderError(f"OpenRouter request failed: {exc}") from exc

    data = response.json()

    if "error" in data:
        raise LLMProviderError(f"OpenRouter returned an error: {data['error']}")

    choices = data.get("choices") or []
    if not choices:
        raise LLMProviderError("OpenRouter returned no choices in its response.")

    text = choices[0].get("message", {}).get("content")
    if not text:
        raise LLMProviderError("OpenRouter's response contained no text content.")

    return text


def call_openrouter(messages, response_format=None):
    """
    Low-level OpenRouter chat-completion call - the ONE place in the app
    that knows how to talk to OpenRouter's API. Every OpenRouter-based
    feature (openrouter_provider() below, and chat.py's chat feature)
    reuses this instead of each becoming its own provider. Returns the raw
    text content of the model's first choice; raises LLMProviderError on
    any failure (missing key, network, malformed response) - the same
    graceful-failure contract as every other provider in this file.

    `response_format` is optional and passed straight through to OpenRouter
    (e.g. {"type": "json_object"} for callers that need JSON back); omitted
    entirely for callers that just want free text, like chat.

    Primary/fallback (Phase D): tries OPENROUTER_MODEL first; if that call
    raises ANY LLMProviderError (rate limit, network error, empty/malformed
    response - the exact free-tier-congestion failure mode this file's
    RateLimitError docstring describes) and OPENROUTER_FALLBACK_MODEL is
    set to a different model, it is tried once, automatically. Callers
    (openrouter_provider(), chat.py) are unchanged - this is entirely
    internal to this one function.
    """
    if not OPENROUTER_API_KEY:
        raise LLMProviderError(
            "No API key configured. Set OPENROUTER_API_KEY to use the OpenRouter provider."
        )

    try:
        return _call_openrouter_once(messages, response_format, OPENROUTER_MODEL)
    except LLMProviderError as primary_error:
        if not OPENROUTER_FALLBACK_MODEL or OPENROUTER_FALLBACK_MODEL == OPENROUTER_MODEL:
            raise
        logger.warning(
            "OpenRouter primary model %r failed (%s) - retrying once with "
            "fallback model %r",
            OPENROUTER_MODEL, primary_error, OPENROUTER_FALLBACK_MODEL,
        )
        return _call_openrouter_once(messages, response_format, OPENROUTER_FALLBACK_MODEL)


def openrouter_provider(profile, agent_context=None):
    """
    Get a meal plan from a free OpenRouter model. Same contract as
    mock_provider()/claude_provider(): same input, same output shape, raises
    LLMProviderError on any failure instead of crashing (so the mock
    provider stays usable if this one is misconfigured or unreachable).

    Unlike claude_provider(), this has NO standalone/legacy mode - it never
    retrieves RAG context and never calls tools itself, regardless of
    whether agent_context is supplied. With agent_context=None it simply
    generates from the profile alone, injecting no reference sections -
    Agent v1's orchestrator is expected to be the one supplying context.

    Also unlike claude_provider(), this does not use Claude's structured-
    output feature (most free OpenRouter models don't support a JSON
    schema the way Claude does) - it relies on build_prompt()'s existing
    "return ONLY valid JSON in exactly this shape" instructions instead,
    plus OpenRouter's best-effort JSON response_format when the model
    supports it. A model that ignores the instructions produces text that
    fails to parse as JSON, which raises LLMProviderError - a real
    limitation of prompt-only JSON on a free model, not something this
    function can guarantee against.
    """
    base_prompt = build_prompt(profile)  # reuse existing prompt logic - no duplication
    rag_context = (agent_context or {}).get("rag_context") or ""
    tool_results = (agent_context or {}).get("tool_results") or {}
    personal_recipes = (agent_context or {}).get("personal_recipes") or []
    recipes_context = _format_recipes(personal_recipes, profile) if personal_recipes else None
    prompt = _compose_prompt(base_prompt, rag_context, tool_results, recipes_context)

    logger.info(
        "openrouter_provider: model=%s, prompt length=%d characters, tool_results=%s",
        OPENROUTER_MODEL, len(prompt), list(tool_results),
    )

    text = call_openrouter(
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    try:
        plan = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMProviderError(f"Could not parse the model's JSON: {exc}") from exc

    _validate_plan_shape(plan)

    return plan
