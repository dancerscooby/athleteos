"""
AthleteOS - "Ask AthleteOS" chat (v1).

Follow-up chat about an athlete's meal plan, food substitutions, nutrition,
and training/match-day fueling - built entirely out of EXISTING pieces of
the pipeline, not a second AI system:

    llm_provider.call_openrouter()        the SAME OpenRouter HTTP call
                                            openrouter_provider() uses for
                                            meal plans - no new provider.
    rag.retrieve_profile_chunks()          the SAME profile-aware RAG path
    rag.format_context_block()             (FAISS + reranking + age/region
                                            top-ups) the meal-plan pipeline
                                            uses, queried with the chat
                                            message as the main search term.
    meal_planner.build_constraint_rules()  the SAME DIET RULE / AGE
                                            DEVELOPMENT RULE text
                                            build_prompt() emits for meal
                                            generation, reused verbatim so
                                            chat enforces identical hard
                                            constraints instead of a second,
                                            possibly-drifting version.
    memory_manager.get() / .list()         the SAME Profile/Preference/
                                            episodic memory store - read
                                            here, written only for the one
                                            narrow case in
                                            extract_preference_update().

Per-request flow (see chat_reply(), the single entry point):
    message
      -> memory: profile + preference + a couple of recent episode summaries
      -> RAG: retrieve a few relevant knowledge_base chunks for THIS
              message, profile-aware (age/region top-ups) exactly like the
              meal-plan path
      -> prompt: safety-boundary system instructions + hard DIET/AGE/
                 ALLERGY constraints (when a profile exists) + that context
                 + recent conversation turns + the new message
      -> llm_provider.call_openrouter() -> plain text reply (chat replies
         are free text, NOT the meal-plan JSON schema)

Audit finding this fixes: chat's profile block always INCLUDED diet/age/
allergy facts, but only as informational text ("Use the athlete's profile...
when relevant") - a soft suggestion, not a constraint - and chat's own RAG
retrieval had no age/region top-up at all, unlike the meal-plan path. A
10-year-old Vegetarian athlete asking a routine fueling question got a reply
suggesting "1 boiled egg (if you're okay with eggs)" with no age-specific
framing, because nothing in the prompt said diet/age were non-negotiable,
and RAG surfaced a diet-agnostic chunk that happened to mention eggs. See
build_constraint_rules() (meal_planner.py) and _allergy_avoidance_rule()
below for the fix, and retrieve_chat_rag_context() for the RAG fix.

Memory writes are NOT automatic. extract_preference_update() recognizes
exactly two narrow, explicit phrasings ("remember I like/prefer X" and
"remember I avoid/dislike/don't eat X") - a deterministic rule, not an LLM
judgment call, so it stays predictable and testable. Anything else is left
alone; that's the "keep this simple for v1" behaviour the task asked for.
"""

import logging

import memory_manager
from llm_provider import LLMProviderError, call_openrouter
from meal_planner import avoid_keywords, build_constraint_rules, classify_personal_recipes
from memory_integration import DEFAULT_ATHLETE_ID
from rag import format_context_block, retrieve_profile_chunks

logger = logging.getLogger("athleteos.chat")

RAG_TOP_K = 3
MAX_HISTORY_TURNS = 6  # most recent turns (user+assistant messages combined)
MAX_RECENT_EPISODES = 3


# ----------------------------------------------------------------------
# Session isolation (Phase 1)
# ----------------------------------------------------------------------
# Chat transcripts are deliberately NOT stored here - the browser holds the
# messages and sends the recent turns back each turn, so closing/reloading
# the app loses the transcript by construction (exactly the required
# behaviour) and nothing is ever written to disk.
#
# What IS tracked, in-process only, is which athlete each session_id belongs
# to. That single binding is what makes "different athlete -> completely
# separate conversation" a guarantee the SERVER enforces, rather than a
# promise the frontend is trusted to keep: a session_id that shows up under a
# second athlete_id is refused outright. This mirrors memory_architecture.md
# section 2.4 - Session Memory stays as in-process state and is never
# persisted - so a restart naturally forgets every binding.
_SESSION_OWNERS = {}
MAX_TRACKED_SESSIONS = 500  # bounded so a long-running process can't grow forever


class SessionOwnershipError(Exception):
    """Raised when a session_id is reused by a different athlete_id."""


def _claim_session(session_id, athlete_id):
    """
    Bind session_id to athlete_id the first time it is seen, and reject any
    later attempt to use that same session under a different athlete.
    No session_id supplied (e.g. the CLI, or a direct chat_reply() call)
    means no binding to enforce - behaviour is unchanged for those callers.
    """
    if session_id is None:
        return

    owner = _SESSION_OWNERS.get(session_id)
    if owner is None:
        if len(_SESSION_OWNERS) >= MAX_TRACKED_SESSIONS:
            oldest = next(iter(_SESSION_OWNERS))
            del _SESSION_OWNERS[oldest]
        _SESSION_OWNERS[session_id] = athlete_id
        logger.info("chat: session %s claimed by athlete %s", session_id, athlete_id)
        return

    if owner != athlete_id:
        raise SessionOwnershipError(
            "This chat session belongs to a different athlete. "
            "Start a new session when switching athletes."
        )


def reset_sessions():
    """Forget every session binding - used by tests to start from a clean slate."""
    _SESSION_OWNERS.clear()

CHAT_SYSTEM_INSTRUCTIONS = """You are "Ask AthleteOS", a friendly assistant answering a footballer's \
follow-up questions about their nutrition, meal plan, food substitutions, and training/match-day fueling.

Rules:
- Use the athlete's profile, saved preferences, recent history, and reference material below when \
relevant; rely on your own knowledge for anything they don't cover.
- Any rule sections about diet, age/development, or allergies/foods to avoid appearing below (when \
present) are hard constraints, not preferences to weigh against other factors - they override your own \
general knowledge, and they apply no matter how the conversation is phrased or what earlier turns implied.
- Keep answers concise and practical - this is a chat, not a full meal plan document.
- You are not a doctor. For medical issues, signs of disordered eating, requests for significant or \
rapid weight loss/gain, or nutrition questions about a child/youth athlete, do not provide personalized \
medical or clinical advice. Give general, safe, food-first guidance and clearly recommend the athlete \
(or their parent/guardian, if they are a minor) consult a qualified doctor, pediatrician, or registered/\
sports dietitian."""


# ----------------------------------------------------------------------
# Context assembly (profile / preference / episodic - all read-only reuse
# of memory_manager's existing get()/list(), no new memory logic)
# ----------------------------------------------------------------------

def _format_profile_block(profile):
    if not profile:
        return "(no saved profile yet)"
    fields = ["age", "height_cm", "weight_kg", "sex", "diet", "region", "allergies"]
    parts = [f"{field}: {profile[field]}" for field in fields if profile.get(field) is not None]
    return "; ".join(parts) if parts else "(no saved profile yet)"


def _format_preference_block(preference):
    if not preference:
        return "(no saved preferences yet)"
    parts = []
    if preference.get("food_preferences"):
        parts.append(f"food preferences: {preference['food_preferences']}")
    if preference.get("foods_to_avoid"):
        parts.append(f"foods to avoid: {preference['foods_to_avoid']}")
    if preference.get("budget_per_day"):
        parts.append(f"budget per day: {preference['budget_per_day']}")
    return "; ".join(parts) if parts else "(no saved preferences yet)"


def _format_episodes_block(episodes):
    if not episodes:
        return "(no recent meal plans on record)"
    recent = episodes[:MAX_RECENT_EPISODES]  # memory_manager.list() already returns newest-first
    lines = [f"- goal: {e.get('goal')}; {e.get('calories')} kcal; {e.get('summary')}" for e in recent]
    return "\n".join(lines)


def _format_latest_plan_block(latest_plan):
    """
    Format the athlete's most recently generated plan (see
    episodic_memory.record_latest_plan_summary() - a bounded, update-in-place
    summary, not the full plan JSON) for the chat prompt: duration/goal/diet/
    region, each day's meal names and calorie/macro totals, and the plan's
    protein/carb sources. Returns a fixed placeholder when nothing has been
    generated yet, so this section behaves like every other memory block
    here (profile/preference/episodes) rather than silently vanishing.
    """
    if not latest_plan:
        return "(no meal plan generated yet)"
    header = (
        f"Duration: {latest_plan.get('duration_days')} day(s); "
        f"Goal: {latest_plan.get('goal')}; Diet: {latest_plan.get('diet')}; "
        f"Region: {latest_plan.get('region')}"
    )
    day_lines = [
        f"- Day {d.get('day')}: {d.get('total_calories')} kcal, "
        f"{d.get('total_protein_g')}g protein, {d.get('total_carbs_g')}g carbs, "
        f"{d.get('total_fats_g')}g fat - {d.get('meals')}"
        for d in latest_plan.get("days", [])
    ]
    lines = [header] + day_lines
    if latest_plan.get("protein_sources"):
        lines.append(f"Protein sources across this plan: {', '.join(latest_plan['protein_sources'])}")
    if latest_plan.get("carb_sources"):
        lines.append(f"Carb/staple sources across this plan: {', '.join(latest_plan['carb_sources'])}")
    return "\n".join(lines)


def _format_recipes_block(recipes, profile):
    """
    Format the athlete's saved personal recipes for the chat prompt -
    classified via meal_planner.classify_personal_recipes(), the SAME
    function mock_provider() and the LLM meal-plan providers use, so chat
    never gives a different safety answer than plan generation would.
    Never sourced from RAG - see personal_recipes.py's module docstring.
    """
    if not recipes:
        return "(no saved recipes yet)"
    classification = classify_personal_recipes(recipes, profile)
    lines = []
    for r in classification["usable"]:
        lines.append(f"- {r['name']} (safe for this profile): {', '.join(r.get('ingredients') or [])}")
    for e in classification["excluded"]:
        lines.append(f"- {e['name']} (NOT suitable right now - {e['reason']})")
    return "\n".join(lines)


def build_memory_context(athlete_id):
    """
    Read (never write) the athlete's Profile/Preference memory, recent
    episodic history, and saved personal recipes via the existing
    memory_manager operations, and return them as short text blocks ready
    to drop into the chat prompt - plus the raw profile/preference dicts
    themselves, so build_messages() can build the hard constraint rules and
    the profile-aware RAG query from real field values rather than
    re-parsing the formatted text blocks.
    """
    profile = memory_manager.get("profile", athlete_id)
    preference = memory_manager.get("preference", athlete_id)
    episodes = memory_manager.list("episodic", athlete_id, limit=MAX_RECENT_EPISODES)
    latest_plan = memory_manager.get("latest_plan", athlete_id)
    recipes = memory_manager.get("recipes", athlete_id) or []
    combined_profile = {**(profile or {}), **(preference or {})}
    return {
        "profile": profile,
        "preference": preference,
        "profile_block": _format_profile_block(profile),
        "preference_block": _format_preference_block(preference),
        "episodes_block": _format_episodes_block(episodes),
        "latest_plan_block": _format_latest_plan_block(latest_plan),
        "recipes_block": _format_recipes_block(recipes, combined_profile),
    }


def _allergy_avoidance_rule(profile):
    """
    Hard allergy/foods-to-avoid constraint for chat.

    meal_planner.build_prompt() has no equivalent section to reuse here - it
    relies on the model reading the raw "Allergies: X" profile line on its
    own. This uses the SAME expansion meal_planner.avoid_keywords() already
    applies to real food selection (e.g. "dairy" -> milk/curd/paneer/cheese/
    ghee/buttermilk, plus singular/plural forms), so chat is told the same
    concrete excluded foods the meal plan itself already enforces, instead
    of being left to infer the expansion itself. Returns "" when there is
    nothing to exclude (no allergies/foods_to_avoid on file), so a profile
    with nothing stored for either field emits no rule text.
    """
    excluded = sorted(avoid_keywords(profile))
    if not excluded:
        return ""
    return f"""
ALLERGY / AVOIDANCE RULE
This athlete must never be recommended, nor have described as suitable, any food matching: \
{", ".join(excluded)}. Treat this as an absolute exclusion, not a preference to weigh against other \
factors - if a food or dish might contain any of these, say so plainly and suggest a safer alternative \
rather than recommending it or assuming it is probably fine."""


def build_constraint_block(profile):
    """
    Assemble the full hard-constraint text for one chat turn: the SAME DIET
    RULE / AGE DEVELOPMENT RULE text meal_planner.build_prompt() emits for
    meal generation (see meal_planner.build_constraint_rules()), plus the
    chat-specific ALLERGY / AVOIDANCE RULE above. Returns "" when there is
    no profile at all yet (a brand-new athlete) - nothing to constrain on,
    and the existing "(no saved profile yet)" text already communicates
    that further down the prompt.
    """
    if not profile:
        return ""
    return build_constraint_rules(profile) + _allergy_avoidance_rule(profile) + "\n\n"


def retrieve_chat_rag_context(message, profile, top_k=RAG_TOP_K):
    """
    Retrieve the top_k knowledge_base chunks most relevant to THIS chat
    message, using the SAME profile-aware retrieval path the meal-plan
    pipeline uses - rag.retrieve_profile_chunks(), which runs the main
    search plus age/region top-ups (see rag.py) - rather than a second,
    message-only retrieval mechanism. The main search still uses the raw
    chat message as the query (so retrieval stays relevant to what was
    actually asked); `profile` only drives the top-ups, exactly like
    orchestrator.py's meal-plan retrieval.

    Audit finding this fixes: chat previously called
    retriever.retrieve_relevant_chunks(message) directly, with no age/region
    top-up at all - a message like "what should I eat before training?"
    could retrieve neither youth guidance nor regional content even for a
    10-year-old Chennai athlete, where the meal-plan path would have
    surfaced both via retrieve_profile_chunks()'s top-ups.
    """
    chunks = retrieve_profile_chunks(profile or {}, message, top_k=top_k)
    logger.info(
        "chat RAG retrieval for %r: %d chunks (%s)",
        message, len(chunks), [c["id"] for c in chunks],
    )
    return format_context_block(chunks)


# ----------------------------------------------------------------------
# Prompt assembly
# ----------------------------------------------------------------------

def build_messages(message, athlete_id, history=None):
    """
    Assemble the full OpenRouter `messages` list for one chat turn: system
    instructions + hard constraint rules (diet/age/allergy, when a profile
    exists) + profile/memory/RAG context, then the most recent conversation
    turns (capped at MAX_HISTORY_TURNS), then the new message.

    Freshly rebuilt on every call from athlete_id (never cached) - the
    active athlete's profile/preference/RAG context is read from memory
    each turn, so switching athletes (a new athlete_id on the next call)
    always uses the newly active athlete's own data, never anything left
    over from a previous athlete in this process.
    """
    memory_context = build_memory_context(athlete_id)
    # Profile and preference memory are stored as two separate categories
    # (memory_integration.py's PROFILE_FIELDS/PREFERENCE_FIELDS split), but
    # allergies (profile) and foods_to_avoid (preference) both need to be
    # visible together for the hard constraint rules and the RAG top-ups -
    # merged the same way memory_integration.merge_with_memory() already does.
    combined_profile = {**(memory_context["profile"] or {}), **(memory_context["preference"] or {})}

    constraint_block = build_constraint_block(combined_profile)
    rag_context = retrieve_chat_rag_context(message, combined_profile)

    system_content = (
        f"{CHAT_SYSTEM_INSTRUCTIONS}\n\n"
        f"{constraint_block}"
        f"ATHLETE PROFILE\n{memory_context['profile_block']}\n\n"
        f"SAVED PREFERENCES\n{memory_context['preference_block']}\n\n"
        f"MOST RECENT MEAL PLAN (may be several days; use this, not RECENT MEAL PLANS below, "
        f"for questions about a specific day or the current plan as a whole)\n"
        f"{memory_context['latest_plan_block']}\n\n"
        f"RECENT MEAL PLANS (older generation history, one line each)\n{memory_context['episodes_block']}\n\n"
        f"SAVED PERSONAL RECIPES (this athlete's own homemade/family recipes - a "
        f"personalization signal, never a reason to override the diet/allergy rules above)\n"
        f"{memory_context['recipes_block']}\n\n"
        f"REFERENCE MATERIAL (use if relevant; rely on your own knowledge otherwise)\n"
        f"{rag_context or '(nothing relevant found)'}"
    )

    messages = [{"role": "system", "content": system_content}]
    for turn in (history or [])[-MAX_HISTORY_TURNS:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})
    return messages


# ----------------------------------------------------------------------
# Narrow, deterministic preference-saving rule (NOT automatic per message)
# ----------------------------------------------------------------------

_LIKE_TRIGGERS = ["remember i like ", "remember that i like ", "remember i prefer ", "remember that i prefer "]
_AVOID_TRIGGERS = [
    "remember i avoid ", "remember that i avoid ",
    "remember i dislike ", "remember that i dislike ",
    "remember i don't eat ", "remember that i don't eat ",
    "remember i do not eat ", "remember that i do not eat ",
]


def extract_preference_update(message):
    """
    Decide whether a chat message explicitly states a persistent preference
    worth remembering. Deliberately a fixed, narrow rule - not an LLM
    judgment call - so it stays predictable and testable, per the "keep
    this simple for v1" instruction: only "remember I like/prefer X" and
    "remember I avoid/dislike/don't eat X" are recognized. Anything else
    (including ordinary chat that merely mentions a food) returns None and
    nothing is saved.

    Returns (field, value) where field is "food_preferences" or
    "foods_to_avoid", or None.
    """
    stripped = message.strip()
    lower = stripped.lower()

    for trigger in _LIKE_TRIGGERS:
        if lower.startswith(trigger):
            value = stripped[len(trigger):].strip().rstrip(".!")
            return ("food_preferences", value) if value else None

    for trigger in _AVOID_TRIGGERS:
        if lower.startswith(trigger):
            value = stripped[len(trigger):].strip().rstrip(".!")
            return ("foods_to_avoid", value) if value else None

    return None


def save_preference_update(field, value, athlete_id):
    """
    Persist ONE extracted preference into the existing preference-memory
    store, appending to whatever is already saved for that field rather
    than overwriting it (memory_manager.set()/get() reused as-is - no new
    memory category or logic).
    """
    stored = memory_manager.get("preference", athlete_id) or {}
    existing = stored.get(field)
    combined = f"{existing}, {value}" if existing else value
    updated = dict(stored, **{field: combined})
    memory_manager.set("preference", athlete_id, updated)
    logger.info(
        "chat: saved preference update for athlete_id=%r: %s += %r (now %r)",
        athlete_id, field, value, combined,
    )
    return combined


# ----------------------------------------------------------------------
# The single entry point
# ----------------------------------------------------------------------

def chat_reply(message, athlete_id=DEFAULT_ATHLETE_ID, history=None, session_id=None):
    """
    Answer one chat message. Returns:
        {"reply": str, "memory_updated": {"field": ..., "value": ...} | None}

    `session_id` (optional) scopes the conversation: the first athlete to use
    a given session owns it, and a different athlete presenting the same
    session_id is refused with SessionOwnershipError. Omitting it (the CLI,
    direct calls, older tests) skips that check entirely - behaviour
    unchanged for those callers.

    Raises llm_provider.LLMProviderError on any failure (missing API key,
    network error, empty response) - the caller (api.py) is expected to
    turn that into a clean HTTP error rather than a raw traceback.
    """
    _claim_session(session_id, athlete_id)

    messages = build_messages(message, athlete_id, history)
    reply = call_openrouter(messages)  # plain text - no response_format/JSON schema

    memory_updated = None
    extracted = extract_preference_update(message)
    if extracted:
        field, value = extracted
        save_preference_update(field, value, athlete_id)
        memory_updated = {"field": field, "value": value}

    return {"reply": reply, "memory_updated": memory_updated}
