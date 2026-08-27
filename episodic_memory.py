"""
AthleteOS - Episodic Memory integration, Stage 2.

Records a short, non-sensitive summary of every successful meal-plan
generation, using memory_manager.py's EXISTING append()/list() operations -
memory_manager.py itself is not modified.

Kept as its own module, separate from memory_integration.py (Profile /
Preference, Stage 1), because episodic memory has a fundamentally different
shape and purpose:
  - Profile/Preference are "current facts about the athlete" - overwritten
    in place, read back every request (see memory_integration.py).
  - Episodic is "a history log of what happened" - append-only, one entry
    per generation, never overwritten. Mixing that into the Stage 1 module
    would blur two genuinely different access patterns into one file.

What's stored per episode - deliberately a SUMMARY, not a transcript:
    timestamp, goal, training_intensity, training_days_per_week,
    calories, confidence_score, and a short one-line plan summary.

What's NEVER stored here: the full prompt sent to the LLM, the raw LLM
response, the RAG context block, tool call logs, or any other debug/
internal data. Those already live in the existing `logging` output
(athleteos.rag, athleteos.tools, athleteos.llm_provider, etc.) - ephemeral
console output, not persisted memory - matching the distinction
memory_architecture.md draws between memory and debug logs (Section 3).
"""

import logging
import time

import memory_manager
from memory_integration import DEFAULT_ATHLETE_ID

logger = logging.getLogger("athleteos.episodic_memory")


def _summarize_meals(plan):
    """
    Build a short, one-line summary of the plan's meals, e.g.
    "Breakfast: Poha with peanuts | Lunch: Dal | Snack: Roasted chana |
    Dinner: Rajma". Only each meal's name and its FIRST ingredient are
    used - not the full ingredient list, quantities, reasoning, or
    resource links from the plan.
    """
    parts = []
    for meal in plan.get("meals", []):
        ingredients = meal.get("ingredients") or []
        main_ingredient = ingredients[0]["item"] if ingredients else "?"
        parts.append(f"{meal.get('meal', '?')}: {main_ingredient}")
    return " | ".join(parts)


def create_episode(profile, plan):
    """
    Build the concise episode summary dict for one successful generation.
    Reads only the specific fields needed from `profile` (the merged
    profile used for this request) and `plan` (generate_meal_plan()'s
    return value) - never stores either of them wholesale.
    """
    day_summary = plan.get("day_summary", {})
    return {
        "timestamp": time.time(),
        "goal": profile.get("goal"),
        "training_intensity": profile.get("training_intensity"),
        "training_days_per_week": profile.get("training_days_per_week"),
        "calories": day_summary.get("total_calories"),
        "confidence_score": plan.get("confidence_score"),
        "summary": _summarize_meals(plan),
    }


def record_episode(profile, plan, athlete_id=DEFAULT_ATHLETE_ID):
    """
    Create and store one episode for this successful generation, via
    memory_manager.append() (category "episodic", unchanged). Pruning
    (max entry count / max age) is entirely memory_manager's existing
    responsibility - this function never touches storage limits itself.
    """
    episode = create_episode(profile, plan)
    memory_manager.append("episodic", athlete_id, episode)
    logger.info(
        "record_episode(athlete_id=%r): stored episode "
        "(goal=%r, calories=%s, confidence_score=%s)",
        athlete_id, episode["goal"], episode["calories"], episode["confidence_score"],
    )
    return episode


# ----------------------------------------------------------------------
# Latest plan summary (Step 9) - a SEPARATE, bounded, update-in-place
# memory, not a change to episodic's append-only schema
# ----------------------------------------------------------------------
# Revisits the Phase 2 decision not to record multi-day plans at all: that
# decision was correct for the "episodic" category specifically (its
# schema - one day_summary, one meals list - is single-day-shaped, and
# forcing a 7-day plan into it would have meant either silently dropping 6
# of the 7 days or inventing a new episodic shape that every existing
# episodic reader would need to special-case). It does NOT mean chat should
# stay blind to the plan an athlete just generated.
#
# So this stores a SEPARATE, compact, bounded summary - via
# memory_manager's "latest_plan" category (get/set, update-in-place, see
# memory_manager.py) - that a NEW generation always overwrites (never
# accumulates), for exactly one day OR many:
#   duration, goal, diet, region, each day's meal names, each day's
#   calorie/macro totals, and the plan's protein/carb sources (deduplicated,
#   so naturally bounded by the regional candidate pool size - at most
#   ~9 items each, never unbounded).
# Deliberately excluded, per the same "summary, not a transcript" principle
# create_episode() already follows: full ingredient lists, quantities,
# reasoning text, resource links, available_ingredients, and notes.

def _plan_days(plan):
    """
    Normalise EITHER response shape generate_meal_plan() can return into a
    uniform list of {"day", "day_summary", "meals"} dicts - a duration_days=1
    plan (top-level day_summary/meals, no "days" key) becomes a single-entry
    list, so the rest of this module handles both shapes with one code path
    instead of two.
    """
    if "days" in plan:
        return plan["days"]
    return [{"day": 1, "day_summary": plan["day_summary"], "meals": plan["meals"]}]


def create_latest_plan_summary(profile, plan):
    """
    Build the compact, bounded multi-day-aware summary - see the module
    note above for exactly what is and is not included. Works for both a
    1-day and a multi-day plan (see _plan_days()).
    """
    days = _plan_days(plan)
    protein_sources = set()
    carb_sources = set()
    days_out = []
    for day in days:
        s = day["day_summary"]
        days_out.append({
            "day": day["day"],
            "total_calories": s.get("total_calories"),
            "total_protein_g": s.get("total_protein_g"),
            "total_carbs_g": s.get("total_carbs_g"),
            "total_fats_g": s.get("total_fats_g"),
            "meals": _summarize_meals(day),
        })
        # Lunch/Dinner ingredients are always [carb, protein, ...] (see
        # meal_planner.build_lunch()/build_dinner()) - Breakfast/Snack don't
        # have an equivalent single "protein"/"carb" slot, so only these two
        # meals contribute to the source lists.
        for meal in day.get("meals", []):
            if meal.get("meal") in ("Lunch", "Dinner"):
                ingredients = meal.get("ingredients") or []
                if len(ingredients) >= 2:
                    carb_sources.add(ingredients[0]["item"])
                    protein_sources.add(ingredients[1]["item"])
    return {
        "timestamp": time.time(),
        "duration_days": len(days),
        "goal": profile.get("goal"),
        "diet": profile.get("diet"),
        "region": profile.get("region"),
        "days": days_out,
        "protein_sources": sorted(protein_sources),
        "carb_sources": sorted(carb_sources),
    }


def record_latest_plan_summary(profile, plan, athlete_id=DEFAULT_ATHLETE_ID):
    """
    Store this generation's compact summary as the athlete's "latest_plan" -
    memory_manager.set() OVERWRITES whatever was there before (unlike
    record_episode()'s append()), so a new plan always supersedes the
    previous one rather than accumulating - exactly the semantics the
    "most recent plan" use case needs, and exactly why this uses a
    different underlying category than episodic history.
    """
    summary = create_latest_plan_summary(profile, plan)
    memory_manager.set("latest_plan", athlete_id, summary)
    logger.info(
        "record_latest_plan_summary(athlete_id=%r): stored (duration_days=%d, goal=%r)",
        athlete_id, summary["duration_days"], summary["goal"],
    )
    return summary
