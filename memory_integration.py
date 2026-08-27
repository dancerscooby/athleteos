"""
AthleteOS - Memory integration, Stage 1 (Profile + Preference only).

Connects memory_manager.py's generic get/set operations to the athlete
profile dict already used throughout the app, WITHOUT modifying
memory_manager.py, generate_meal_plan(), or any request/response schema.
Two functions, used one on each side of a request:

    merge_with_memory(profile, athlete_id)
        called BEFORE generate_meal_plan() - fills gaps in the incoming
        profile from the athlete's stored memory.

    update_memory_from_profile(profile, athlete_id)
        called AFTER a successful generate_meal_plan() call - persists
        whatever changed.

Scope for this stage (see memory_architecture.md):
  - Only the "profile" and "preference" categories - no episodic, no
    session, no cache yet.
  - PROFILE_FIELDS and PREFERENCE_FIELDS match memory_architecture.md's own
    example field lists for those two categories exactly.
  - "goal", "training_days_per_week", "training_intensity", and "notes" are
    intentionally EXCLUDED from memory. They describe what THIS request is
    about (e.g. "match tomorrow at 9 AM"), not a stable fact about the
    athlete - silently reapplying an old "notes: match tomorrow" to an
    unrelated future request would be actively wrong, not helpful. This
    mirrors memory_architecture.md's distinction between Profile/Preference
    Memory and the (not-yet-implemented) Session Memory.
  - "food_preferences" and "foods_to_avoid" are STORED as preference memory
    (still logged/updated below) but are NOT filled back into a later
    request that leaves them blank - the same "don't silently reapply an
    old value" reasoning as goal/notes above, applied after real usage
    showed it in practice: toggling these fields back and forth got
    confusingly "stuck" on whatever was last submitted, since a blank
    field could never be told apart from "no opinion this time". Only
    budget_per_day still fills forward, since that one hasn't shown the
    same problem and is closer to a stable fact than a per-request choice.
  - No authentication exists yet (see the project's original MVP
    principles), so every request is currently treated as the same
    DEFAULT_ATHLETE_ID - exactly the placeholder memory_architecture.md
    Section 5 describes until real accounts exist.
"""

import logging

import memory_manager

logger = logging.getLogger("athleteos.memory_integration")

DEFAULT_ATHLETE_ID = "default"

# Exactly the fields memory_architecture.md lists as Profile / Preference
# Memory content (Section 2.1 / 2.2). Deliberately excludes goal/notes/
# training_days_per_week/training_intensity - see module docstring.
PROFILE_FIELDS = ["age", "height_cm", "weight_kg", "sex", "diet", "region", "allergies"]
PREFERENCE_FIELDS = ["budget_per_day", "food_preferences", "foods_to_avoid"]

# The subset of the above that a later, blank request is allowed to inherit
# from memory - see the module docstring for why food_preferences/
# foods_to_avoid are deliberately left out of this list even though they're
# still stored.
MERGE_FIELDS = PROFILE_FIELDS + ["budget_per_day"]


def merge_with_memory(profile, athlete_id=DEFAULT_ATHLETE_ID):
    """
    Return a NEW profile dict with Profile/Preference gaps filled from
    stored memory.

    Merge rule, applied per field in MERGE_FIELDS (see that constant for why
    food_preferences/foods_to_avoid are deliberately excluded even though
    they're stored):
      - profile[field] is not None  -> keep it (request wins)
      - profile[field] is None AND memory has a stored value -> use it
      - profile[field] is None AND memory has nothing either -> leave None
        (identical to today's behaviour)

    All other fields (goal, notes, training_*, food_preferences,
    foods_to_avoid, etc.) pass through completely untouched - this function
    never fills them from memory, though update_memory_from_profile() may
    still store some of them.
    """
    stored_profile = memory_manager.get("profile", athlete_id) or {}
    stored_preference = memory_manager.get("preference", athlete_id) or {}
    stored = {**stored_profile, **stored_preference}

    merged = dict(profile)
    filled_from_memory = []
    for field in MERGE_FIELDS:
        if merged.get(field) is None and stored.get(field) is not None:
            merged[field] = stored[field]
            filled_from_memory.append(field)

    if filled_from_memory:
        logger.info(
            "merge_with_memory(athlete_id=%r): filled %s from stored memory",
            athlete_id, filled_from_memory,
        )
    else:
        logger.info(
            "merge_with_memory(athlete_id=%r): request already complete, "
            "nothing filled from memory",
            athlete_id,
        )
    return merged


def update_memory_from_profile(profile, athlete_id=DEFAULT_ATHLETE_ID):
    """
    Persist the resolved (post-merge) Profile/Preference values - but only
    the fields that actually changed versus what's already stored, so an
    unchanged field is never rewritten.
    """
    _update_category("profile", PROFILE_FIELDS, profile, athlete_id)
    _update_category("preference", PREFERENCE_FIELDS, profile, athlete_id)


def _update_category(category, fields, profile, athlete_id):
    """Shared logic for updating one category (profile or preference)."""
    stored = memory_manager.get(category, athlete_id) or {}
    updated = dict(stored)
    changed_fields = []

    for field in fields:
        new_value = profile.get(field)
        if new_value is None:
            continue  # nothing new to remember for this field
        if stored.get(field) != new_value:
            updated[field] = new_value
            changed_fields.append(field)

    if changed_fields:
        memory_manager.set(category, athlete_id, updated)
        logger.info(
            "update_memory_from_profile(athlete_id=%r): updated %s in %r memory",
            athlete_id, changed_fields, category,
        )
    else:
        logger.info(
            "update_memory_from_profile(athlete_id=%r): no changes in %r memory",
            athlete_id, category,
        )
