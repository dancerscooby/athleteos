"""
AthleteOS v0.1 - Step 1: The swappable AI seam.

What this script does (all offline: no API key, no external packages,
standard library only):

  1. Asks you for an athlete's details in the terminal.
  2. Builds the exact prompt we WOULD send to an LLM and prints it, so you
     can paste it into Claude.ai and judge the meal-plan quality for free.
  3. Runs a MOCK "AI" provider that returns a realistic meal plan in a
     fixed JSON shape - the SAME shape a real LLM will return later.

The key architectural idea: every part of the app talks to ONE function,

    generate_meal_plan(profile)

so we can swap the mock for a real Claude call later by setting one
environment variable (ATHLETEOS_PROVIDER=claude) without touching anything
else.
"""

import hashlib
import json
import os

from regional_foods import REGIONS, SAFE_VEG_PROTEIN, detect_region
from recipe_links import get_recipe_resource
from nutrition_data import get_nutrition_info
from tools.ingredients_tool import NON_VEGETARIAN


# Shown at the end of every generated meal plan (see mock_provider / build_prompt).
DISCLAIMER = (
    "This meal plan is AI-generated for educational purposes and should not replace "
    "advice from a qualified sports nutritionist or medical professional."
)


def _format_number(value):
    """
    For DISPLAY only: show a whole number without a trailing '.0'
    (72.0 -> '72') but keep genuine decimals (72.5 -> '72.5').
    Does not change how the value is stored in the profile.
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


# ----------------------------------------------------------------------
# 1. COLLECT THE ATHLETE PROFILE  (interactive terminal input)
# ----------------------------------------------------------------------

def ask_required(label):
    """Keep asking until the user types something non-empty."""
    while True:
        value = input(label + ": ").strip()
        if value:
            return value
        print("  (this field is required - please enter a value)")


def ask_optional(label):
    """Return the typed value, or None if the user just presses Enter."""
    value = input(label + " (optional - press Enter to skip): ").strip()
    return value or None


def ask_int(label, low, high):
    """
    Ask for a WHOLE number between low and high (inclusive).
    Re-prompts with a simple message if the value is blank, not a whole
    number, or outside the sensible range.
    """
    while True:
        raw = input(f"{label} ({low}-{high}): ").strip()
        if not raw:
            print("  (this field is required - please enter a value)")
            continue
        try:
            value = int(raw)
        except ValueError:
            print(f"  (please enter a whole number, e.g. {low})")
            continue
        if value < low or value > high:
            print(f"  (that seems off - please enter a value between {low} and {high})")
            continue
        return value


def ask_number(label, low, high):
    """
    Ask for a positive, realistic number between low and high (decimals
    allowed, e.g. a weight of 72.5). Re-prompts on blank, non-numbers, or
    unrealistic values.
    """
    while True:
        raw = input(f"{label} ({low}-{high}): ").strip()
        if not raw:
            print("  (this field is required - please enter a value)")
            continue
        try:
            value = float(raw)
        except ValueError:
            print(f"  (please enter a number, e.g. {low})")
            continue
        if value < low or value > high:
            print(f"  (that seems off - please enter a value between {low} and {high})")
            continue
        return value


def ask_choice(label, options):
    """
    Ask until the answer matches one of the allowed options. Matching is
    case-insensitive (so 'high' is accepted), and the clean version from
    'options' is returned (so it is stored as 'High').
    """
    shown = " / ".join(options)
    while True:
        raw = input(f"{label} ({shown}): ").strip()
        if not raw:
            print("  (this field is required - please enter a value)")
            continue
        for option in options:
            if raw.lower() == option.lower():
                return option
        print(f"  (please choose one of: {shown})")


def collect_profile():
    """Ask the user for every field and return a profile dictionary."""
    print("=" * 64)
    print("AthleteOS - enter the athlete's details")
    print("=" * 64)

    profile = {
        # --- mandatory fields (with value checks, not just presence) ---
        "age": ask_int("Age", 10, 60),
        "height_cm": ask_number("Height (cm)", 120, 230),
        "weight_kg": ask_number("Weight (kg)", 30, 200),
        "sex": ask_choice("Sex", ["Male", "Female", "Other"]),
        "goal": ask_required("Goal"),
        "diet": ask_choice("Diet", ["Vegetarian", "Non-Vegetarian"]),
        "training_days_per_week": ask_int("Training frequency (days/week)", 1, 7),
        "training_intensity": ask_choice("Training intensity", ["Low", "Moderate", "High"]),
        "region": ask_required("Region (city / state)"),
        "allergies": ask_required("Allergies (type 'none' if no allergies)"),
        # --- optional fields (blank -> None) ---
        "budget_per_day": ask_optional("Budget per day (e.g. Rs 300)"),
        "food_preferences": ask_optional("Food preferences"),
        "foods_to_avoid": ask_optional("Foods to avoid"),
        "notes": ask_optional("Additional notes"),
    }
    return profile


# ----------------------------------------------------------------------
# 2. BUILD THE PROMPT  (pure text - no network, no API)
# ----------------------------------------------------------------------

def build_constraint_rules(profile):
    """
    The DIET RULE and AGE / DEVELOPMENT RULE text, exactly as build_prompt()
    has always emitted it - extracted so chat.py can reuse the identical,
    already-tested hard-constraint wording for its own system prompt instead
    of writing a second version that could drift out of sync with this one.
    build_prompt() below calls this and interpolates the result in exactly
    the position this text has always occupied, so its own output is
    unchanged, character for character. Uses profile.get('age') rather than
    profile['age'] so chat.py (which may call this with an incomplete or
    not-yet-saved profile) never crashes on a missing key - build_prompt()'s
    own caller always supplies a validated profile with age present, so this
    is a no-op change for build_prompt()'s existing output.
    """
    age = profile.get("age")
    age_text = age if age is not None else "unknown"
    return f"""DIET RULE
In this Indian context, "Vegetarian" means no meat, fish, or eggs by default -
this is stricter than some Western definitions. Only include eggs for a
Vegetarian athlete if "Food preferences" above explicitly mentions eggs;
otherwise never include eggs, meat, or fish anywhere in the plan for a
Vegetarian diet.

AGE / DEVELOPMENT RULE
This athlete's age is {age_text}. If they are under 18 - and
especially if they are a child (roughly under 13) - do NOT apply adult
calorie, protein, portion, or muscle-gain/bulking norms to them: a growing
child or teenager is not a smaller adult. If reference material about youth
athlete nutrition appears above, treat it as authoritative for this
athlete's food choices and portions rather than optional background, and
favor age-appropriate, growth-supportive food selection over an adult-style
bulking or cutting approach. Do not state a specific calorie, protein, or
supplement recommendation for a minor beyond what the reference material or
precomputed values above actually support - say the exact number depends on
age, sex, growth stage, and individual factors instead of inventing one. If
the goal involves significant weight change, restrictive eating, or
muscle-building for a minor, note that this is worth discussing with a
parent/guardian and a pediatrician or sports dietitian, rather than treating
it as a plan you can fully resolve alone."""


def build_prompt(profile):
    """
    Turn the profile into the full instruction text for an LLM.

    'Light guidance' style: we describe WHO the athlete is and the
    constraints, and trust the model's own nutrition knowledge instead
    of hardcoding calorie / macro formulas.
    """
    # Include only the optional fields the user actually filled in.
    optional_lines = []
    for key, label in [
        ("budget_per_day", "Budget per day"),
        ("food_preferences", "Food preferences"),
        ("foods_to_avoid", "Foods to avoid"),
        ("notes", "Additional notes"),
    ]:
        if profile.get(key):
            optional_lines.append(f"- {label}: {profile[key]}")
    optional_block = "\n".join(optional_lines) if optional_lines else "- (none provided)"

    # Only emitted when the athlete actually listed ingredients. A request
    # without them must read exactly as it did before this feature existed -
    # no stray instructions about a pantry nobody described.
    ingredients_rule = ""
    if profile.get("available_ingredients"):
        ingredients_rule = """
AVAILABLE INGREDIENTS RULE
This athlete listed ingredients they already have at home, and a "HOME
INGREDIENTS" block may appear in the precomputed values above with each one
already checked against their diet and allergies.

- Make good use of what they already have where it genuinely fits the meal.
  Do NOT force every listed ingredient into the plan - a sensible plan that
  uses some of them beats an awkward one that uses all of them.
- Anything listed as "MUST NOT be used" is excluded by an allergy, a food
  they avoid, or their diet. Never include it, and never suggest it as a
  substitute. Those rules outrank availability every time.
- Anything listed as "could not be verified" was typed by the athlete but is
  not a confirmed usable food. Do not build a meal around it and do not
  assume it is suitable or nutritious just because they mentioned it.
- Having an ingredient is not a reason to break anything else: the diet rule,
  the age/development rule, regional suitability and the day's nutritional
  needs all still apply in full.
- If what they have is not enough for a sensible, balanced day, say so plainly
  and name a small number of realistic, affordable additions to pick up -
  rather than pretending the available ingredients are sufficient.
"""

    return f"""You are a sports nutritionist for Indian footballers who cannot afford a
personal nutritionist. Give realistic, practical advice using foods that are
actually available in the athlete's region and within their budget and culture.
Prefer home-cooked, easily available Indian ingredients. Do not assume expensive
supplements or imported foods. Be honest about uncertainty.

ATHLETE PROFILE
- Age: {profile['age']}
- Height: {_format_number(profile['height_cm'])} cm
- Weight: {_format_number(profile['weight_kg'])} kg
- Sex: {profile['sex']}
- Goal: {profile['goal']}
- Diet: {profile['diet']}
- Training: {profile['training_days_per_week']} days/week, intensity {profile['training_intensity']}
- Region: {profile['region']}
- Allergies: {profile['allergies']}
{optional_block}

{build_constraint_rules(profile)}
{ingredients_rule}
TASK
Generate today's meal plan with exactly four meals: Breakfast, Lunch, Snack, Dinner.
For each meal give ingredients with quantities, calories, protein, carbs and fats,
a short reason why it suits THIS athlete, and any useful resources (recipe videos
or articles). Also give an overall daily summary and an honest confidence score.

Return ONLY valid JSON in exactly this shape:
{{
  "confidence_score": <number 0-10>,
  "confidence_reason": [<short strings explaining the score>],
  "day_summary": {{
    "total_calories": <int>, "total_protein_g": <int>,
    "total_carbs_g": <int>, "total_fats_g": <int>, "notes": "<string>"
  }},
  "meals": [
    {{
      "meal": "Breakfast",
      "ingredients": [{{"item": "<string>", "quantity": "<string>"}}],
      "calories": <int>, "protein_g": <int>, "carbs_g": <int>, "fats_g": <int>,
      "reasoning": "<why this meal for this athlete>",
      "resources": [{{"type": "youtube", "title": "<string>", "url": "<string>", "item": "<string>"}}]
    }}
  ],
  "disclaimer": "<string>"
}}
Each resource's "item" must be an EXACT match of one of that same meal's
ingredient "item" names above - it says which ingredient the recipe link is
for, so do not invent a new name or use a dish name that isn't also an
ingredient.
Lower the confidence_score when important inputs (budget, allergies, training
intensity) are missing or unclear, and say so in confidence_reason.
For "disclaimer", use exactly this text:
"{DISCLAIMER}"
"""


# ----------------------------------------------------------------------
# 3. THE AI PROVIDERS  (interchangeable implementations)
# ----------------------------------------------------------------------

def _confidence(profile):
    """
    An honest, simple confidence score based on how complete the profile is.

    A real LLM will later report its OWN score; the mock computes one the
    way a careful human would - more known inputs means more trust. We return
    (score, reasons) so the app can explain the number instead of just stating
    it.
    """
    score = 6.0
    reasons = []

    if profile.get("budget_per_day"):
        score += 1.5
        reasons.append("Budget provided")
    else:
        reasons.append("Budget missing")

    if profile.get("training_intensity"):
        score += 1.0
        reasons.append("Training intensity known")
    else:
        reasons.append("Training intensity unclear")

    if profile.get("allergies"):
        score += 0.8
        reasons.append("Allergies specified")
    else:
        reasons.append("Allergies unknown")

    if profile.get("food_preferences"):
        score += 0.4
        reasons.append("Food preferences given")

    score = min(round(score, 1), 10.0)
    return score, reasons


# --- Rule-based mock data (lightweight, deterministic - NOT a nutrition DB) ---

# The regional food candidate lists live in regional_foods.py (Phase 2) -
# that file is pure data, this one is the logic that selects from it.
# Re-exported here under the names the rest of this module already used, so
# every builder below reads exactly as before.

# Day-level macro targets per goal (chosen by detect_primary_goal's priority).
GOAL_TARGETS = {
    "match":       {"cal": 2800, "p": 100, "c": 430, "f": 60,
                    "note": "Higher carbohydrate focus for match-day energy; lighter, easy-to-digest meals."},
    "performance": {"cal": 2750, "p": 115, "c": 380, "f": 70,
                    "note": "Balanced energy to support high-intensity football performance."},
    "recovery":    {"cal": 2700, "p": 130, "c": 360, "f": 70,
                    "note": "Higher protein and carbs to support recovery between sessions."},
    "muscle_gain": {"cal": 2950, "p": 140, "c": 360, "f": 75,
                    "note": "Calorie surplus with high protein to support muscle gain."},
    "maintenance": {"cal": 2600, "p": 100, "c": 340, "f": 70,
                    "note": "Balanced intake to maintain fitness through training weeks."},
    "fat_loss":    {"cal": 2150, "p": 125, "c": 200, "f": 60,
                    "note": "Moderate calorie deficit with protein kept high to preserve muscle."},
}

# ----------------------------------------------------------------------
# Narrative-text variety (Step 7): reasoning/notes phrasing pools
# ----------------------------------------------------------------------
# The problem this solves: a 7-day plan's food genuinely rotates, but every
# meal's "reasoning" and every day's summary "notes" were built from a
# single fixed sentence per goal/meal-slot - so scrolling a week showed the
# exact same sentence seven times under seven different breakfasts, reading
# as templated even though the food itself was not.
#
# Fix is the SAME deterministic-rotation principle the food selection
# already uses (see variety_seed()/slot_seed()/rotate() above), applied to
# text instead of food: each pool's INDEX 0 IS the original, pre-existing
# sentence, and _phrase() picks pool[day_index % len(pool)]. A single-day
# caller never passes day_index (it defaults to 0), so 1-day output is
# byte-identical to before this existed - only a multi-day plan, where
# day_index actually varies, ever sees index > 0. Nothing here is random;
# the same day_index always yields the same phrasing.
#
# The FACTUAL clause in _day_summary() ("Portions tuned for a {portion}
# training load...") and the minor-safety note are deliberately NOT
# rotated - they are true statements that legitimately don't change day to
# day, not filler reasoning standing in for one.

GOAL_NOTE_VARIANTS = {
    "match": [
        "Higher carbohydrate focus for match-day energy; lighter, easy-to-digest meals.",
        "Extra carbohydrate emphasis to be ready for the match; kept light and easy to digest.",
    ],
    "performance": [
        "Balanced energy to support high-intensity football performance.",
        "Energy balanced to keep up with high-intensity football demands.",
    ],
    "recovery": [
        "Higher protein and carbs to support recovery between sessions.",
        "Extra protein and carbs to help you bounce back between sessions.",
    ],
    "muscle_gain": [
        "Calorie surplus with high protein to support muscle gain.",
        "A calorie surplus, protein-forward, to support building muscle.",
    ],
    "maintenance": [
        "Balanced intake to maintain fitness through training weeks.",
        "Steady, balanced intake to hold fitness through the training week.",
    ],
    "fat_loss": [
        "Moderate calorie deficit with protein kept high to preserve muscle.",
        "A modest calorie deficit, protein kept high to protect muscle.",
    ],
}

BREAKFAST_REASON_TRAINING = [
    "Carb-rich start to fuel morning training",
    "Quick-digesting carbs to top up energy before training",
]
BREAKFAST_REASON_GENERAL = [
    "Balanced breakfast with protein to start the day",
    "Steady start with protein and carbs to begin the day",
]
LUNCH_REASON = [
    "Largest meal around training for steady energy and protein",
    "The day's biggest plate, timed around training for lasting energy",
]
SNACK_REASON_PROTEIN = [
    "Protein-focused snack to support muscle repair",
    "Extra protein between meals to support recovery",
]
SNACK_REASON_GENERAL = [
    "Light snack to keep energy up between meals",
    "A small bite to bridge the gap between meals",
]
DINNER_REASON_PROTEIN = [
    "High-protein evening meal for overnight recovery",
    "Protein-rich dinner to support recovery overnight",
]
DINNER_REASON_GENERAL = [
    "Balanced dinner to round out the day",
    "A balanced evening meal to close out the day",
]


def _phrase(pool, day_index):
    """Deterministically pick a phrasing by day index - see the module note
    above. day_index=0 (every existing single-day caller) always yields
    pool[0], the original wording."""
    return pool[day_index % len(pool)]


# Foods dropped first when the budget is low (kept only if the user prefers them).
EXPENSIVE = ["paneer", "chicken", "fish", "mutton", "cheese"]

# Share of the day's macros allocated to each meal (sums to 1.0).
MEAL_SPLIT = {"Breakfast": 0.25, "Lunch": 0.33, "Snack": 0.15, "Dinner": 0.27}


# --- Small helpers: read the profile into simple "signals" ------------

# detect_region() now lives in regional_foods.py next to the data it matches
# against, and is imported at the top of this file - so `detect_region` still
# resolves here for every existing caller (print_debug, mock_provider, tests).


# ----------------------------------------------------------------------
# Controlled variety (Phase 2)
# ----------------------------------------------------------------------
# The problem this solves: choose_items() always took the FIRST candidates
# after filtering, so every athlete whose filters happened to match ended up
# eating the same thing - two similar profiles got near-identical plans.
#
# The fix is a deterministic ROTATION of each candidate list, not randomness.
# Randomness would make plans irreproducible and untestable, and would change
# a plan every time the page was refreshed. Instead each profile derives a
# stable integer seed from its own contents, and each meal slot rotates its
# candidate list by a different amount derived from that seed. So:
#   - the same profile always produces the same plan (reproducible, testable)
#   - two similar-but-not-identical profiles land on different seeds and
#     therefore different foods
#   - the slots decorrelate from each other, instead of every slot shifting
#     in lockstep and just producing a different uniform plan
#
# Filtering still happens BEFORE rotation and preferences are applied AFTER,
# so allergies, foods-to-avoid, budget limits and stated preferences all keep
# exactly the authority they had before variety existed.

# Fields whose values distinguish one athlete from another. Deliberately
# excludes allergies/foods_to_avoid/food_preferences: those already change
# the plan through filtering, and folding them into the seed as well would
# make an unrelated food swap look like it was caused by an allergy edit.
VARIETY_FIELDS = ["age", "height_cm", "weight_kg", "sex", "goal", "region",
                  "diet", "training_days_per_week", "training_intensity"]

# Per-slot nudges so breakfast, snack, carbs, veg and the two proteins do not
# all rotate by the same amount for a given athlete.
SLOT_BREAKFAST = 0
SLOT_BREAKFAST_SIDE = 1
SLOT_SNACK = 2
SLOT_LUNCH_PROTEIN = 3
SLOT_DINNER_PROTEIN = 4
SLOT_LUNCH_VEG = 5
SLOT_DINNER_VEG = 6
SLOT_LUNCH_CARB = 7
SLOT_DINNER_CARB = 8

# ----------------------------------------------------------------------
# Multi-day round-robin cycling (Step 8, replacing Step 6's per-day hash)
# ----------------------------------------------------------------------
# Step 6 originally re-hashed a completely independent seed for each day
# (slot_seed(seed, DAY_SEED_BASE + day_index)), which is why it needed
# explicit prev-day exclusion as a safety net: an independent hash per day
# has no memory of the pool's overall coverage, so a 9-item pool could by
# chance land on the same handful of items for most of a 7-day week while
# other items were never touched at all - the audit that preceded this
# measured that effect directly (an ~8-item pool sampled this way covers
# only ~65% of its own items across 7 draws, not all 7).
#
# The fix: every slot's offset below is now `slot_seed(seed, SLOT_X) +
# day_index` instead of a fresh hash per day - i.e. a genuine ROUND ROBIN
# through that athlete's own seed-rotated pool order, not independent
# sampling. day_index=0 (every single-day caller, unchanged) always
# resolves to exactly slot_seed(seed, SLOT_X) - today's existing offset -
# so 1-day output is untouched. For day_index > 0 this sweeps through
# every remaining position in the pool before any repeat is possible
# (guaranteed: rotate()'s offset increases by exactly 1 each day, so no
# two DIFFERENT day_index values 0..len(pool)-1 can land on the same
# position), and consecutive days can never coincide either (consecutive
# integers are never congruent mod a pool of size >= 2) - so immediate
# repetition is prevented as a mathematical property of the cycle itself,
# not only by the prev-day `exclude` safety net (which is kept anyway, as
# a second line of defence for edge cases like a pool that shrank to a
# single item after allergy filtering).
#
# The existing per-slot SLOT_* hash still gives each athlete their OWN
# starting point in the cycle (so two athletes still diverge - unchanged
# from Phase 2), and lunch/dinner keep their own independent SLOT
# constants, so a repeated protein on, say, day 4 of a 4-item pool almost
# always pairs with a different carb than it did on day 0, since the carb
# slot cycles from its own independent starting point at its own pace.


def variety_seed(profile):
    """
    A stable non-negative integer derived from the athlete's own profile.

    Hashed (rather than, say, summing the numbers) so that a small change -
    72 kg to 73 kg - produces a completely different rotation instead of an
    adjacent one, which is what makes similar profiles diverge. sha256 is
    used only because it is in the standard library and stable across runs;
    nothing security-sensitive depends on it.
    """
    basis = "|".join(str(profile.get(field)) for field in VARIETY_FIELDS)
    return int(hashlib.sha256(basis.encode("utf-8")).hexdigest()[:8], 16)


def slot_seed(seed, slot):
    """
    Derive an independent rotation offset for one meal slot.

    Re-hashing rather than plain `seed + slot` arithmetic matters: candidate
    lists are short (7-10 items), so `seed + slot` leaves every slot sharing
    the same `seed % len` base and the slots move in lockstep - two athletes
    who collide on breakfast then collide on snack and veg too. Hashing the
    (seed, slot) pair decorrelates them, so a collision in one slot says
    nothing about the others.
    """
    return int(hashlib.sha256(f"{seed}:{slot}".encode("utf-8")).hexdigest()[:8], 16)


def rotate(items, offset):
    """Rotate a list left by `offset` positions (no-op for empty lists)."""
    if not items:
        return items
    pivot = offset % len(items)
    return items[pivot:] + items[:pivot]


def detect_primary_goal(profile):
    """
    Pick ONE goal by fixed priority - football before aesthetics:
    match > performance > recovery > muscle_gain > maintenance > fat_loss.
    This is why "performance + muscle gain" resolves to performance.
    """
    text = (profile.get("goal", "") + " " + (profile.get("notes") or "")).lower()
    if "match" in text:
        return "match"
    if any(w in text for w in ["performance", "football", "fitness", "stamina",
                               "endurance", "agility", "speed", "on-field"]):
        return "performance"
    if any(w in text for w in ["recovery", "recover", "repair"]):
        return "recovery"
    if any(w in text for w in ["muscle", "gain", "build", "bulk", "size",
                               "strength", "stronger"]):
        return "muscle_gain"
    if any(w in text for w in ["maintain", "maintenance", "stay", "keep"]):
        return "maintenance"
    if any(w in text for w in ["fat loss", "lose", "cut", "lean", "weight loss", "slim"]):
        return "fat_loss"
    return "maintenance"


def avoid_keywords(profile):
    """Words that must not appear in any ingredient (allergies + foods to avoid)."""
    raw = (profile.get("allergies") or "") + "," + (profile.get("foods_to_avoid") or "")
    words = {w.strip().lower() for w in raw.replace(" and ", ",").split(",") if w.strip()}
    words.discard("none")
    # "Dairy" is one word but many foods - expand it so the filter catches them.
    if any(d in words for d in ["dairy", "lactose", "milk"]):
        words.update(["milk", "curd", "paneer", "cheese", "ghee", "yogurt", "buttermilk"])
    # Match both singular and plural (e.g. an allergy typed as "eggs" must
    # also catch "Egg curry"/"Egg bhurji", which use the singular) - simple,
    # deliberately limited English plural handling, not a real stemmer.
    for word in list(words):
        if word.endswith("s") and len(word) > 1:
            words.add(word[:-1])
        else:
            words.add(word + "s")
    return words


def _recipe_is_non_veg(recipe):
    """True if this saved recipe's own ingredient list contains a
    non-vegetarian word - the SAME word list tools/ingredients_tool.py
    already uses to classify home ingredients, imported rather than
    duplicated, so the two checks can never quietly drift apart."""
    ingredients_text = " ".join(recipe.get("ingredients") or []).lower()
    return any(word in ingredients_text for word in NON_VEGETARIAN)


def classify_personal_recipes(recipes, profile):
    """
    Sort an athlete's saved recipes (personal_recipes.py) into usable /
    excluded-with-reason for THIS profile, using the exact same
    avoid_keywords() check and diet rule every other food candidate in this
    file is filtered through - checked against the recipe's NAME and its
    ingredients together, so "Paneer curry" is caught even if "paneer"
    happens to be spelled differently in the ingredient list.

    This is the ONE place personal-recipe safety is decided. It is called
    independently by mock_provider() (never trusting any upstream
    filtering - see its docstring) AND by llm_provider.py's providers (for
    the prompt's "must not use" section) AND by chat.py - one function, one
    answer, everywhere a saved recipe might reach an athlete.

    Mirrors tools.ingredients_tool.assess_ingredients()'s usable/excluded
    shape on purpose, for the same reason that tool's docstring gives:
    anything excluded must say why, never be silently dropped.
    """
    avoid = avoid_keywords(profile)
    is_veg = profile.get("diet") == "Vegetarian"
    usable, excluded = [], []
    for recipe in (recipes or []):
        name = recipe.get("name", "")
        ingredients_text = " ".join(recipe.get("ingredients") or []).lower()
        combined = f"{name.lower()} {ingredients_text}"
        blocked_by = next((a for a in avoid if a and a in combined), None)
        if blocked_by:
            excluded.append({"name": name, "reason": f"excluded by allergy/foods-to-avoid ({blocked_by})"})
            continue
        if is_veg and _recipe_is_non_veg(recipe):
            excluded.append({"name": name, "reason": "not suitable for a Vegetarian diet"})
            continue
        usable.append(recipe)
    return {"usable": usable, "excluded": excluded}


def prefer_tokens(profile):
    """
    Foods the user likes, used to bias selection to the front.

    Typing "none" explicitly clears a stored preference (same convention
    already used by allergies/foods_to_avoid) - leaving the field truly
    blank instead means "inherit whatever memory already has", so returning
    athletes don't have to retype it every time. Only an explicit "none"
    overrides that.
    """
    raw = profile.get("food_preferences") or ""
    tokens = [w.strip().lower() for w in raw.replace(" and ", ",").split(",") if w.strip()]
    return [t for t in tokens if t != "none"]


def ingredient_tokens(profile):
    """
    Foods the athlete says they already have at home (Phase 4), parsed the
    same way as prefer_tokens() and fed into the same `prefer` list.

    Availability is treated as a REASON TO PREFER a food, never as a reason
    to relax a rule. Because `prefer` is applied inside choose_items() only
    AFTER allergy/foods-to-avoid filtering, after the diet-specific
    candidate list has been chosen, and within the region's own candidates,
    an ingredient can only ever promote something the athlete was already
    allowed to eat. Listing "chicken" cannot put chicken in a vegetarian
    plan, and listing an allergen cannot reintroduce it.

    Ingredients that match no candidate food simply have no effect - the
    plan never claims to have used them (see ingredients_tool for the
    honest accounting of what was and was not usable).

    Optional, like every other field here: no ingredients means an empty
    list, and the plan is byte-identical to one built without this feature.
    """
    raw = profile.get("available_ingredients") or ""
    tokens = [w.strip().lower() for w in raw.replace(" and ", ",").split(",") if w.strip()]
    return [t for t in tokens if t != "none"]


# NOTE: a previous version scanned Agent v1's retrieved RAG prose for food
# keywords and fed the matches in as "preferred" foods. That is deliberately
# gone. Substring matching cannot read a sentence's intent, so guidance that
# said a food should NOT be assumed was extracted as a preference for it -
# the northeast section's "Curd, paneer and milk-based sides should not be
# assumed as defaults" yielded exactly ['paneer', 'curd', 'milk'] as things
# to prefer, pushing dairy onto the very athletes the text warns about.
#
# The fix is a separation of concerns rather than a smarter parser:
#   - RAG prose is CONTEXTUAL GUIDANCE, and goes to the LLM providers, which
#     can actually read it (see llm_provider._compose_prompt).
#   - Structured regional data (regional_foods.py) decides which foods the
#     deterministic mock provider may choose from.
# So no arbitrary prose can promote a food here any more. Explicit athlete
# input still can: food_preferences (prefer_tokens) and home ingredients
# (ingredient_tokens), both of which remain subject to allergy, diet,
# foods-to-avoid and budget filtering inside choose_items().


def budget_level(profile):
    """low (<150) / high (>300) / medium (in between, or unknown)."""
    raw = (profile.get("budget_per_day") or "").replace(",", "")  # drop thousands separators
    digits = "".join(c if c.isdigit() else " " for c in raw).split()
    if not digits:
        return "medium"
    amount = int(digits[0])
    if amount < 150:
        return "low"
    if amount > 300:
        return "high"
    return "medium"


# --- Small helpers: pick ingredients ----------------------------------

def _allowed(item, avoid):
    """True if the ingredient name contains none of the avoided words."""
    low = item.lower()
    return not any(bad in low for bad in avoid)


def choose_items(candidates, avoid, prefer, limit, level, offset=0, exclude=None):
    """
    Filter out avoided foods, drop pricey items on a low budget, rotate the
    survivors for variety, exclude same-day/previous-day picks, move
    preferred foods to the front, then return up to `limit` items.

    Order matters and is deliberate:
      1. avoid/allergy filtering   - safety, must never be overridden
      2. budget filtering          - affordability constraint
      3. rotation by `offset`      - variety across athletes AND (Step 8)
                                     the round-robin cycle across a
                                     multi-day plan's days
      4. `exclude` filtering       - AFTER rotation, not before (see below)
      5. preferences to the front  - the athlete's explicit wishes outrank
                                     whatever the rotation happened to pick
    `offset=0` reproduces the exact pre-Phase-2 ordering, so any caller that
    doesn't opt into variety behaves as before.

    `exclude` (default None) intentionally filters AFTER rotation, not
    before. Filtering before rotation would shrink the candidate list by
    exactly one item on every day that supplies an exclusion (i.e. every
    day except day 0), which changes the modulus the round-robin cycle
    computes positions against - day 0 cycles through a pool of size N,
    every later day would cycle through N-1, breaking the clean "every
    position visited once before any repeat" guarantee the round-robin
    offset exists to provide (this was measured directly: it dropped a
    10-item pool's 7-day coverage from a possible 7/7 to 5-6/7). Filtering
    after rotation instead means the excluded item is skipped only on the
    rare day the cycle would have actually landed on it - which, given
    the round-robin offset already guarantees no two of the days it's
    covering land on the same position, should not happen in the normal
    case at all; `exclude` remains purely a defensive fallback for edge
    cases (e.g. a pool reduced to one surviving item by allergy filtering).
    """
    items = [c for c in candidates if _allowed(c, avoid)]
    if level == "low":
        cheap = [c for c in items
                 if not any(x in c.lower() for x in EXPENSIVE)
                 or any(p in c.lower() for p in prefer)]
        items = cheap or items
    items = rotate(items, offset)
    if exclude:
        excluded = _as_exclude_set(exclude)
        filtered = [c for c in items if c not in excluded]
        items = filtered or items  # never let exclusion empty the list entirely
    preferred = [c for c in items if any(p in c.lower() for p in prefer)]
    rest = [c for c in items if c not in preferred]
    return (preferred + rest)[:limit]


def _as_exclude_set(exclude):
    """
    Normalise `exclude` to a set for filtering: None -> empty set, a single
    string -> a one-item set (the original single-item contract, unchanged
    for every existing caller), or an iterable of strings/None -> a set with
    any None entries dropped. The None-dropping is what lets the multi-day
    planner (Step 6) pass e.g. [lunch_protein, prev_day.get("dinner_protein")]
    on day 1, where prev_day is None and that second slot is simply absent.
    """
    if exclude is None:
        return set()
    if isinstance(exclude, str):
        return {exclude}
    return {e for e in exclude if e is not None}


def pick_carb(region_data, avoid, prefer, level, exclude=None, offset=0):
    """
    Choose the staple carbohydrate for one main meal.

    `exclude` lets dinner avoid repeating lunch's carb (within-day variety),
    and the multi-day planner also passes the previous day's same-slot carb
    here (cross-day variety) - passed straight through to choose_items(),
    which applies it AFTER rotation (see its docstring for why that
    ordering matters for the round-robin cycle). A single string still
    works exactly as before for every existing caller.
    """
    chosen = choose_items(region_data["carbs"], avoid, prefer, 1, level, offset, exclude=exclude)
    return chosen[0] if chosen else "Rice"


def pick_protein(region_data, is_veg, avoid, prefer, level, exclude=None, offset=0,
                  extra_candidates=None):
    """
    Choose one main protein suited to diet, budget, allergies and preferences.
    `exclude` lets a later meal (dinner) avoid repeating an earlier one (lunch)
    within a day, AND lets the multi-day planner also exclude the previous
    day's same-slot protein - passed straight through to choose_items(),
    which applies it AFTER rotation (see its docstring). For non-vegetarians
    the region's list already includes eggs/chicken/fish, so those appear
    naturally without the user having to request them.

    `extra_candidates` (Personal Recipes feature, default None - every
    existing caller unaffected): dish names to add to the region's own pool
    for this pick, e.g. an athlete's saved recipes already filtered safe for
    this profile (see meal_planner.classify_personal_recipes()). They are
    appended to the SAME list choose_items() rotates over - not boosted to
    the front, not force-selected - so a saved recipe only ever wins this
    slot the same way any regional dish would: when the existing rotation/
    exclusion math actually lands on it.
    """
    source = region_data["veg_protein"] if is_veg else region_data["nonveg_protein"]
    if extra_candidates:
        source = source + extra_candidates
    chosen = choose_items(source, avoid, prefer, 1, level, offset, exclude=exclude)
    if chosen:
        return chosen[0]
    # Everything was avoided (e.g. dairy allergy) - fall back to a safe protein.
    safe = choose_items(SAFE_VEG_PROTEIN, avoid, prefer, 1, level, offset)
    return safe[0] if safe else "Moong dal"


def training_load(profile):
    """
    Turn training days/week + intensity into (energy_factor, portion_label).
    More days and higher intensity => a bigger multiplier and larger portions.
    This is what makes the plan visibly react to how hard the athlete trains.
    """
    days = profile.get("training_days_per_week") or 0
    intensity = (profile.get("training_intensity") or "").lower()
    volume = 0.9 if days <= 2 else (1.0 if days <= 4 else 1.1)
    intensity_factor = {"low": 0.9, "moderate": 1.0, "high": 1.15}.get(intensity, 1.0)
    factor = volume * intensity_factor
    if factor <= 0.96:
        portion = "light"
    elif factor >= 1.10:
        portion = "large"
    else:
        portion = "moderate"
    return factor, portion


def split_macros(goal_key, factor=1.0, overrides=None):
    """
    Split the day target into per-meal macros; the day total = their sum.
    `factor` scales the whole day up/down for training load (see training_load).

    `overrides` (optional, default None - existing callers unaffected) is a
    {"cal","p","c"} dict of precise day-level targets - e.g. from Agent v1's
    nutrition_tool, which computes weight/intensity-specific numbers. When
    a key is present, it REPLACES the GOAL_TARGETS lookup for that macro,
    used AS-IS (not multiplied by `factor` again, since the tool's own
    formula already accounts for training intensity). Fat ("f") always
    comes from GOAL_TARGETS * factor - the tool doesn't compute a fat
    target, so there is nothing to override it with.
    """
    t = GOAL_TARGETS[goal_key]
    overrides = overrides or {}
    scaled = {
        "cal": overrides["cal"] if overrides.get("cal") is not None else t["cal"] * factor,
        "p": overrides["p"] if overrides.get("p") is not None else t["p"] * factor,
        "c": overrides["c"] if overrides.get("c") is not None else t["c"] * factor,
        "f": t["f"] * factor,
    }
    meals = {}
    for meal, frac in MEAL_SPLIT.items():
        meals[meal] = {
            "calories": round(scaled["cal"] * frac),
            "protein_g": round(scaled["p"] * frac),
            "carbs_g": round(scaled["c"] * frac),
            "fats_g": round(scaled["f"] * frac),
        }
    # Match-day: move some carbs to breakfast so dinner stays light.
    if goal_key == "match":
        meals["Breakfast"]["carbs_g"] += 15
        meals["Dinner"]["carbs_g"] = max(meals["Dinner"]["carbs_g"] - 15, 0)
    return meals


# Portion words per training load, so quantities read differently by load.
PLATE = {"light": "1 small plate", "moderate": "1 plate", "large": "1 large plate"}
BOWL = {"light": "1 small bowl", "moderate": "1 bowl", "large": "1 large bowl"}


def _carb_servings(portion, goal_key, is_minor=False):
    """
    Serving count for the main carb, scaled by training load and goal.

    The extra "bulking" serving for a muscle_gain goal is an adult-style
    surplus strategy - skipped for a minor (is_minor=True) rather than
    stacking an adult bulking portion onto a child's plate. Everything else
    about the serving count is unchanged for a minor.
    """
    count = {"light": 2, "moderate": 3, "large": 4}[portion]
    if goal_key == "muscle_gain" and not is_minor:
        count += 1
    return f"{count} servings"


def _dedupe(ingredients):
    """Drop repeated ingredients (by item name), keeping first occurrence."""
    seen = set()
    unique = []
    for ing in ingredients:
        if ing["item"] not in seen:
            seen.add(ing["item"])
            unique.append(ing)
    return unique


def _resources_for_ingredients(ingredients):
    """
    One recipe-link resource per distinct complex dish among this meal's
    (already deduped) ingredients, each tagged with which ingredient it's
    for via resource["item"]. Simple items contribute nothing - see
    recipe_links.py for the complexity registry.
    """
    resources = []
    seen = set()
    for ing in ingredients:
        item = ing["item"]
        if item in seen:
            continue
        resource = get_recipe_resource(item)
        if resource:
            resources.append(resource)
            seen.add(item)
    return resources


def _attach_nutrition(ingredients):
    """
    Add an optional "nutrition" field to each ingredient dict where a
    sourced entry exists (see nutrition_data.py) - additive only, exactly
    like _resources_for_ingredients(). Ingredients with no verified entry
    are left completely unchanged (no "nutrition" key at all), so this
    never implies data that wasn't actually sourced. Does NOT touch this
    meal's own calories/protein_g/carbs_g/fats_g - those stay exactly what
    split_macros() computed, untouched by this per-ingredient addition.
    """
    for ing in ingredients:
        info = get_nutrition_info(ing["item"])
        if info:
            ing["nutrition"] = info
    return ingredients


def _meal(name, ingredients, macros, reasoning):
    """Assemble one meal dict in the exact schema order."""
    deduped = _attach_nutrition(_dedupe(ingredients))
    meal = {"meal": name, "ingredients": deduped}
    meal.update(macros)  # calories, protein_g, carbs_g, fats_g
    meal["reasoning"] = reasoning
    meal["resources"] = _resources_for_ingredients(deduped)
    return meal


# --- Small helpers: build each meal -----------------------------------

def build_breakfast(region_data, is_veg, avoid, prefer, level, macros, goal_key, portion,
                    eggs_ok=False, seed=0, exclude_main=None, day_index=0, exclude_side=None):
    """
    `exclude_main` (default None - every existing caller unaffected): the
    multi-day planner passes yesterday's breakfast main here so today's
    doesn't repeat it. Falls back to the full candidate list if excluding it
    would leave nothing (e.g. a region with only one breakfast option).

    `exclude_side` (default None, same pattern as `exclude_main`): yesterday's
    breakfast side, so it doesn't repeat either - the audit found sides had
    no cross-day protection at all before this.

    `day_index` (default 0): both selects the reasoning phrasing (see the
    GOAL_NOTE_VARIANTS/_phrase() note above GOAL_TARGETS) AND advances the
    main/side round-robin cycle by one position per day - see the note above
    SLOT_LUNCH_CARB. day_index=0 reproduces the exact pre-existing offset.
    """
    picks = choose_items(region_data["breakfast"], avoid, prefer, 1, level,
                         slot_seed(seed, SLOT_BREAKFAST) + day_index, exclude=exclude_main)
    main = picks[0] if picks else "Vegetable upma"
    ingredients = [{"item": main, "quantity": PLATE[portion]}]
    # The side comes from the region's own list (Phase 2) rather than a fixed
    # north-Indian default, so e.g. a northeast athlete isn't handed curd as
    # though every Indian breakfast were dairy-based. Exclusion is applied
    # AFTER rotation (same reasoning as choose_items()'s exclude parameter -
    # see its docstring) so the round-robin cycle's modulus stays constant
    # across days instead of shrinking by one every day a side is excluded.
    sides = rotate(region_data["breakfast_sides"], slot_seed(seed, SLOT_BREAKFAST_SIDE) + day_index)
    if exclude_side:
        sides = [s for s in sides if s != exclude_side] or sides
    if is_veg and not eggs_ok:
        for side in sides:
            if _allowed(side, avoid):
                ingredients.append({"item": side, "quantity": BOWL[portion]})
                break
    elif _allowed("Boiled eggs", avoid):
        ingredients.append({"item": "Boiled eggs", "quantity": "3" if portion == "large" else "2"})
    else:
        # Eggs are avoided/allergic - fall back to a safe side instead of
        # forcing them in regardless (this branch covers non-veg athletes
        # and veg athletes with the eggs_ok exception alike).
        for side in [s for s in sides if "curd" not in s.lower()]:
            if _allowed(side, avoid):
                ingredients.append({"item": side, "quantity": BOWL[portion]})
                break
    if _allowed("Banana", avoid):
        ingredients.append({"item": "Banana", "quantity": "1"})
    reason = _phrase(
        BREAKFAST_REASON_TRAINING if goal_key in ("match", "performance") else BREAKFAST_REASON_GENERAL,
        day_index,
    )
    return _meal("Breakfast", ingredients, macros, reason)


def build_lunch(region_data, is_veg, avoid, prefer, level, macros, goal_key, portion, protein,
                is_minor=False, seed=0, carb=None, day_index=0, exclude_veg=None):
    carb = carb or pick_carb(region_data, avoid, prefer, level,
                             offset=slot_seed(seed, SLOT_LUNCH_CARB) + day_index)
    veg = choose_items(region_data["veg"], avoid, prefer, 1, level,
                       slot_seed(seed, SLOT_LUNCH_VEG) + day_index, exclude=exclude_veg)
    ingredients = [
        {"item": carb, "quantity": _carb_servings(portion, goal_key, is_minor)},
        {"item": protein, "quantity": BOWL[portion]},
    ]
    if veg:
        ingredients.append({"item": veg[0], "quantity": "1 bowl"})
    # The dairy accompaniment is whatever that region customarily serves, and
    # nothing at all where a dairy side is not the custom (see
    # regional_foods.dairy_side). This used to be a hard-coded "Curd" for
    # every vegetarian, which put curd on a northeast athlete's plate at every
    # single lunch - directly contradicting the knowledge base shipped in the
    # same request, which says dairy should not be assumed as a default there.
    dairy_side = region_data.get("dairy_side")
    if is_veg and dairy_side and _allowed(dairy_side, avoid):
        ingredients.append({"item": dairy_side, "quantity": "1 small bowl"})
    reason = _phrase(LUNCH_REASON, day_index)
    return _meal("Lunch", ingredients, macros, reason)


def nutrition_tool_overrides(tool_results):
    """
    Turn Agent v1's nutrition_tool result (if present in tool_results) into
    a {"cal","p","c"} override dict for split_macros(). Returns None when
    there's nothing usable - split_macros() then falls back to its normal
    GOAL_TARGETS-based calculation, byte-identical to before this existed.
    Missing/partial sub-fields are handled gracefully (per-field, not
    all-or-nothing) so a partial tool result never crashes this.
    """
    if not tool_results:
        return None
    nutrition = tool_results.get("nutrition_tool")
    if not nutrition:
        return None
    overrides = {}
    if nutrition.get("calories") is not None:
        overrides["cal"] = nutrition["calories"]
    if nutrition.get("protein") is not None:
        overrides["p"] = nutrition["protein"]
    if nutrition.get("carbs") is not None:
        overrides["c"] = nutrition["carbs"]
    return overrides or None


def budget_tool_snack_suggestion(tool_results, level):
    """
    Offer Agent v1's budget_tool protein suggestion as a bonus snack item -
    but only when the athlete is actually budget-constrained (level ==
    "low"). A well-funded athlete doesn't need the nudge, and offering it
    unconditionally would add a generic staple regardless of relevance.
    Returns None (no bonus item) for any other case, including missing/
    partial tool_results - never crashes on absent data.
    """
    if level != "low" or not tool_results:
        return None
    budget = tool_results.get("budget_tool")
    if not budget:
        return None
    suggestions = budget.get("suggestions") or {}
    protein_pick = suggestions.get("protein")
    return protein_pick.get("name") if protein_pick else None


def build_snack(region_data, is_veg, avoid, prefer, level, macros, goal_key, portion,
                 budget_suggestion=None, eggs_ok=False, is_minor=False, seed=0, exclude_main=None,
                 day_index=0):
    """
    `exclude_main` (default None - every existing caller unaffected): the
    multi-day planner passes yesterday's snack main here so today's doesn't
    repeat it. Only the first (main) candidate is affected - a short
    candidate list still falls back to the full list, same pattern as
    build_breakfast()'s exclude_main.
    """
    picks = choose_items(region_data["snack"], avoid, prefer, 2, level,
                         slot_seed(seed, SLOT_SNACK) + day_index, exclude=exclude_main) or ["Banana"]
    ingredients = [{"item": p, "quantity": "1 serving"} for p in picks]
    # Heavy training load -> add extra quick carbs.
    if portion == "large":
        for extra in ["Dates (5-6)", "Banana"]:
            if _allowed(extra, avoid):
                ingredients.append({"item": extra, "quantity": "1 serving"})
                break
    # The muscle_gain/recovery protein "booster" is an adult bulking-style
    # addition on top of an already-complete snack - skipped for a minor
    # rather than stacking extra adult-style protein onto a child's plate.
    if goal_key in ("muscle_gain", "recovery") and not is_minor:
        booster = "Boiled eggs" if (not is_veg or eggs_ok) else "Roasted chana"
        if _allowed(booster, avoid):
            ingredients.append({"item": booster, "quantity": "2 eggs / 1 handful"})

    # Agent v1 integration: offer the budget tool's cheap protein pick as a
    # bonus snack item, subject to the SAME allergy/avoid filtering as
    # everything else, and only if not already present in the snack.
    if budget_suggestion and _allowed(budget_suggestion, avoid):
        already_present = any(
            budget_suggestion.lower() in ing["item"].lower() for ing in ingredients
        )
        if not already_present:
            ingredients.append({"item": budget_suggestion, "quantity": "1 serving (budget pick)"})

    reason = _phrase(
        SNACK_REASON_PROTEIN if goal_key in ("muscle_gain", "recovery") else SNACK_REASON_GENERAL,
        day_index,
    )
    return _meal("Snack", ingredients, macros, reason)


def build_dinner(region_data, is_veg, avoid, prefer, level, macros, goal_key, portion, protein,
                 is_minor=False, seed=0, carb=None, day_index=0, apply_match_override=True,
                 exclude_veg=None):
    """
    `apply_match_override` (default True - every existing single-day caller
    unaffected): the multi-day planner (Step 7) only sets this False for
    days that aren't the LAST day of the plan, so a "match" goal's light
    khichdi dinner appears once, the night before the match (the last day
    of the plan), rather than identically on all 7 days merely because the
    goal mentions "match". A single day IS the last day of its own 1-day
    plan, so this is a no-op for every pre-existing caller.
    """
    # Match-day: a light, easy-to-digest dinner the night before.
    if goal_key == "match" and apply_match_override:
        main = "Khichdi (rice + moong dal)"
        ingredients = [{"item": main, "quantity": "1 large bowl"}]
        dairy_side = region_data.get("dairy_side")  # region-aware, see build_lunch
        if is_veg and dairy_side and _allowed(dairy_side, avoid):
            ingredients.append({"item": dairy_side, "quantity": "1 small bowl"})
        reason = "Light, easy-to-digest dinner before match day"
        return _meal("Dinner", ingredients, macros, reason)
    carb = carb or pick_carb(region_data, avoid, prefer, level,
                             offset=slot_seed(seed, SLOT_DINNER_CARB) + day_index)
    veg = choose_items(region_data["veg"], avoid, prefer, 1, level,
                       slot_seed(seed, SLOT_DINNER_VEG) + day_index, exclude=exclude_veg)
    ingredients = [
        {"item": carb, "quantity": _carb_servings(portion, goal_key, is_minor)},
        {"item": protein, "quantity": BOWL[portion]},
    ]
    if veg:
        ingredients.append({"item": veg[0], "quantity": "1 bowl"})
    reason = _phrase(
        DINNER_REASON_PROTEIN if goal_key in ("muscle_gain", "recovery") else DINNER_REASON_GENERAL,
        day_index,
    )
    return _meal("Dinner", ingredients, macros, reason)


def _day_summary(meals, goal_key, portion, profile, is_minor, day_index=0):
    """
    Sum one day's meals into the day_summary block, with the same training-
    load/goal/minor-appropriate note mock_provider() has always built.
    Extracted so the multi-day planner (Step 6) can build the identical
    per-day summary for each of its days without duplicating this text.

    `day_index` (default 0, matching every existing caller): selects the
    goal-note phrasing via GOAL_NOTE_VARIANTS/_phrase() - see the note above
    GOAL_TARGETS. The factual training-load clause and minor-safety note are
    NOT rotated - they're true statements that don't change day to day.
    """
    training_days = profile.get("training_days_per_week")
    intensity = profile.get("training_intensity")
    note = (_phrase(GOAL_NOTE_VARIANTS[goal_key], day_index)
            + f" Portions tuned for a {portion} training load "
            + f"({training_days} days/week, {intensity} intensity).")
    if is_minor:
        note += " Growth-appropriate portions for a young athlete, not an adult bulking plan."
    return {
        "total_calories": sum(m["calories"] for m in meals),
        "total_protein_g": sum(m["protein_g"] for m in meals),
        "total_carbs_g": sum(m["carbs_g"] for m in meals),
        "total_fats_g": sum(m["fats_g"] for m in meals),
        "notes": note,
    }


def _build_day(region_data, is_veg, avoid, prefer, level, goal_key, portion, is_minor, eggs_ok,
               macros, seed, budget_suggestion, prev=None, day_index=0, is_last_day=True,
               recent_proteins=None, recent_carbs=None, recipe_candidates=None):
    """
    Build one day's four meals - the exact food-selection logic
    mock_provider() has always used for its single day, extracted here so
    the multi-day planner (Step 6) can call it once per day with a
    day-specific `seed` and the previous day's picks, without a second
    food-selection system.

    `prev` (None for day 1, and for every call from the unchanged single-day
    mock_provider()): the previous day's {"breakfast", "breakfast_side",
    "lunch_protein", "lunch_carb", "lunch_veg", "dinner_protein",
    "dinner_carb", "dinner_veg", "snack"} choices. Each slot's candidate pool
    excludes that item IN ADDITION to its existing within-day exclusion
    (dinner already avoids repeating lunch's protein/carb) - so two
    consecutive days never repeat the same breakfast/protein/carb/snack/
    side/veg main, on top of the round-robin cycling `day_index` already
    provides (see the note above SLOT_LUNCH_CARB) - this exclusion is now a
    second line of defence rather than the only protection, but is kept for
    edge cases (e.g. a pool shrunk to one item by allergy filtering). None
    (the default) reproduces the exact pre-existing single-day behaviour -
    _as_exclude_set(None) is an empty set, so nothing is excluded beyond
    what already was.

    `day_index` (default 0): selects reasoning phrasing for this day (see
    _phrase() above GOAL_TARGETS) AND advances every slot's round-robin
    cycle position by one - day_index=0 reproduces the exact pre-existing
    single-day offsets for every slot.

    `is_last_day` (default True, matching every existing single-day caller -
    a 1-day plan's only day IS its last day): only the last day of a
    multi-day plan may apply build_dinner()'s match-day override, so a
    "match" goal's light khichdi dinner appears once (the night before the
    match), not identically on all 7 days - see build_dinner()'s docstring.

    `recent_proteins`/`recent_carbs` (Phase 2A, default None - every existing
    caller unaffected): dishes this athlete was served in a PAST generation
    (read from memory_manager's existing "latest_plan"/"episodic" categories
    by orchestrator.py, never by this function - see mock_provider()'s
    docstring). Folded into the same `exclude` lists `prev` already
    populates, not a new mechanism - choose_items()'s existing "never let
    exclusion empty the list entirely" fallback means this deprioritizes
    rather than hard-blocks: a small pool still produces a meal even if
    every candidate was recently served.

    `recipe_candidates` (Personal Recipes feature, default None - every
    existing caller unaffected): this athlete's saved recipes, ALREADY
    filtered safe for this profile (see mock_provider()'s docstring) - only
    ever reaches the LUNCH and DINNER protein pick (see pick_protein()'s
    `extra_candidates`), matching the worked example (a curry) this feature
    was scoped around. Breakfast/snack/veg/carb slots are unaffected.

    Returns (meals, choices): `meals` is the familiar 4-item list; `choices`
    is this day's own picks, in the same shape `prev` expects, so the caller
    can hand it to the NEXT day's _build_day() call.
    """
    prev = prev or {}
    recent_proteins = recent_proteins or []
    recent_carbs = recent_carbs or []

    # Pick two different mains AND two different staples so lunch and dinner
    # are not near-identical plates (within-day), and - when `prev` is
    # supplied - so today doesn't repeat yesterday's same-slot choice either.
    # `+ day_index` on every offset is the round-robin cycle advance - see
    # the note above SLOT_LUNCH_CARB. `recent_proteins`/`recent_carbs` widen
    # the exclusion beyond this single call's own days, to the athlete's
    # actual generation history (Phase 2A).
    lunch_protein = pick_protein(region_data, is_veg, avoid, prefer, level,
                                 exclude=[prev.get("lunch_protein"), *recent_proteins],
                                 offset=slot_seed(seed, SLOT_LUNCH_PROTEIN) + day_index,
                                 extra_candidates=recipe_candidates)
    dinner_protein = pick_protein(region_data, is_veg, avoid, prefer, level,
                                  exclude=[lunch_protein, prev.get("dinner_protein"), *recent_proteins],
                                  offset=slot_seed(seed, SLOT_DINNER_PROTEIN) + day_index,
                                  extra_candidates=recipe_candidates)
    lunch_carb = pick_carb(region_data, avoid, prefer, level,
                           exclude=[prev.get("lunch_carb"), *recent_carbs],
                           offset=slot_seed(seed, SLOT_LUNCH_CARB) + day_index)
    dinner_carb = pick_carb(region_data, avoid, prefer, level,
                            exclude=[lunch_carb, prev.get("dinner_carb"), *recent_carbs],
                            offset=slot_seed(seed, SLOT_DINNER_CARB) + day_index)

    breakfast = build_breakfast(region_data, is_veg, avoid, prefer, level, macros["Breakfast"],
                                goal_key, portion, eggs_ok, seed, exclude_main=prev.get("breakfast"),
                                day_index=day_index, exclude_side=prev.get("breakfast_side"))
    lunch = build_lunch(region_data, is_veg, avoid, prefer, level, macros["Lunch"], goal_key,
                        portion, lunch_protein, is_minor, seed, lunch_carb, day_index=day_index,
                        exclude_veg=prev.get("lunch_veg"))
    snack = build_snack(region_data, is_veg, avoid, prefer, level, macros["Snack"], goal_key,
                        portion, budget_suggestion, eggs_ok, is_minor, seed,
                        exclude_main=prev.get("snack"), day_index=day_index)
    dinner = build_dinner(region_data, is_veg, avoid, prefer, level, macros["Dinner"], goal_key,
                          portion, dinner_protein, is_minor, seed, dinner_carb, day_index=day_index,
                          apply_match_override=is_last_day, exclude_veg=prev.get("dinner_veg"))

    meals = [breakfast, lunch, snack, dinner]

    # Extracted by membership rather than fixed position, since a slot may
    # be absent (e.g. a match-day dinner has no "veg" concept at all, and a
    # breakfast in the eggs-branch has no side drawn from breakfast_sides) -
    # membership against the region's own pool correctly identifies "was
    # this slot actually used today" regardless of ingredient order.
    def _used(meal, pool):
        return next((i["item"] for i in meal["ingredients"] if i["item"] in pool), None)

    choices = {
        "breakfast": breakfast["ingredients"][0]["item"],
        "breakfast_side": _used(breakfast, region_data["breakfast_sides"]),
        "lunch_protein": lunch_protein,
        "lunch_carb": lunch_carb,
        "lunch_veg": _used(lunch, region_data["veg"]),
        "dinner_protein": dinner_protein,
        "dinner_carb": dinner_carb,
        "dinner_veg": _used(dinner, region_data["veg"]),
        "snack": snack["ingredients"][0]["item"],
    }
    return meals, choices


def mock_provider(profile, agent_context=None):
    """
    Deterministic, rule-based stand-in for a real LLM. It reads the profile
    into simple signals (region, diet, budget, allergies, preferences, the
    highest-priority goal and the training load) and builds a meal plan from
    small rules - returning the exact same JSON shape a real provider will. It
    is realistic enough to test the app against different profiles, NOT an
    accurate nutrition engine.

    agent_context (optional, default None - Agent v1's orchestrator): when
    supplied, this mock ALSO reacts to it, layered on top of its existing
    profile-only rules - it is still a deterministic simulation, not an
    LLM, just extended to also look at pre-computed context when available:
      - nutrition_tool results REPLACE the day-level calorie/protein/carb
        targets (fat stays goal-based - the tool doesn't compute a fat
        target). See nutrition_tool_overrides().
      - budget_tool's cheapest protein suggestion is offered as a bonus
        snack item, but only when the athlete is budget-constrained
        (see budget_tool_snack_suggestion()).
    agent_context's rag_context is deliberately IGNORED here. Retrieved prose
    is contextual guidance for a model that can read it, not a food list -
    see the note above budget_level() for why scanning it for keywords was
    removed. Which foods this provider may choose from comes solely from
    regional_foods.py.

    With agent_context=None (the default), none of the above triggers, and
    this function behaves EXACTLY as it did before agent_context existed.

    Phase 2A (memory-aware variety): agent_context may also carry
    "variety_context" - {"generation_index": int, "recent_proteins": [...],
    "recent_carbs": [...]}, computed by orchestrator.py from this athlete's
    EXISTING episodic/latest_plan memory (memory_manager.list("episodic", ...)
    / memory_manager.get("latest_plan", ...) - no new memory category, no
    change to memory_manager.py). This function never reads memory itself;
    it only consumes whatever orchestrator.py already computed, the same
    pattern tool_results already follows.
      - generation_index becomes this call's day_index (0 for an athlete's
        first-ever generation - byte-identical to pre-Phase-2A behaviour -
        advancing by one for each generation since, reusing the EXACT
        round-robin cycle math _build_day()/slot_seed() already use for
        multi-day plans, not a second rotation system).
      - recent_proteins/recent_carbs (this athlete's most recently served
        protein/carb dishes) are folded into the same `exclude` lists
        _build_day() already builds from `prev` - see its docstring.
    With no "variety_context" key (agent_context=None, or any existing
    caller/test that doesn't supply it), generation_index defaults to 0 and
    recent_proteins/recent_carbs default to empty - behaviour is EXACTLY
    what it was before Phase 2A.

    Age (audit finding fix - age used to reach only the calorie number via
    nutrition_tool, never the food-selection logic below): for a minor
    (age < 18), this also (a) caps the training-load portion scaling at
    "moderate" instead of letting it reach "large" purely from training
    frequency, and (b) skips the adult-style muscle_gain/recovery bulking
    additions in build_snack()/_carb_servings() - see is_minor below. Adult
    behaviour (age is None or >= 18) is completely unchanged by this.
    """
    tool_results = (agent_context or {}).get("tool_results") or {}
    # agent_context["rag_context"] is intentionally not read - see docstring.

    is_veg = profile.get("diet") == "Vegetarian"
    region_data = REGIONS[detect_region(profile)]
    goal_key = detect_primary_goal(profile)
    avoid = avoid_keywords(profile)
    own_food_prefs = prefer_tokens(profile)
    # Home ingredients (Phase 4) join the same preference list: "I already
    # have this" biases selection exactly like "I like this" does, and is
    # subject to the identical filtering. Deliberately NOT folded into
    # own_food_prefs - that variable gates the ovo-vegetarian egg exception
    # below, and having eggs in the house is not the same as a vegetarian
    # athlete asking for eggs (they may be for someone else in the house).
    prefer = own_food_prefs + ingredient_tokens(profile)
    level = budget_level(profile)
    factor, portion = training_load(profile)
    macros = split_macros(goal_key, factor, overrides=nutrition_tool_overrides(tool_results))

    age = profile.get("age")
    is_minor = age is not None and age < 18
    if is_minor and portion == "large":
        # A child/teen's portions shouldn't scale to the adult "large" tier
        # purely from training frequency/intensity - training_load() has no
        # concept of age, so this caps its output rather than changing it.
        portion = "moderate"

    # Ovo-vegetarian exception: eggs may appear for an otherwise-Vegetarian
    # athlete ONLY when THEY explicitly asked for eggs via food_preferences -
    # never just because RAG text happens to mention eggs (own_food_prefs is
    # deliberately profile-only, not the RAG-widened `prefer` above). Default
    # (no such preference) stays strictly no-eggs, unchanged from before.
    eggs_ok = is_veg and any("egg" in tok for tok in own_food_prefs)

    # Phase 2: one stable seed per athlete, nudged per meal slot inside each
    # builder, so two similar profiles no longer collapse onto the same foods.
    seed = variety_seed(profile)

    budget_suggestion = budget_tool_snack_suggestion(tool_results, level)

    # Phase 2A: generation_index/recent_proteins/recent_carbs come solely
    # from agent_context - this function does not read memory_manager
    # itself (see docstring). Absent (or no agent_context at all), every
    # value below defaults to exactly what it was before Phase 2A existed.
    variety_context = (agent_context or {}).get("variety_context") or {}
    generation_index = variety_context.get("generation_index", 0)
    recent_proteins = variety_context.get("recent_proteins") or []
    recent_carbs = variety_context.get("recent_carbs") or []

    # Personal Recipes: agent_context carries the athlete's RAW saved
    # recipes (orchestrator.py fetches them from memory but does not filter
    # them - see its docstring). This function re-derives eligibility
    # itself via classify_personal_recipes(), the SAME function the LLM
    # providers and chat.py use, rather than trusting any upstream
    # filtering - this is the deterministic path's own enforcement, not a
    # relay of someone else's decision.
    raw_recipes = (agent_context or {}).get("personal_recipes") or []
    recipe_candidates = [r["name"] for r in classify_personal_recipes(raw_recipes, profile)["usable"]]

    # _build_day() with prev=None reproduces exactly the single-day logic
    # this function has always used - see its docstring. day_index is now
    # generation_index (0 on an athlete's first-ever generation, reusing
    # the same round-robin math a multi-day plan's day_index already uses).
    meals, _choices = _build_day(region_data, is_veg, avoid, prefer, level, goal_key, portion,
                                 is_minor, eggs_ok, macros, seed, budget_suggestion, prev=None,
                                 day_index=generation_index,
                                 recent_proteins=recent_proteins, recent_carbs=recent_carbs,
                                 recipe_candidates=recipe_candidates)
    day_summary = _day_summary(meals, goal_key, portion, profile, is_minor, day_index=generation_index)

    score, reasons = _confidence(profile)
    return {
        "confidence_score": score,
        "confidence_reason": reasons,
        "day_summary": day_summary,
        "meals": meals,
        "disclaimer": DISCLAIMER,
    }


def mock_provider_multi_day(profile, days, agent_context=None):
    """
    Multi-day extension of mock_provider() (Step 6): generates `days`
    consecutive daily plans for the same athlete, reusing every existing
    food-selection helper (choose_items/pick_protein/pick_carb/build_* via
    _build_day()) - there is no second food-selection system here, only a
    loop that calls the same one repeatedly with day-aware inputs.

    Every day shares the SAME athlete seed (unchanged from single-day) - it
    is `day_index` alone that advances each slot's round-robin position by
    one per day (see the note above SLOT_LUNCH_CARB): day 0 reproduces
    today's exact single-day offsets, day 1 moves one step further around
    each pool, and so on, sweeping through every position before any repeat
    is possible - not an independent per-day hash (that was Step 6's
    approach; the audit that preceded this step measured it under-covering
    each pool by about a third over a 7-day week). On top of that, each day
    after the first is still told the previous day's picks (see
    _build_day()'s `prev` parameter) as a second line of defence, not the
    only protection anymore.

    Determinism: every offset is derived purely from profile fields
    (variety_seed) and the day index - nothing random - so the same profile
    + the same `days` always produces the exact same multi-day plan, and two
    similar-but-different profiles still diverge, exactly like the
    single-day mock.

    Profile-level signals (region, diet, goal, budget, training load,
    macros, minor status, ovo-vegetarian exception) are computed ONCE, not
    per day - an athlete's diet/allergies/region/age don't change day to
    day, only which specific foods express them.

    Narrative text (Step 7): each day also gets its own `day_index`, which
    _build_day()/_day_summary() use to pick a deterministic reasoning
    phrasing (see _phrase() above GOAL_TARGETS) - so a week's worth of
    meal "reasoning" and day "notes" don't read as the exact same sentence
    seven times, even though the underlying food-selection logic is
    unchanged. Also (Step 7): only the LAST day may apply build_dinner()'s
    match-day override, so a "match" goal doesn't produce the identical
    khichdi dinner on all 7 days - see build_dinner()'s docstring.

    Phase 2A (memory-aware variety): the SAME generation_index/
    recent_proteins/recent_carbs mock_provider() reads from
    agent_context["variety_context"] apply here too - see its docstring.
    generation_index shifts every day's round-robin position by this
    athlete's generation count instead of always starting a fresh multi-day
    plan back at position 0, so requesting a new 7-day plan for the same
    athlete continues the rotation rather than repeating last week's plan
    exactly. `day_offset` (this loop's own 0..days-1 counter) still decides
    the "Day N" label and which day is last (match-day override) - only the
    round-robin math and narrative phrasing shift by generation_index.
    """
    tool_results = (agent_context or {}).get("tool_results") or {}

    is_veg = profile.get("diet") == "Vegetarian"
    region_data = REGIONS[detect_region(profile)]
    goal_key = detect_primary_goal(profile)
    avoid = avoid_keywords(profile)
    own_food_prefs = prefer_tokens(profile)
    prefer = own_food_prefs + ingredient_tokens(profile)
    level = budget_level(profile)
    factor, portion = training_load(profile)
    macros = split_macros(goal_key, factor, overrides=nutrition_tool_overrides(tool_results))

    age = profile.get("age")
    is_minor = age is not None and age < 18
    if is_minor and portion == "large":
        portion = "moderate"

    eggs_ok = is_veg and any("egg" in tok for tok in own_food_prefs)

    seed = variety_seed(profile)
    budget_suggestion = budget_tool_snack_suggestion(tool_results, level)

    # Phase 2A - see mock_provider()'s docstring for what these mean and
    # where they come from (orchestrator.py, via existing memory_manager
    # reads - this function never touches memory_manager itself). Absent
    # variety_context, every value below is exactly what it was before
    # Phase 2A: generation_index=0, no recent-item exclusion.
    variety_context = (agent_context or {}).get("variety_context") or {}
    generation_index = variety_context.get("generation_index", 0)
    recent_proteins = variety_context.get("recent_proteins") or []
    recent_carbs = variety_context.get("recent_carbs") or []

    # Personal Recipes - see mock_provider()'s docstring: this function
    # re-derives eligibility itself, every day, rather than trusting any
    # upstream filtering.
    raw_recipes = (agent_context or {}).get("personal_recipes") or []
    recipe_candidates = [r["name"] for r in classify_personal_recipes(raw_recipes, profile)["usable"]]

    days_out = []
    prev_choices = None
    for day_offset in range(days):
        is_last_day = day_offset == days - 1
        effective_day_index = generation_index + day_offset
        meals, choices = _build_day(region_data, is_veg, avoid, prefer, level, goal_key, portion,
                                    is_minor, eggs_ok, macros, seed, budget_suggestion,
                                    prev=prev_choices, day_index=effective_day_index,
                                    is_last_day=is_last_day,
                                    recent_proteins=recent_proteins, recent_carbs=recent_carbs,
                                    recipe_candidates=recipe_candidates)
        days_out.append({
            "day": day_offset + 1,
            "day_summary": _day_summary(meals, goal_key, portion, profile, is_minor,
                                        day_index=effective_day_index),
            "meals": meals,
        })
        prev_choices = choices

    score, reasons = _confidence(profile)
    return {
        "duration_days": days,
        "confidence_score": score,
        "confidence_reason": reasons,
        "days": days_out,
        "disclaimer": DISCLAIMER,
    }


# ----------------------------------------------------------------------
# 4. THE SINGLE INTERFACE  (the "socket" the whole app talks to)
# ----------------------------------------------------------------------

# Deployment config (Phase 0): overridable via ATHLETEOS_PROVIDER so a hosted
# deployment can switch providers with an environment variable instead of an
# edit + redeploy. Default stays "mock" so local/test behaviour is unchanged
# for every existing caller that doesn't set this env var.
PROVIDER = os.environ.get("ATHLETEOS_PROVIDER", "mock")
DEBUG = False       # set True to print internal signals while testing (never affects the JSON)


def generate_meal_plan(profile, agent_context=None, days=1):
    """
    The one function the rest of the app will ever call to get a plan.

    `agent_context` is optional and defaults to None, so every existing
    caller that only passes `profile` (the CLI below, tests, direct calls)
    behaves EXACTLY as before. When orchestrator.py supplies it, it carries
    RAG/tool results the orchestrator already computed - mock_provider(),
    llm_provider.claude_provider(), and llm_provider.openrouter_provider()
    each know how to use it (see each function's own docstring for what it
    does with it); with agent_context=None (the default), all three behave
    exactly as they did before agent_context existed.

    `days` (optional, default 1 - Step 6/multi-day planning) is likewise
    additive: with days=1 this function takes EXACTLY the code path it
    always has (single-day dict: confidence_score/day_summary/meals/
    disclaimer), for every provider. days > 1 is currently only implemented
    for PROVIDER="mock" (see mock_provider_multi_day()) - the deterministic
    provider is what the multi-day variety requirement (no second food-
    selection system, cross-day awareness, full reproducibility) was built
    against. Requesting days > 1 under "claude"/"openrouter" raises
    NotImplementedError rather than silently ignoring `days` or calling the
    LLM once per day with no cross-day awareness.
    """
    if days == 1:
        if PROVIDER == "mock":
            return mock_provider(profile, agent_context)
        if PROVIDER == "claude":
            # Imported lazily, inside the branch that needs it, so mock-only
            # usage (the default) never pulls in llm_provider.py or the
            # 'anthropic' package, and so this avoids a circular import at
            # module-load time (llm_provider.py itself imports from this file).
            from llm_provider import claude_provider
            return claude_provider(profile, agent_context)
        if PROVIDER == "openrouter":
            # Same lazy-import reasoning as the "claude" branch above.
            from llm_provider import openrouter_provider
            return openrouter_provider(profile, agent_context)
        raise ValueError(f"Unknown provider: {PROVIDER!r}")

    if PROVIDER != "mock":
        raise NotImplementedError(
            f"Multi-day plans (days={days}) are only implemented for "
            f"PROVIDER='mock' right now; got PROVIDER={PROVIDER!r}."
        )
    return mock_provider_multi_day(profile, days, agent_context)


# ----------------------------------------------------------------------
# 5. RUN IT
# ----------------------------------------------------------------------

def print_debug(profile):
    """
    Developer-only view of the internal signals the mock derives from the
    profile. It re-computes them with the same helpers the mock uses, so it
    reflects the real logic WITHOUT adding anything to the JSON response.
    """
    score, reasons = _confidence(profile)
    print("=" * 64)
    print("DEBUG  (developer info only - NOT part of the JSON response)")
    print("=" * 64)
    print(f"Diet:              {profile.get('diet')}")
    print(f"Detected region:   {detect_region(profile)}")
    print(f"Primary goal:      {detect_primary_goal(profile)}")
    print(f"Budget level:      {budget_level(profile)}")
    print(f"Avoid keywords:    {sorted(avoid_keywords(profile)) or '(none)'}")
    print(f"Food preferences:  {prefer_tokens(profile) or '(none)'}")
    print(f"Confidence score:  {score}")
    print(f"Confidence reason: {reasons}")
    print()


def main():
    profile = collect_profile()

    print("\n" + "=" * 64)
    print("PROMPT  (copy everything below into Claude.ai to test quality)")
    print("=" * 64)
    print(build_prompt(profile))

    if DEBUG:
        print_debug(profile)

    print("=" * 64)
    print("MOCK MEAL PLAN  (a real LLM will return this same JSON shape)")
    print("=" * 64)
    plan = generate_meal_plan(profile)
    print(json.dumps(plan, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
