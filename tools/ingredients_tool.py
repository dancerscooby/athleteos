"""
AthleteOS - Home ingredients tool (Phase 4, step 2).

Given the ingredients an athlete says they already have at home, works out
which are actually usable for them, which food groups those cover, and which
groups are still missing. Deterministic table lookups and string matching -
no LLM, no network, no nutrition maths (that stays in nutrition_tool.py).

Deliberately standalone, like every other tool in this package: it takes the
already-computed `avoid` words as an argument rather than importing
meal_planner to recompute them, so tools/ keeps no dependency on the
application. The caller (orchestrator._run_ingredients_tool) passes in
meal_planner.avoid_keywords(profile), which is the single source of truth
for allergy/avoid expansion - including its dairy expansion and singular/
plural handling.

Honesty rule baked into the output: anything this tool cannot confidently
place in a food group comes back under "unrecognised" rather than being
quietly counted as covering something. The point is to report what the
ingredients genuinely provide, not to flatter the input.
"""

from tools.base_tool import BaseTool

# Keyword -> food group. Substring matching, in the same deliberately simple
# style as meal_planner's avoid/prefer handling. An ingredient may legitimately
# belong to more than one group (peanuts are protein and fat; potato is a
# carbohydrate and a vegetable), so groups are not mutually exclusive.
INGREDIENT_GROUPS = {
    "protein": [
        "dal", "lentil", "rajma", "kidney bean", "chana", "chickpea", "chole",
        "paneer", "curd", "yogurt", "yoghurt", "milk", "soy", "soya", "tofu",
        "egg", "chicken", "fish", "mutton", "prawn", "pork", "sattu",
        "black gram", "urad", "moong", "masoor", "toor", "arhar", "sprout",
        "peanut", "groundnut",
    ],
    "carbohydrate": [
        "rice", "roti", "chapati", "atta", "wheat", "flour", "poha", "oats",
        "bread", "potato", "aloo", "bhakri", "jowar", "bajra", "ragi",
        "millet", "idli", "dosa", "upma", "suji", "semolina", "rava",
        "chira", "muri", "noodle", "pasta", "maize", "corn", "sabudana",
    ],
    "fat": [
        "oil", "ghee", "butter", "peanut", "groundnut", "almond", "cashew",
        "walnut", "seed", "coconut", "til", "sesame",
    ],
    "fruit": [
        "banana", "apple", "orange", "guava", "papaya", "mango", "amla",
        "lemon", "lime", "grape", "pomegranate", "date", "watermelon",
        "chikoo", "sapota", "pear", "fruit",
    ],
    "vegetable": [
        "onion", "tomato", "potato", "spinach", "palak", "cabbage",
        "cauliflower", "gobi", "carrot", "bean", "pea", "brinjal", "baingan",
        "bhindi", "okra", "lauki", "gourd", "cucumber", "green", "methi",
        "capsicum", "pumpkin", "bamboo", "radish", "beetroot", "saag",
        "vegetable", "mushroom",
    ],
}

# Ingredients a Vegetarian athlete cannot use. Kept separate from the group
# table because this is a DIET rule, not a nutrition classification.
NON_VEGETARIAN = [
    "egg", "chicken", "fish", "mutton", "prawn", "pork", "meat", "beef",
    "shrimp", "crab", "bacon", "ham",
]

ALL_GROUPS = ["protein", "carbohydrate", "fat", "fruit", "vegetable"]


def parse_ingredients(raw):
    """
    Turn free text ("rice, dal and 2 tomatoes") or an already-split list into
    a clean list of lowercase ingredient strings. Mirrors the comma/"and"
    splitting meal_planner already uses for preferences and avoid lists.
    """
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        parts = [str(item) for item in raw]
    else:
        parts = str(raw).replace(" and ", ",").split(",")
    cleaned = [p.strip().lower() for p in parts if p and p.strip()]
    return [c for c in cleaned if c != "none"]


def _groups_for(ingredient):
    """Every food group this ingredient plausibly contributes to (may be empty)."""
    return [
        group for group, keywords in INGREDIENT_GROUPS.items()
        if any(keyword in ingredient for keyword in keywords)
    ]


def _is_non_vegetarian(ingredient):
    return any(word in ingredient for word in NON_VEGETARIAN)


def assess_ingredients(ingredients, diet="Vegetarian", avoid=None):
    """
    Sort the supplied ingredients into usable / excluded / unrecognised, and
    report which food groups they cover and which are still missing.

    `avoid` is the caller's already-expanded set of allergy and
    foods-to-avoid words. Exclusions are reported with a reason so the
    decision is auditable rather than silent - and so a caller can tell the
    athlete *why* something they own was not used.
    """
    items = parse_ingredients(ingredients)
    avoid_words = [a.lower() for a in (avoid or [])]

    usable, excluded, unrecognised = [], [], []
    covered = {group: [] for group in ALL_GROUPS}

    for ingredient in items:
        blocked_by = next((a for a in avoid_words if a and a in ingredient), None)
        if blocked_by:
            excluded.append({
                "ingredient": ingredient,
                "reason": f"excluded by allergy/foods-to-avoid ({blocked_by})",
            })
            continue

        if diet == "Vegetarian" and _is_non_vegetarian(ingredient):
            excluded.append({
                "ingredient": ingredient,
                "reason": "not suitable for a Vegetarian diet",
            })
            continue

        usable.append(ingredient)

        groups = _groups_for(ingredient)
        if not groups:
            # Usable, but we cannot claim it covers any food group.
            unrecognised.append(ingredient)
            continue
        for group in groups:
            covered[group].append(ingredient)

    missing = [group for group in ALL_GROUPS if not covered[group]]

    return {
        "provided": items,
        "usable": usable,
        "excluded": excluded,
        "unrecognised": unrecognised,
        "covered_groups": {g: v for g, v in covered.items() if v},
        "missing_groups": missing,
        "diet": diet,
    }


class IngredientsTool(BaseTool):
    """Assesses what an athlete's home ingredients can and cannot cover."""

    name = "ingredients_tool"
    description = (
        "Given a list of ingredients the athlete already has at home, their "
        "diet, and their allergy/foods-to-avoid words, report which "
        "ingredients are usable, which are excluded and why, which food "
        "groups (protein, carbohydrate, fat, fruit, vegetable) they cover, "
        "and which groups are still missing."
    )

    def run(self, ingredients, diet="Vegetarian", avoid=None, **kwargs):
        return assess_ingredients(ingredients, diet, avoid)
