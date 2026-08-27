"""
AthleteOS - per-food nutrition data registry (Stage 2, first pass).

Structured, hand-curated, sourced nutrition entries - NOT a nutrition
database in the general sense, and deliberately tiny right now. Follows the
same convention as recipe_links.py: a flat dict keyed by the exact item
string used elsewhere in the app, returning None for anything not yet
sourced (additive, never blocking).

METHODOLOGY (see the Stage 1 audit for the full research trail)
    Every entry here was pulled from a real, cited source - IFCT 2017
    (ICMR-National Institute of Nutrition, decrypted/text-extracted
    directly from the official PDF at nin.res.in/ebooks/IFCT2017.pdf) or
    USDA FoodData Central (live API query, fdc.nal.usda.gov) - never
    estimated, derived from a recipe's ingredients, or recalled from
    memory. Composite/prepared dishes (curries, sabzis, etc.) are
    categorically excluded: neither source catalogs them, and computing a
    value from raw ingredients would be exactly the kind of invented number
    this registry exists to avoid.

WHY THIS IS SO SMALL
    Two hard constraints eliminated almost every candidate:
    1. IFCT 2017's 528 foods are essentially all RAW/uncooked ingredients
       (confirmed by direct search of the extracted text - no curd, no
       buttermilk, no roti/chapati, no idli/dosa, no cooked dal anywhere).
       Applying a raw-ingredient value to how this app actually serves
       food (cooked, prepared) would require a cooking-yield conversion
       neither source documents - not applied, per explicit instruction.
    2. Where a source value genuinely matches the food's STATE (e.g. a
       cooked/boiled entry, or a raw-eaten-as-is fruit), the app's
       QUANTITY LABEL still often doesn't correspond to any documented
       gram weight (e.g. "1" banana, "2 eggs" - neither source states how
       many grams "1 banana" or "1 egg" is). Rescaling a per-100g value to
       a serving would require inventing that gram figure - not done.
       Where this applies, the entry stays "per 100g" and is NOT rescaled
       to the app's quantity - see each entry's "basis" field.

CURRENT COVERAGE: Banana only. See the entry below for exactly why it
qualifies and every other Stage-1-priority item does not (yet).
"""

NUTRITION = {
    "Banana": {
        # IFCT reports energy in kJ only; kcal = kJ / 4.184 (the standard
        # physical kJ-per-kcal conversion constant - a unit conversion, not
        # a nutritional estimate). 440 kJ / 4.184 = 105.16, rounded to 105.
        "basis": (
            "Per 100g edible portion of raw banana (IFCT food code E012, "
            "\"Banana, ripe, robusta\"). NOT rescaled to the app's \"1\" "
            "quantity - no authoritative gram weight for a single banana "
            "was sourced, so no serving-size conversion has been applied. "
            "Treat this as reference composition for banana generally, not "
            "a claim about the exact size of the banana in this meal."
        ),
        "kcal": 105,
        "protein_g": 1.23,
        "carbs_g": 23.63,
        "fat_g": 0.33,
        "source": "IFCT 2017 (ICMR-National Institute of Nutrition)",
        "source_ref": "Food code E012, Table 1 (Proximate Principles and Dietary Fibre), PDF p.49",
    },
}


def get_nutrition_info(item_name):
    """
    Return this food's sourced nutrition entry, or None if nothing has been
    verified yet. Deliberately item-only (not item+quantity) right now -
    every current entry is explicitly "per 100g", not rescaled to any
    app quantity label, so there is nothing quantity-specific to key on.
    """
    return NUTRITION.get(item_name)
