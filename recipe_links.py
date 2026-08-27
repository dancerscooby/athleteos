"""
AthleteOS - Recipe link registry (Step 10 / Phase 4C).

Structured, deterministic classification of which dishes in regional_foods.py
are "complex" enough to warrant a recipe link, and the link itself - no LLM
involved, matching the same "structured data, not generated text" principle
regional_foods.py itself follows.

WHAT THIS IS
    A single flat set of exact dish-name strings (COMPLEX_DISHES), keyed
    EXACTLY as those strings appear in regional_foods.py, plus one function
    (get_recipe_resource) that turns a dish name into a resource dict or
    None. Deliberately flat and region-agnostic: regional_foods.py reuses
    identical strings for common items across regions ("Dal", "Rice",
    "Curd", ...), so one shared set automatically gives the same dish the
    same classification everywhere it appears - there is no per-region
    registry to keep in sync.

CLASSIFICATION PRINCIPLE (approved by the user before this file was written)
    Complex = the dish has a real name implying a distinct technique or
    multiple prep steps - a curry, a stuffed/fermented/steamed item, a named
    regional specialty - the kind of thing someone would plausibly search
    "how to make X" for.
    Simple = raw/as-is foods, plain dairy, plain staples (rice/roti/roti
    variants), plain undistinguished dal, and everyday basic stir-fry
    sabzis that don't need a tutorial to get right.
    Explicit exceptions approved after review: Poha (all variants), Vegetable
    upma, Semiya upma, and Ragi mudde are everyday quick preparations where
    a recipe link adds little value despite technically having some
    technique - kept Simple rather than Complex for that reason.

URLS (Step 10, this pass only)
    YouTube SEARCH-RESULTS links, generated deterministically from the dish
    name - never a fabricated specific page. This is intentionally the
    smaller-effort option: swapping any entry to a specific verified recipe
    URL later is a one-line data change (see get_recipe_resource), not a
    redesign - the registry shape (dish name -> resource dict) doesn't
    change either way.

WHAT THIS IS NOT
    Not a nutrition source, not a claim that a specific recipe is
    authoritative - purely "does this dish plausibly need a recipe link,
    and if so, what search points at one."
"""

# Dishes considered complex enough to warrant a recipe link. Exact strings,
# copied from regional_foods.py's own candidate lists - not normalized,
# because every current dish string is already the single canonical name
# used everywhere it appears (regional_foods.py itself reuses identical
# strings for the same dish across regions, e.g. "Dal", "Rice", "Curd").
COMPLEX_DISHES = {
    # --- Breakfast mains ---
    "Idli with sambar", "Masala dosa", "Ven pongal", "Uttapam",
    "Appam with vegetable stew", "Rava idli", "Set dosa", "Ragi dosa",
    "Stuffed paratha with curd", "Besan chilla", "Chana masala with kulcha",
    "Aloo paratha", "Sooji halwa with chana", "Thepla with curd",
    "Misal pav", "Dhokla", "Handvo", "Sabudana khichdi", "Methi thepla",
    "Vegetable paratha", "Sattu paratha", "Ghugni with muri",
    "Luchi with aloo dum", "Bhutte ka kees",
    "Muthia (steamed rice dumplings)", "Pitha (rice cake)",
    "Vegetable thukpa", "Steamed veg momos",

    # --- Snacks ---
    "Groundnut chikki", "Peanut chikki", "Sukha bhel",
    "Muri (puffed rice) with chana", "Jhalmuri", "Ghugni (white peas)",

    # --- Carbs (staples) ---
    "Litti",

    # --- Vegetarian protein (named curry/technique dishes) ---
    "Sambar (toor dal)", "Rasam with dal", "Moong dal kootu",
    "Soya chunk curry", "Peanut sundal", "Rajma", "Paneer sabzi",
    "Chana masala", "Kadhi with besan", "Kadhi", "Chole", "Paneer bhurji",
    "Matki usal", "Sprouts usal", "Cholar dal",
    "Fermented soybean (axone) with dal", "Bamboo shoot with dal",
    "Kholar (Naga rajma)",

    # --- Non-vegetarian protein (every entry is a seasoned/cooked dish) ---
    "Egg curry", "Chicken curry", "Fish curry", "Chicken Chettinad",
    "Egg roast", "Prawn curry", "Egg bhurji", "Chicken tikka",
    "Mutton keema", "Grilled chicken", "Chicken sukka", "Prawn masala",
    "Fish curry (macher jhol)", "Rohu fish fry",
    "Smoked pork with bamboo shoot", "Chicken with local herbs",
    "Dried fish curry", "Boiled chicken with greens",

    # --- Veg sides (distinctive regional specialties) ---
    "Mixed vegetable kootu", "Avial", "Pumpkin erissery", "Undhiyu",
    "Sarson ka saag", "Saag", "Shukto", "Bamboo shoot fry",

    # --- Veg sides: prepared sabzi/fry/curry dishes (Step 4C follow-up,
    # approved after audit) - these have real technique (tempering,
    # sauteing, mashing, roasting) even though the name is generic, and
    # match the user's own bar: "if someone would reasonably need a
    # recipe to prepare it, give them one." ---
    "Beans poriyal", "Cabbage thoran", "Bhindi fry", "Carrot poriyal",
    "Keerai masiyal",
    "Aloo-gobi", "Palak sabzi", "Mixed veg", "Bhindi masala",
    "Baingan bharta", "Lauki sabzi", "Tinda sabzi",
    "Mixed veg shaak", "Bhindi sabzi", "Cabbage sabzi", "Gawar sabzi",
    "Dudhi (lauki) sabzi", "Tendli sabzi",
    "Aloo-potol sabzi", "Mixed vegetable ghonto", "Begun bhaja",
    "Lau (bottle gourd) ghonto", "Bandhakopir torkari",
    "Aloo-baingan sabzi", "Chech bhaji (greens)",
    "Lai xaak (mustard greens)", "Squash curry", "Stir-fried greens",

    # --- Match-day dinner ---
    # (deliberately NOT included: "Khichdi (rice + moong dal)" is a
    # one-pot rice+lentil boil, same treatment as the other khichdi/
    # khichuri variants below - Simple.)
}

# Explicit note (not code - COMPLEX_DISHES above is authoritative) on dishes
# that might look like candidates but were deliberately left Simple, so a
# future editor doesn't "fix" this file by adding them back:
#   Poha, Poha with peanuts, Poha with sev, Vegetable upma, Semiya upma,
#   Ragi mudde - approved exceptions: everyday quick preparations where a
#   recipe link adds little value despite some technique being involved.
#   Moong khichdi, Vegetable khichdi, Vegetable khichuri, Chira with curd,
#   Jolpan (rice flakes with curd) - one-pot/no-cook items, same category
#   as the other khichdi variants and plain rice-flakes dishes.
#   Palak (East) - listed as a bare ingredient name (unlike North's
#   "Palak sabzi"), read as plain steamed spinach rather than a named
#   preparation - kept Simple, unlike its North/Central/West "Palak sabzi"
#   counterpart.
#   Pumpkin with herbs (Northeast) - descriptive rather than a named
#   technique (contrast "Squash curry", added as Complex) - kept Simple.
#   Boiled seasonal vegetables, Boiled beans with herbs (Northeast) -
#   explicitly "boiled" - same treatment as any other boiled/plain item.


def _search_url(dish_name):
    """
    Deterministically build a YouTube search-results URL from the dish
    name - the exact same query-shaping logic the single-resource-per-meal
    version of this feature already used (lowercase, drop any parenthetical
    qualifier, spaces -> '+'), so existing URL formatting is unchanged.
    Never a fabricated specific page - a search query is always a real,
    valid destination.
    """
    query = dish_name.lower().split("(")[0].strip().replace(" ", "+")
    return f"https://www.youtube.com/results?search_query={query}+recipe"


def get_recipe_resource(item_name):
    """
    Return this ingredient's recipe resource dict, or None if it's
    classified Simple (no link). The `item` field names which ingredient
    this resource belongs to - callers building a meal's full resources
    list use it to attribute each link to the right dish (see
    meal_planner._resources_for_ingredients()).
    """
    if item_name not in COMPLEX_DISHES:
        return None
    return {
        "type": "youtube",
        "title": f"How to make {item_name}",
        "url": _search_url(item_name),
        "item": item_name,
    }
