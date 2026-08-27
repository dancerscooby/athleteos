"""
AthleteOS - Regional food data (Phase 2).

The candidate foods the deterministic meal builder picks from, organised by
Indian region. Extracted out of meal_planner.py because this is *data*, not
logic - the same "one file, one job" split the project already uses for
ingest/embeddings/retriever and for tools/.

WHAT THIS IS
    Culinary/cultural data only: which everyday foods are commonly eaten in
    which part of India, grouped by the slot they fill in a day's meals.
    There are deliberately NO nutrition numbers here - calories, protein and
    portion logic all stay in meal_planner.py / nutrition_tool.py, so nothing
    in this file can silently become an unsourced nutrition claim.

WHAT THIS IS NOT
    This maps LOCATION -> CUISINE, and nothing else. It says what food is
    commonly available and culturally familiar in a region. It makes no claim
    whatsoever about the people who live there - not their genetics, not their
    metabolism, not their nutritional requirements. Two athletes with the same
    body stats and training load get the same nutrition targets regardless of
    region; only the foods those targets are expressed in differ.

REGIONS
    Six Indian zones - north, south, east, west, central, northeast - plus
    (Phase 2B) a seventh, non-Indian "international" pool used only as the
    DEFAULT_REGION fallback for an athlete outside India or with no/
    unrecognised location text (previously this silently fell through to
    "north", which is not culturally accurate for a non-Indian athlete).
    Central (MP/Chhattisgarh) and northeast were previously swept into
    "north" by default too, which is why athletes there got food from the
    wrong kitchen entirely - the same class of bug "international" now
    fixes for non-Indian athletes.

SCHEMA (every region has the same keys, so the builders stay generic)
    keywords         cities/states used to detect the region from free text
    breakfast        breakfast mains
    breakfast_sides  the small protein-ish side added to breakfast, in
                     regional preference order (northeast is deliberately
                     low-dairy, so it does not lead with curd)
    dairy_side       the dairy accompaniment customarily served with a main
                     meal in this region, or None where that is not the
                     custom. Northeast is None: dairy is eaten there, but it
                     is not the default alongside lunch the way curd is in
                     the north or chaas is in the west, so the builders must
                     not add one automatically. This replaced a hard-coded
                     "Curd" in build_lunch()/build_dinner() that applied to
                     every vegetarian in every region.
    snack            between-meal options
    carbs            staple carbohydrate for lunch/dinner
    veg_protein      vegetarian mains
    nonveg_protein   non-vegetarian mains
    veg              vegetable dishes

Sources consulted for the two newly-added regions (culinary attribution, not
nutrition): regional cuisine overviews for Northeast India (rice, bamboo
shoot, fermented soybean/fish, smoked pork) and for Central India (wheat/
jowar/maize rotis, arhar dal, Indori poha, bhutte ka kees, Chhattisgarhi
muthia; MP is a major soybean-growing state, so soya is genuinely common).
"""

REGIONS = {
    "south": {
        "keywords": [
            "kerala", "tamil", "chennai", "karnataka", "bangalore", "bengaluru",
            "hyderabad", "telangana", "andhra", "kochi", "madurai", "mysore",
            "coimbatore", "trivandrum", "thiruvananthapuram", "vijayawada",
            "visakhapatnam", "mangalore", "kozhikode", "salem", "tiruchirappalli",
        ],
        "breakfast": [
            "Idli with sambar", "Masala dosa", "Vegetable upma", "Ven pongal",
            "Uttapam", "Appam with vegetable stew", "Rava idli", "Set dosa",
            "Semiya upma", "Ragi dosa",
        ],
        "breakfast_sides": ["Curd", "Buttermilk", "Roasted peanuts", "Boiled chana", "Coconut chutney"],
        "dairy_side": "Curd",
        "snack": [
            "Roasted chana (sundal)", "Banana", "Buttermilk", "Ragi malt",
            "Steamed sweet corn", "Groundnut chikki", "Tender coconut water",
            "Boiled peanuts", "Seasonal fruit",
        ],
        "carbs": ["Rice", "Ragi mudde", "Whole-wheat roti", "Red rice", "Idiyappam"],
        "veg_protein": [
            "Sambar (toor dal)", "Rasam with dal", "Boiled chana", "Moong dal kootu",
            "Curd rice", "Soya chunk curry", "Urad dal", "Peanut sundal",
        ],
        "nonveg_protein": [
            "Egg curry", "Chicken curry", "Fish curry", "Chicken Chettinad",
            "Egg roast", "Prawn curry",
        ],
        "veg": [
            "Beans poriyal", "Cabbage thoran", "Mixed vegetable kootu", "Avial",
            "Bhindi fry", "Pumpkin erissery", "Keerai masiyal", "Carrot poriyal",
        ],
    },
    "north": {
        "keywords": [
            "delhi", "punjab", "haryana", "lucknow", "uttar pradesh", " up",
            "rajasthan", "jaipur", "chandigarh", "kanpur", "noida", "himachal",
            "uttarakhand", "agra", "varanasi", "amritsar", "ludhiana", "jodhpur",
            "udaipur", "dehradun", "meerut", "gurgaon", "gurugram", "kashmir",
        ],
        "breakfast": [
            "Poha with peanuts", "Stuffed paratha with curd", "Vegetable upma",
            "Besan chilla", "Daliya (broken wheat porridge)", "Chana masala with kulcha",
            "Aloo paratha", "Vegetable oats", "Sooji halwa with chana",
        ],
        "breakfast_sides": ["Curd", "Roasted peanuts", "Boiled chana", "Pickled vegetables (achaar)"],
        "dairy_side": "Curd",
        "snack": [
            "Roasted chana", "Banana with milk", "Sprouts chaat", "Lassi",
            "Makhana (fox nuts)", "Peanut chikki", "Fruit chaat", "Roasted peanuts",
            "Seasonal fruit",
        ],
        "carbs": ["Whole-wheat roti", "Rice", "Missi roti", "Bajra roti", "Jowar roti"],
        "veg_protein": [
            "Dal", "Rajma", "Paneer sabzi", "Chana masala", "Kadhi with besan",
            "Moong dal", "Chole", "Soya chunk curry", "Sprouted moong salad",
        ],
        "nonveg_protein": [
            "Egg bhurji", "Chicken curry", "Egg curry", "Chicken tikka",
            "Mutton keema", "Grilled chicken",
        ],
        "veg": [
            "Aloo-gobi", "Palak sabzi", "Mixed veg", "Bhindi masala",
            "Baingan bharta", "Lauki sabzi", "Sarson ka saag", "Tinda sabzi",
        ],
    },
    "west": {
        "keywords": [
            "mumbai", "maharashtra", "pune", "gujarat", "ahmedabad", "surat",
            "goa", "nagpur", "rajkot", "nashik", "vadodara", "thane", "kolhapur",
            "aurangabad", "panaji", "bhavnagar", "jamnagar",
        ],
        "breakfast": [
            "Poha", "Thepla with curd", "Vegetable upma", "Misal pav", "Dhokla",
            "Handvo", "Sabudana khichdi", "Methi thepla", "Vegetable paratha",
        ],
        "breakfast_sides": ["Curd", "Buttermilk (chaas)", "Roasted peanuts", "Boiled chana"],
        "dairy_side": "Buttermilk (chaas)",
        # "Dhokla" and "Buttermilk (chaas)" used to also appear here,
        # duplicating breakfast/breakfast_sides+dairy_side respectively -
        # removed from this pool (Phase 2C) so each stays in the ONE pool
        # where it best fits, instead of being reachable from 2-3
        # independently-picked slots in the same day.
        "snack": [
            "Roasted chana", "Banana", "Khakhra", "Sukha bhel",
            "Peanut chikki", "Seasonal fruit", "Roasted peanuts",
        ],
        "carbs": ["Bhakri (jowar)", "Whole-wheat roti", "Rice", "Bajra bhakri", "Nachni bhakri (ragi)"],
        "veg_protein": [
            "Dal", "Chana masala", "Paneer bhurji", "Moong khichdi", "Matki usal",
            "Kadhi", "Soya chunk curry", "Val dal", "Sprouts usal",
        ],
        "nonveg_protein": [
            "Egg curry", "Chicken sukka", "Fish curry", "Egg bhurji",
            "Chicken curry", "Prawn masala",
        ],
        "veg": [
            "Mixed veg shaak", "Bhindi sabzi", "Cabbage sabzi", "Undhiyu",
            "Baingan bharta", "Gawar sabzi", "Dudhi (lauki) sabzi", "Tendli sabzi",
        ],
    },
    "east": {
        "keywords": [
            "kolkata", "west bengal", "bengal", "bihar", "patna", "odisha",
            "bhubaneswar", "jharkhand", "ranchi", "siliguri", "cuttack",
            "howrah", "durgapur", "gaya", "muzaffarpur", "jamshedpur", "dhanbad",
        ],
        "breakfast": [
            "Chire (rice flakes) with banana", "Vegetable upma", "Roti with sabzi",
            "Sattu paratha", "Ghugni with muri", "Luchi with aloo dum",
            "Vegetable khichuri", "Chira with curd",
        ],
        "breakfast_sides": ["Curd", "Roasted chana", "Roasted peanuts"],
        "dairy_side": "Curd",
        "snack": [
            "Muri (puffed rice) with chana", "Banana", "Roasted chana",
            "Sattu drink", "Jhalmuri", "Ghugni (white peas)", "Seasonal fruit",
            "Roasted peanuts",
        ],
        "carbs": ["Rice", "Whole-wheat roti", "Litti", "Chira (rice flakes)", "Panta bhat (fermented rice)"],
        "veg_protein": [
            "Cholar dal", "Moong dal", "Chana masala", "Sattu (roasted gram)",
            "Ghugni (white peas)", "Soya chunk curry", "Masoor dal",
        ],
        "nonveg_protein": [
            "Fish curry (macher jhol)", "Egg curry", "Chicken curry",
            "Prawn curry", "Egg bhurji", "Rohu fish fry",
        ],
        "veg": [
            "Aloo-potol sabzi", "Mixed vegetable ghonto", "Palak", "Begun bhaja",
            "Shukto", "Saag", "Lau (bottle gourd) ghonto", "Bandhakopir torkari",
        ],
    },
    # --- Newly added in Phase 2 (previously fell through to "north") ---
    "central": {
        "keywords": [
            "madhya pradesh", "bhopal", "indore", "jabalpur", "gwalior", "ujjain",
            "chhattisgarh", "raipur", "bilaspur", "sagar", "rewa", "durg",
            "korba", "satna", "ratlam",
        ],
        "breakfast": [
            "Poha with sev", "Bhutte ka kees", "Vegetable upma", "Besan chilla",
            "Daliya (broken wheat porridge)", "Muthia (steamed rice dumplings)",
            "Jowar roti with sabzi", "Vegetable paratha",
        ],
        "breakfast_sides": ["Curd", "Roasted chana", "Roasted peanuts"],
        "dairy_side": "Curd",
        "snack": [
            "Roasted chana", "Banana", "Peanut chikki", "Muri", "Roasted peanuts",
            "Buttermilk", "Seasonal fruit", "Boiled corn",
        ],
        "carbs": ["Jowar roti", "Whole-wheat roti", "Rice", "Makka (maize) roti", "Bafla (wheat dumpling)"],
        "veg_protein": [
            "Arhar dal", "Chana dal", "Rajma", "Moong dal", "Soya chunk curry",
            "Kadhi", "Chana masala", "Urad dal",
        ],
        "nonveg_protein": [
            "Egg curry", "Chicken curry", "Egg bhurji", "Chicken sukka",
            "Mutton curry", "Fish curry",
        ],
        "veg": [
            "Aloo-baingan sabzi", "Palak sabzi", "Mixed veg", "Lauki sabzi",
            "Bhindi sabzi", "Chech bhaji (greens)", "Cabbage sabzi",
            "Bharwa baingan (stuffed eggplant)", "Bhutte ki sabzi (corn kernel sabzi)",
        ],
    },
    "northeast": {
        "keywords": [
            "assam", "guwahati", "dibrugarh", "silchar", "manipur", "imphal",
            "nagaland", "kohima", "dimapur", "meghalaya", "shillong", "tripura",
            "agartala", "mizoram", "aizawl", "arunachal", "itanagar", "sikkim",
            "gangtok", "jorhat", "tezpur",
        ],
        "breakfast": [
            "Rice with boiled vegetables", "Pitha (rice cake)", "Jolpan (rice flakes with curd)",
            "Vegetable thukpa", "Steamed veg momos", "Rice flakes with banana",
            "Vegetable khichdi", "Poita bhaat (fermented rice) with vegetables",
            "Black rice porridge (chak-hao)",
        ],
        # Northeast cuisine is comparatively low-dairy, so curd is not the
        # default side here the way it is in the north/west.
        "breakfast_sides": ["Boiled black gram", "Roasted peanuts", "Curd"],
        "dairy_side": None,
        # "Boiled black gram" used to also appear here, duplicating
        # breakfast_sides AND veg_protein - removed from this pool (Phase
        # 2C) so it stays reachable from breakfast_sides (a genuine side)
        # and veg_protein (its more substantial protein role), not all
        # three independently-picked slots in the same day.
        "snack": [
            "Steamed sweet corn", "Banana", "Roasted peanuts",
            "Seasonal fruit", "Steamed veg momos", "Rice flakes",
        ],
        "carbs": ["Rice", "Sticky rice", "Rice flakes", "Whole-wheat roti"],
        "veg_protein": [
            "Dal", "Boiled black gram", "Fermented soybean (axone) with dal",
            "Bamboo shoot with dal", "Kholar (Naga rajma)", "Moong dal",
            "Soya chunk curry",
        ],
        "nonveg_protein": [
            "Fish curry", "Smoked pork with bamboo shoot", "Chicken with local herbs",
            "Egg curry", "Dried fish curry", "Boiled chicken with greens",
        ],
        "veg": [
            "Boiled seasonal vegetables", "Bamboo shoot fry", "Lai xaak (mustard greens)",
            "Pumpkin with herbs", "Squash curry", "Boiled beans with herbs",
            "Stir-fried greens", "Fiddlehead fern curry (dhekia xaak)",
        ],
    },
    # --- Phase 2B: the non-Indian / unmatched-location fallback ---
    # Deliberately NOT any single national cuisine - see the module
    # docstring's REGIONS note above for why, and why "keywords" is empty
    # (this pool is reached only via DEFAULT_REGION, never by a text match -
    # see detect_region()).
    #
    # Egg dishes are deliberately kept OUT of breakfast/breakfast_sides/snack/
    # veg_protein here, exactly like every Indian region already does -
    # those pools are used directly by is_veg-unaware builders (only
    # allergy/avoid filtering applies to them), so an egg item placed there
    # would leak into a vegetarian athlete's plan. Eggs only ever appear via
    # nonveg_protein (gated by is_veg in pick_protein()) or the separate,
    # explicitly eggs_ok-gated "Boiled eggs" addition build_breakfast()
    # already has for every region - unchanged here.
    "international": {
        "keywords": [],
        "breakfast": [
            "Oatmeal with fruit", "Whole-grain cereal with milk",
            "Greek yogurt with granola", "Pancakes with fruit", "Peanut butter toast",
            "Fruit and nut porridge", "Fruit smoothie with oats",
        ],
        "breakfast_sides": ["Yogurt", "Roasted nuts", "Baked beans"],
        # Unlike most of the Indian regions, a dairy side is not
        # automatically added to lunch/dinner in a generic international
        # meal pattern the way curd/chaas is in India - so this stays None,
        # the same treatment northeast already gets for the same reason
        # (see the "dairy_side" note in the module docstring above).
        "dairy_side": None,
        "snack": [
            "Mixed nuts", "Seasonal fruit", "Yogurt with honey", "Granola bar",
            "Hummus with vegetables", "Peanut butter with crackers", "Trail mix",
            "Roasted chickpeas",
        ],
        "carbs": ["Rice", "Pasta", "Bread", "Potatoes"],
        "veg_protein": [
            "Lentil soup", "Bean stew", "Tofu stir-fry", "Chickpea salad",
            "Hummus with pita", "Split pea soup", "Black bean stew",
        ],
        "nonveg_protein": [
            "Grilled chicken", "Baked fish", "Roast chicken", "Grilled salmon",
            "Turkey breast", "Beef stew", "Scrambled eggs",
        ],
        "veg": [
            "Steamed mixed vegetables", "Roasted vegetables", "Garden salad",
            "Sauteed spinach", "Grilled vegetables", "Vegetable soup", "Coleslaw",
        ],
    },
}

# Non-dairy vegetarian proteins used when a dairy allergy / avoid-list removes
# the region's usual protein. Region-agnostic on purpose: this is the safety
# net that runs when regional options have all been filtered out.
SAFE_VEG_PROTEIN = ["Soya chunk curry", "Rajma", "Boiled chana", "Moong dal"]

# Phase 2B: an athlete outside India (or with no/unrecognised location text)
# now gets a genuinely international/general food pool, not silently North
# Indian food. "international" is reached ONLY via this default - its own
# "keywords" list is empty, so it is never matched by detect_region()'s text
# scan below, exactly like every other region is matched.
DEFAULT_REGION = "international"


def detect_region(profile):
    """
    Map the free-text region to one of the six Indian zones, or the
    "international" fallback (Phase 2B) for anything outside India or left
    blank/unrecognised.

    Substring matching on a space-padded lowercase string, unchanged from
    the original implementation - the only difference from Phase 2 is what
    happens when nothing matches: previously that silently meant "north"
    (culturally wrong for a non-Indian athlete), now it means the dedicated
    "international" pool. Every Indian region keyword match is completely
    unaffected - an Indian location still resolves to its own regional
    cuisine exactly as before.
    """
    text = " " + (profile.get("region") or "").lower()
    for key, data in REGIONS.items():
        if any(k in text for k in data["keywords"]):
            return key
    return DEFAULT_REGION
