"""
AthleteOS - Phase 2 tests: food variety + regional food architecture.

Two guarantees:

  VARIETY   - similar-but-not-identical profiles get genuinely different
              meals, while the SAME profile stays reproducible. Crucially,
              variety must never override safety: allergies, foods-to-avoid,
              diet type, budget and stated preferences all keep the exact
              authority they had before variety existed.

  REGIONAL  - six Indian zones (north/south/east/west/central/northeast)
              instead of four, each producing food from its own kitchen.
              Central and northeast previously fell through to "north".

Everything here is offline and deterministic - no network, no API calls.
"""

import unittest

import meal_planner as mp
from regional_foods import DEFAULT_REGION, REGIONS, detect_region

BASE = dict(
    age=22, height_cm=178, weight_kg=72, sex="Male",
    goal="Improve football performance",
    diet="Vegetarian", training_days_per_week=5, training_intensity="High",
    region="Lucknow, Uttar Pradesh", allergies="none",
    budget_per_day=None, food_preferences=None, foods_to_avoid=None, notes=None,
)


def all_items(plan):
    return [ing["item"] for meal in plan["meals"] for ing in meal["ingredients"]]


def plan_signature(profile):
    """Every ingredient in the plan, as a comparable tuple."""
    return tuple(all_items(mp.mock_provider(dict(profile))))


# ----------------------------------------------------------------------
# 1. Regional coverage
# ----------------------------------------------------------------------

class TestSixRegions(unittest.TestCase):
    def test_all_six_indian_regions_exist(self):
        self.assertEqual(
            {"north", "south", "east", "west", "central", "northeast"} - set(REGIONS),
            set(),
        )

    def test_international_fallback_region_exists(self):
        """Phase 2B: a genuine 7th pool for non-Indian/unrecognised locations."""
        self.assertIn("international", REGIONS)

    def test_every_region_has_the_same_schema(self):
        must_be_populated = {"keywords", "breakfast", "breakfast_sides", "snack",
                             "carbs", "veg_protein", "nonveg_protein", "veg"}
        # dairy_side is the one key allowed to be None: it names the dairy
        # accompaniment customarily served with a main meal, and "there isn't
        # one" is a legitimate answer (northeast, and now international -
        # see regional_foods.py's "dairy_side" note).
        required = must_be_populated | {"dairy_side"}
        for name, data in REGIONS.items():
            with self.subTest(region=name):
                self.assertEqual(set(data), required)
                for key in must_be_populated:
                    # "keywords" is the one field allowed to be empty, and
                    # only for "international" - it is reached purely via
                    # DEFAULT_REGION, never by a text match, so it has
                    # nothing to match against (see detect_region()).
                    if key == "keywords" and name == "international":
                        self.assertEqual(data[key], [])
                        continue
                    self.assertTrue(data[key], f"{name}.{key} must not be empty")

    def test_central_india_is_detected_and_no_longer_falls_back_to_north(self):
        for city in ["Bhopal, Madhya Pradesh", "Indore", "Raipur, Chhattisgarh"]:
            with self.subTest(city=city):
                self.assertEqual(detect_region({"region": city}), "central")

    def test_northeast_is_detected_and_no_longer_falls_back_to_north(self):
        for city in ["Guwahati, Assam", "Imphal, Manipur", "Shillong, Meghalaya"]:
            with self.subTest(city=city):
                self.assertEqual(detect_region({"region": city}), "northeast")

    def test_original_four_regions_still_detect_correctly(self):
        cases = {
            "Lucknow, Uttar Pradesh": "north",
            "Chennai, Tamil Nadu": "south",
            "Kolkata, West Bengal": "east",
            "Pune, Maharashtra": "west",
        }
        for city, expected in cases.items():
            with self.subTest(city=city):
                self.assertEqual(detect_region({"region": city}), expected)

    def test_unknown_region_falls_back_to_the_default(self):
        self.assertEqual(detect_region({"region": "Atlantis"}), DEFAULT_REGION)
        self.assertEqual(detect_region({"region": ""}), DEFAULT_REGION)
        self.assertEqual(detect_region({}), DEFAULT_REGION)

    def test_each_region_produces_food_from_its_own_candidate_lists(self):
        """Nothing regional should ever leak in from another zone's kitchen."""
        cities = {
            "north": "Lucknow, Uttar Pradesh", "south": "Chennai, Tamil Nadu",
            "east": "Kolkata, West Bengal", "west": "Pune, Maharashtra",
            "central": "Indore, Madhya Pradesh", "northeast": "Guwahati, Assam",
        }
        for region, city in cities.items():
            with self.subTest(region=region):
                data = REGIONS[region]
                own = set(data["breakfast"] + data["breakfast_sides"] + data["snack"]
                          + data["carbs"] + data["veg_protein"] + data["veg"])
                # Items the builders can add regardless of region.
                generic = {"Banana", "Dates (5-6)", "Roasted chana", "Curd",
                           "Khichdi (rice + moong dal)", "Vegetable upma",
                           "Boiled eggs", "Moong dal", "Rice"}
                plan = mp.mock_provider(dict(BASE, region=city))
                for item in all_items(plan):
                    self.assertTrue(
                        item in own or item in generic,
                        f"{region}: unexpected item {item!r} not from this region",
                    )

    def test_northeast_breakfast_side_is_not_dairy_by_default(self):
        """Northeast cuisine is comparatively low-dairy, so curd must not be
        the automatic breakfast side the way it is up north."""
        self.assertNotIn("curd", REGIONS["northeast"]["breakfast_sides"][0].lower())

    def test_regions_offer_meaningfully_more_choice_than_before(self):
        """Phase 2 exists because 3-item lists forced repetition."""
        for name, data in REGIONS.items():
            with self.subTest(region=name):
                self.assertGreaterEqual(len(data["breakfast"]), 6)
                self.assertGreaterEqual(len(data["snack"]), 6)
                self.assertGreaterEqual(len(data["veg_protein"]), 6)
                self.assertGreaterEqual(len(data["carbs"]), 3)


# ----------------------------------------------------------------------
# 2. Variety across similar profiles
# ----------------------------------------------------------------------

class TestVarietyAcrossSimilarProfiles(unittest.TestCase):
    def test_small_profile_differences_produce_different_plans(self):
        variants = [
            dict(BASE),
            dict(BASE, weight_kg=73),
            dict(BASE, height_cm=175),
            dict(BASE, age=24),
            dict(BASE, sex="Female"),
            dict(BASE, training_days_per_week=4),
        ]
        signatures = [plan_signature(p) for p in variants]
        # Not demanding all-distinct (short lists can legitimately collide on
        # a slot), but near-identical profiles must not collapse to one plan.
        self.assertGreaterEqual(
            len(set(signatures)), 5,
            "similar profiles are still collapsing onto the same foods",
        )

    def test_the_same_profile_is_reproducible(self):
        """Variety is deterministic rotation, not randomness - refreshing
        must not reshuffle someone's plan."""
        self.assertEqual(plan_signature(BASE), plan_signature(BASE))
        self.assertEqual(mp.mock_provider(dict(BASE)), mp.mock_provider(dict(BASE)))

    def test_different_regions_produce_different_plans(self):
        north = plan_signature(dict(BASE, region="Lucknow, Uttar Pradesh"))
        south = plan_signature(dict(BASE, region="Chennai, Tamil Nadu"))
        northeast = plan_signature(dict(BASE, region="Guwahati, Assam"))
        self.assertNotEqual(north, south)
        self.assertNotEqual(north, northeast)
        self.assertNotEqual(south, northeast)

    def test_lunch_and_dinner_proteins_still_differ(self):
        plan = mp.mock_provider(dict(BASE))
        lunch_protein = plan["meals"][1]["ingredients"][1]["item"]
        dinner_protein = plan["meals"][3]["ingredients"][1]["item"]
        self.assertNotEqual(lunch_protein, dinner_protein)

    def test_lunch_and_dinner_carbs_differ_in_every_region(self):
        """A day reading "roti ... roti" looks like the generator gave up -
        dinner excludes lunch's staple, mirroring the protein slots."""
        for region in REGIONS:
            with self.subTest(region=region):
                # "international" (Phase 2B) has no keywords - it's reached
                # via blank/unrecognised region text, not a city name, like
                # every other region here.
                city = REGIONS[region]["keywords"][0] if REGIONS[region]["keywords"] else ""
                plan = mp.mock_provider(dict(BASE, region=city))
                self.assertNotEqual(
                    plan["meals"][1]["ingredients"][0]["item"],
                    plan["meals"][3]["ingredients"][0]["item"],
                )


# ----------------------------------------------------------------------
# 3. The variety primitives themselves
# ----------------------------------------------------------------------

class TestVarietyPrimitives(unittest.TestCase):
    def test_rotate_is_a_rotation_not_a_reordering(self):
        items = ["a", "b", "c", "d"]
        self.assertEqual(mp.rotate(items, 0), ["a", "b", "c", "d"])
        self.assertEqual(mp.rotate(items, 1), ["b", "c", "d", "a"])
        self.assertEqual(mp.rotate(items, 4), ["a", "b", "c", "d"])
        self.assertEqual(sorted(mp.rotate(items, 3)), sorted(items))

    def test_rotate_handles_empty_and_single_lists(self):
        self.assertEqual(mp.rotate([], 5), [])
        self.assertEqual(mp.rotate(["only"], 7), ["only"])

    def test_variety_seed_is_stable_and_sensitive(self):
        self.assertEqual(mp.variety_seed(dict(BASE)), mp.variety_seed(dict(BASE)))
        self.assertNotEqual(mp.variety_seed(dict(BASE)),
                            mp.variety_seed(dict(BASE, weight_kg=73)))

    def test_variety_seed_ignores_fields_that_already_filter(self):
        """Allergies/preferences change the plan through filtering. Folding
        them into the seed too would make an unrelated food swap look like it
        was caused by editing an allergy."""
        for field, value in [("allergies", "peanuts"),
                             ("foods_to_avoid", "rajma"),
                             ("food_preferences", "paneer")]:
            with self.subTest(field=field):
                self.assertEqual(mp.variety_seed(dict(BASE)),
                                 mp.variety_seed(dict(BASE, **{field: value})))

    def test_slot_seeds_are_decorrelated(self):
        """Plain `seed + slot` left every slot sharing the same `seed % len`
        base, so slots moved in lockstep. Different slots must not track
        each other across seeds."""
        lengths = 9
        pairs = set()
        for seed in range(50):
            a = mp.slot_seed(seed, mp.SLOT_BREAKFAST) % lengths
            b = mp.slot_seed(seed, mp.SLOT_SNACK) % lengths
            pairs.add((a - b) % lengths)
        self.assertGreater(len(pairs), 1, "slot offsets are moving in lockstep")

    def test_offset_zero_preserves_the_original_ordering(self):
        """Any caller that doesn't opt into variety behaves exactly as before."""
        candidates = ["First", "Second", "Third"]
        self.assertEqual(
            mp.choose_items(candidates, set(), [], 3, "medium"),
            candidates,
        )


# ----------------------------------------------------------------------
# 4. Variety must never override safety or preference rules
# ----------------------------------------------------------------------

class TestVarietyNeverOverridesExistingRules(unittest.TestCase):
    def test_allergies_are_respected_in_every_region(self):
        for region in REGIONS:
            with self.subTest(region=region):
                # "international" (Phase 2B) has no keywords - reached via
                # blank region text, not a city name, like every other
                # region tested here.
                city = REGIONS[region]["keywords"][0] if REGIONS[region]["keywords"] else ""
                plan = mp.mock_provider(dict(BASE, region=city, allergies="dairy"))
                names = [i.lower() for i in all_items(plan)]
                for blocked in ("milk", "curd", "paneer", "cheese", "ghee"):
                    self.assertFalse(any(blocked in n for n in names),
                                     f"{region}: {blocked} leaked past a dairy allergy")

    def test_vegetarian_diet_is_respected_in_every_region(self):
        for region in REGIONS:
            with self.subTest(region=region):
                city = REGIONS[region]["keywords"][0] if REGIONS[region]["keywords"] else ""
                plan = mp.mock_provider(dict(BASE, region=city, diet="Vegetarian"))
                names = [i.lower() for i in all_items(plan)]
                for meat in ("chicken", "fish", "pork", "mutton", "prawn", "egg"):
                    self.assertFalse(any(meat in n for n in names),
                                     f"{region}: {meat} leaked into a vegetarian plan")

    def test_foods_to_avoid_are_respected_despite_rotation(self):
        plan = mp.mock_provider(dict(BASE, foods_to_avoid="paneer, rajma"))
        names = [i.lower() for i in all_items(plan)]
        self.assertFalse(any("paneer" in n for n in names))
        self.assertFalse(any("rajma" in n for n in names))

    def test_stated_preferences_still_outrank_the_rotation(self):
        """Rotation happens before preferences are applied, so an explicit
        preference must still win."""
        plan = mp.mock_provider(dict(BASE, food_preferences="paneer"))
        self.assertTrue(any("paneer" in i.lower() for i in all_items(plan)))

    def test_low_budget_still_drops_expensive_foods(self):
        plan = mp.mock_provider(dict(BASE, budget_per_day="Rs 100"))
        names = [i.lower() for i in all_items(plan)]
        for pricey in ("paneer", "chicken", "fish"):
            self.assertFalse(any(pricey in n for n in names),
                             f"{pricey} survived a low budget")

    def test_minor_portion_rules_still_apply_with_variety(self):
        child = dict(BASE, age=10, weight_kg=32, height_cm=145, goal="muscle growth")
        adult = dict(child, age=25)
        child_plan = mp.mock_provider(child)
        adult_plan = mp.mock_provider(adult)
        self.assertIn("Growth-appropriate", child_plan["day_summary"]["notes"])
        self.assertNotIn("Growth-appropriate", adult_plan["day_summary"]["notes"])


# ----------------------------------------------------------------------
# 5. Region reaches the RAG retrieval query
# ----------------------------------------------------------------------
# build_retrieval_query() previously never mentioned region, so every
# regional knowledge-base document was unreachable through the real
# pipeline - a Chennai athlete and a Lucknow athlete produced identical
# queries. Same class of gap as the earlier age one.

class TestRegionReachesTheRetrievalQuery(unittest.TestCase):
    def _query(self, region):
        from rag import build_retrieval_query
        return build_retrieval_query(dict(BASE, region=region))

    def test_region_text_appears_in_the_query(self):
        self.assertIn("Chennai, Tamil Nadu", self._query("Chennai, Tamil Nadu"))

    def test_zone_label_appears_in_the_query(self):
        self.assertIn("South Indian", self._query("Chennai, Tamil Nadu"))
        self.assertIn("North Indian", self._query("Lucknow, Uttar Pradesh"))
        self.assertIn("Northeast Indian", self._query("Guwahati, Assam"))
        self.assertIn("Central Indian", self._query("Indore, Madhya Pradesh"))

    def test_different_regions_produce_different_queries(self):
        self.assertNotEqual(
            self._query("Chennai, Tamil Nadu"),
            self._query("Lucknow, Uttar Pradesh"),
        )

    def test_unrecognised_region_gets_no_zone_label(self):
        """detect_region() defaults unknown text to "north", which is the
        right default for PICKING FOOD but must not be asserted as fact in a
        retrieval query - an athlete in London is not "North Indian"."""
        query = self._query("London, UK")
        self.assertIn("London, UK", query)
        for label in ("North Indian", "South Indian", "East Indian",
                      "West Indian", "Central Indian", "Northeast Indian"):
            self.assertNotIn(label, query)

    def test_missing_region_leaves_the_query_unchanged(self):
        from rag import build_retrieval_query
        without_region = dict(BASE)
        without_region.pop("region")
        expected = (
            f"{BASE['goal']} - {BASE['diet']} footballer - "
            f"training {BASE['training_days_per_week']} days/week, "
            f"{BASE['training_intensity']} intensity"
        )
        self.assertEqual(build_retrieval_query(without_region), expected)
        self.assertEqual(build_retrieval_query(dict(BASE, region="")), expected)

    def test_age_phrase_still_works_alongside_region(self):
        """The region addition must not disturb the earlier age fix."""
        from rag import build_retrieval_query
        query = build_retrieval_query(dict(BASE, age=10, region="Chennai, Tamil Nadu"))
        self.assertIn("10 year old child athlete", query)
        self.assertIn("South Indian", query)

    def test_regional_document_is_now_reachable_through_the_real_query(self):
        """The point of the fix: a profile-driven query must be able to
        surface regional knowledge, which it previously never could."""
        from rag import build_retrieval_query
        from retriever import retrieve_relevant_chunks
        for region in ["Chennai, Tamil Nadu", "Lucknow, Uttar Pradesh", "Guwahati, Assam"]:
            with self.subTest(region=region):
                query = build_retrieval_query(dict(BASE, region=region))
                sources = [c["source"] for c in retrieve_relevant_chunks(query, top_k=3)]
                self.assertIn("indian_regional_eating_patterns.md", sources)

    def test_mock_provider_food_selection_is_unaffected_by_this_change(self):
        """This change touches retrieval only. The deterministic structured
        food selection reads the profile's region directly via
        regional_foods.detect_region(), never through rag.py."""
        for region in ["Chennai, Tamil Nadu", "Guwahati, Assam"]:
            with self.subTest(region=region):
                plan = mp.mock_provider(dict(BASE, region=region))
                data = REGIONS[detect_region({"region": region})]
                own = set(data["breakfast"] + data["carbs"] + data["veg_protein"])
                self.assertTrue(any(i in own for i in all_items(plan)))


# ----------------------------------------------------------------------
# 6. Region-SPECIFIC sections are actually retrievable
# ----------------------------------------------------------------------
# Getting the region into the query was only half the job: the per-region
# sections still never surfaced. Two causes, both fixed here - chunking
# split every section into a name-only chunk plus a body chunk that never
# named the region, and a profile query is so dominated by training
# vocabulary that a short region tail is averaged away in the embedding.

class TestSectionAwareChunking(unittest.TestCase):
    def test_heading_stays_with_the_content_beneath_it(self):
        from ingest import split_into_chunks
        text = (
            "## South India\n*Tamil Nadu, Kerala*\n\n"
            "- **Staples:** rice dominates, idli and dosa are everyday breakfasts.\n\n"
            "## North India\n*Punjab, Delhi*\n\n"
            "- **Staples:** whole-wheat roti at main meals.\n"
        )
        chunks = split_into_chunks(text)
        south = [c for c in chunks if "South India" in c]
        self.assertEqual(len(south), 1, "the South heading should appear in exactly one chunk")
        self.assertIn("rice dominates", south[0], "heading was separated from its own content")
        self.assertNotIn("North India", south[0], "one chunk straddled two sections")

    def test_a_heading_is_never_left_as_its_own_chunk(self):
        from ingest import split_into_chunks
        long_body = "- " + ("food item, " * 120)
        chunks = split_into_chunks("## Section\n\n" + long_body)
        self.assertIn("Section", chunks[0])
        self.assertGreater(len(chunks[0]), 60, "heading was orphaned into a tiny chunk")

    def test_every_region_section_is_a_self_contained_chunk(self):
        from ingest import create_chunk_records, load_documents
        records = [r for r in create_chunk_records(load_documents())
                   if r["source"] == "indian_regional_eating_patterns.md"]
        for zone in ["North India", "South India", "East India",
                     "West India", "Central India", "Northeast India"]:
            with self.subTest(zone=zone):
                owning = [r for r in records if f"## {zone}" in r["text"]]
                self.assertEqual(len(owning), 1)
                # The chunk must carry real content, not just the heading.
                self.assertIn("Staples", owning[0]["text"])


class TestRegionSpecificRetrieval(unittest.TestCase):
    ZONES = [
        ("Chennai, Tamil Nadu", "## South India"),
        ("Lucknow, Uttar Pradesh", "## North India"),
        ("Guwahati, Assam", "## Northeast India"),
        ("Indore, Madhya Pradesh", "## Central India"),
        ("Pune, Maharashtra", "## West India"),
        ("Kolkata, West Bengal", "## East India"),
    ]

    def test_each_region_retrieves_its_own_section(self):
        from rag import build_retrieval_query, retrieve_profile_chunks
        for region, marker in self.ZONES:
            with self.subTest(region=region):
                profile = dict(BASE, region=region)
                chunks = retrieve_profile_chunks(
                    profile, build_retrieval_query(profile), top_k=3
                )
                self.assertTrue(
                    any(marker in c["text"] for c in chunks),
                    f"{region}: {marker} section not retrieved",
                )

    def test_a_region_does_not_retrieve_another_regions_section(self):
        from rag import build_retrieval_query, retrieve_profile_chunks
        profile = dict(BASE, region="Guwahati, Assam")
        text = " ".join(
            c["text"] for c in retrieve_profile_chunks(
                profile, build_retrieval_query(profile), top_k=3)
        )
        self.assertIn("## Northeast India", text)
        for other in ["## South India", "## West India", "## Central India"]:
            self.assertNotIn(other, text)

    def test_profile_without_a_region_is_unchanged(self):
        """No region means no top-up - identical to the plain retrieval."""
        from rag import build_retrieval_query, retrieve_profile_chunks
        from retriever import retrieve_relevant_chunks
        profile = dict(BASE)
        profile.pop("region")
        query = build_retrieval_query(profile)
        self.assertEqual(
            [c["id"] for c in retrieve_profile_chunks(profile, query, top_k=3)],
            [c["id"] for c in retrieve_relevant_chunks(query, top_k=3)],
        )

    def test_region_top_up_does_not_displace_youth_content(self):
        """The region top-up ADDS chunks; it must never push youth guidance
        out for a minor."""
        from rag import build_retrieval_query, retrieve_profile_chunks
        child = dict(BASE, age=10, weight_kg=32, height_cm=145,
                     goal="muscle growth", training_days_per_week=4,
                     training_intensity="Moderate", region="Chennai, Tamil Nadu")
        chunks = retrieve_profile_chunks(child, build_retrieval_query(child), top_k=3)
        sources = [c["source"] for c in chunks]
        self.assertIn("youth_athlete_nutrition.md", sources)
        self.assertIn("indian_regional_eating_patterns.md", sources)


# ----------------------------------------------------------------------
# 7. Youth guidance reaches a minor regardless of their stated goal
# ----------------------------------------------------------------------
# Same root cause and same remedy as the region gap: the developmental-stage
# phrase is a short tail in a training-dominated query, so it was averaged
# away in the single query embedding. A 10-year-old whose goal read
# "Improve football performance" previously retrieved NO youth guidance at
# all (confirmed against the older chunking too, so this predated the
# section-aware chunking work). The phrase searched on its own retrieves the
# youth document at rank 1 for every age from 8 to 17.

MINOR = dict(
    age=10, height_cm=145, weight_kg=32, sex="Male", diet="Vegetarian",
    training_days_per_week=4, training_intensity="Moderate",
    region="Chennai, Tamil Nadu", allergies="none",
    budget_per_day=None, food_preferences=None, foods_to_avoid=None, notes=None,
)

FOOTBALL_GOALS = [
    "muscle growth",
    "Improve football performance",
    "build stamina and endurance",
    "match preparation",
    "get stronger for football",
]


def _chunks_for(profile):
    from rag import build_retrieval_query, retrieve_profile_chunks
    return retrieve_profile_chunks(profile, build_retrieval_query(profile), top_k=3)


class TestYouthGuidanceReachesMinors(unittest.TestCase):
    def test_a_minor_gets_youth_guidance_for_any_normal_goal(self):
        for goal in FOOTBALL_GOALS:
            with self.subTest(goal=goal):
                sources = [c["source"] for c in _chunks_for(dict(MINOR, goal=goal))]
                self.assertIn("youth_athlete_nutrition.md", sources)

    def test_adolescents_also_get_youth_guidance(self):
        for age in [13, 15, 17]:
            with self.subTest(age=age):
                profile = dict(MINOR, age=age, goal="Improve football performance")
                sources = [c["source"] for c in _chunks_for(profile)]
                self.assertIn("youth_athlete_nutrition.md", sources)

    def test_an_adult_never_receives_the_youth_top_up(self):
        """An adult may still match the youth document organically for a
        query about growth or recovery - that is ordinary retrieval. What
        must not happen is the top-up FORCING it in, so an adult's results
        must contain nothing beyond the main search plus the region top-up."""
        from rag import (REGION_TOP_UP, build_retrieval_query,
                         retrieve_profile_chunks)
        from retriever import retrieve_relevant_chunks
        for goal in FOOTBALL_GOALS:
            with self.subTest(goal=goal):
                adult = dict(MINOR, age=25, weight_kg=72, height_cm=178, goal=goal)
                query = build_retrieval_query(adult)
                chunks = retrieve_profile_chunks(adult, query, top_k=3)
                baseline = retrieve_relevant_chunks(query, top_k=3)
                # 3 from the main search + at most the region top-up, and
                # never the extra youth slot a minor would receive.
                self.assertLessEqual(len(chunks), len(baseline) + REGION_TOP_UP)

    def test_exactly_18_is_treated_as_an_adult_for_retrieval(self):
        from rag import _developmental_stage_phrase
        self.assertEqual(_developmental_stage_phrase(18), "")
        self.assertNotEqual(_developmental_stage_phrase(17), "")

    def test_youth_and_regional_context_coexist_in_every_region(self):
        for region, marker in TestRegionSpecificRetrieval.ZONES:
            with self.subTest(region=region):
                profile = dict(MINOR, region=region, goal="Improve football performance")
                chunks = _chunks_for(profile)
                sources = [c["source"] for c in chunks]
                self.assertIn("youth_athlete_nutrition.md", sources)
                self.assertTrue(
                    any(marker in c["text"] for c in chunks),
                    f"{region}: regional section lost when youth top-up was added",
                )

    def test_minor_food_and_portion_logic_is_untouched(self):
        """This is a retrieval-only change - the deterministic minor
        portion/bulking rules must behave exactly as before."""
        child = mp.mock_provider(dict(MINOR, goal="muscle growth"))
        adult = mp.mock_provider(dict(MINOR, age=25, weight_kg=72, height_cm=178,
                                      goal="muscle growth"))
        self.assertIn("Growth-appropriate", child["day_summary"]["notes"])
        self.assertNotIn("Growth-appropriate", adult["day_summary"]["notes"])


# ----------------------------------------------------------------------
# 8. Region-aware dairy (audit fix)
# ----------------------------------------------------------------------
# build_lunch()/build_dinner() used to hard-code a "Curd" side for every
# vegetarian in every region, so a northeast athlete got curd at every
# single lunch - contradicting the knowledge base shipped in the same
# request, which says dairy should not be assumed as a default there.

class TestRegionAwareDairy(unittest.TestCase):
    NE = "Guwahati, Assam"

    def test_northeast_lunch_does_not_force_a_dairy_side(self):
        for weight in range(70, 82):
            with self.subTest(weight=weight):
                plan = mp.mock_provider(dict(BASE, region=self.NE, weight_kg=weight))
                lunch = [i["item"].lower() for i in plan["meals"][1]["ingredients"]]
                for dairy in ("curd", "paneer", "milk", "buttermilk"):
                    self.assertFalse(
                        any(dairy in item for item in lunch),
                        f"northeast lunch force-added {dairy}: {lunch}",
                    )

    def test_northeast_declares_no_customary_dairy_side(self):
        self.assertIsNone(REGIONS["northeast"]["dairy_side"])

    def test_regions_where_dairy_is_customary_still_get_it(self):
        expected = {
            "Lucknow, Uttar Pradesh": "Curd",
            "Chennai, Tamil Nadu": "Curd",
            "Pune, Maharashtra": "Buttermilk (chaas)",
            "Kolkata, West Bengal": "Curd",
            "Indore, Madhya Pradesh": "Curd",
        }
        for region, side in expected.items():
            with self.subTest(region=region):
                lunch = [i["item"] for i in
                         mp.mock_provider(dict(BASE, region=region))["meals"][1]["ingredients"]]
                self.assertIn(side, lunch)

    def test_a_dairy_allergy_still_removes_the_customary_side(self):
        lunch = [i["item"].lower() for i in mp.mock_provider(
            dict(BASE, region="Lucknow, Uttar Pradesh", allergies="dairy")
        )["meals"][1]["ingredients"]]
        for dairy in ("curd", "paneer", "milk"):
            self.assertFalse(any(dairy in item for item in lunch))

    def test_non_vegetarians_never_got_the_dairy_side_and_still_do_not(self):
        lunch = [i["item"].lower() for i in mp.mock_provider(
            dict(BASE, diet="Non-Vegetarian", region="Lucknow, Uttar Pradesh")
        )["meals"][1]["ingredients"]]
        self.assertFalse(any("curd" in item for item in lunch))

    def test_northeast_keeps_its_genuine_dairy_containing_dishes(self):
        """Not a blanket dairy ban - jolpan with curd is authentically
        Assamese and curd remains a valid (non-default) breakfast side."""
        self.assertIn("Curd", REGIONS["northeast"]["breakfast_sides"])


# ----------------------------------------------------------------------
# 9. One canonical profile-aware retrieval path (audit fix)
# ----------------------------------------------------------------------

class TestUnifiedRetrievalPath(unittest.TestCase):
    CHILD = dict(BASE, age=10, weight_kg=32, height_cm=145, region="Chennai, Tamil Nadu")

    def test_standalone_and_orchestrator_contexts_match(self):
        """get_context_block() (Claude standalone, CLI) must produce the same
        context as the orchestrator path for the same athlete."""
        from rag import (build_retrieval_query, format_context_block,
                         get_context_block, retrieve_profile_chunks)
        profile = dict(self.CHILD)
        standalone = get_context_block(profile)
        orchestrated = format_context_block(
            retrieve_profile_chunks(profile, build_retrieval_query(profile), top_k=3))
        self.assertEqual(standalone, orchestrated)

    def test_standalone_path_now_includes_youth_guidance_for_a_minor(self):
        from rag import get_context_block
        self.assertIn("youth_athlete_nutrition.md", get_context_block(dict(self.CHILD)))

    def test_standalone_path_now_includes_the_region_section(self):
        from rag import get_context_block
        self.assertIn("## South India", get_context_block(dict(self.CHILD)))

    def test_adult_gets_no_forced_youth_top_up_on_the_standalone_path(self):
        from rag import REGION_TOP_UP, get_context_block
        from retriever import retrieve_relevant_chunks
        from rag import build_retrieval_query
        adult = dict(BASE, age=25, region="Chennai, Tamil Nadu")
        context = get_context_block(adult)
        baseline = retrieve_relevant_chunks(build_retrieval_query(adult), top_k=3)
        # 3 main + at most the region top-up, never the extra youth slot.
        self.assertLessEqual(context.count("[Source:"), len(baseline) + REGION_TOP_UP)

    def test_unknown_region_does_not_invent_a_region_on_either_path(self):
        from rag import build_retrieval_query, get_context_block
        profile = dict(BASE, region="London, UK")
        query = build_retrieval_query(profile)
        for label in ("North Indian", "South Indian", "Northeast Indian"):
            self.assertNotIn(label, query)
        get_context_block(profile)  # must not raise

    def test_retrieval_logic_is_not_duplicated(self):
        """get_context_block must delegate, not reimplement - so patching the
        canonical helper changes its behaviour."""
        from unittest import mock
        import rag
        with mock.patch.object(rag, "retrieve_profile_chunks", return_value=[]) as canonical:
            rag.get_context_block(dict(self.CHILD))
        canonical.assert_called_once()


# ----------------------------------------------------------------------
# Northeast variety additions (Step 4C follow-up - 3 high-confidence dishes)
# ----------------------------------------------------------------------
# Approved additions only: "Fiddlehead fern curry (dhekia xaak)" (veg),
# "Poita bhaat (fermented rice) with vegetables" (breakfast), and
# "Black rice porridge (chak-hao)" (breakfast). Lower-confidence candidates
# from the audit were deliberately NOT added in this pass.

NORTHEAST_ADDITIONS = {
    "breakfast": ["Poita bhaat (fermented rice) with vegetables", "Black rice porridge (chak-hao)"],
    "veg": ["Fiddlehead fern curry (dhekia xaak)"],
}

NORTHEAST_BASE = dict(
    age=22, height_cm=178, weight_kg=72, sex="Male",
    goal="Improve football performance",
    diet="Vegetarian", training_days_per_week=5, training_intensity="High",
    region="Guwahati, Assam", allergies="none",
    budget_per_day=None, food_preferences=None, foods_to_avoid=None, notes=None,
)


class TestNortheastVarietyAdditions(unittest.TestCase):
    def test_the_three_approved_dishes_are_present_in_the_registry(self):
        northeast = REGIONS["northeast"]
        for category, dishes in NORTHEAST_ADDITIONS.items():
            for dish in dishes:
                self.assertIn(dish, northeast[category], msg=f"{dish} missing from northeast[{category}]")

    def test_no_lower_confidence_candidates_were_added(self):
        northeast = REGIONS["northeast"]
        not_added = [
            "Masoor dal", "Black gram fritters (bora)", "Matikaloi dal (black gram lentil)",
            "Banana flower curry (kol phul)", "Alu pitika (mashed potato, mustard oil, chili)",
            "Boiled soybean", "Til (sesame) chutney", "Muri with jaggery", "Roasted soybean",
        ]
        all_items = {item for values in northeast.values() if isinstance(values, list) for item in values}
        for dish in not_added:
            self.assertNotIn(dish, all_items, msg=f"{dish} should not have been added in this pass")

    def test_new_dishes_appear_through_the_normal_regional_selection_path(self):
        """Generate a real 7-day plan for a Northeast athlete (no selection-
        logic changes here - just exercising the existing deterministic
        rotation) and confirm at least one of the 3 new dishes surfaces
        somewhere across the week, proving they are reachable, not just
        present in the data file."""
        plan = mp.mock_provider_multi_day(dict(NORTHEAST_BASE), days=7)
        seen_items = {
            ing["item"]
            for day in plan["days"]
            for meal in day["meals"]
            for ing in meal["ingredients"]
        }
        new_dishes = {d for dishes in NORTHEAST_ADDITIONS.values() for d in dishes}
        self.assertTrue(
            seen_items & new_dishes,
            f"expected at least one of {new_dishes} to appear across a 7-day Northeast plan, got {seen_items}",
        )

    def test_detect_region_still_maps_assam_to_northeast(self):
        self.assertEqual(detect_region(NORTHEAST_BASE), "northeast")


if __name__ == "__main__":
    unittest.main(verbosity=2)
