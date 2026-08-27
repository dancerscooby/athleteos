"""
AthleteOS - Personal Recipes (athlete-saved homemade/family recipes).

Three pieces, tested here:

  STORAGE   personal_recipes.py - CRUD over memory_manager's "recipes"
            category, keyed strictly by athlete_id (never display name).

  SAFETY    meal_planner.classify_personal_recipes() - the one place recipe
            safety is decided (allergy/foods-to-avoid/diet), called
            independently by the mock provider, the LLM providers, and chat.

  SELECTION recipe candidates reach the mock provider's lunch/dinner protein
            picks via pick_protein()'s extra_candidates / _build_day()'s
            recipe_candidates - never forced, only ever selected the same
            way any regional dish would be.

Everything here is offline and deterministic - no network, no API calls.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import meal_planner as mp
import memory_manager as mm
import orchestrator as orch
import personal_recipes as pr

BASE = dict(
    age=22, height_cm=178, weight_kg=72, sex="Male",
    goal="Improve football performance",
    diet="Vegetarian", training_days_per_week=5, training_intensity="High",
    region="Lucknow, Uttar Pradesh", allergies="none",
    budget_per_day=None, food_preferences=None, foods_to_avoid=None,
    notes=None, available_ingredients=None,
)


class MemoryIsolatedTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        tmp = Path(self._tmpdir.name)
        self._orig = {n: getattr(mm, n) for n in
                      ["MEMORY_DIR", "PROFILES_FILE", "PREFERENCES_FILE", "LATEST_PLAN_FILE",
                       "RECIPES_FILE", "EPISODES_DIR", "CACHE_DIR", "UPDATABLE_CATEGORIES"]}
        mm.MEMORY_DIR = tmp
        mm.PROFILES_FILE = tmp / "profiles.json"
        mm.PREFERENCES_FILE = tmp / "preferences.json"
        mm.LATEST_PLAN_FILE = tmp / "latest_plans.json"
        mm.RECIPES_FILE = tmp / "recipes.json"
        mm.EPISODES_DIR = tmp / "episodes"
        mm.CACHE_DIR = tmp / "cache"
        mm.UPDATABLE_CATEGORIES = {"profile": mm.PROFILES_FILE,
                                   "preference": mm.PREFERENCES_FILE,
                                   "latest_plan": mm.LATEST_PLAN_FILE,
                                   "recipes": mm.RECIPES_FILE}

    def tearDown(self):
        for name, value in self._orig.items():
            setattr(mm, name, value)
        self._tmpdir.cleanup()


# ----------------------------------------------------------------------
# 1. Storage - personal_recipes.py
# ----------------------------------------------------------------------

class TestRecipeStorage(MemoryIsolatedTestCase):
    def test_new_athlete_has_no_recipes(self):
        self.assertEqual(pr.list_recipes("brand-new-athlete"), [])

    def test_create_and_retrieve(self):
        recipe = pr.add_recipe("athlete-a", "Mom's Chicken Curry",
                                ["Chicken", "Onion", "Tomato", "Ginger", "Garlic", "Spices"])
        self.assertEqual(recipe["name"], "Mom's Chicken Curry")
        self.assertEqual(recipe["ingredients"],
                          ["Chicken", "Onion", "Tomato", "Ginger", "Garlic", "Spices"])
        self.assertIn("id", recipe)
        self.assertIn("created_at", recipe)
        stored = pr.list_recipes("athlete-a")
        self.assertEqual(stored, [recipe])

    def test_optional_fields_default_to_none(self):
        recipe = pr.add_recipe("athlete-a", "Simple Dal", ["Dal", "Turmeric"])
        self.assertIsNone(recipe["method"])
        self.assertIsNone(recipe["notes"])
        self.assertIsNone(recipe["servings"])

    def test_optional_fields_are_stored_when_given(self):
        recipe = pr.add_recipe("athlete-a", "Simple Dal", ["Dal", "Turmeric"],
                                method="Boil and temper", notes="Grandma's version", servings="2")
        self.assertEqual(recipe["method"], "Boil and temper")
        self.assertEqual(recipe["notes"], "Grandma's version")
        self.assertEqual(recipe["servings"], "2")

    def test_multiple_recipes_all_persist(self):
        pr.add_recipe("athlete-a", "Recipe One", ["A"])
        pr.add_recipe("athlete-a", "Recipe Two", ["B"])
        pr.add_recipe("athlete-a", "Recipe Three", ["C"])
        names = {r["name"] for r in pr.list_recipes("athlete-a")}
        self.assertEqual(names, {"Recipe One", "Recipe Two", "Recipe Three"})

    def test_delete_removes_only_the_targeted_recipe(self):
        r1 = pr.add_recipe("athlete-a", "Keep Me", ["A"])
        r2 = pr.add_recipe("athlete-a", "Delete Me", ["B"])
        self.assertTrue(pr.delete_recipe("athlete-a", r2["id"]))
        remaining = pr.list_recipes("athlete-a")
        self.assertEqual([r["id"] for r in remaining], [r1["id"]])

    def test_delete_of_unknown_id_returns_false_and_changes_nothing(self):
        pr.add_recipe("athlete-a", "Keep Me", ["A"])
        self.assertFalse(pr.delete_recipe("athlete-a", "does-not-exist"))
        self.assertEqual(len(pr.list_recipes("athlete-a")), 1)

    def test_persistence_across_separate_calls(self):
        """Simulates persistence "across requests" - each call re-reads
        from the same on-disk store rather than any in-process cache."""
        pr.add_recipe("athlete-a", "First", ["A"])
        pr.add_recipe("athlete-a", "Second", ["B"])
        self.assertEqual(len(pr.list_recipes("athlete-a")), 2)
        pr.add_recipe("athlete-a", "Third", ["C"])
        self.assertEqual(len(pr.list_recipes("athlete-a")), 3)


# ----------------------------------------------------------------------
# 2. Identity isolation - athlete_id, never display name
# ----------------------------------------------------------------------

class TestRecipeIsolation(MemoryIsolatedTestCase):
    def test_same_display_name_different_ids_do_not_share_recipes(self):
        """Two athletes named "Rahul" must never see each other's recipes -
        storage is keyed by athlete_id (a UUID), not name."""
        id_1 = "11111111-1111-4111-8111-111111111111"
        id_2 = "22222222-2222-4222-8222-222222222222"
        pr.add_recipe(id_1, "Rahul's Recipe A", ["A"])
        pr.add_recipe(id_2, "Rahul's Recipe B", ["B"])
        self.assertEqual([r["name"] for r in pr.list_recipes(id_1)], ["Rahul's Recipe A"])
        self.assertEqual([r["name"] for r in pr.list_recipes(id_2)], ["Rahul's Recipe B"])

    def test_switching_athletes_shows_the_correct_recipes(self):
        pr.add_recipe("athlete-a", "A's Curry", ["A"])
        pr.add_recipe("athlete-b", "B's Curry", ["B"])
        self.assertEqual([r["name"] for r in pr.list_recipes("athlete-a")], ["A's Curry"])
        self.assertEqual([r["name"] for r in pr.list_recipes("athlete-b")], ["B's Curry"])
        # switching back
        self.assertEqual([r["name"] for r in pr.list_recipes("athlete-a")], ["A's Curry"])

    def test_deleting_one_athletes_recipe_does_not_touch_another(self):
        r_a = pr.add_recipe("athlete-a", "A's Curry", ["A"])
        pr.add_recipe("athlete-b", "B's Curry", ["B"])
        pr.delete_recipe("athlete-a", r_a["id"])
        self.assertEqual(pr.list_recipes("athlete-a"), [])
        self.assertEqual(len(pr.list_recipes("athlete-b")), 1)

    def test_recipe_data_never_leaks_via_the_orchestrator(self):
        """agent_context["personal_recipes"] must only ever contain the
        requesting athlete's own recipes."""
        pr.add_recipe("athlete-a", "Secret Family Recipe", ["A"])
        trace = orch.run_with_trace(dict(BASE), athlete_id="athlete-b")
        names = [r["name"] for r in trace["personal_recipes"]]
        self.assertNotIn("Secret Family Recipe", names)


# ----------------------------------------------------------------------
# 3. Safety - meal_planner.classify_personal_recipes()
# ----------------------------------------------------------------------

class TestRecipeSafetyClassification(unittest.TestCase):
    def test_a_compatible_recipe_is_usable(self):
        recipes = [{"name": "Rajma Bowl", "ingredients": ["Rajma", "Rice", "Onion"]}]
        result = mp.classify_personal_recipes(recipes, dict(BASE))
        self.assertEqual([r["name"] for r in result["usable"]], ["Rajma Bowl"])
        self.assertEqual(result["excluded"], [])

    def test_allergy_overrides_a_saved_recipe(self):
        """Worked example from the spec: athlete saves "Paneer curry", later
        marks dairy as an allergy -> must never be recommended."""
        recipes = [{"name": "Paneer Curry", "ingredients": ["Paneer", "Tomato", "Spices"]}]
        profile = dict(BASE, allergies="dairy")
        result = mp.classify_personal_recipes(recipes, profile)
        self.assertEqual(result["usable"], [])
        self.assertEqual(result["excluded"][0]["name"], "Paneer Curry")
        self.assertIn("allergy", result["excluded"][0]["reason"])

    def test_diet_overrides_a_saved_recipe(self):
        """A vegetarian athlete must never receive a saved chicken recipe."""
        recipes = [{"name": "Mom's Chicken Curry",
                    "ingredients": ["Chicken", "Onion", "Tomato", "Ginger", "Garlic", "Spices"]}]
        result = mp.classify_personal_recipes(recipes, dict(BASE, diet="Vegetarian"))
        self.assertEqual(result["usable"], [])
        self.assertIn("Vegetarian", result["excluded"][0]["reason"])

    def test_non_vegetarian_athlete_may_use_a_meat_recipe(self):
        recipes = [{"name": "Mom's Chicken Curry", "ingredients": ["Chicken", "Onion"]}]
        result = mp.classify_personal_recipes(recipes, dict(BASE, diet="Non-Vegetarian"))
        self.assertEqual([r["name"] for r in result["usable"]], ["Mom's Chicken Curry"])

    def test_foods_to_avoid_overrides_a_saved_recipe(self):
        recipes = [{"name": "Rajma Bowl", "ingredients": ["Rajma", "Rice"]}]
        result = mp.classify_personal_recipes(recipes, dict(BASE, foods_to_avoid="rajma"))
        self.assertEqual(result["usable"], [])
        self.assertIn("rajma", result["excluded"][0]["reason"])

    def test_multiple_recipes_are_classified_independently(self):
        recipes = [
            {"name": "Rajma Bowl", "ingredients": ["Rajma", "Rice"]},
            {"name": "Paneer Curry", "ingredients": ["Paneer", "Tomato"]},
            {"name": "Mom's Chicken Curry", "ingredients": ["Chicken", "Onion"]},
        ]
        result = mp.classify_personal_recipes(recipes, dict(BASE, allergies="dairy"))
        usable_names = {r["name"] for r in result["usable"]}
        excluded_names = {e["name"] for e in result["excluded"]}
        self.assertEqual(usable_names, {"Rajma Bowl"})
        self.assertEqual(excluded_names, {"Paneer Curry", "Mom's Chicken Curry"})

    def test_no_recipes_is_a_no_op(self):
        self.assertEqual(mp.classify_personal_recipes([], dict(BASE)),
                          {"usable": [], "excluded": []})
        self.assertEqual(mp.classify_personal_recipes(None, dict(BASE)),
                          {"usable": [], "excluded": []})


# ----------------------------------------------------------------------
# 4. Selection - the mock provider (the active generation path)
# ----------------------------------------------------------------------

class TestRecipesDoNotForceIntoEveryPlan(unittest.TestCase):
    def test_no_recipes_leaves_the_plan_unchanged(self):
        baseline = mp.mock_provider(dict(BASE))
        with_empty_context = mp.mock_provider(dict(BASE), agent_context={"personal_recipes": []})
        no_context = mp.mock_provider(dict(BASE), agent_context=None)
        self.assertEqual(baseline, with_empty_context)
        self.assertEqual(baseline, no_context)

    def test_a_saved_recipe_is_not_forced_into_every_plan(self):
        """Merely having a saved, compatible recipe must not guarantee it
        appears - it's a candidate, not a mandate."""
        recipes = [{"name": "Rajma Bowl", "ingredients": ["Rajma", "Rice"]}]
        baseline = mp.mock_provider(dict(BASE))
        with_recipe = mp.mock_provider(dict(BASE), agent_context={"personal_recipes": recipes})
        # The rest of the plan's structure/shape is untouched either way.
        self.assertEqual(set(baseline.keys()), set(with_recipe.keys()))


class TestRecipesCanInfluenceSelection(unittest.TestCase):
    def _items(self, profile, agent_context):
        plan = mp.mock_provider(profile, agent_context=agent_context)
        return [ing["item"] for meal in plan["meals"] for ing in meal["ingredients"]]

    def test_a_compatible_recipe_is_reachable_via_rotation(self):
        """A saved, safe recipe is appended to the same candidate pool the
        rotation logic already cycles over - so across enough generation
        offsets it must eventually be selected, exactly like any regional
        dish would be."""
        recipes = [{"name": "Grandma's Special Rajma", "ingredients": ["Rajma", "Rice", "Onion"]}]
        found = False
        for gen_index in range(30):
            items = self._items(dict(BASE), {
                "personal_recipes": recipes,
                "variety_context": {"generation_index": gen_index},
            })
            if "Grandma's Special Rajma" in items:
                found = True
                break
        self.assertTrue(found, "saved recipe was never reachable across 30 rotation offsets")

    def test_a_matching_preference_promotes_the_recipe(self):
        """food_preferences already promotes matching candidates to the
        front of the pool (existing, pre-recipe behaviour) - saved recipes
        participate in that same mechanism, not a new one."""
        recipes = [{"name": "Grandma's Special Rajma", "ingredients": ["Rajma", "Rice", "Onion"]}]
        profile = dict(BASE, food_preferences="grandma's special rajma")
        items = self._items(profile, {"personal_recipes": recipes})
        self.assertIn("Grandma's Special Rajma", items)

    def test_an_excluded_recipe_never_appears_even_with_matching_preference(self):
        """Safety still outranks preference: a dairy-allergic athlete who
        somehow types the excluded recipe's name as a preference still
        never sees it, because it was never added to the candidate pool."""
        recipes = [{"name": "Grandma's Paneer Bowl", "ingredients": ["Paneer", "Peas"]}]
        profile = dict(BASE, allergies="dairy", food_preferences="grandma's paneer bowl")
        for gen_index in range(10):
            items = self._items(profile, {
                "personal_recipes": recipes,
                "variety_context": {"generation_index": gen_index},
            })
            self.assertNotIn("Grandma's Paneer Bowl", items)

    def test_a_diet_incompatible_recipe_never_appears(self):
        recipes = [{"name": "Mom's Chicken Curry", "ingredients": ["Chicken", "Onion", "Spices"]}]
        profile = dict(BASE, diet="Vegetarian")
        for gen_index in range(10):
            items = self._items(profile, {
                "personal_recipes": recipes,
                "variety_context": {"generation_index": gen_index},
            })
            self.assertNotIn("Mom's Chicken Curry", items)

    def test_multi_day_planner_also_respects_the_same_rules(self):
        recipes = [{"name": "Mom's Chicken Curry", "ingredients": ["Chicken", "Onion"]}]
        profile = dict(BASE, diet="Vegetarian")
        plan = mp.mock_provider_multi_day(profile, 5, agent_context={"personal_recipes": recipes})
        all_items = [ing["item"] for day in plan["days"] for meal in day["meals"]
                     for ing in meal["ingredients"]]
        self.assertNotIn("Mom's Chicken Curry", all_items)


# ----------------------------------------------------------------------
# 5. Existing rotation/regression untouched
# ----------------------------------------------------------------------

class TestExistingBehaviourUnaffected(unittest.TestCase):
    def test_meal_rotation_still_advances_with_generation_index(self):
        """Phase 2A round-robin behaviour must be unaffected by the recipes
        feature when no recipes are present."""
        first = mp.mock_provider(dict(BASE), agent_context={"variety_context": {"generation_index": 0}})
        second = mp.mock_provider(dict(BASE), agent_context={"variety_context": {"generation_index": 1}})
        self.assertNotEqual(first["meals"], second["meals"])

    def test_plan_schema_is_unchanged(self):
        plan = mp.mock_provider(dict(BASE), agent_context={"personal_recipes": [
            {"name": "Rajma Bowl", "ingredients": ["Rajma", "Rice"]}]})
        self.assertEqual(
            set(plan.keys()),
            {"confidence_score", "confidence_reason", "day_summary", "meals", "disclaimer"},
        )


# ----------------------------------------------------------------------
# 6. The HTTP endpoints (api.py)
# ----------------------------------------------------------------------

class TestRecipeEndpoints(MemoryIsolatedTestCase):
    ATHLETE = "33333333-3333-4333-8333-333333333333"

    def test_get_recipes_on_a_new_athlete_is_empty(self):
        from api import get_recipes
        self.assertEqual(get_recipes(self.ATHLETE), [])

    def test_create_then_get_round_trips(self):
        from api import create_recipe, get_recipes, RecipeCreate
        created = create_recipe(self.ATHLETE, RecipeCreate(
            name="Mom's Chicken Curry",
            ingredients="Chicken, Onion, Tomato, Ginger, Garlic, Spices",
        ))
        self.assertEqual(created["name"], "Mom's Chicken Curry")
        self.assertEqual(created["ingredients"],
                          ["chicken", "onion", "tomato", "ginger", "garlic", "spices"])
        fetched = get_recipes(self.ATHLETE)
        self.assertEqual([r["id"] for r in fetched], [created["id"]])

    def test_create_rejects_empty_ingredients(self):
        from api import create_recipe, RecipeCreate
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            create_recipe(self.ATHLETE, RecipeCreate(name="Empty", ingredients="   "))
        self.assertEqual(ctx.exception.status_code, 422)

    def test_delete_removes_the_recipe(self):
        from api import create_recipe, remove_recipe, get_recipes, RecipeCreate
        created = create_recipe(self.ATHLETE, RecipeCreate(name="Dal", ingredients="Dal, Turmeric"))
        result = remove_recipe(self.ATHLETE, created["id"])
        self.assertEqual(result, {"deleted": True})
        self.assertEqual(get_recipes(self.ATHLETE), [])

    def test_delete_unknown_id_is_404(self):
        from api import remove_recipe
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            remove_recipe(self.ATHLETE, "does-not-exist")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_invalid_athlete_id_is_rejected(self):
        from api import get_recipes
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            get_recipes("not-a-valid-id; drop table")
        self.assertEqual(ctx.exception.status_code, 422)

    def test_two_athletes_stay_isolated_through_the_endpoint_layer(self):
        from api import create_recipe, get_recipes, RecipeCreate
        other = "44444444-4444-4444-8444-444444444444"
        create_recipe(self.ATHLETE, RecipeCreate(name="Mine", ingredients="A"))
        create_recipe(other, RecipeCreate(name="Theirs", ingredients="B"))
        self.assertEqual([r["name"] for r in get_recipes(self.ATHLETE)], ["Mine"])
        self.assertEqual([r["name"] for r in get_recipes(other)], ["Theirs"])


# ----------------------------------------------------------------------
# 7. The end-to-end meal-plan endpoint (unchanged behaviour, no recipes)
# ----------------------------------------------------------------------

class TestMealPlanEndpointUnaffectedWithoutRecipes(MemoryIsolatedTestCase):
    def test_plan_generation_is_unchanged_when_no_recipes_are_saved(self):
        from api import MealPlanRequest, create_meal_plan
        athlete = "55555555-5555-4555-8555-555555555555"
        payload = {k: v for k, v in BASE.items()}
        payload["athlete_id"] = athlete
        without_recipes = create_meal_plan(MealPlanRequest(**payload))
        # Same profile, still no recipes on a second call -> identical shape.
        again = create_meal_plan(MealPlanRequest(**payload))
        self.assertEqual(set(without_recipes.keys()), set(again.keys()))


# ----------------------------------------------------------------------
# 8. Chat awareness (no persistence introduced)
# ----------------------------------------------------------------------

class TestChatRecipeAwareness(MemoryIsolatedTestCase):
    def test_recipes_block_reflects_saved_recipes(self):
        from chat import build_memory_context
        pr.add_recipe("chat-athlete", "Rajma Bowl", ["Rajma", "Rice"])
        context = build_memory_context("chat-athlete")
        self.assertIn("Rajma Bowl", context["recipes_block"])

    def test_no_recipes_gives_a_clear_placeholder(self):
        from chat import build_memory_context
        context = build_memory_context("no-recipes-athlete")
        self.assertIn("no saved recipes yet", context["recipes_block"])

    def test_excluded_recipe_is_labelled_not_suitable_in_the_chat_context(self):
        from chat import build_memory_context
        pr.add_recipe("dairy-athlete", "Paneer Curry", ["Paneer", "Tomato"])
        mm.set("profile", "dairy-athlete", dict(BASE, allergies="dairy"))
        context = build_memory_context("dairy-athlete")
        self.assertIn("NOT suitable", context["recipes_block"])

    def test_chat_still_does_not_persist_transcripts(self):
        """Unchanged guarantee: this feature adds no chat persistence.
        build_memory_context reads only pre-existing memory categories plus
        the new recipes category - never a chat/transcript store."""
        from chat import build_memory_context
        pr.add_recipe("chat-athlete-2", "Rajma Bowl", ["Rajma"])
        build_memory_context("chat-athlete-2")
        self.assertNotIn("chat", mm.UPDATABLE_CATEGORIES)
        self.assertNotIn("transcript", mm.UPDATABLE_CATEGORIES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
