"""
AthleteOS - Personal recipes (athlete-saved homemade/family recipes).

A thin CRUD layer over memory_manager's existing "recipes" category - the
same update-in-place get()/set() pattern profile/preference/latest_plan
already use (see memory_manager.py: recipes is one more entry in
UPDATABLE_CATEGORIES, no new operation was added to support this). Mirrors
episodic_memory.py's role: this file owns the STORAGE shape for one memory
category; it has no opinion on diet/allergy/selection rules - that decision
belongs to meal_planner.classify_personal_recipes(), which is the one place
recipe SAFETY is enforced (see its docstring).

WHAT THIS IS
    Read/add/delete for one athlete's list of saved recipes, keyed strictly
    by athlete_id (a UUID) - never by display name, so two athletes named
    "Rahul" never share a recipe box.

WHAT THIS IS NOT
    Not a food-selection system, not a nutrition calculator, not RAG. A
    recipe saved here is never ingested into the knowledge_base/FAISS index
    - it is athlete-specific memory, not general domain knowledge.
"""

import time
import uuid

import memory_manager

CATEGORY = "recipes"


def _new_id():
    """A short, unique-enough id for one recipe within an athlete's list -
    not a security boundary (athlete_id already is one), just enough to
    unambiguously reference one saved recipe for deletion."""
    return uuid.uuid4().hex[:12]


def list_recipes(athlete_id):
    """Return this athlete's saved recipes, or an empty list if none yet."""
    return memory_manager.get(CATEGORY, athlete_id) or []


def add_recipe(athlete_id, name, ingredients, method=None, notes=None, servings=None):
    """
    Save one new recipe for this athlete. `ingredients` must already be a
    clean list of strings (the API layer parses the raw textarea text via
    tools.ingredients_tool.parse_ingredients() before calling this, the same
    parser home-ingredients already uses - no second parsing scheme).

    Read-modify-write: fetches the athlete's current list, appends the new
    recipe, writes the whole list back via memory_manager.set() - exactly
    the same pattern used everywhere else update-in-place memory is edited.
    """
    recipe = {
        "id": _new_id(),
        "name": name.strip(),
        "ingredients": list(ingredients),
        "method": (method or "").strip() or None,
        "notes": (notes or "").strip() or None,
        "servings": (servings or "").strip() or None,
        "created_at": time.time(),
    }
    recipes = list_recipes(athlete_id)
    recipes.append(recipe)
    memory_manager.set(CATEGORY, athlete_id, recipes)
    return recipe


def delete_recipe(athlete_id, recipe_id):
    """Remove one recipe by id. Returns True if a recipe was actually
    removed, False if no recipe with that id existed for this athlete."""
    recipes = list_recipes(athlete_id)
    remaining = [r for r in recipes if r.get("id") != recipe_id]
    if len(remaining) == len(recipes):
        return False
    memory_manager.set(CATEGORY, athlete_id, remaining)
    return True
