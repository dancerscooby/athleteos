"""
AthleteOS - Budget tool (tool-calling foundation).

Given a daily food budget (in Rs) and a diet type, suggests one affordable
Indian food per category (protein, carbohydrate, fat, fruit, vegetable).
This is a small, self-contained cost table for the tool-calling foundation -
NOT the nutrition database itself (see knowledge_base/indian_protein_sources.md
for the fuller reference), and it is not wired into the meal planner yet.
"""

from tools.base_tool import BaseTool

# Approximate cost (Rs) for one typical serving/day-portion of each food.
# "diet" marks whether the food requires a non-vegetarian diet.
FOOD_DATABASE = {
    "protein": [
        {"name": "Dal (1 bowl)", "cost": 10, "diet": "Vegetarian"},
        {"name": "Soy chunks (50g dry)", "cost": 8, "diet": "Vegetarian"},
        {"name": "Rajma (1 bowl)", "cost": 12, "diet": "Vegetarian"},
        {"name": "Curd (200g)", "cost": 15, "diet": "Vegetarian"},
        {"name": "Eggs (2)", "cost": 15, "diet": "Non-Vegetarian"},
        {"name": "Paneer (100g)", "cost": 40, "diet": "Vegetarian"},
        {"name": "Chicken (100g)", "cost": 25, "diet": "Non-Vegetarian"},
    ],
    "carbohydrate": [
        {"name": "Roti (3 pieces)", "cost": 6, "diet": "Vegetarian"},
        {"name": "Rice (200g cooked)", "cost": 8, "diet": "Vegetarian"},
        {"name": "Poha (1 plate)", "cost": 10, "diet": "Vegetarian"},
        {"name": "Oats (50g)", "cost": 12, "diet": "Vegetarian"},
    ],
    "fat": [
        {"name": "Peanuts (30g)", "cost": 5, "diet": "Vegetarian"},
        {"name": "Cooking oil (for the day)", "cost": 5, "diet": "Vegetarian"},
        {"name": "Ghee (1 tsp)", "cost": 5, "diet": "Vegetarian"},
    ],
    "fruit": [
        {"name": "Banana (1)", "cost": 5, "diet": "Vegetarian"},
        {"name": "Seasonal fruit (1 serving)", "cost": 15, "diet": "Vegetarian"},
    ],
    "vegetable": [
        {"name": "Seasonal leafy greens (1 bowl)", "cost": 10, "diet": "Vegetarian"},
        {"name": "Mixed vegetable sabzi (1 bowl)", "cost": 15, "diet": "Vegetarian"},
    ],
}


def suggest_affordable_foods(budget_per_day, diet="Vegetarian"):
    """
    For each category (protein, carbohydrate, fat, fruit, vegetable), pick the
    cheapest food allowed by `diet`. Returns the per-category picks, the
    combined estimated cost, and whether that total fits within budget_per_day.
    """
    suggestions = {}
    total_cost = 0

    for category, foods in FOOD_DATABASE.items():
        allowed = [f for f in foods if diet == "Non-Vegetarian" or f["diet"] == "Vegetarian"]
        allowed.sort(key=lambda f: f["cost"])
        cheapest = allowed[0] if allowed else None
        if cheapest:
            suggestions[category] = {"name": cheapest["name"], "cost": cheapest["cost"]}
            total_cost += cheapest["cost"]

    return {
        "budget_per_day": budget_per_day,
        "diet": diet,
        "suggestions": suggestions,
        "estimated_total_cost": total_cost,
        "within_budget": (total_cost <= budget_per_day) if budget_per_day is not None else None,
    }


class BudgetTool(BaseTool):
    """Suggests one affordable food per category for a given daily budget and diet."""

    name = "budget_tool"
    description = (
        "Given a daily food budget in Rs and a diet type (Vegetarian or "
        "Non-Vegetarian), suggest one affordable Indian food per category "
        "(protein, carbohydrate, fat, fruit, vegetable) and the combined cost."
    )

    def run(self, budget_per_day, diet="Vegetarian", **kwargs):
        return suggest_affordable_foods(budget_per_day, diet)
