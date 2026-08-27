"""
AthleteOS - tools package (tool-calling foundation).

Exposes the base interface, the two current tools, and a ready-made default
registry, so callers can do:

    from tools import build_default_registry
    registry = build_default_registry()
    registry.execute("nutrition_tool", operation="protein", weight_kg=72)

Nothing here is wired into the LLM provider yet - this package is only the
foundation for future tool calling.
"""

from tools.base_tool import BaseTool
from tools.nutrition_tool import NutritionTool
from tools.budget_tool import BudgetTool
from tools.ingredients_tool import IngredientsTool
from tools.tool_registry import ToolRegistry, build_default_registry

__all__ = [
    "BaseTool",
    "NutritionTool",
    "BudgetTool",
    "IngredientsTool",
    "ToolRegistry",
    "build_default_registry",
]
