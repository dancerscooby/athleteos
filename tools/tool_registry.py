"""
AthleteOS - Tool registry (tool-calling foundation).

A simple lookup table of available tools. Responsible for:
    register()  - add a tool
    list_tools()- describe every registered tool (name + description)
    find_tool() - look a tool up by name
    execute()   - run a tool by name, with logging and timing

This is deliberately NOT an agent: nothing here decides *which* tool to call.
It only provides the plumbing so that, once something (application code today,
an LLM later) decides "call nutrition_tool with operation=protein", there is a
single place that does it safely and logs what happened.
"""

import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("athleteos.tools")


class ToolRegistry:
    """Holds a set of BaseTool instances, keyed by their `name`."""

    def __init__(self):
        self._tools = {}

    def register(self, tool):
        """Add a tool to the registry (keyed by tool.name)."""
        self._tools[tool.name] = tool

    def list_tools(self):
        """Return [{"name": ..., "description": ...}, ...] for every registered tool."""
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]

    def find_tool(self, name):
        """Look up a tool by name. Returns None if not found."""
        return self._tools.get(name)

    def execute(self, name, **kwargs):
        """
        Run the named tool with `kwargs`, logging the call, its inputs,
        its outputs (or error), and how long it took.
        """
        tool = self.find_tool(name)
        if tool is None:
            raise ValueError(f"Unknown tool: '{name}'. Registered tools: {list(self._tools)}")

        logger.info("Tool called: %s | inputs: %s", name, kwargs)
        start = time.perf_counter()
        try:
            result = tool.run(**kwargs)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.info("Tool failed: %s | error: %s | time: %.4fs", name, exc, elapsed)
            raise
        elapsed = time.perf_counter() - start
        logger.info("Tool result: %s | outputs: %s | time: %.4fs", name, result, elapsed)
        return result


def build_default_registry():
    """Create a registry with the current tools already registered."""
    from tools.nutrition_tool import NutritionTool
    from tools.budget_tool import BudgetTool
    from tools.ingredients_tool import IngredientsTool

    registry = ToolRegistry()
    registry.register(NutritionTool())
    registry.register(BudgetTool())
    registry.register(IngredientsTool())
    return registry


# ----------------------------------------------------------------------
# TEST SECTION  -  run with: python -m tools.tool_registry
# ----------------------------------------------------------------------

if __name__ == "__main__":
    registry = build_default_registry()

    print("=" * 64)
    print("REGISTERED TOOLS")
    print("=" * 64)
    for tool_info in registry.list_tools():
        print(f"  - {tool_info['name']}: {tool_info['description']}")

    print("\n" + "=" * 64)
    print("EXAMPLE CALLS")
    print("=" * 64)

    protein = registry.execute(
        "nutrition_tool", operation="protein", weight_kg=72, training_intensity="High"
    )
    print("\nprotein ->", protein)

    calories = registry.execute(
        "nutrition_tool", operation="calories",
        weight_kg=72, height_cm=178, age=22, sex="Male", training_intensity="High",
    )
    print("calories ->", calories)

    hydration = registry.execute(
        "nutrition_tool", operation="hydration", weight_kg=72, training_intensity="High"
    )
    print("hydration ->", hydration)

    carbs = registry.execute(
        "nutrition_tool", operation="carbs", weight_kg=72, training_intensity="High"
    )
    print("carbs ->", carbs)

    budget = registry.execute("budget_tool", budget_per_day=120, diet="Vegetarian")
    print("budget ->", budget)

    print("\n" + "=" * 64)
    print("UNKNOWN TOOL (should raise cleanly)")
    print("=" * 64)
    try:
        registry.execute("does_not_exist")
    except ValueError as exc:
        print("Caught expected error:", exc)
