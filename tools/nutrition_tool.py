"""
AthleteOS - Nutrition tool (tool-calling foundation).

Simple, formula-based nutrition math - NOT an LLM call, NOT a nutrition
database. This is the kind of calculation an LLM could delegate to a tool
instead of guessing at arithmetic itself. Each calculation is also exposed as
a plain function so it can be unit-tested or used directly without going
through the tool/registry machinery.

Formulas used (standard, widely published estimates - not medical advice):
- Protein:  weight_kg x g/kg, scaled by training intensity.
- Calories: Mifflin-St Jeor BMR x an activity multiplier for training intensity.
- Hydration: a baseline (35 ml/kg) plus extra fluid for training load.
- Carbohydrate: weight_kg x g/kg, scaled by training intensity (see
  knowledge_base/football_nutrition.md for the same target ranges).
"""

from tools.base_tool import BaseTool

# g of protein per kg body weight, by training intensity.
PROTEIN_FACTOR = {"Low": 1.2, "Moderate": 1.6, "High": 1.8}

# Activity multiplier applied to BMR, by training intensity.
CALORIE_ACTIVITY_MULTIPLIER = {"Low": 1.4, "Moderate": 1.6, "High": 1.8}

# Extra hydration (ml) on top of the baseline, by training intensity.
HYDRATION_EXTRA_ML = {"Low": 300, "Moderate": 600, "High": 1000}

# g of carbohydrate per kg body weight, by training intensity.
CARB_FACTOR = {"Low": 4, "Moderate": 6, "High": 8}


def calculate_protein_intake(weight_kg, training_intensity="Moderate"):
    """Daily protein target in grams."""
    factor = PROTEIN_FACTOR.get(training_intensity, PROTEIN_FACTOR["Moderate"])
    return round(weight_kg * factor, 1)


def calculate_calorie_requirement(weight_kg, height_cm, age, sex, training_intensity="Moderate"):
    """Daily calorie target (kcal), via Mifflin-St Jeor BMR x activity multiplier."""
    if sex == "Female":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    else:  # Male / Other - Mifflin-St Jeor has no third term; use the male formula
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    multiplier = CALORIE_ACTIVITY_MULTIPLIER.get(training_intensity, CALORIE_ACTIVITY_MULTIPLIER["Moderate"])
    return round(bmr * multiplier)


def estimate_hydration_requirement(weight_kg, training_intensity="Moderate"):
    """Daily hydration target in litres."""
    baseline_ml = weight_kg * 35
    extra_ml = HYDRATION_EXTRA_ML.get(training_intensity, HYDRATION_EXTRA_ML["Moderate"])
    return round((baseline_ml + extra_ml) / 1000, 2)


def estimate_carbohydrate_intake(weight_kg, training_intensity="Moderate"):
    """Daily carbohydrate target in grams."""
    factor = CARB_FACTOR.get(training_intensity, CARB_FACTOR["Moderate"])
    return round(weight_kg * factor, 1)


class NutritionTool(BaseTool):
    """
    Runs one of four nutrition calculations, selected by `operation`:
        "protein"    -> calculate_protein_intake
        "calories"   -> calculate_calorie_requirement
        "hydration"  -> estimate_hydration_requirement
        "carbs"      -> estimate_carbohydrate_intake
    """

    name = "nutrition_tool"
    description = (
        "Calculate daily protein (g), calorie (kcal), hydration (litres), or "
        "carbohydrate (g) targets for an athlete from their body stats and "
        "training intensity. operation must be one of: protein, calories, "
        "hydration, carbs."
    )

    OPERATIONS = {
        "protein": calculate_protein_intake,
        "calories": calculate_calorie_requirement,
        "hydration": estimate_hydration_requirement,
        "carbs": estimate_carbohydrate_intake,
    }

    def run(self, operation, **kwargs):
        func = self.OPERATIONS.get(operation)
        if func is None:
            raise ValueError(
                f"Unknown operation '{operation}'. Must be one of: {list(self.OPERATIONS)}"
            )
        return {"operation": operation, "result": func(**kwargs)}
