"""
AthleteOS - Tool-selection layer (connects tools/ to the LLM provider).

Bridges the standalone tools/ package (nutrition_tool, budget_tool) to
Claude's tool-use API. This is NOT an autonomous agent: the MODEL decides
whether to call a tool and which one; this layer only (a) tells Claude what
tools exist, (b) executes whichever tool Claude picks via the ToolRegistry,
and (c) hands the result back so Claude can write the final answer. Nothing
here decides on its own to call a tool.

Flow per request:
    1. Send the prompt to Claude, with the tools available.
    2. If Claude's response says "I want to call a tool" (stop_reason ==
       "tool_use"), run it through the tool registry and log everything.
    3. Send the tool's result back to Claude as part of the conversation.
    4. Repeat until Claude stops asking for tools and returns its answer.

A safety cap (MAX_TOOL_ITERATIONS) prevents a stuck loop from calling tools
forever.
"""

import json
import logging

from tools import build_default_registry

logger = logging.getLogger("athleteos.tool_selector")

MAX_TOOL_ITERATIONS = 4

# The tools/ package (base_tool.py etc.) knows nothing about Claude's API -
# these schemas are the translation from "what our tools accept" to "what
# Claude's tool-use format requires". Kept here, not in tools/, so the
# standalone tool package stays independent of any specific LLM provider.
TOOL_SPECS = [
    {
        "name": "nutrition_tool",
        "description": (
            "Calculate a precise daily nutrition target using formulas (not a "
            "guess): protein (g), calories (kcal), hydration (litres), or "
            "carbohydrate (g). Use this whenever the athlete's exact numeric "
            "target matters, instead of estimating it yourself."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["protein", "calories", "hydration", "carbs"],
                    "description": "Which calculation to run.",
                },
                "weight_kg": {
                    "type": "number",
                    "description": "Body weight in kg. Required for every operation.",
                },
                "height_cm": {
                    "type": "number",
                    "description": "Height in cm. Required only for operation=calories.",
                },
                "age": {
                    "type": "integer",
                    "description": "Age in years. Required only for operation=calories.",
                },
                "sex": {
                    "type": "string",
                    "enum": ["Male", "Female", "Other"],
                    "description": "Required only for operation=calories.",
                },
                "training_intensity": {
                    "type": "string",
                    "enum": ["Low", "Moderate", "High"],
                    "description": "Training intensity, used by every operation.",
                },
            },
            "required": ["operation", "weight_kg"],
        },
    },
    {
        "name": "budget_tool",
        "description": (
            "Given a daily food budget in Rs and a diet type, suggest one "
            "affordable Indian food per category (protein, carbohydrate, fat, "
            "fruit, vegetable) and whether the total fits the budget. Use this "
            "for budget-constrained athletes instead of guessing food costs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "budget_per_day": {
                    "type": "number",
                    "description": "Daily food budget in Rs.",
                },
                "diet": {
                    "type": "string",
                    "enum": ["Vegetarian", "Non-Vegetarian"],
                    "description": "Defaults to Vegetarian if omitted.",
                },
            },
            "required": ["budget_per_day"],
        },
    },
]

_registry = build_default_registry()  # the same registry from tools/, unmodified


def execute_tool_call(tool_use_block):
    """
    Run one Claude tool_use block through the tool registry and return the
    tool_result content block to send back (success, or is_error on failure).
    """
    name = tool_use_block.name
    tool_input = tool_use_block.input
    logger.info("Model selected tool: %s | arguments: %s", name, tool_input)

    try:
        result = _registry.execute(name, **tool_input)
        logger.info("Returning tool result to model: %s -> %s", name, result)
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_block.id,
            "content": json.dumps(result),
        }
    except Exception as exc:
        logger.info("Tool call failed, returning error to model: %s -> %s", name, exc)
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_block.id,
            "content": f"Error: {exc}",
            "is_error": True,
        }


def run_with_tools(client, model, max_tokens, output_config, messages):
    """
    Run the tool-use loop: call Claude with the tools available, execute
    whatever it selects (if anything), feed results back, and repeat until
    Claude stops calling tools and returns its final response.

    `messages` is mutated in place (tool_use / tool_result turns are
    appended) so the caller can inspect the full conversation afterwards.
    Returns the final response object from the Anthropic SDK.
    """
    response = None
    for iteration in range(1, MAX_TOOL_ITERATIONS + 1):
        logger.info("Requesting model response (iteration %d)", iteration)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            tools=TOOL_SPECS,
            output_config=output_config,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            logger.info(
                "Model finished (stop_reason=%s); no further tool calls",
                response.stop_reason,
            )
            return response

        logger.info("Model requested tool use (iteration %d)", iteration)
        messages.append({"role": "assistant", "content": response.content})

        tool_results = [
            execute_tool_call(block)
            for block in response.content
            if block.type == "tool_use"
        ]
        messages.append({"role": "user", "content": tool_results})

    logger.info(
        "Reached MAX_TOOL_ITERATIONS (%d) without a final answer; returning last response as-is",
        MAX_TOOL_ITERATIONS,
    )
    return response
