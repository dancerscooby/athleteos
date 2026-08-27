"""
AthleteOS - Base tool interface (tool-calling foundation).

Every tool in this package inherits from BaseTool, so the registry (and
eventually an LLM deciding which tool to call) can treat all tools the same
way: read `name`/`description` to know what a tool is for, then call
`run(**kwargs)` to execute it.
"""

from abc import ABC, abstractmethod


class BaseTool(ABC):
    """Minimal interface every tool must implement."""

    name: str = "base_tool"
    description: str = "Base tool interface - override name/description/run() in subclasses."

    @abstractmethod
    def run(self, **kwargs):
        """Execute the tool with the given inputs and return a result (usually a dict)."""
        raise NotImplementedError
