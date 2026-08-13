"""
agentkit.errors
===============

One base class so callers can `except AgentKitError` and catch everything the
framework raises, with specific subclasses so they can be surgical when they
want to be.

Error *messages* here are written for the person debugging at 2am: they say
what happened, what was expected, and where to look.
"""

from __future__ import annotations


class AgentKitError(Exception):
    """Base class for every error raised by this framework."""


class GraphError(AgentKitError):
    """Structural problems: unknown nodes, unreachable ends, bad edges."""


class RecursionLimitError(GraphError):
    """The graph took more steps than `RunConfig.recursion_limit` allows.

    Nearly always means an agent is stuck in a model->tool->model loop. Raise
    the limit only after you have looked at the trace and understood *why*.
    """

    def __init__(self, limit: int, path: list[str] | None = None):
        self.limit = limit
        self.path = path or []
        tail = " -> ".join(self.path[-8:]) if self.path else "unknown"
        super().__init__(
            f"Recursion limit of {limit} steps exceeded. Recent path: {tail}. "
            "Check for a node that always routes back to the model, or a tool "
            "whose output never satisfies the stop condition."
        )


class ToolNotFoundError(AgentKitError):
    """The model asked for a tool that is not registered.

    Usually a hallucinated name, or a registry/prompt mismatch where the
    prompt advertises a tool the node was not given.
    """

    def __init__(self, name: str, available: list[str] | None = None):
        self.name = name
        self.available = available or []
        avail = ", ".join(self.available) if self.available else "<none registered>"
        super().__init__(f"No tool named {name!r}. Available: {avail}")


class ToolExecutionError(AgentKitError):
    """A tool was found but could not be run with the given arguments."""

    def __init__(self, name: str, detail: str):
        self.name = name
        super().__init__(f"Tool {name!r} failed: {detail}")


class ModelError(AgentKitError):
    """The provider adapter failed — network, auth, rate limit, bad response."""


class InterruptError(AgentKitError):
    """Raised internally to pause a graph for human input.

    Not a failure. The engine catches this, checkpoints, and returns control
    to the caller so a human can approve, edit, or reject before resuming.
    """

    def __init__(self, node: str, payload: dict | None = None):
        self.node = node
        self.payload = payload or {}
        super().__init__(f"Graph interrupted at node {node!r} awaiting human input.")


class ConfigurationError(AgentKitError):
    """Something was wired up wrong before the run even started."""
