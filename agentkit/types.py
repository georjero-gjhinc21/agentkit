"""
agentkit.types
==============

The vocabulary of the framework. Everything else in the package speaks in
terms of the handful of dataclasses defined here.

DESIGN NOTE — why our own message classes instead of raw provider dicts?
------------------------------------------------------------------------
Every LLM vendor has a slightly different wire format (Anthropic wants
`content` blocks, OpenAI wants `tool_calls` on the assistant message and a
separate `tool` role, Google wants `parts`, ...). If those differences leak
into your agent logic you have permanently married one vendor.

So we define a *neutral* internal representation. Provider adapters
(`agentkit.models`) translate neutral -> wire on the way out and
wire -> neutral on the way back. The graph, the tools, the tracer and your
own nodes only ever see the neutral form.

This is the single most important portability decision in the whole repo.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------
# We deliberately keep the role set small. "system" is instructions, "user" is
# input from the outside world, "assistant" is model output, "tool" is the
# result of executing a tool the assistant asked for.
Role = Literal["system", "user", "assistant", "tool"]


def _new_id(prefix: str) -> str:
    """Short, sortable-ish, human-greppable IDs. Not cryptographic."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------
@dataclass
class ToolCall:
    """A model's *request* to run a tool. It has not been executed yet.

    `id` matters: the result message must carry the same id so the model can
    match request to response when several tools run in parallel.
    """

    name: str
    args: dict[str, Any]
    id: str = field(default_factory=lambda: _new_id("call"))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ToolCall({self.name}, {self.args})"


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------
@dataclass
class Message:
    """One turn in a conversation.

    Fields are a superset across roles; which ones are populated depends on
    `role`:

    * system/user    -> content
    * assistant      -> content and/or tool_calls (a model may do both)
    * tool           -> content (the result) + tool_call_id (which request)

    `metadata` is a free-form escape hatch. Middleware and tracers stash
    things there (latency, token counts, guardrail verdicts) without needing
    us to grow the dataclass every time.
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None  # tool name, or a named agent in multi-agent setups
    id: str = field(default_factory=lambda: _new_id("msg"))
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- ergonomic constructors ---------------------------------------------
    # These exist purely so user code reads like prose:
    #     Message.user("hello")   instead of   Message(role="user", content=...)

    @classmethod
    def system(cls, content: str, **kw: Any) -> "Message":
        return cls(role="system", content=content, **kw)

    @classmethod
    def user(cls, content: str, **kw: Any) -> "Message":
        return cls(role="user", content=content, **kw)

    @classmethod
    def assistant(
        cls, content: str = "", tool_calls: list[ToolCall] | None = None, **kw: Any
    ) -> "Message":
        return cls(role="assistant", content=content, tool_calls=tool_calls or [], **kw)

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str | None = None, **kw: Any) -> "Message":
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name, **kw)

    # -- helpers -------------------------------------------------------------
    @property
    def has_tool_calls(self) -> bool:
        """The core branch condition of every tool-calling agent loop."""
        return bool(self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form. Used by checkpointers and the tracer."""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "tool_calls": [{"id": c.id, "name": c.name, "args": c.args} for c in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        return cls(
            role=d["role"],
            content=d.get("content", ""),
            tool_calls=[ToolCall(**c) for c in d.get("tool_calls", [])],
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
            id=d.get("id", _new_id("msg")),
            created_at=d.get("created_at", time.time()),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Usage / cost accounting
# ---------------------------------------------------------------------------
@dataclass
class Usage:
    """Token accounting, aggregated across a whole run.

    Agents are loops, so a single `.invoke()` can hide a dozen model calls.
    Surfacing usage at the run level is what makes cost debuggable.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            model_calls=self.model_calls + other.model_calls,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# Model response
# ---------------------------------------------------------------------------
@dataclass
class ModelResponse:
    """What a provider adapter returns: one neutral message plus metadata."""

    message: Message
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    raw: Any = None  # the untouched provider payload, for debugging


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    """Per-invocation knobs, threaded through every node and middleware hook.

    `thread_id` is the conversation key used by checkpointers — pass the same
    one twice and the agent resumes where it left off. `recursion_limit` is
    the seatbelt that stops a misbehaving agent from looping forever.
    """

    thread_id: str = field(default_factory=lambda: _new_id("thread"))
    run_id: str = field(default_factory=lambda: _new_id("run"))
    recursion_limit: int = 25
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Arbitrary values your nodes may want (user id, tenant, db handle, ...).
    # Kept separate from graph *state* because it is input-only: nodes read it
    # but never write it, so it never needs a reducer.
    context: dict[str, Any] = field(default_factory=dict)
