"""
agentkit.tracing
================

You cannot debug an agent by reading its final answer.

A single `invoke()` can hide fifteen model calls, nine tool executions and a
routing decision that quietly sent everything down the wrong branch. When the
output is wrong, the question is never "what did it say" — it is "at which
step did it go wrong, and what did it see at that moment". That requires a
trace: a tree of steps with inputs, outputs and timings.

This module gives you a local one. It is a `Middleware`, so you attach it the
same way as anything else, and it can export to JSON for a viewer or to a
hosted platform (LangSmith, OpenTelemetry, your own) by subclassing
`BaseTracer.export`.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, TextIO

from .middleware import Middleware
from .types import Message, RunConfig


@dataclass
class Span:
    """One timed unit of work: a node, a model call, a tool execution."""

    name: str
    kind: str  # "run" | "node" | "model" | "tool"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str | None = None
    start: float = field(default_factory=time.time)
    end: float | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return round(((self.end or time.time()) - self.start) * 1000, 2)

    def finish(self, outputs: dict[str, Any] | None = None, error: str | None = None) -> None:
        self.end = time.time()
        if outputs:
            self.outputs = outputs
        if error:
            self.error = error


def _summarize(value: Any, limit: int = 300) -> Any:
    """Traces are for reading. A 40k-token transcript in every span makes the
    trace unusable, so we keep shapes and truncate contents."""
    if isinstance(value, Message):
        return {
            "role": value.role,
            "content": (value.content[:limit] + "…") if len(value.content) > limit else value.content,
            "tool_calls": [c.name for c in value.tool_calls],
        }
    if isinstance(value, list):
        head = [_summarize(v, limit) for v in value[:3]]
        return head + ([f"… +{len(value) - 3} more"] if len(value) > 3 else [])
    if isinstance(value, dict):
        return {k: _summarize(v, limit) for k, v in value.items()}
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "…"
    return value


class BaseTracer(Middleware):
    """Collects spans. Subclass and override `export` to ship them anywhere."""

    def __init__(self) -> None:
        self.spans: list[Span] = []
        self._stack: list[str] = []
        self._open: dict[str, Span] = {}

    # -- span plumbing -------------------------------------------------------
    def start_span(self, name: str, kind: str, inputs: dict[str, Any] | None = None) -> Span:
        span = Span(
            name=name,
            kind=kind,
            parent_id=self._stack[-1] if self._stack else None,
            inputs=_summarize(inputs or {}),
        )
        self.spans.append(span)
        self._open[span.id] = span
        self._stack.append(span.id)
        return span

    def end_span(self, span: Span, outputs: dict[str, Any] | None = None, error: str | None = None) -> None:
        span.finish(_summarize(outputs or {}), error)
        self._open.pop(span.id, None)
        if self._stack and self._stack[-1] == span.id:
            self._stack.pop()

    # -- middleware hooks ----------------------------------------------------
    def on_start(self, state, config: RunConfig):
        self._run_span = self.start_span(
            f"run:{config.run_id}", "run", {"thread_id": config.thread_id, "tags": config.tags}
        )

    def before_node(self, node, state, config):
        self._open[f"node:{node}"] = self.start_span(node, "node", {"state_keys": sorted(state)})

    def after_node(self, node, state, update, config):
        span = self._open.pop(f"node:{node}", None)
        if span:
            self.end_span(span, {"update": update})
        return None

    def on_error(self, node, error, state, config):
        span = self._open.pop(f"node:{node}", None)
        if span:
            self.end_span(span, error=f"{type(error).__name__}: {error}")

    def on_end(self, state, config):
        run = getattr(self, "_run_span", None)
        if run:
            self.end_span(run, {"final_keys": sorted(state)})
        self.export()

    # -- output --------------------------------------------------------------
    def export(self) -> None:
        """Called once at the end of a run. Default: do nothing."""

    def to_json(self, indent: int = 2) -> str:
        return json.dumps([asdict(s) | {"duration_ms": s.duration_ms} for s in self.spans], indent=indent, default=str)


class ConsoleTracer(BaseTracer):
    """Prints an indented tree at the end of the run.

    Zero setup, works over SSH, good enough for most debugging:

        run:run_abc123                              412.3ms
          model                                     280.1ms
          tools                                     118.7ms
          model                                      13.5ms
    """

    def __init__(self, stream: TextIO | None = None, show_io: bool = False):
        super().__init__()
        self.stream = stream or sys.stdout
        self.show_io = show_io

    def export(self) -> None:
        by_parent: dict[str | None, list[Span]] = {}
        for s in self.spans:
            by_parent.setdefault(s.parent_id, []).append(s)

        def walk(parent: str | None, depth: int) -> None:
            for s in by_parent.get(parent, []):
                pad = "  " * depth
                mark = " ✗" if s.error else ""
                print(f"{pad}{s.name}{mark}".ljust(46) + f"{s.duration_ms:>8.1f}ms", file=self.stream)
                if s.error:
                    print(f"{pad}  ! {s.error}", file=self.stream)
                if self.show_io and s.outputs:
                    print(f"{pad}  -> {json.dumps(s.outputs, default=str)[:200]}", file=self.stream)
                walk(s.id, depth + 1)

        print("\n--- trace ---", file=self.stream)
        walk(None, 0)


class JSONFileTracer(BaseTracer):
    """Append one JSON document per run to a file.

    A poor man's observability backend. Enough to build a regression suite:
    keep traces for known-good runs and diff routing decisions when you change
    a prompt.
    """

    def __init__(self, path: str = "traces.jsonl"):
        super().__init__()
        self.path = path

    def export(self) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"spans": [asdict(s) for s in self.spans]}, default=str) + "\n")
