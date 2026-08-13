"""
agentkit.tools
==============

Turning ordinary Python functions into things an LLM can call.

DESIGN NOTE — the function *is* the contract
--------------------------------------------
A tool definition sent to a model is three things: a name, a description, and
a JSON Schema for its arguments. All three already exist in a well-written
Python function — the name, the docstring, and the type hints. So we derive
the schema by introspection instead of asking you to write it twice and let
the two drift apart.

    @tool
    def get_weather(city: str, units: str = "celsius") -> str:
        '''Look up current weather for a city.'''

...becomes a complete, model-ready tool. Write the docstring as if the reader
is the model, because it is: the description is the *only* thing telling the
model when to reach for this tool.

SAFETY NOTE
-----------
Tool arguments are attacker-influenced data. A model can be talked into
calling `delete_account(id="*")` by text it read from a web page. Two
mitigations live here: `requires_approval` (pause for a human), and the fact
that every tool failure is captured and returned to the model as a normal
tool message rather than crashing the run — models recover from a clear error
string remarkably well.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Union, get_args, get_origin, get_type_hints

from .errors import ToolExecutionError, ToolNotFoundError
from .types import Message, ToolCall

# ---------------------------------------------------------------------------
# Python type -> JSON Schema
# ---------------------------------------------------------------------------
_PRIMITIVES: dict[Any, dict[str, Any]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    type(None): {"type": "null"},
    Any: {},
}


def python_type_to_json_schema(tp: Any) -> dict[str, Any]:
    """Best-effort conversion of a type annotation into JSON Schema.

    Handles primitives, list[X], dict[str, X], Literal[...], Optional[X] and
    unions. Anything exotic degrades to an unconstrained `{}` rather than
    raising — a slightly loose schema is far better than a tool you cannot
    register at all.
    """
    if tp in _PRIMITIVES:
        return dict(_PRIMITIVES[tp])

    origin = get_origin(tp)
    args = get_args(tp)

    if origin is Literal:
        # Literal["a","b"] -> an enum. Great for constraining model choices.
        return {"type": "string", "enum": [str(a) for a in args]}

    if origin in (list, set, tuple):
        item = python_type_to_json_schema(args[0]) if args else {}
        return {"type": "array", "items": item}

    if origin is dict:
        value = python_type_to_json_schema(args[1]) if len(args) == 2 else {}
        return {"type": "object", "additionalProperties": value}

    if origin is Union:
        # Optional[X] is Union[X, None]; drop the None and describe X.
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return python_type_to_json_schema(non_none[0])
        return {"anyOf": [python_type_to_json_schema(a) for a in non_none]}

    # Dataclasses and pydantic-ish objects: expose their fields if we can.
    if hasattr(tp, "__dataclass_fields__"):
        props, required = {}, []
        for name, f in tp.__dataclass_fields__.items():
            props[name] = python_type_to_json_schema(f.type)
            required.append(name)
        return {"type": "object", "properties": props, "required": required}

    return {}


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a docstring into (summary, {param: description}).

    Understands the two conventions people actually use:
        Google style:  "    city: The city to look up."   under an Args: header
        Sphinx style:  ":param city: The city to look up."
    """
    if not doc:
        return "", {}
    doc = inspect.cleandoc(doc)
    params: dict[str, str] = {}

    for m in re.finditer(r"^:param\s+(\w+)\s*:\s*(.+)$", doc, re.MULTILINE):
        params[m.group(1)] = m.group(2).strip()

    lines = doc.splitlines()
    summary_lines: list[str] = []
    in_args = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"^(Args|Arguments|Parameters)\s*:$", stripped, re.IGNORECASE):
            in_args = True
            continue
        if in_args:
            if re.match(r"^(Returns|Raises|Yields|Examples?)\s*:$", stripped, re.IGNORECASE):
                in_args = False
                continue
            m = re.match(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$", stripped)
            if m:
                params.setdefault(m.group(1), m.group(2).strip())
            continue
        if stripped.startswith(":param"):
            continue
        summary_lines.append(line)

    summary = "\n".join(summary_lines).strip()
    return summary, params


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------
@dataclass
class Tool:
    """A callable the model may invoke, plus everything needed to describe it."""

    name: str
    description: str
    parameters: dict[str, Any]          # JSON Schema (object)
    func: Callable[..., Any]
    is_async: bool = False
    requires_approval: bool = False     # human-in-the-loop gate; see graph interrupts
    tags: list[str] = field(default_factory=list)
    # If True the raw Python return value is passed through untouched; otherwise
    # we stringify it for the model. Keep False unless you know why.
    return_raw: bool = False

    # -- what we send to the provider ---------------------------------------
    def to_schema(self) -> dict[str, Any]:
        """Neutral tool schema. Provider adapters reshape this as needed."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    # -- execution ----------------------------------------------------------
    def run(self, args: dict[str, Any]) -> Any:
        """Synchronously execute. Async tools are driven on a private loop."""
        if self.is_async:
            return asyncio.run(self.func(**args))
        return self.func(**args)

    async def arun(self, args: dict[str, Any]) -> Any:
        """Await the tool. Sync tools run in a worker thread so one slow
        blocking call cannot stall the whole event loop."""
        if self.is_async:
            return await self.func(**args)
        return await asyncio.to_thread(self.func, **args)

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Cheap guard rail before we hand model-generated args to real code.

        We check for missing required params and drop unexpected ones. We do
        NOT do deep JSON Schema validation — plug in `jsonschema` here if your
        tools touch anything dangerous.
        """
        props = self.parameters.get("properties", {})
        required = set(self.parameters.get("required", []))
        missing = required - set(args)
        if missing:
            raise ToolExecutionError(
                self.name, f"missing required argument(s): {', '.join(sorted(missing))}"
            )
        return {k: v for k, v in args.items() if k in props} if props else dict(args)


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    requires_approval: bool = False,
    tags: list[str] | None = None,
) -> Any:
    """Decorator that promotes a function to a `Tool`.

    Usage:
        @tool
        def add(a: int, b: int) -> int:
            '''Add two numbers.'''
            return a + b

        @tool(requires_approval=True, tags=["write"])
        def send_email(to: str, body: str) -> str:
            '''Send an email. Requires human approval.'''
            ...
    """

    def build(f: Callable[..., Any]) -> Tool:
        sig = inspect.signature(f)
        try:
            hints = get_type_hints(f)
        except Exception:  # forward refs we cannot resolve — fall back to raw
            hints = getattr(f, "__annotations__", {})

        summary, param_docs = _parse_docstring(f.__doc__)

        properties: dict[str, Any] = {}
        required: list[str] = []
        for pname, param in sig.parameters.items():
            # *args/**kwargs cannot be expressed in a tool schema; skip them.
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            schema = python_type_to_json_schema(hints.get(pname, Any))
            if pname in param_docs:
                schema["description"] = param_docs[pname]
            if param.default is not inspect.Parameter.empty:
                schema["default"] = param.default
            else:
                required.append(pname)
            properties[pname] = schema

        return Tool(
            name=name or f.__name__,
            description=description or summary or f"Call {f.__name__}.",
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
            },
            func=f,
            is_async=inspect.iscoroutinefunction(f),
            requires_approval=requires_approval,
            tags=list(tags or []),
        )

    # Support both @tool and @tool(...)
    return build(func) if func is not None else build


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class ToolRegistry:
    """A named collection of tools, with filtering.

    Filtering matters more than it sounds. Handing a model 60 tools measurably
    degrades selection accuracy; exposing a task-relevant subset per node is
    one of the cheapest quality wins available. `subset()` and `by_tag()` are
    there so a graph can give different nodes different toolboxes.
    """

    def __init__(self, tools: list[Tool | Callable] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, t: Tool | Callable) -> Tool:
        # Accept bare functions for convenience — wrap them on the fly.
        if not isinstance(t, Tool):
            t = tool(t)
        if t.name in self._tools:
            raise ValueError(f"Duplicate tool name: {t.name!r}")
        self._tools[t.name] = t
        return t

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolNotFoundError(name, sorted(self._tools))
        return self._tools[name]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_schema() for t in self._tools.values()]

    def subset(self, names: list[str]) -> "ToolRegistry":
        return ToolRegistry([self.get(n) for n in names])

    def by_tag(self, tag: str) -> "ToolRegistry":
        return ToolRegistry([t for t in self._tools.values() if tag in t.tags])


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def _stringify(value: Any) -> str:
    """Tool results must reach the model as text. Prefer JSON when possible so
    structured results stay machine-readable on the model's side too."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def execute_tool_call(
    registry: ToolRegistry,
    call: ToolCall,
    *,
    on_error: str = "return",
) -> Message:
    """Run one tool call and package the outcome as a `tool` Message.

    `on_error="return"` (default) converts exceptions into an error string the
    model can read and react to — retry with different args, apologise, or try
    another approach. `on_error="raise"` propagates instead, which is what you
    want in tests and in strict pipelines.

    Either way the returned message carries `tool_call_id`, without which the
    provider will reject the next request.
    """
    started = time.perf_counter()
    try:
        t = registry.get(call.name)
        args = t.validate_args(call.args)
        result = t.run(args)
        content = result if t.return_raw else _stringify(result)
        msg = Message.tool(content=content, tool_call_id=call.id, name=call.name)
        msg.metadata["ok"] = True
    except Exception as exc:  # noqa: BLE001 - intentional catch-all boundary
        if on_error == "raise":
            raise
        msg = Message.tool(
            content=f"Error executing tool {call.name!r}: {type(exc).__name__}: {exc}",
            tool_call_id=call.id,
            name=call.name,
        )
        msg.metadata["ok"] = False
        msg.metadata["error"] = repr(exc)
    msg.metadata["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return msg
