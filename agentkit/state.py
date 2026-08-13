"""
agentkit.state
==============

State is the single shared object that flows through a graph. Nodes do not
call each other; they read state and return a *partial update*, and the
engine merges that update in.

THE KEY IDEA — REDUCERS
-----------------------
"Merge" is ambiguous, and getting it wrong is the #1 source of agent bugs.
If a node returns `{"messages": [new_msg]}`, did it mean

    (a) "append this message to history", or
    (b) "history is now exactly this one message"?

Almost always (a) for messages and (b) for scalars like `next_step`. So each
key in the schema declares a **reducer**: a function `(old, update) -> new`.
The engine never guesses.

This is what makes parallel nodes safe. Two branches can both emit
`{"messages": [...]}` and the append-reducer combines them deterministically
instead of one silently clobbering the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .types import Message

# A reducer takes the current value (possibly None on first write) and the
# value a node returned, and produces the new stored value.
Reducer = Callable[[Any, Any], Any]


# ---------------------------------------------------------------------------
# Built-in reducers
# ---------------------------------------------------------------------------
def last_value(old: Any, new: Any) -> Any:
    """Overwrite. The sane default for scalars, flags and counters-by-assignment."""
    return new


def add_messages(old: list[Message] | None, new: Any) -> list[Message]:
    """Append-with-upsert reducer for conversation history.

    Behaviour:
      * accepts a single Message or an iterable of Messages
      * appends by default
      * if an incoming message has the same `id` as an existing one, it
        *replaces* it in place. That upsert is what makes streaming and
        message-editing middleware possible without special-casing.
    """
    old = list(old or [])
    if isinstance(new, Message):
        new = [new]
    elif new is None:
        new = []

    index = {m.id: i for i, m in enumerate(old)}
    for msg in new:
        if not isinstance(msg, Message):
            raise TypeError(f"add_messages expected Message, got {type(msg).__name__}")
        if msg.id in index:
            old[index[msg.id]] = msg
        else:
            index[msg.id] = len(old)
            old.append(msg)
    return old


def append(old: list | None, new: Any) -> list:
    """Generic list concatenation, for scratchpads, citations, errors, etc."""
    old = list(old or [])
    if isinstance(new, (list, tuple)):
        old.extend(new)
    elif new is not None:
        old.append(new)
    return old


def merge_dict(old: dict | None, new: dict | None) -> dict:
    """Shallow dict merge. Handy for accumulating per-key scratch data."""
    out = dict(old or {})
    out.update(new or {})
    return out


def add_int(old: int | None, new: int) -> int:
    """Numeric accumulation — retry counts, token tallies, loop counters."""
    return (old or 0) + (new or 0)


# ---------------------------------------------------------------------------
# Channel + schema
# ---------------------------------------------------------------------------
@dataclass
class Channel:
    """One key in the state schema: how to merge it and what it starts as.

    `default` is a *factory* when mutable (list/dict) so instances never share
    the classic Python mutable-default bug.
    """

    reducer: Reducer = last_value
    default: Any = None
    description: str = ""

    def initial(self) -> Any:
        return self.default() if callable(self.default) else self.default


class StateSchema:
    """Declares the shape of a graph's state.

    Unknown keys returned by a node raise by default (`strict=True`). That is
    deliberate: a typo like `{"mesages": [...]}` should fail loudly on the
    first run, not silently do nothing for three weeks.
    """

    def __init__(self, channels: dict[str, Channel] | None = None, strict: bool = True):
        self.channels: dict[str, Channel] = dict(channels or {})
        self.strict = strict

    def add(self, name: str, reducer: Reducer = last_value, default: Any = None, description: str = "") -> "StateSchema":
        """Fluent registration: `schema.add("plan", default=list)`."""
        self.channels[name] = Channel(reducer=reducer, default=default, description=description)
        return self

    def initial_state(self, **overrides: Any) -> dict[str, Any]:
        """Build a fresh state dict, then apply any caller-supplied seed values."""
        state = {name: ch.initial() for name, ch in self.channels.items()}
        return self.apply(state, overrides) if overrides else state

    def apply(self, state: dict[str, Any], update: dict[str, Any] | None) -> dict[str, Any]:
        """Merge `update` into `state` using each channel's reducer.

        Returns a NEW dict rather than mutating. Immutability here is what lets
        the checkpointer snapshot history cheaply and lets you diff steps in a
        trace without worrying about aliasing.
        """
        if not update:
            return state
        if not isinstance(update, dict):
            raise TypeError(
                f"Nodes must return a dict of state updates (or None), got {type(update).__name__}"
            )

        new_state = dict(state)
        for key, value in update.items():
            channel = self.channels.get(key)
            if channel is None:
                if self.strict:
                    known = ", ".join(sorted(self.channels)) or "<none>"
                    raise KeyError(
                        f"Node wrote unknown state key {key!r}. Declared keys: {known}. "
                        f"Add it with schema.add({key!r}, ...) or use strict=False."
                    )
                # Non-strict mode: accept it with overwrite semantics.
                channel = Channel()
                self.channels[key] = channel
            new_state[key] = channel.reducer(new_state.get(key), value)
        return new_state


# ---------------------------------------------------------------------------
# The default schema every prebuilt agent uses
# ---------------------------------------------------------------------------
def default_agent_state() -> StateSchema:
    """A conversation-shaped state that covers ~90% of agents.

    Add your own channels on top for domain data (`retrieved_docs`, `plan`,
    `budget_remaining`, ...) — the graph does not care what is in there.
    """
    return StateSchema(
        {
            "messages": Channel(
                reducer=add_messages,
                default=list,
                description="Full conversation history, neutral Message objects.",
            ),
            "steps": Channel(
                reducer=add_int,
                default=0,
                description="How many model<->tool cycles have run this invocation.",
            ),
            "scratchpad": Channel(
                reducer=merge_dict,
                default=dict,
                description="Free-form working memory shared between nodes.",
            ),
            "errors": Channel(
                reducer=append,
                default=list,
                description="Recoverable errors recorded rather than raised.",
            ),
            "done": Channel(
                reducer=last_value,
                default=False,
                description="Set True by a node to request an early, clean stop.",
            ),
        }
    )


def last_message(state: dict[str, Any]) -> Message | None:
    """Convenience accessor used constantly in routing functions."""
    msgs: Iterable[Message] = state.get("messages") or []
    msgs = list(msgs)
    return msgs[-1] if msgs else None
