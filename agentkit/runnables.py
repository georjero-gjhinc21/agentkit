"""
agentkit.runnables
==================

Composable pipelines — the `prompt | model | parser` idea.

WHY A PIPE OPERATOR IS NOT JUST SUGAR
--------------------------------------
Written out longhand, a chat pipeline is:

    text   = template.format(question=q)
    reply  = model.invoke(text)
    answer = parse(reply)

That is fine until you want it to *stream*, or run 200 of them as a *batch*,
or run three of them in *parallel*, or run any of it *async*. Now every one of
those three lines needs four variants, and you are writing the same plumbing
in every project.

The fix is to make every stage implement one small interface:

    invoke(x)   -> y            one input, one output
    stream(x)   -> Iterator[y]  incremental output
    batch(xs)   -> [y]          many inputs, concurrently
    ainvoke(x)  -> await y      non-blocking

...and then define composition over that interface. Once `A | B` produces
something that is itself a Runnable, streaming, batching and async come free
for the whole pipeline, at every depth, forever. That is the actual payoff:
not shorter code, but *uniform* code.

WHAT COMPOSES
-------------
    Runnable          passes through unchanged
    BaseChatModel     wrapped so it takes a prompt and returns a Message
    plain function    wrapped as RunnableLambda
    dict              wrapped as RunnableParallel (branches run concurrently)

So all of these work:

    chain = prompt | model | StrOutputParser()
    chain = {"context": retriever, "question": passthrough} | prompt | model
    chain = prompt | model | (lambda m: m.content.upper())
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterator, Sequence

from .errors import ConfigurationError
from .types import Message


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class Runnable(ABC):
    """One stage of a pipeline.

    Subclasses MUST implement `invoke`. The other three methods have working
    defaults derived from it, so a two-line custom stage is immediately
    streamable, batchable and awaitable — just not *efficiently* so. Override
    them when you can do better (a model can really stream; `invoke` can only
    pretend by yielding one chunk).
    """

    @abstractmethod
    def invoke(self, input: Any, config: dict[str, Any] | None = None) -> Any:
        """Transform one input into one output."""

    def stream(self, input: Any, config: dict[str, Any] | None = None) -> Iterator[Any]:
        """Yield output incrementally. Default: one chunk, the whole result."""
        yield self.invoke(input, config)

    def batch(
        self,
        inputs: Sequence[Any],
        config: dict[str, Any] | None = None,
        max_workers: int = 8,
    ) -> list[Any]:
        """Run many inputs concurrently, preserving input order.

        Threads rather than processes because the bottleneck is network I/O,
        not CPU. Order is preserved because callers almost always zip results
        back against their inputs.
        """
        if len(inputs) <= 1:
            return [self.invoke(i, config) for i in inputs]
        with ThreadPoolExecutor(max_workers=min(max_workers, len(inputs))) as pool:
            return list(pool.map(lambda i: self.invoke(i, config), inputs))

    async def ainvoke(self, input: Any, config: dict[str, Any] | None = None) -> Any:
        """Await the result. Default runs `invoke` in a worker thread so a
        blocking stage cannot stall the event loop."""
        return await asyncio.to_thread(self.invoke, input, config)

    # -- composition ---------------------------------------------------------
    def __or__(self, other: Any) -> "RunnableSequence":
        """`a | b` — feed a's output into b."""
        return RunnableSequence([self, coerce_runnable(other)])

    def __ror__(self, other: Any) -> "RunnableSequence":
        """`dict | runnable` — lets a plain dict start a chain."""
        return RunnableSequence([coerce_runnable(other), self])

    # -- introspection -------------------------------------------------------
    @property
    def name(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{self.name}>"


def coerce_runnable(obj: Any) -> Runnable:
    """Turn anything pipe-able into a Runnable.

    This function is why chains read naturally: you never have to wrap a
    lambda or a dict by hand, and a model can sit in a chain despite having a
    different native signature.
    """
    from .models import BaseChatModel  # local import: avoids a cycle

    if isinstance(obj, Runnable):
        return obj
    if isinstance(obj, BaseChatModel):
        return RunnableModel(obj)
    if isinstance(obj, dict):
        return RunnableParallel(obj)
    if callable(obj):
        return RunnableLambda(obj)
    raise ConfigurationError(
        f"Cannot use {type(obj).__name__} in a chain. Expected a Runnable, a "
        "chat model, a dict of branches, or a callable."
    )


# ---------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------
class RunnableSequence(Runnable):
    """Stages executed in order, each fed the previous one's output.

    Flattening in `__or__` keeps `a | b | c` a single three-stage sequence
    rather than a nested pair — which matters for readable traces and for
    streaming (only the LAST stage can stream meaningfully; everything before
    it must complete to produce its input).
    """

    def __init__(self, steps: Sequence[Any]):
        self.steps: list[Runnable] = [coerce_runnable(s) for s in steps]
        if not self.steps:
            raise ConfigurationError("A chain needs at least one step.")

    def __or__(self, other: Any) -> "RunnableSequence":
        return RunnableSequence([*self.steps, coerce_runnable(other)])

    def invoke(self, input: Any, config: dict[str, Any] | None = None) -> Any:
        value = input
        for step in self.steps:
            value = step.invoke(value, config)
        return value

    def stream(self, input: Any, config: dict[str, Any] | None = None) -> Iterator[Any]:
        """Run all but the last stage eagerly, then stream the last one.

        This is the honest implementation. You cannot stream through a stage
        that needs its complete input (a parser cannot parse half a JSON
        object), so the streaming boundary is the final stage.
        """
        value = input
        for step in self.steps[:-1]:
            value = step.invoke(value, config)
        yield from self.steps[-1].stream(value, config)

    async def ainvoke(self, input: Any, config: dict[str, Any] | None = None) -> Any:
        value = input
        for step in self.steps:
            value = await step.ainvoke(value, config)
        return value

    @property
    def name(self) -> str:
        return " | ".join(s.name for s in self.steps)


# ---------------------------------------------------------------------------
# Parallel
# ---------------------------------------------------------------------------
class RunnableParallel(Runnable):
    """Run several branches on the SAME input and collect a dict of results.

    The canonical use is assembling a RAG prompt's inputs in one shot:

        {"context": retriever, "question": RunnablePassthrough()} | prompt | model

    Both branches see the user's question; one retrieves documents, the other
    passes it through untouched. Because they are independent, they run
    concurrently — a retrieval round-trip overlaps with anything else.
    """

    def __init__(self, branches: dict[str, Any]):
        if not branches:
            raise ConfigurationError("RunnableParallel needs at least one branch.")
        self.branches = {k: coerce_runnable(v) for k, v in branches.items()}

    def invoke(self, input: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
        if len(self.branches) == 1:
            k, r = next(iter(self.branches.items()))
            return {k: r.invoke(input, config)}
        with ThreadPoolExecutor(max_workers=len(self.branches)) as pool:
            futures = {k: pool.submit(r.invoke, input, config) for k, r in self.branches.items()}
            return {k: f.result() for k, f in futures.items()}

    async def ainvoke(self, input: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
        keys = list(self.branches)
        results = await asyncio.gather(*(self.branches[k].ainvoke(input, config) for k in keys))
        return dict(zip(keys, results))

    @property
    def name(self) -> str:
        return "{" + ", ".join(self.branches) + "}"


# ---------------------------------------------------------------------------
# Leaf stages
# ---------------------------------------------------------------------------
class RunnableLambda(Runnable):
    """Wrap a plain function. The escape hatch for anything ad hoc.

    Accepts a one-arg `f(x)` or a two-arg `f(x, config)`; we sniff which.
    """

    def __init__(self, func: Callable[..., Any], name: str | None = None):
        self.func = func
        self._name = name or getattr(func, "__name__", "lambda")

    def invoke(self, input: Any, config: dict[str, Any] | None = None) -> Any:
        try:
            return self.func(input, config)  # type: ignore[call-arg]
        except TypeError as exc:
            if "positional argument" not in str(exc):
                raise
            return self.func(input)

    @property
    def name(self) -> str:
        return self._name


class RunnablePassthrough(Runnable):
    """Return the input unchanged.

    Sounds useless; is essential. Inside a `RunnableParallel` it is how you
    carry the original input forward alongside derived values.

    `assign` extends a dict input with computed keys instead of replacing it:

        RunnablePassthrough.assign(word_count=lambda d: len(d["text"].split()))
    """

    def __init__(self, extra: dict[str, Any] | None = None):
        self.extra = {k: coerce_runnable(v) for k, v in (extra or {}).items()}

    @classmethod
    def assign(cls, **kwargs: Any) -> "RunnablePassthrough":
        return cls(kwargs)

    def invoke(self, input: Any, config: dict[str, Any] | None = None) -> Any:
        if not self.extra:
            return input
        if not isinstance(input, dict):
            raise ConfigurationError("RunnablePassthrough.assign requires a dict input.")
        return {**input, **{k: r.invoke(input, config) for k, r in self.extra.items()}}


class RunnableModel(Runnable):
    """Adapts a `BaseChatModel` to the Runnable interface.

    Accepts a str, a single Message, or a list of Messages, and returns the
    assistant Message. Created automatically when you pipe a model, so you
    rarely construct one yourself.

    Returning a Message (not a raw string) is deliberate: the next stage may
    need `tool_calls` or `metadata`, and a parser can always take `.content`.
    """

    def __init__(self, model: Any, **call_kwargs: Any):
        self.model = model
        self.call_kwargs = call_kwargs

    @staticmethod
    def _to_messages(input: Any) -> list[Message]:
        if isinstance(input, str):
            return [Message.user(input)]
        if isinstance(input, Message):
            return [input]
        if isinstance(input, list) and all(isinstance(m, Message) for m in input):
            return list(input)
        # A dict usually means someone forgot the prompt stage — say so clearly.
        raise ConfigurationError(
            f"A model stage needs a string or Messages, got {type(input).__name__}. "
            "Did you forget a prompt template before the model?"
        )

    def invoke(self, input: Any, config: dict[str, Any] | None = None) -> Message:
        return self.model.invoke(self._to_messages(input), **self.call_kwargs).message

    def stream(self, input: Any, config: dict[str, Any] | None = None) -> Iterator[str]:
        yield from self.model.stream(self._to_messages(input), **self.call_kwargs)

    @property
    def name(self) -> str:
        return f"model:{getattr(self.model, 'model', '?')}"


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
class RunnableBranch(Runnable):
    """Dynamic routing: pick a downstream chain based on the input.

    Cheaper and far more debuggable than asking an agent to decide. Classify
    once, then run the specialised chain:

        RunnableBranch(
            (lambda x: "refund" in x.lower(), refund_chain),
            (lambda x: "bug" in x.lower(),    support_chain),
            default_chain,                    # required fallback
        )

    Conditions are tried in order; the first match wins.
    """

    def __init__(self, *branches: Any):
        if len(branches) < 2:
            raise ConfigurationError("RunnableBranch needs at least one condition and a default.")
        *pairs, default = branches
        self.branches = [(cond, coerce_runnable(r)) for cond, r in pairs]
        self.default = coerce_runnable(default)

    def invoke(self, input: Any, config: dict[str, Any] | None = None) -> Any:
        for condition, runnable in self.branches:
            if condition(input):
                return runnable.invoke(input, config)
        return self.default.invoke(input, config)


class RunnableRetry(Runnable):
    """Wrap any stage with retries. Composable, so it applies at any depth:

        chain = prompt | RunnableRetry(model, attempts=3) | parser
    """

    def __init__(self, runnable: Any, attempts: int = 3, delay: float = 0.5):
        self.runnable = coerce_runnable(runnable)
        self.attempts = attempts
        self.delay = delay

    def invoke(self, input: Any, config: dict[str, Any] | None = None) -> Any:
        import time

        last: BaseException | None = None
        for i in range(self.attempts):
            try:
                return self.runnable.invoke(input, config)
            except Exception as exc:  # noqa: BLE001
                last = exc
                if i < self.attempts - 1:
                    time.sleep(self.delay * (2**i))
        raise last  # type: ignore[misc]


class RunnableFallback(Runnable):
    """Try a primary stage; on failure fall through to alternatives.

    The standard use is provider redundancy — when the primary API is rate
    limited or down, degrade to a second provider or a smaller model rather
    than failing the user's request:

        RunnableFallback(gpt4_chain, [claude_chain, local_chain])
    """

    def __init__(self, primary: Any, fallbacks: Sequence[Any]):
        self.primary = coerce_runnable(primary)
        self.fallbacks = [coerce_runnable(f) for f in fallbacks]

    def invoke(self, input: Any, config: dict[str, Any] | None = None) -> Any:
        errors: list[str] = []
        for stage in [self.primary, *self.fallbacks]:
            try:
                return stage.invoke(input, config)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{stage.name}: {type(exc).__name__}: {exc}")
        raise RuntimeError("All fallbacks failed:\n  " + "\n  ".join(errors))


# ---------------------------------------------------------------------------
# Memory wrapper
# ---------------------------------------------------------------------------
class RunnableWithMessageHistory(Runnable):
    """Give a stateless chain conversational memory.

    A chain is a pure function: same input, same output, no recollection. Real
    chatbots need the previous turns. This wrapper does the bookkeeping:

        1. load this session's history
        2. splice it into the chain's input
        3. run the chain
        4. append the new user turn and the reply back to history

    Sessions are keyed by `session_id`, so one wrapped chain serves every
    concurrent user without their conversations bleeding into each other.

    Note the difference from a graph checkpointer: a checkpointer snapshots
    *entire graph state* for resumability; this stores *only messages* for a
    linear chain. Use whichever matches the shape of what you built.
    """

    def __init__(
        self,
        runnable: Any,
        get_history: Callable[[str], Any],
        input_key: str = "input",
        history_key: str = "history",
    ):
        self.runnable = coerce_runnable(runnable)
        self.get_history = get_history
        self.input_key = input_key
        self.history_key = history_key

    def invoke(self, input: Any, config: dict[str, Any] | None = None) -> Any:
        config = config or {}
        session_id = config.get("session_id", "default")
        history = self.get_history(session_id)

        payload = dict(input) if isinstance(input, dict) else {self.input_key: input}
        payload[self.history_key] = history.messages

        result = self.runnable.invoke(payload, config)

        # Persist the turn only after a successful call, so a failed request
        # does not leave a user message with no reply corrupting the history.
        user_text = payload.get(self.input_key)
        if isinstance(user_text, str):
            history.add(Message.user(user_text))
        history.add(result if isinstance(result, Message) else Message.assistant(str(result)))
        return result


class ChatMessageHistory:
    """Message store for one session. Trivial on purpose.

    Swap in Redis or Postgres by implementing `messages`, `add` and `clear`.
    Bounded by `max_messages` because unbounded history is how chatbots end up
    sending a 200k-token request on turn 90.
    """

    def __init__(self, max_messages: int | None = 100):
        self._messages: list[Message] = []
        self.max_messages = max_messages

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def add(self, message: Message) -> None:
        self._messages.append(message)
        if self.max_messages and len(self._messages) > self.max_messages:
            del self._messages[: -self.max_messages]

    def clear(self) -> None:
        self._messages.clear()


class InMemoryHistoryStore:
    """session_id -> ChatMessageHistory. Pass `.get` to RunnableWithMessageHistory."""

    def __init__(self, max_messages: int | None = 100):
        self._sessions: dict[str, ChatMessageHistory] = {}
        self.max_messages = max_messages

    def get(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatMessageHistory(self.max_messages)
        return self._sessions[session_id]

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
