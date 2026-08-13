"""
agentkit.middleware
===================

Cross-cutting concerns, kept out of node bodies.

Logging, cost caps, PII redaction, guardrails and human-approval policies all
have the same shape: they need to see every step but they are not *about* any
particular step. Pushing them into nodes means copy-pasting the same five
lines into every node and forgetting one.

Middleware is a plain object with optional hooks. Implement only what you
need; the engine uses `getattr(mw, "hook", None)` so absent hooks cost
nothing:

    on_start(state, config)
    before_node(node_name, state, config)
    after_node(node_name, state, update, config) -> update | None
    on_error(node_name, exception, state, config)
    on_end(state, config)

ORDERING
--------
`before_node` hooks run in the order given; `after_node` hooks run in reverse,
like an onion / ASGI stack. So the outermost middleware sees the request first
and the response last, which is what you want for timing and for anything that
wraps.

`after_node` may return a *modified* update, which makes it a genuine
intercept point: redact secrets, clamp values, or veto a change before it ever
reaches the state.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Iterable

from .errors import AgentKitError
from .types import Message, RunConfig

logger = logging.getLogger("agentkit")


class Middleware:
    """Optional base class. Subclass and override what you care about.

    You do not have to inherit from this — duck typing is enough — but
    subclassing documents intent and gives you no-op defaults.
    """

    def on_start(self, state: dict[str, Any], config: RunConfig) -> None: ...
    def before_node(self, node: str, state: dict[str, Any], config: RunConfig) -> None: ...
    def after_node(
        self, node: str, state: dict[str, Any], update: dict[str, Any], config: RunConfig
    ) -> dict[str, Any] | None:
        return None
    def on_error(
        self, node: str, error: BaseException, state: dict[str, Any], config: RunConfig
    ) -> None: ...
    def on_end(self, state: dict[str, Any], config: RunConfig) -> None: ...


# ---------------------------------------------------------------------------
class LoggingMiddleware(Middleware):
    """Structured per-node logging with timings. The first thing to add when
    an agent misbehaves and the last thing you should remove."""

    def __init__(self, level: int = logging.INFO, log: logging.Logger | None = None):
        self.level = level
        self.log = log or logger
        self._t: dict[str, float] = {}

    def on_start(self, state, config):
        self.log.log(self.level, "run.start run_id=%s thread=%s", config.run_id, config.thread_id)

    def before_node(self, node, state, config):
        self._t[node] = time.perf_counter()
        self.log.log(self.level, "node.start %s", node)

    def after_node(self, node, state, update, config):
        dt = (time.perf_counter() - self._t.pop(node, time.perf_counter())) * 1000
        self.log.log(self.level, "node.end %s %.1fms keys=%s", node, dt, sorted(update))
        return None

    def on_error(self, node, error, state, config):
        self.log.error("node.error %s %s: %s", node, type(error).__name__, error)

    def on_end(self, state, config):
        self.log.log(self.level, "run.end run_id=%s steps=%s", config.run_id, state.get("steps"))


# ---------------------------------------------------------------------------
class BudgetMiddleware(Middleware):
    """Hard caps on steps, wall-clock time and tokens.

    An agent with a bug and a credit card is a genuinely expensive combination.
    Treat this as a seatbelt you always wear, not a thing you add after the
    first surprising invoice. It raises rather than truncating, because a run
    that silently returns half an answer is worse than one that fails loudly.
    """

    def __init__(
        self,
        max_steps: int | None = None,
        max_seconds: float | None = None,
        max_tokens: int | None = None,
    ):
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.max_tokens = max_tokens
        self._start = 0.0
        self._steps = 0

    def on_start(self, state, config):
        self._start = time.time()
        self._steps = 0

    def before_node(self, node, state, config):
        self._steps += 1
        if self.max_steps and self._steps > self.max_steps:
            raise AgentKitError(f"Budget exceeded: more than {self.max_steps} node executions.")
        if self.max_seconds and (time.time() - self._start) > self.max_seconds:
            raise AgentKitError(f"Budget exceeded: run took longer than {self.max_seconds}s.")
        if self.max_tokens:
            used = (state.get("scratchpad") or {}).get("total_tokens", 0)
            if used > self.max_tokens:
                raise AgentKitError(f"Budget exceeded: {used} > {self.max_tokens} tokens.")


# ---------------------------------------------------------------------------
class RedactionMiddleware(Middleware):
    """Scrub sensitive patterns out of message content.

    Runs on `after_node`, so it catches data *before* it lands in state — which
    means before it is checkpointed to disk, before it is traced, and before it
    is fed back to the model on the next turn.

    The default patterns are illustrative, not a compliance solution. Real PII
    detection is a hard problem; wire in a proper classifier if you need one.
    """

    DEFAULT_PATTERNS = {
        "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
        "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "api_key": r"\b(sk|pk|api|key)[-_][A-Za-z0-9]{16,}\b",
    }

    def __init__(self, patterns: dict[str, str] | None = None, placeholder: str = "[REDACTED:{}]"):
        self.patterns = {k: re.compile(v) for k, v in (patterns or self.DEFAULT_PATTERNS).items()}
        self.placeholder = placeholder

    def _clean(self, text: str) -> str:
        for label, rx in self.patterns.items():
            text = rx.sub(self.placeholder.format(label), text)
        return text

    def after_node(self, node, state, update, config):
        msgs = update.get("messages")
        if not msgs:
            return None
        msgs = msgs if isinstance(msgs, list) else [msgs]
        for m in msgs:
            if isinstance(m, Message) and m.content:
                cleaned = self._clean(m.content)
                if cleaned != m.content:
                    m.content = cleaned
                    m.metadata["redacted"] = True
        return {**update, "messages": msgs}


# ---------------------------------------------------------------------------
class GuardrailMiddleware(Middleware):
    """Run arbitrary validators over each node's output.

    A validator takes the update and returns None (pass) or a string
    describing the problem. `on_violation` decides the consequence:

      "raise"  -> abort the run (use for policy violations)
      "log"    -> record and continue (use while tuning a new rule)
      "annotate" -> push the complaint into `errors` state so the agent itself
                    can see it on the next turn and self-correct

    "annotate" is the interesting one: it turns a guardrail from a wall into a
    feedback signal.
    """

    def __init__(
        self,
        validators: Iterable[Callable[[dict[str, Any]], str | None]],
        on_violation: str = "raise",
    ):
        self.validators = list(validators)
        self.on_violation = on_violation

    def after_node(self, node, state, update, config):
        problems = [msg for v in self.validators if (msg := v(update))]
        if not problems:
            return None
        detail = f"Guardrail violation in node {node!r}: " + "; ".join(problems)
        if self.on_violation == "raise":
            raise AgentKitError(detail)
        if self.on_violation == "log":
            logger.warning(detail)
            return None
        return {**update, "errors": problems}


# ---------------------------------------------------------------------------
class UsageTrackingMiddleware(Middleware):
    """Accumulate token usage into `scratchpad` so BudgetMiddleware and your
    dashboards have a single number to read.

    Depends on the model node stashing `usage` in the assistant message's
    metadata — which `prebuilt.react` does.
    """

    def after_node(self, node, state, update, config):
        msgs = update.get("messages") or []
        msgs = msgs if isinstance(msgs, list) else [msgs]
        delta = 0
        for m in msgs:
            u = getattr(m, "metadata", {}).get("usage")
            if u:
                delta += u.get("input_tokens", 0) + u.get("output_tokens", 0)
        if not delta:
            return None
        pad = state.get("scratchpad") or {}
        return {**update, "scratchpad": {"total_tokens": pad.get("total_tokens", 0) + delta}}
