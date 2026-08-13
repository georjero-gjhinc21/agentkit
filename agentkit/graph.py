"""
agentkit.graph
==============

The execution engine. This is the heart of the framework.

WHY A GRAPH AND NOT A CHAIN OF FUNCTION CALLS?
----------------------------------------------
The first thing anyone builds is a while-loop:

    while True:
        reply = model(messages)
        if not reply.tool_calls:
            break
        messages += run_tools(reply.tool_calls)

That works, and then reality arrives. You need to pause for human approval.
You need to resume a run three hours later on a different machine. You need
to retry one step without replaying the other twelve. You need two branches to
run in parallel. You need to see, in production, exactly which step went
wrong. A while-loop gives you none of that, because its state lives on the
Python call stack, and a call stack cannot be paused, serialised, inspected or
resumed.

So: make the control flow *data*. Nodes are steps, edges are transitions,
state is an explicit dict. Once the loop is a graph, all of the above becomes
possible, because the engine can stop between any two nodes and write the
state down.

THE EXECUTION MODEL — SUPERSTEPS
--------------------------------
Execution is bulk-synchronous (the Pregel model, same as LangGraph):

    1. There is a *frontier*: the set of nodes to run right now.
    2. All frontier nodes run against the SAME immutable snapshot of state.
    3. Their updates are collected, then applied together via the reducers.
    4. Edges are evaluated to compute the next frontier.
    5. Repeat until the frontier is empty or only contains END.

Step 2 is the subtle one. Because every node in a superstep sees the same
input state, two parallel nodes cannot see each other's half-finished work,
and results do not depend on which finished first. Combined with reducers,
that is what makes parallel branches deterministic.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence

from .errors import ConfigurationError, GraphError, RecursionLimitError
from .state import StateSchema, default_agent_state
from .types import RunConfig

# Sentinel node names. START is where execution begins; END terminates a path.
START = "__start__"
END = "__end__"

# A node receives (state, config) and returns a partial state update, or None
# to mean "I changed nothing".
NodeFn = Callable[[dict[str, Any], RunConfig], dict[str, Any] | None]

# A router receives (state, config) and returns the next node name, or a list
# of names to fan out to.
RouterFn = Callable[[dict[str, Any], RunConfig], str | list[str]]


@dataclass
class Node:
    """A unit of work in the graph."""

    name: str
    fn: NodeFn
    #: Retries for transient failures inside this node specifically.
    retries: int = 0
    retry_delay: float = 0.5
    #: Human-readable purpose, surfaced in `graph.describe()` and traces.
    description: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class ConditionalEdge:
    """A branch. `router` decides at runtime which node(s) come next.

    `path_map` translates the router's return value into node names. It exists
    for two reasons: routers can then return domain-meaningful strings
    ("needs_tools", "finished") instead of node names, and the set of possible
    destinations becomes statically known — which is what lets us validate the
    graph at compile time and draw it.
    """

    source: str
    router: RouterFn
    path_map: dict[str, str] | None = None

    def destinations(self) -> list[str]:
        return list(self.path_map.values()) if self.path_map else []

    def resolve(self, state: dict[str, Any], config: RunConfig) -> list[str]:
        raw = self.router(state, config)
        keys = raw if isinstance(raw, list) else [raw]
        out: list[str] = []
        for k in keys:
            if self.path_map is not None:
                if k not in self.path_map:
                    raise GraphError(
                        f"Router on node {self.source!r} returned {k!r}, which is not in "
                        f"path_map keys {sorted(self.path_map)}."
                    )
                out.append(self.path_map[k])
            else:
                out.append(k)  # router returned node names directly
        return out


@dataclass
class StepEvent:
    """One superstep's worth of observability, yielded by `stream()`.

    This is the raw material for tracing, debugging UIs and evaluation: what
    ran, what it changed, how long it took, and the resulting state.
    """

    step: int
    nodes: list[str]
    updates: dict[str, dict[str, Any]]  # node name -> its partial update
    state: dict[str, Any]               # state AFTER merging
    duration_ms: float
    interrupted: bool = False


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
class StateGraph:
    """Declarative graph builder.

        g = StateGraph(default_agent_state())
        g.add_node("model", call_model)
        g.add_node("tools", run_tools)
        g.set_entry_point("model")
        g.add_conditional_edges("model", should_continue,
                                {"tools": "tools", "done": END})
        g.add_edge("tools", "model")     # the loop
        app = g.compile()

    Build-then-compile is deliberate. `compile()` validates the whole
    structure once, so mistakes surface at startup rather than halfway through
    a customer's request.
    """

    def __init__(self, schema: StateSchema | None = None):
        self.schema = schema or default_agent_state()
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, list[str]] = {}                 # static fan-out
        self.conditional: dict[str, ConditionalEdge] = {}     # dynamic branch
        self.entry: str | None = None

    # -- construction --------------------------------------------------------
    def add_node(
        self,
        name: str,
        fn: NodeFn,
        *,
        retries: int = 0,
        retry_delay: float = 0.5,
        description: str = "",
        tags: list[str] | None = None,
    ) -> "StateGraph":
        if name in (START, END):
            raise GraphError(f"{name!r} is a reserved node name.")
        if name in self.nodes:
            raise GraphError(f"Node {name!r} already exists.")
        if not callable(fn):
            raise GraphError(f"Node {name!r} must be callable.")
        self.nodes[name] = Node(name, fn, retries, retry_delay, description, list(tags or []))
        return self

    def add_edge(self, source: str, target: str) -> "StateGraph":
        """Unconditional transition. Adding several from one source fans out
        into parallel branches within the next superstep."""
        self.edges.setdefault(source, []).append(target)
        return self

    def add_conditional_edges(
        self,
        source: str,
        router: RouterFn,
        path_map: dict[str, str] | None = None,
    ) -> "StateGraph":
        if source in self.conditional:
            raise GraphError(f"Node {source!r} already has conditional edges.")
        self.conditional[source] = ConditionalEdge(source, router, path_map)
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        self.entry = name
        return self.add_edge(START, name)

    def set_finish_point(self, name: str) -> "StateGraph":
        return self.add_edge(name, END)

    # -- validation ----------------------------------------------------------
    def _validate(self) -> None:
        if not self.nodes:
            raise GraphError("Graph has no nodes.")
        if self.entry is None:
            raise GraphError("No entry point. Call set_entry_point(...).")

        known = set(self.nodes) | {START, END}

        for src, targets in self.edges.items():
            if src not in known:
                raise GraphError(f"Edge from unknown node {src!r}.")
            for t in targets:
                if t not in known:
                    raise GraphError(f"Edge {src!r} -> unknown node {t!r}.")

        for src, ce in self.conditional.items():
            if src not in known:
                raise GraphError(f"Conditional edge from unknown node {src!r}.")
            for t in ce.destinations():
                if t not in known:
                    raise GraphError(f"Conditional edge {src!r} -> unknown node {t!r}.")

        # A node with no outgoing edge at all is almost always a forgotten
        # `add_edge(name, END)`. Catch it now rather than as a silent halt.
        for name in self.nodes:
            if name not in self.edges and name not in self.conditional:
                raise GraphError(
                    f"Node {name!r} has no outgoing edges. Add one, or "
                    f"add_edge({name!r}, END) if it is terminal."
                )

        # Warn-by-erroring if END is unreachable — an agent that can never stop
        # is a bug you want to know about before it burns tokens.
        reachable_end = any(END in t for t in self.edges.values()) or any(
            END in ce.destinations() for ce in self.conditional.values()
        )
        has_bare_router = any(ce.path_map is None for ce in self.conditional.values())
        if not reachable_end and not has_bare_router:
            raise GraphError(
                "No path reaches END. The graph would run until the recursion "
                "limit. Add an edge to END or a router branch that returns it."
            )

    def compile(
        self,
        *,
        checkpointer: Any | None = None,
        interrupt_before: Sequence[str] = (),
        interrupt_after: Sequence[str] = (),
        parallel: bool = True,
        middleware: Sequence[Any] = (),
        reset_on_new_turn: Sequence[str] = (),
    ) -> "CompiledGraph":
        """Freeze and validate the graph into something runnable."""
        self._validate()
        for n in list(interrupt_before) + list(interrupt_after):
            if n not in self.nodes:
                raise ConfigurationError(f"Cannot interrupt at unknown node {n!r}.")
        return CompiledGraph(
            self,
            checkpointer=checkpointer,
            interrupt_before=set(interrupt_before),
            interrupt_after=set(interrupt_after),
            parallel=parallel,
            middleware=list(middleware),
            reset_on_new_turn=reset_on_new_turn,
        )

    # -- introspection -------------------------------------------------------
    def to_mermaid(self) -> str:
        """Render as a Mermaid flowchart.

        Being able to *see* the graph is not a nicety. Paste the output into
        any Markdown renderer and control-flow bugs become obvious in seconds.
        """
        lines = ["flowchart TD", f'    {START}(["START"])', f'    {END}(["END"])']
        for name, node in self.nodes.items():
            label = f"{name}<br/><i>{node.description}</i>" if node.description else name
            lines.append(f'    {name}["{label}"]')
        for src, targets in self.edges.items():
            for t in targets:
                lines.append(f"    {src} --> {t}")
        for src, ce in self.conditional.items():
            if ce.path_map:
                for key, t in ce.path_map.items():
                    lines.append(f"    {src} -.->|{key}| {t}")
            else:
                lines.append(f"    {src} -.-> ?")
        return "\n".join(lines)

    def describe(self) -> str:
        """Plain-text summary for logs and `--help`-style output."""
        out = [f"StateGraph: {len(self.nodes)} nodes, entry={self.entry}"]
        out.append(f"  state channels: {', '.join(sorted(self.schema.channels))}")
        for name, node in self.nodes.items():
            targets = self.edges.get(name, [])
            if name in self.conditional:
                targets = targets + [f"?{d}" for d in self.conditional[name].destinations()]
            out.append(f"  - {name}: -> {', '.join(targets) or '(dynamic)'}"
                       + (f"   # {node.description}" if node.description else ""))
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
class CompiledGraph:
    """A validated, executable graph. Treat instances as immutable and reuse
    them across requests — all per-run data lives in `state` and `RunConfig`,
    never on `self`, which is what makes a single compiled graph safe to share
    between threads."""

    def __init__(
        self,
        builder: StateGraph,
        *,
        checkpointer: Any | None = None,
        interrupt_before: set[str] | None = None,
        interrupt_after: set[str] | None = None,
        parallel: bool = True,
        middleware: list[Any] | None = None,
        reset_on_new_turn: Sequence[str] = (),
    ):
        self.builder = builder
        self.schema = builder.schema
        self.checkpointer = checkpointer
        self.interrupt_before = interrupt_before or set()
        self.interrupt_after = interrupt_after or set()
        self.parallel = parallel
        self.middleware = middleware or []
        #: Channels reset when a finished thread receives new input. Use for
        #: per-turn counters that should not accumulate across a conversation.
        self.reset_on_new_turn = tuple(reset_on_new_turn)

    # -- helpers -------------------------------------------------------------
    def _next_nodes(self, node: str, state: dict[str, Any], config: RunConfig) -> list[str]:
        """Resolve outgoing edges for one node. Static and conditional edges
        both apply if both are declared."""
        out: list[str] = list(self.builder.edges.get(node, []))
        ce = self.builder.conditional.get(node)
        if ce is not None:
            out.extend(ce.resolve(state, config))
        return out

    def _run_node(self, node: Node, state: dict[str, Any], config: RunConfig) -> dict[str, Any]:
        """Execute a single node with its retry policy.

        Retries live here rather than inside node bodies so that every node
        gets the behaviour for free and so the trace can record attempt counts.
        """
        attempts = node.retries + 1
        last: BaseException | None = None
        for i in range(attempts):
            try:
                for mw in self.middleware:
                    hook = getattr(mw, "before_node", None)
                    if hook:
                        hook(node.name, state, config)

                update = node.fn(state, config) or {}

                for mw in reversed(self.middleware):
                    hook = getattr(mw, "after_node", None)
                    if hook:
                        update = hook(node.name, state, update, config) or update
                return update
            except Exception as exc:  # noqa: BLE001
                last = exc
                for mw in self.middleware:
                    hook = getattr(mw, "on_error", None)
                    if hook:
                        hook(node.name, exc, state, config)
                if i < attempts - 1:
                    time.sleep(node.retry_delay * (2**i))
        raise GraphError(f"Node {node.name!r} failed after {attempts} attempt(s): {last}") from last

    # -- streaming execution -------------------------------------------------
    def stream(
        self,
        input: dict[str, Any] | None = None,
        config: RunConfig | None = None,
    ) -> Iterator[StepEvent]:
        """Run the graph, yielding a `StepEvent` after every superstep.

        `stream` is the primitive; `invoke` is a thin wrapper that drains it.
        Writing it this way means observability is not bolted on — anything
        that wants progress (a UI, a tracer, a test) consumes the same events
        the engine already produces.
        """
        config = config or RunConfig()

        # -- resume-or-start ---------------------------------------------
        # If a checkpointer holds state for this thread_id, continue from it.
        # This one branch is what turns a stateless function into a durable,
        # multi-turn, human-interruptible agent.
        state: dict[str, Any] | None = None
        frontier: list[str] = []
        # True when the previous call ended at an `interrupt_before` pause.
        # Without this flag, resuming would hit the same interrupt again and
        # the run could never move past it — the approval would never "take".
        # It applies to the FIRST superstep only, then clears.
        resumed_from_interrupt = False
        if self.checkpointer is not None:
            saved = self.checkpointer.get(config.thread_id)
            if saved:
                state = self.schema.apply(saved["state"], input)
                # A saved frontier of just [END] means the previous run finished.
                # Strip it so the "new turn" branch below fires.
                frontier = [n for n in (saved.get("next") or []) if n != END]
                resumed_from_interrupt = bool(saved.get("interrupted"))

                if not frontier:
                    # The saved run finished. New input on the same thread is a
                    # NEW TURN of the same conversation: history is kept, but we
                    # re-enter at the entry point and reset per-turn counters so
                    # (for example) `max_iterations` measures this turn's work
                    # rather than the whole conversation's.
                    for key in self.reset_on_new_turn:
                        if key in self.schema.channels:
                            state[key] = self.schema.channels[key].initial()
                    frontier = self._next_nodes(START, state, config)

        if state is None:
            state = self.schema.initial_state(**(input or {}))
            frontier = self._next_nodes(START, state, config)

        for mw in self.middleware:
            hook = getattr(mw, "on_start", None)
            if hook:
                hook(state, config)

        step = 0
        path: list[str] = []

        while frontier:
            # Reaching END on every branch means we are finished.
            frontier = [n for n in frontier if n != END]
            if not frontier:
                break

            step += 1
            if step > config.recursion_limit:
                raise RecursionLimitError(config.recursion_limit, path)

            # -- interrupt BEFORE ----------------------------------------
            # Save and hand control back to the caller. The pending frontier
            # is persisted so a later `stream()` with the same thread_id picks
            # up exactly here. This is the human-in-the-loop mechanism.
            pending = [] if resumed_from_interrupt else [n for n in frontier if n in self.interrupt_before]
            resumed_from_interrupt = False  # consumed; later steps pause normally
            if pending:
                self._save(config, state, frontier, interrupted=True)
                yield StepEvent(step, pending, {}, state, 0.0, interrupted=True)
                return

            # -- run the superstep ---------------------------------------
            started = time.perf_counter()
            to_run = [self.builder.nodes[n] for n in frontier if n in self.builder.nodes]
            unknown = [n for n in frontier if n not in self.builder.nodes]
            if unknown:
                raise GraphError(f"Routed to unknown node(s): {unknown}")

            if self.parallel and len(to_run) > 1:
                # Every node sees the same pre-superstep snapshot, so ordering
                # of completion cannot affect the result.
                with ThreadPoolExecutor(max_workers=len(to_run)) as pool:
                    futures = {pool.submit(self._run_node, n, state, config): n for n in to_run}
                    updates = {futures[f].name: f.result() for f in futures}
            else:
                updates = {n.name: self._run_node(n, state, config) for n in to_run}

            # -- merge -----------------------------------------------------
            # Deterministic order (sorted by node name) so two parallel
            # branches writing the same channel always merge the same way.
            for name in sorted(updates):
                state = self.schema.apply(state, updates[name])

            path.extend(sorted(updates))
            duration = round((time.perf_counter() - started) * 1000, 2)

            # -- compute the next frontier --------------------------------
            # `done` is a universal escape hatch any node can set.
            if state.get("done"):
                next_frontier: list[str] = []
            else:
                next_frontier = []
                for name in sorted(updates):
                    for nxt in self._next_nodes(name, state, config):
                        if nxt not in next_frontier:
                            next_frontier.append(nxt)

            self._save(config, state, next_frontier)
            yield StepEvent(step, sorted(updates), updates, state, duration)

            # -- interrupt AFTER -------------------------------------------
            if any(n in self.interrupt_after for n in updates):
                yield StepEvent(step, sorted(updates), {}, state, 0.0, interrupted=True)
                return

            frontier = next_frontier

        for mw in self.middleware:
            hook = getattr(mw, "on_end", None)
            if hook:
                hook(state, config)

    def invoke(
        self,
        input: dict[str, Any] | None = None,
        config: RunConfig | None = None,
    ) -> dict[str, Any]:
        """Run to completion (or to an interrupt) and return the final state."""
        final: dict[str, Any] = self.schema.initial_state(**(input or {}))
        for event in self.stream(input, config):
            final = event.state
        return final

    # -- persistence ---------------------------------------------------------
    def _save(
        self,
        config: RunConfig,
        state: dict[str, Any],
        next_nodes: list[str],
        interrupted: bool = False,
    ) -> None:
        if self.checkpointer is not None:
            self.checkpointer.put(
                config.thread_id,
                {"state": state, "next": next_nodes, "interrupted": interrupted},
            )

    def get_state(self, thread_id: str) -> dict[str, Any] | None:
        """Inspect a paused or finished run without resuming it."""
        if self.checkpointer is None:
            return None
        saved = self.checkpointer.get(thread_id)
        return saved["state"] if saved else None

    def update_state(self, thread_id: str, update: dict[str, Any]) -> dict[str, Any]:
        """Edit a paused run's state before resuming.

        This is how a human approves, corrects, or overrides an agent mid-run:
        pause with `interrupt_before`, patch the state here, then call
        `stream()`/`invoke()` again with the same thread_id.
        """
        if self.checkpointer is None:
            raise ConfigurationError("update_state requires a checkpointer.")
        saved = self.checkpointer.get(thread_id)
        if not saved:
            raise GraphError(f"No saved state for thread {thread_id!r}.")
        saved["state"] = self.schema.apply(saved["state"], update)
        self.checkpointer.put(thread_id, saved)
        return saved["state"]

    def to_mermaid(self) -> str:
        return self.builder.to_mermaid()
