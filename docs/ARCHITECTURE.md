# Architecture

Why the pieces are shaped the way they are. Read `README.md` first for the
tour; this file is the reasoning behind it.

---

## The two halves

The package contains two execution models, and knowing which one you need is
the most important decision you will make with it.

```
CHAINS (agentkit.runnables)        GRAPHS (agentkit.graph)
─────────────────────────          ───────────────────────
linear, acyclic                    cyclic
stateless (pure function)          stateful (explicit state dict)
composed with |                    composed with nodes and edges
streaming/batch/async built in     pausable, resumable, checkpointed
no loop, no branching back         loops, human-in-the-loop, parallel joins

prompt | model | parser            model ⇄ tools until done
```

**Use a chain** when the flow is known in advance: prompt → model → parse.
RAG question answering, classification, extraction, summarisation. Most
"AI features" are chains, and people reach for agents far too early.

**Use a graph** when the flow depends on what the model decides at runtime, or
when you need to stop mid-flight. Tool-calling agents, approval workflows,
revise-until-good-enough loops, multi-agent systems.

They compose: a node inside a graph can invoke a chain, and a chain stage can
invoke a compiled graph. Neither knows about the other's internals.

---

## Layers

```
┌───────────────────────────────────────────────────────────┐
│  prebuilt/       create_agent, create_supervisor          │
├───────────────────────────────────────────────────────────┤
│  graph.py        StateGraph, CompiledGraph, supersteps    │
│  runnables.py    Runnable, |, parallel, branch, history   │
├───────────────────────────────────────────────────────────┤
│  prompts.py  parsers.py  rag.py  tools.py  memory.py      │
├───────────────────────────────────────────────────────────┤
│  models.py       provider adapters (the vendor boundary)  │
├───────────────────────────────────────────────────────────┤
│  types.py        Message, ToolCall, Usage, RunConfig      │
└───────────────────────────────────────────────────────────┘
     middleware.py / tracing.py cut across every layer
```

Dependencies point downward only. `types.py` imports nothing from the package;
`models.py` imports only `types` and `errors`. That is what keeps the vendor
boundary a boundary.

---

## Decision 1: a neutral message format

Every provider disagrees about the wire format. Anthropic puts the system
prompt in a top-level parameter and tool calls in `tool_use` content blocks;
OpenAI puts the system prompt in the message list and tool calls in a
`tool_calls` array with JSON-string arguments; tool *results* go in a user
message for one and a `tool`-role message for the other.

If those differences reach your agent logic, you have chosen a vendor
permanently. So there is one internal `Message`, and translation happens in
exactly two functions per adapter (`_convert` in and the response mapping out).

**Cost:** a small translation layer, and features unique to one provider need
either a `metadata` entry or an adapter-specific keyword argument.

**Benefit:** switching providers is a one-line change, and — more valuable in
practice — you can run the *same* agent against a cheap model in CI and an
expensive one in production, or fall back between providers at runtime
(`RunnableFallback`).

---

## Decision 2: reducers on state

The alternative is dict-update semantics, where a node returning
`{"messages": [x]}` replaces the history. Everyone who has done this has then
written `{"messages": state["messages"] + [x]}` in every node, and eventually
forgotten it in one.

Reducers move that decision to the schema, declared once:

```python
"messages": Channel(add_messages, list)   # append, upsert by id
"steps":    Channel(add_int, 0)           # accumulate
"draft":    Channel()                     # overwrite
```

The deeper reason is parallelism. Two nodes in the same superstep both return
`{"findings": [...]}`. With overwrite semantics the result depends on which
finished first — a race. With a declared `append` reducer it is deterministic.
You cannot have safe parallel branches without something like this.

`add_messages` upserts by id rather than blindly appending, which is what lets
a human edit a pending message and write it back (`update_state`) without
creating a duplicate.

---

## Decision 3: bulk-synchronous supersteps

Execution proceeds in rounds:

1. A *frontier* of nodes is scheduled.
2. All of them run against the **same immutable snapshot** of state.
3. Their updates are merged together through the reducers.
4. Edges are evaluated to compute the next frontier.

Step 2 is the whole design. Because no node can observe another node's
half-finished work within a superstep, results do not depend on scheduling.
The consequence worth internalising: **a node cannot see the output of a node
running beside it.** If B needs A's result, B goes downstream of A, not
alongside it.

Updates are merged in sorted node-name order so that even reducers which are
not commutative produce a stable result.

---

## Decision 4: checkpoint the frontier, not just the state

A checkpoint stores `{"state": ..., "next": [node names], "interrupted": bool}`.

Storing the pending frontier is what makes a resumed run continue *mid-graph*
rather than restarting at the entry point. Storing the `interrupted` flag is
what stops a resumed run from immediately re-triggering the interrupt it just
returned from — a real bug this codebase had, now covered by
`test_interrupt_and_resume`.

An empty (or END-only) frontier means the run finished. New input on that same
thread is therefore a **new turn**: history is preserved, execution re-enters
at the entry point, and the channels named in `reset_on_new_turn` are cleared
so per-turn counters like `steps` do not accumulate across a conversation.

---

## Decision 5: one interface, four methods

`Runnable` requires `invoke` and provides working defaults for `stream`,
`batch` and `ainvoke`. Composition (`|`) returns another Runnable.

The payoff is not terseness. It is that adding a stage anywhere in a pipeline —
at any nesting depth — does not require re-implementing streaming, batching or
async for that pipeline. Uniformity compounds; syntax does not.

One honest limitation: `RunnableSequence.stream` runs every stage but the last
eagerly, then streams the last. You cannot stream through a stage that needs
its complete input, and a parser cannot parse half a JSON object.

---

## Decision 6: middleware instead of hooks in nodes

Logging, budget caps, redaction, guardrails and tracing all need to observe
every step while belonging to no particular step. Implemented inside nodes,
they are five lines copy-pasted everywhere and forgotten once.

Hooks are looked up with `getattr`, so middleware objects implement only what
they need and absent hooks cost nothing. `before_node` runs outside-in,
`after_node` inside-out — an onion, like ASGI — so the outermost middleware
sees the request first and the response last.

`after_node` may return a **modified** update. That makes it a real intercept
point rather than a listener: `RedactionMiddleware` scrubs content *before* it
enters state, which means before it is checkpointed to disk, before it is
traced, and before it is sent back to the model next turn.

---

## Decision 7: fail at compile time

`StateGraph.compile()` rejects:

- edges pointing at nodes that do not exist
- nodes with no outgoing edge (a forgotten `add_edge(name, END)`)
- graphs where no path reaches `END`

And `StateSchema` rejects writes to undeclared keys.

These are all mistakes that otherwise surface as an agent that silently halts,
or loops to the recursion limit, or ignores a state write for three weeks
because of one transposed letter. Startup is a much cheaper place to find them
than production.

---

## What is deliberately missing

**Async throughout.** `ainvoke` exists on Runnables; the graph engine is
synchronous with a thread pool for parallel nodes. For LLM workloads the
bottleneck is network latency and threads handle that fine. A fully async
engine would roughly double the surface area of `graph.py` for a benefit most
users would never measure.

**Real embeddings.** `HashingEmbeddings` is lexical, not semantic — it matches
"refund" to "refunds" and misses "money back guarantee" entirely. It exists so
the RAG pipeline runs offline in tests and examples. Ship real embeddings.

**A hosted trace/eval platform.** `tracing.py` produces a span tree and can
export JSONL. Wiring it to LangSmith or OpenTelemetry is one `export()` method.

**Streaming from the graph.** `CompiledGraph.stream` yields one event per
superstep, not per token. Token streaming lives at the model/chain layer.

**Retries with circuit breaking, rate limiters, caches.** Each is a natural
middleware or Runnable wrapper. They are left out because the right policy is
workload-specific and a wrong default is worse than none.
