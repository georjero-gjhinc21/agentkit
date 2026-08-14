# agentkit

A small, readable, provider-agnostic framework for building **any kind of agent** —
chat pipelines, RAG chatbots, tool-callers, human-in-the-loop approvals,
multi-agent teams.

Plus the loop for getting one into a business and proving it works: audit,
evals, staged rollout, impact in dollars.

Zero required dependencies. Runs offline. 98 tests, 7 runnable examples.

```python
from agentkit import create_agent, tool, AnthropicModel, Message

@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    return {"city": city, "temp_c": 24}

agent = create_agent(
    model=AnthropicModel("claude-sonnet-4-5"),
    tools=[get_weather],
    system_prompt="You are a concise assistant.",
)

result = agent.invoke({"messages": [Message.user("Weather in Tokyo?")]})
print(result["messages"][-1].content)
```

```python
# ...or compose a pipeline with the pipe operator
chain = (
    ChatPromptTemplate.from_template("Summarise: {text}", system="Be terse.")
    | model
    | StrOutputParser()
)
chain.batch([{"text": a}, {"text": b}])     # concurrent, order preserved
```

---

## Two shapes, and knowing which you need

```
CHAINS (runnables.py)              GRAPHS (graph.py)
─────────────────────              ─────────────────
linear, stateless                  cyclic, stateful
prompt | model | parser            model ⇄ tools until done
streaming/batch/async free         pausable, resumable, checkpointed
```

**Chains** for flows known in advance: RAG, classification, extraction,
summarisation. Most "AI features" are chains, and people reach for agents far
too early — an agent given a job a chain could do is slower, costlier, harder
to debug, and can fail in ways a chain cannot.

**Graphs** when the flow depends on what the model decides at runtime, or when
you need to stop mid-flight for a human.

They nest: a graph node can invoke a chain, a chain stage can invoke a graph.

And a third concern that is neither: **getting it deployed.** Building an agent
is the easy half. `workflow.py`, `evals.py` and `deployment.py` cover the audit
→ evals → rollout loop — see [`docs/FDE_PLAYBOOK.md`](docs/FDE_PLAYBOOK.md).

---

## Why this exists

The first agent anyone builds is a while-loop:

```python
while True:
    reply = model(messages)
    if not reply.tool_calls:
        break
    messages += run_tools(reply.tool_calls)
```

It works, and then reality arrives. You need to **pause for human approval**.
You need to **resume** a run three hours later on a different machine. You need to
**retry one step** without replaying the other twelve. You need two things to run
**in parallel**. You need to know, in production, **which step went wrong**.

A while-loop gives you none of that, because its state lives on the Python call
stack — and a call stack cannot be paused, serialised, inspected or resumed.

So: make control flow **data**. Nodes are steps, edges are transitions, state is an
explicit dict with declared merge semantics. That single change is what buys you
durability, parallelism, human-in-the-loop, and observability.

This repo is a compact, heavily commented implementation of that idea — the same
architecture LangChain's LangGraph popularised, rebuilt small enough to read in an
afternoon and modify without fear.

---

## Architecture

### System Layers

```
┌─────────────────────────────────────────────────────────────┐
│  APPLICATIONS                                               │
│  prebuilt/       create_agent, create_supervisor, ReAct     │
├─────────────────────────────────────────────────────────────┤
│  ORCHESTRATION                                              │
│  graph.py        StateGraph, supersteps, checkpoints        │
│  runnables.py    Chains, | composition, batch, streaming    │
├─────────────────────────────────────────────────────────────┤
│  COMPONENTS                                                 │
│  prompts.py  parsers.py  rag.py  tools.py  memory.py        │
│  workflow.py  evals.py  deployment.py  bitwarden.py         │
├─────────────────────────────────────────────────────────────┤
│  PROVIDERS (vendor boundary)                                │
│  models.py       AnthropicModel, OpenAIModel, LiteLLMModel  │
├─────────────────────────────────────────────────────────────┤
│  FOUNDATION                                                 │
│  types.py        Message, ToolCall, Usage, ModelResponse    │
│  errors.py       Exception hierarchy                        │
└─────────────────────────────────────────────────────────────┘
     middleware.py / tracing.py cut across all layers
```

**Key principles:**
- Dependencies point **downward only** - no cycles
- Vendor abstraction at **one layer** - models.py is the boundary
- Everything above models.py is **provider-agnostic**
- Each layer **independently testable**

### Core Design Decisions

#### 1. Neutral Message Format

One internal `Message` type, adapters translate at the edge:

```python
# Internal (vendor-neutral)
Message.user("What's the weather?")
Message.assistant("Let me check", tool_calls=[...])
Message.tool("Weather data", tool_call_id="...")

# Adapters handle provider quirks:
# - Anthropic: system in top-level param, tool_use blocks
# - OpenAI: system in messages array, tool_calls with JSON strings
# - LiteLLM: OpenAI-compatible format for 100+ providers
```

**Benefit:** Swap providers in one line. Same agent, different model.

#### 2. State with Declared Reducers

Nodes return **partial updates**. State keys declare **how updates merge**:

```python
schema = StateSchema({
    "messages": Channel(add_messages, list),   # append (upsert by id)
    "findings": Channel(append, list),         # concatenate
    "quality":  Channel(),                     # overwrite (default)
    "steps":    Channel(add_int, 0),           # accumulate
})
```

**Why:** Enables deterministic parallel execution. Two branches both write `messages` → reducer defines the merge, not last-writer-wins.

#### 3. Control Flow as Data (Graphs)

```python
graph = StateGraph(schema)
graph.add_node("model", call_model)
graph.add_node("tools", run_tools)
graph.add_conditional_edges("model", should_continue, {
    "tools": "tools",
    "done": END
})
app = graph.compile(checkpointer=FileCheckpointer())
```

**Execution model:** Bulk-synchronous (Pregel)
- Supersteps run all nodes at current frontier
- Nodes read same state snapshot
- Updates merged, then next frontier computed
- Parallel branches are **deterministic**

**Benefits:**
- Pausable/resumable (state is data)
- Human-in-the-loop (interrupt before node)
- Observable (every step recorded)
- Testable (FakeModel + scripted responses)

#### 4. Composition via Runnables

Everything implements `invoke | stream | batch | ainvoke`:

```python
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)

# Streaming/batching/async work at any nesting depth
for chunk in chain.stream(input):
    print(chunk)
```

**Benefit:** Add stages anywhere without re-implementing the four methods.

#### 5. Cross-Cutting Concerns as Middleware

Logging, budgets, guardrails attach **from outside**:

```python
app = graph.compile(middleware=[
    ConsoleTracer(),                           # Observability
    BudgetMiddleware(max_tokens=100_000),      # Cost controls
    RedactionMiddleware(),                     # PII protection
    GuardrailMiddleware([...]),                # Safety checks
    AuditMiddleware(trail),                    # Compliance
])
```

**Benefit:** Cross-cutting logic doesn't pollute node code.

### Data Flow

```
User Input
    ↓
[Middleware: before_run]
    ↓
Graph Execution (supersteps)
  ├─ Read state snapshot
  ├─ Execute frontier nodes (parallel)
  ├─ [Middleware: before_node / after_node]
  ├─ Merge updates via reducers
  ├─ Compute next frontier
  └─ Repeat until END
    ↓
[Middleware: after_run]
    ↓
Final State (with full message history)
```

### Provider Integration

#### Model Abstraction

```python
class BaseChatModel:
    def invoke(self, messages, tools=None, **kw) -> ModelResponse:
        """Neutral in, neutral out. Provider details hidden."""
```

**Implementations:**
- `AnthropicModel` - Claude API
- `OpenAIModel` - OpenAI / vLLM / Ollama / any compatible
- `LiteLLMModel` - 100+ providers via LiteLLM
- `FakeModel` - Scripted responses for tests

**Adding a provider:** Implement one method. That's it.

#### Credential Management

```python
# Option 1: Environment variables
model = AnthropicModel()  # Reads ANTHROPIC_API_KEY

# Option 2: Bitwarden vault (secure)
from agentkit import get_secret
model = AnthropicModel(api_key=get_secret("ANTHROPIC_API_KEY"))

# Option 3: Direct (not recommended)
model = AnthropicModel(api_key="sk-...")
```

**Bitwarden integration:** Encrypted vault, cross-device sync, team sharing, audit logs.

### Testing Architecture

```python
# Agent code is testable because interesting logic
# is separate from non-deterministic network calls

model = FakeModel(responses=[
    Message.assistant("", tool_calls=[...]),
    Message.assistant("Final answer"),
])

agent = create_agent(model=model, tools=[...])
result = agent.invoke({"messages": [...]})

# Assert on behavior, inspect model.calls
assert result["messages"][-1].content == "Final answer"
assert len(model.calls) == 2
```

**Every routing decision, retry path, and termination condition becomes a unit test.**

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│  DEVELOPMENT                                            │
│  • FakeModel for offline testing                        │
│  • 98 passing tests                                     │
│  • Local Ollama for free iteration                      │
└─────────────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│  EVALUATION                                             │
│  • Golden datasets from production history              │
│  • Scorers: exact_match, contains, LLM judge            │
│  • Regression detection                                 │
│  • Critical case tracking                               │
└─────────────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│  STAGED ROLLOUT                                         │
│  1. SHADOW      → Agent runs, changes nothing           │
│  2. SUGGEST     → Output shown as suggestion            │
│  3. APPROVE     → Every action needs approval           │
│  4. AUTO+EXCEPT → High confidence auto, rest escalate   │
│  5. AUTONOMOUS  → Fully autonomous (audited)            │
└─────────────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│  PRODUCTION                                             │
│  • FileCheckpointer or DB for durability                │
│  • BudgetMiddleware for cost controls                   │
│  • GuardrailMiddleware for safety                       │
│  • AuditTrail for compliance                            │
│  • Impact metrics (cost/risk/revenue)                   │
└─────────────────────────────────────────────────────────┘
```

See [`docs/FDE_PLAYBOOK.md`](docs/FDE_PLAYBOOK.md) for the complete deployment methodology.

### Security Architecture

```
API Keys & Credentials
         ↓
    [Bitwarden Vault]
    (AES-256 encrypted)
         ↓
    BitwardenSecrets
         ↓
    Model Constructors
         ↓
    Agents & Chains
         ↓
    [Middleware Layer]
    ├─ RedactionMiddleware (strip PII)
    ├─ GuardrailMiddleware (safety checks)
    └─ AuditMiddleware (compliance logs)
         ↓
    Provider APIs
```

**Security features:**
- ✅ Encrypted credential storage (Bitwarden)
- ✅ PII redaction middleware
- ✅ Guardrails with "annotate" mode
- ✅ Audit trails (append-only JSONL)
- ✅ Budget limits (token/cost caps)
- ✅ No secrets in git (never .env files)

### Extension Points

```python
# New provider: Implement one method
class MyModel(BaseChatModel):
    def invoke(self, messages, tools=None, **kw) -> ModelResponse:
        ...

# New checkpointer: Implement three methods  
class PostgresCheckpointer(BaseCheckpointer):
    def get(self, thread_id): ...
    def put(self, thread_id, checkpoint): ...
    def delete(self, thread_id): ...

# New middleware: All hooks optional
class MyMiddleware(Middleware):
    def before_node(self, node, state, config): ...
    def after_node(self, node, state, update, config): ...

# New architecture: Copy a prebuilt, edit the graph
# (Don't add flags - fork the pattern)
```

For detailed architecture rationale, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Install

```bash
git clone <your-fork-url> agentkit && cd agentkit
pip install -e .

# Optional provider SDKs — imported lazily, only when you use them:
pip install anthropic     # AnthropicModel
pip install openai        # OpenAIModel (also any OpenAI-compatible server)
pip install litellm       # LiteLLMModel (100+ providers: OpenAI, Anthropic, Azure, Bedrock, Ollama, etc.)

# Optional: Bitwarden CLI for secure credential management
npm install -g @bitwarden/cli  # Store API keys in encrypted Bitwarden vault
```

Nothing is required to run the tests or the first four examples:

```bash
python tests/test_agentkit.py      # 24 tests: graph runtime
python tests/test_chains.py        # 34 tests: chains, prompts, parsers, RAG
python tests/test_fde.py           # 40 tests: audit, evals, deployment
python examples/01_minimal_agent.py
```

---

## The four ideas

Understand these and you understand the whole package.

### 1. Messages are neutral — `types.py`

Every provider has a different wire format. If those differences leak into your
agent logic, you are married to one vendor. So there is one internal `Message`
type, and adapters translate at the edge.

Swapping Anthropic → OpenAI → a local vLLM server is one line.

### 2. State is explicit; merges are declared — `state.py`

Nodes never call each other. They read state and return a **partial update**, and
each state key declares a **reducer** saying how updates combine:

```python
schema = StateSchema({
    "messages": Channel(add_messages, list),   # append (and upsert by id)
    "findings": Channel(append, list),         # concatenate
    "quality":  Channel(),                     # overwrite (default)
    "steps":    Channel(add_int, 0),           # accumulate
})
```

This is the part people skip and then spend a week debugging. If two parallel
branches both write `messages`, the reducer decides what happens — deterministically,
instead of last-writer-wins.

### 3. Control flow is a graph — `graph.py`

```python
g = StateGraph(schema)
g.add_node("model", call_model)
g.add_node("tools", run_tools)
g.set_entry_point("model")
g.add_conditional_edges("model", should_continue, {"tools": "tools", "done": END})
g.add_edge("tools", "model")
app = g.compile(checkpointer=FileCheckpointer())
```

Execution is **bulk-synchronous** (the Pregel model): all nodes in the current
frontier run against the *same* state snapshot, their updates are merged together,
then the next frontier is computed. That is why parallel branches are deterministic —
no node can see another's half-finished work.

`compile()` validates the whole structure up front: unknown edge targets, dangling
nodes, and graphs with no path to `END` all fail at startup rather than in
production.

### 4. Composition over one interface — `runnables.py`

Every stage implements `invoke` / `stream` / `batch` / `ainvoke`, and `A | B`
produces something that is itself a stage. So streaming, batching and async
work on the whole pipeline, at any nesting depth, for free:

```python
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | model | parser
```

The payoff is uniformity, not brevity. Adding a stage anywhere never means
re-implementing the four methods for that pipeline.

### 5. Cross-cutting concerns are middleware — `middleware.py`, `tracing.py`

Logging, spend caps, PII redaction, guardrails and tracing all need to see every
step but belong to no particular step. They attach from outside:

```python
app = g.compile(middleware=[
    ConsoleTracer(),
    BudgetMiddleware(max_steps=30, max_seconds=120, max_tokens=100_000),
    RedactionMiddleware(),
    GuardrailMiddleware([no_empty_answers], on_violation="annotate"),
])
```

---

## What you get

| Capability | Where | Notes |
|---|---|---|
| Prompt templates | `prompts.py` | Basic, chat (with role placeholders), few-shot; validated variables |
| LCEL-style chains | `runnables.py` | `\|` composition, parallel branches, routing, retries, provider fallback |
| Output parsers | `parsers.py` | JSON (fence/prose tolerant), structured w/ schema, list, boolean, self-correcting retry |
| RAG | `rag.py` | Recursive splitter, embeddings interface, vector store w/ metadata filtering, citation formatting |
| Conversational memory | `runnables.py` | Session-keyed history wrapper for chains |
| Tool calling from plain functions | `tools.py` | JSON Schema inferred from type hints + docstring |
| Provider adapters | `models.py` | Anthropic, OpenAI, LiteLLM (100+ providers), `FakeModel` for tests |
| Graph engine | `graph.py` | Supersteps, conditional edges, parallel fan-out, per-node retries |
| Durable state | `memory.py` | In-memory and atomic file checkpointers; implement 3 methods for your own |
| Long-term memory | `memory.py` | Namespaced store; swap the search for embeddings |
| Human-in-the-loop | `graph.py` | `interrupt_before` + `update_state` → approve, edit, or reject mid-run |
| Context management | `memory.py` | `trim_messages` (orphan-safe) and `summarize_and_trim` |
| Observability | `tracing.py` | Span tree, console renderer, JSONL export |
| Guardrails & budgets | `middleware.py` | Including "annotate" mode, which feeds violations back to the agent |
| Workflow audit | `workflow.py` | Map the real process; decide deterministic vs LLM vs human per step |
| Evals | `evals.py` | Golden datasets from history, scorers, LLM judge, critical-case tracking, regression detection |
| Audit trail | `deployment.py` | Append-only JSONL with reasoning — for the client, not for debugging |
| Staged rollout | `deployment.py` | Shadow mode, autonomy ladder, per-tool gates, promotion criteria |
| Impact measurement | `deployment.py` | Cost saved / risk reduced / revenue enabled, with honest denominators |
| ReAct agent | `prebuilt/react.py` | ~120 readable lines |
| Multi-agent supervisor | `prebuilt/supervisor.py` | Workers are compiled graphs, so it nests |

---

## Examples

Run in order; each one introduces exactly one new idea.

| File | Shows |
|---|---|
| `01_minimal_agent.py` | Tools, the agent loop, tracing. Offline. |
| `02_custom_workflow.py` | Custom state, **parallel branches**, a quality-gated revision loop. Offline. |
| `03_human_in_the_loop.py` | Pause → human edits the tool arguments → resume. Offline. |
| `04_multi_agent.py` | A supervisor routing to two specialist agents. Offline. |
| `05_production_agent.py` | Real provider + checkpoints + budget + redaction + guardrails. Needs an API key. |
| `06_rag_chatbot.py` | Templates, LCEL, batch, routing, RAG, session memory, structured output. Offline. |
| `07_fde_loop.py` | Audit an AP workflow, build evals, run shadow mode, measure impact, gate the rollout. Offline. |
| `08_litellm_multi_provider.py` | Use LiteLLM to access 100+ providers (OpenAI, Anthropic, Bedrock, Ollama, etc.). Offline demo. |
| `09_bitwarden_secrets.py` | Secure credential management with Bitwarden vault (encrypted API key storage). Needs bw CLI. |

---

## Testing agents

Agent code is normally untestable because the interesting logic is entangled with a
non-deterministic network call. `FakeModel` unties that knot:

```python
model = FakeModel(responses=[
    Message.assistant("", tool_calls=[ToolCall(name="multiply", args={"a": 6, "b": 7})]),
    Message.assistant("The answer is 42."),
])

out = create_agent(model=model, tools=[multiply]).invoke(
    {"messages": [Message.user("6 times 7?")]}
)
assert [m.role for m in out["messages"]] == ["user", "assistant", "tool", "assistant"]
assert model.calls[1]  # inspect exactly what the agent sent on the second call
```

Or pass `handler=` a function of the history for adaptive behaviour. Every routing
decision, retry path and termination condition becomes an ordinary unit test.

`tests/test_agentkit.py` is written to double as documentation of the framework's
contracts — read it alongside the source.

---

## Extending it

The framework is defined by five small interfaces. Implement one and everything
else keeps working:

```python
class BaseChatModel:       # a new provider — one method
    def invoke(self, messages, tools=None, **kw) -> ModelResponse: ...

class BaseCheckpointer:    # Postgres/Redis durability — three methods
    def get(self, thread_id): ...
    def put(self, thread_id, checkpoint): ...
    def delete(self, thread_id): ...

class BaseStore:           # vector-backed long-term memory — four methods
    def put(self, namespace, key, value): ...
    def get(self, namespace, key): ...
    def delete(self, namespace, key): ...
    def search(self, namespace, query="", limit=10): ...

class Middleware:          # any cross-cutting concern — all hooks optional
    def before_node(self, node, state, config): ...
    def after_node(self, node, state, update, config): ...

class BaseTracer(Middleware):   # ship spans to LangSmith / OTel / your own
    def export(self): ...
```

For new **architectures** (planner-executor, reflection, debate, RAG pipelines),
copy a file from `prebuilt/` and edit the graph. That is the intended workflow —
a prebuilt that grows twenty flags has become a framework of its own, which is
exactly what this package is trying not to be.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design rationale,
[`docs/PATTERNS.md`](docs/PATTERNS.md) for recipes, and
[`docs/FDE_PLAYBOOK.md`](docs/FDE_PLAYBOOK.md) for deploying into a real
business.

---

## Design decisions, stated plainly

**Why no dependencies?** So the graph engine can be read, audited and vendored
without pulling in a hundred transitive packages. Provider SDKs are imported lazily
inside `__init__`, so you only install what you use.

**Why strict state keys?** A typo like `{"mesages": [...]}` should fail on the first
run, not silently do nothing for three weeks. Pass `strict=False` if you disagree.

**Why does `compile()` refuse a graph with no path to END?** Because an agent that
cannot terminate is a bug, and it is cheaper to catch at startup than on a bill.

**Why are tool errors returned to the model instead of raised?** Models recover from
a clear error string remarkably well — they retry with corrected arguments or try
another approach. Set `tool_error_handling="raise"` in tests where you want the
opposite.

**Why is multi-agent discouraged in its own module's docstring?** Because it is
usually the wrong answer. It costs an extra model call per hop and much harder
debugging. It earns its keep past ~15–20 tools, or when specialists genuinely need
different models. Below that, one agent with good tools wins.

**Why is `HashingEmbeddings` the default?** So the RAG pipeline runs offline, in
CI, with no API key. It is lexical, not semantic — it matches "refund" to
"refunds" and misses "money back guarantee" entirely. Build the plumbing against
it; ship real embeddings.

**Why does `recommend_placement()` default to deterministic code?** Because most
steps do not want a model, and the expensive mistake in applied AI is not a bad
model — it is pointing one at a step that never needed it. Roughly 95% of
generative AI pilots fail to reach production, and the failures cluster around
workflows nobody observed before automating them.

**Why do parsers tolerate markdown fences and stray prose?** Because models are
*nearly* compliant, and the gap between "valid JSON" and "valid JSON in a code
fence after the word Sure!" is where a surprising share of production incidents
live. Be liberal about surface, strict about shape.

---

## Component map

If you arrived here from a LangChain walkthrough, this is where each concept
lives:

| Concept | Module | Notes |
|---|---|---|
| Vendor independence (`ChatOpenAI` → `ChatAnthropic`) | `models.py` | One neutral `Message`; adapters translate at the edge |
| Prompt templates | `prompts.py` | `PromptTemplate`, `ChatPromptTemplate`, `FewShotPromptTemplate` |
| LCEL (`prompt \| model \| parser`) | `runnables.py` | Plus parallel, routing, retry, fallback |
| Output parsers | `parsers.py` | Str, JSON, structured, list, boolean, retry |
| Memory / chat history | `runnables.py`, `memory.py` | Session history for chains; checkpointers for graphs |
| Embeddings + vector store | `rag.py` | Interface + offline default; swap in FAISS/Chroma/pgvector |
| Retrieval and RAG chains | `rag.py` | Splitter, retriever, `create_rag_chain` with citations |
| Tools | `tools.py` | Schema inferred from signature + docstring |
| Graph workflows (LangGraph) | `graph.py` | Supersteps, checkpoints, interrupts |
| Tracing (LangSmith) | `tracing.py` | Span tree; `export()` to ship anywhere |
| Evaluation (LangSmith) | `evals.py` | Datasets, scorers, reports, regression gates |

---

## Relationship to LangChain / LangGraph

The architecture here follows the ideas LangChain's ecosystem established — a
graph-structured agent runtime (LangGraph), a neutral message and tool abstraction
(LangChain), and the observe/evaluate/deploy lifecycle around it (LangSmith).

This is a from-scratch educational reimplementation of that shape, not a fork and
not a drop-in replacement. Use it to **understand** the model, to build on something
small you fully control, or as a starting point for an in-house framework. If you
need production breadth — hundreds of integrations, a hosted trace/eval platform,
managed deployment — use the real thing at [langchain.com](https://www.langchain.com).

---

## License

MIT. See `LICENSE`.
