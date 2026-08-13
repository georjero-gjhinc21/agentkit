# Patterns

Recipes for things people actually build. Each is short enough to paste and
edit.

---

## Choosing a shape

| You need | Use |
|---|---|
| Prompt → model → parse | a chain |
| Answer from your documents | a chain (RAG) |
| Model picks tools until done | `create_agent` |
| Human approves before a side effect | graph + checkpointer + `interrupt_before` |
| Fan out, then combine | graph with parallel edges |
| Retry until quality passes | graph with a conditional edge back |
| Route to specialists | `RunnableBranch` first; `create_supervisor` only if that fails |

Reach for the simplest one that fits. An agent given a job a chain could do is
slower, costlier and harder to debug, and it can fail in ways a chain cannot.

---

## Classification with structured output

```python
parser = StructuredOutputParser({
    "category": "one of: billing, technical, account",
    "urgency":  "one of: low, medium, high",
})

chain = (
    ChatPromptTemplate
        .from_messages([("system", "Classify the ticket.\n{format_instructions}"),
                        ("user", "{ticket}")])
        .partial(format_instructions=parser.get_format_instructions())
    | model
    | parser
)

chain.invoke({"ticket": "I was charged twice"})
# {"category": "billing", "urgency": "high"}
```

Generating the format instructions from the same dict that validates the
result is the point — prompt and check cannot drift apart.

For bulk work, `chain.batch([...])` runs concurrently and preserves order.

---

## RAG with citations

```python
chunks = RecursiveTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
store = InMemoryVectorStore(OpenAIEmbeddings())
store.add_documents(chunks)

chain = create_rag_chain(model, store.as_retriever(k=4, score_threshold=0.3))
chain.invoke("What is the refund window?")
```

Three things matter more than the vector store you pick:

1. **Chunk on semantic boundaries.** The splitter tries paragraphs, then
   lines, then sentences, before resorting to character counts.
2. **Set a relevance floor.** Four irrelevant chunks are worse than one good
   one — the model will try to use whatever you give it.
3. **Instruct grounding explicitly.** `RAG_SYSTEM_PROMPT` tells the model to
   say "I don't have that information" rather than filling the gap. Without
   that line you have built a confident liar.

**Multi-tenancy:** pass `filter={"tenant": tenant_id}` to the retriever. It
applies before scoring, so a filtered document can never leak — filtering
after retrieval silently returns fewer than `k` results and leaks the
existence of documents the user cannot see.

---

## Conversational RAG

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer from the context.\n\nContext:\n{context}"),
    MessagesPlaceholder("history"),
    ("user", "{input}"),
])

chain = (
    RunnablePassthrough.assign(
        context=RunnableLambda(lambda d: d["input"]) | retriever | format_documents
    )
    | prompt | model | StrOutputParser()
)

bot = RunnableWithMessageHistory(chain, InMemoryHistoryStore().get, input_key="input")
bot.invoke({"input": "Can I get a refund?"}, {"session_id": user_id})
```

Known weakness: retrieval uses the raw latest message, so a follow-up like
"and how long does that take?" retrieves on a query with no subject. The fix
is a query-rewriting stage that condenses history + question into a
standalone question before it reaches the retriever.

---

## Tool-calling agent

```python
@tool
def lookup_order(order_id: str) -> dict:
    """Look up an order by its ID. Use this before discussing any specific order."""
    return db.get(order_id)

agent = create_agent(model, tools=[lookup_order], system_prompt=SYSTEM, max_iterations=8)
agent.invoke({"messages": [Message.user("Where is order A-1234?")]})
```

Tool quality dominates agent quality — far more than prompt wording. A good
tool has a narrow job, an unambiguous description written *for the model*,
typed arguments, and a return value the model can act on. If the agent is
picking wrong, rewrite the descriptions before you touch the system prompt.

Past roughly 15–20 tools, selection accuracy degrades. Split by task with
`registry.subset([...])` or `registry.by_tag("read")` and give different nodes
different toolboxes.

---

## Human approval before side effects

```python
@tool(requires_approval=True, tags=["write"])
def issue_refund(order_id: str, amount: float) -> str:
    """Issue a refund. Irreversible."""
    ...

agent = create_agent(model, tools=[issue_refund],
                     checkpointer=FileCheckpointer(),
                     interrupt_before_tools=True)

cfg = RunConfig(thread_id=conversation_id)
list(agent.stream({"messages": [Message.user(request)]}, cfg))   # pauses

pending = agent.get_state(conversation_id)["messages"][-1]
call = pending.tool_calls[0]                                     # show to a human

# approve (optionally with edits)
call.args["amount"] = 49.00
agent.update_state(conversation_id, {"messages": [pending]})
list(agent.stream(None, cfg))                                    # resumes

# or reject
agent.update_state(conversation_id, {"messages": [
    Message.tool("Denied by reviewer: amount exceeds policy.",
                 tool_call_id=call.id, name=call.name)
]})
```

The two passes can be different processes, hours apart. All they share is the
`thread_id`.

---

## Retry until quality passes

```python
def evaluate(state, config):
    verdict = judge.invoke([
        Message.system("Score this draft 0-1 for accuracy and clarity. Reply with the number only."),
        Message.user(state["draft"]),
    ]).message.content
    return {"quality": float(verdict.strip()), "rounds": 1}

def good_enough(state, config):
    if state["quality"] >= 0.8:  return "done"
    if state["rounds"] >= 3:     return "done"   # ALWAYS have this second exit
    return "revise"

g.add_conditional_edges("evaluate", good_enough, {"revise": "write", "done": END})
```

The round cap is not optional. A quality-only exit is how an agent spends a
thousand dollars chasing a rubric it can never satisfy.

---

## Fan out and combine

```python
g.add_edge("plan", "search_web")
g.add_edge("plan", "search_internal")
g.add_edge("plan", "check_database")
for n in ("search_web", "search_internal", "check_database"):
    g.add_edge(n, "synthesize")
```

All three run in one superstep against the same state snapshot, and their
updates merge through the reducers before `synthesize` runs once. Declare the
channel they all write with `append`, not the default overwrite.

---

## Provider fallback

```python
chain = RunnableFallback(
    prompt | AnthropicModel("claude-sonnet-4-5") | StrOutputParser(),
    [
        prompt | OpenAIModel("gpt-4o") | StrOutputParser(),
        prompt | OpenAIModel(base_url="http://localhost:11434/v1") | StrOutputParser(),
    ],
)
```

Degrade to a second provider or a local model instead of failing the user's
request. Only possible because the message format is neutral.

---

## Cheap routing before expensive work

```python
chain = RunnableBranch(
    (lambda x: "refund" in x.lower(), refund_chain),
    (lambda x: "error"  in x.lower(), support_chain),
    general_chain,
)
```

A keyword check, or a small model with `BooleanOutputParser`, costs
milliseconds. Asking a large model to decide costs a full round trip and is
harder to debug. Classify first, then run the specialised chain.

---

## Budgets and guardrails

```python
def cites_a_source(update):
    msgs = update.get("messages") or []
    for m in msgs:
        if m.role == "assistant" and m.content and "[" not in m.content:
            return "answer did not cite a source"
    return None

app = g.compile(middleware=[
    BudgetMiddleware(max_steps=30, max_seconds=120, max_tokens=100_000),
    GuardrailMiddleware([cites_a_source], on_violation="annotate"),
])
```

`on_violation="annotate"` writes the complaint into `errors` state, where the
agent sees it on the next turn and can self-correct. That turns a guardrail
from a wall into a feedback signal. Use `"raise"` for policy violations where
continuing is not acceptable.

---

## Long conversations

```python
agent = create_agent(model, tools, max_history=40)                # drop old turns
messages = summarize_and_trim(messages, summarizer=cheap_model)   # or compress them
```

`trim_messages` never orphans a tool result from the assistant message that
requested it — providers reject that, and it is the bug you hit at turn 30,
not turn 3.

---

## Testing

```python
model = FakeModel(responses=[
    Message.assistant("", tool_calls=[ToolCall(name="lookup_order", args={"order_id": "A-1"})]),
    Message.assistant("Your order ships tomorrow."),
])

out = create_agent(model=model, tools=[lookup_order]).invoke(
    {"messages": [Message.user("Where is A-1?")]}
)

assert [m.role for m in out["messages"]] == ["user", "assistant", "tool", "assistant"]
assert "A-1" in str(model.calls[1])      # assert on what the agent actually sent
```

Use `handler=` instead of `responses=` when the reply should depend on the
history — adaptive fakes let you test routing and recovery paths:

```python
def handler(history):
    if any(m.role == "tool" for m in history):
        return Message.assistant("Final answer.")
    return Message.assistant("", tool_calls=[ToolCall(name="search", args={"q": "x"})])
```

Set `tool_error_handling="raise"` in tests so a broken tool fails the test
instead of being quietly narrated back to the model.

---

## Adding a provider

```python
class MyModel(BaseChatModel):
    model = "my-model-v1"

    def invoke(self, messages, tools=None, **kwargs):
        payload = self._convert(messages)          # neutral -> your wire format
        resp = self._client.generate(**payload)
        return ModelResponse(
            message=Message.assistant(resp.text, tool_calls=self._calls(resp)),
            usage=Usage(resp.in_tokens, resp.out_tokens, 1),
            finish_reason=resp.stop,
            raw=resp,
        )
```

One method. Everything else — agents, chains, RAG, tracing — works unchanged.
Override `stream` if the provider supports real token streaming.

---

## Adding a durable checkpointer

```python
class PostgresCheckpointer(BaseCheckpointer):
    def get(self, thread_id):
        row = self.db.fetchone("select payload from checkpoints where thread_id=%s", thread_id)
        return json.loads(row[0]) if row else None

    def put(self, thread_id, checkpoint):
        self.db.execute(
            "insert into checkpoints (thread_id, payload) values (%s, %s) "
            "on conflict (thread_id) do update set payload = excluded.payload",
            thread_id, json.dumps(checkpoint, default=str),
        )

    def delete(self, thread_id):
        self.db.execute("delete from checkpoints where thread_id=%s", thread_id)
```

`put` is called after every superstep, so keep it cheap. See
`memory.py::_encode` for making `Message` objects JSON-safe and round-trippable.
