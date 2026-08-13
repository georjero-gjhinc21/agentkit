"""
Test suite.

Run with `pytest -q`, or `python tests/test_agentkit.py` for a
zero-dependency run.

WHY THESE TESTS EXIST
---------------------
Agent frameworks are usually untested because the logic is tangled up with a
non-deterministic network call. `FakeModel` unties that knot, and once you can
script the model, all the genuinely tricky behaviour becomes ordinary unit
testing: does the loop terminate, do parallel branches merge correctly, does a
tool failure reach the model, does a resumed run pick up where it left off.

Treat this file as executable documentation of the framework's contracts.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentkit import (  # noqa: E402
    END,
    Channel,
    FakeModel,
    GraphError,
    InMemoryCheckpointer,
    InMemoryStore,
    Message,
    RecursionLimitError,
    RunConfig,
    StateGraph,
    StateSchema,
    ToolCall,
    ToolRegistry,
    add_messages,
    append,
    create_agent,
    default_agent_state,
    execute_tool_call,
    tool,
    trim_messages,
)


# ===========================================================================
# state / reducers
# ===========================================================================
def test_add_messages_appends():
    a, b = Message.user("one"), Message.user("two")
    assert [m.content for m in add_messages([a], [b])] == ["one", "two"]


def test_add_messages_upserts_by_id():
    """Same id replaces in place. This is what makes streaming edits work."""
    a = Message.user("draft")
    edited = Message(role="user", content="final", id=a.id)
    out = add_messages([a], [edited])
    assert len(out) == 1 and out[0].content == "final"


def test_schema_rejects_unknown_keys():
    """Typos in state keys must fail loudly, not silently do nothing."""
    schema = default_agent_state()
    try:
        schema.apply(schema.initial_state(), {"mesages": []})
    except KeyError as e:
        assert "mesages" in str(e)
    else:
        raise AssertionError("expected KeyError for unknown state key")


def test_apply_does_not_mutate_input():
    schema = default_agent_state()
    s0 = schema.initial_state()
    s1 = schema.apply(s0, {"messages": [Message.user("hi")]})
    assert s0["messages"] == [] and len(s1["messages"]) == 1


# ===========================================================================
# tools
# ===========================================================================
@tool
def multiply(a: int, b: int = 2) -> int:
    """Multiply two integers.

    Args:
        a: First number.
        b: Second number.
    """
    return a * b


def test_schema_inference():
    s = multiply.to_schema()
    assert s["name"] == "multiply"
    assert s["description"].startswith("Multiply two integers")
    props = s["parameters"]["properties"]
    assert props["a"] == {"type": "integer", "description": "First number."}
    assert props["b"]["default"] == 2
    assert s["parameters"]["required"] == ["a"]  # b has a default


def test_tool_execution_and_error_capture():
    reg = ToolRegistry([multiply])

    ok = execute_tool_call(reg, ToolCall(name="multiply", args={"a": 3, "b": 4}))
    assert ok.content == "12" and ok.metadata["ok"] is True

    # Missing required arg: the error comes back as a readable tool message
    # instead of crashing the run, so the model can retry.
    bad = execute_tool_call(reg, ToolCall(name="multiply", args={}))
    assert bad.metadata["ok"] is False and "missing required" in bad.content

    # Hallucinated tool name: same treatment.
    ghost = execute_tool_call(reg, ToolCall(name="nope", args={}))
    assert ghost.metadata["ok"] is False and "No tool named" in ghost.content


def test_bare_functions_are_wrapped():
    def echo(text: str) -> str:
        """Echo the input."""
        return text

    reg = ToolRegistry([echo])
    assert "echo" in reg and reg.get("echo").parameters["required"] == ["text"]


# ===========================================================================
# graph
# ===========================================================================
def test_linear_graph():
    schema = StateSchema({"n": Channel(default=0), "log": Channel(append, list)})
    g = StateGraph(schema)
    g.add_node("inc", lambda s, c: {"n": s["n"] + 1, "log": "inc"})
    g.add_node("double", lambda s, c: {"n": s["n"] * 2, "log": "double"})
    g.set_entry_point("inc")
    g.add_edge("inc", "double")
    g.add_edge("double", END)

    out = g.compile().invoke({"n": 5})
    assert out["n"] == 12 and out["log"] == ["inc", "double"]


def test_parallel_branches_merge_via_reducer():
    """Both branches see the SAME input state and both contribute."""
    schema = StateSchema({"seen": Channel(append, list), "base": Channel(default=1)})
    g = StateGraph(schema)
    g.add_node("fork", lambda s, c: {})
    g.add_node("left", lambda s, c: {"seen": f"left saw {s['base']}"})
    g.add_node("right", lambda s, c: {"seen": f"right saw {s['base']}"})
    g.add_node("join", lambda s, c: {})
    g.set_entry_point("fork")
    g.add_edge("fork", "left")
    g.add_edge("fork", "right")
    g.add_edge("left", "join")
    g.add_edge("right", "join")
    g.add_edge("join", END)

    out = g.compile(parallel=True).invoke({"base": 7})
    assert sorted(out["seen"]) == ["left saw 7", "right saw 7"]


def test_join_runs_once_not_twice():
    """Two edges into one node must not run it twice in a superstep."""
    schema = StateSchema({"count": Channel(lambda o, n: (o or 0) + n, 0)})
    g = StateGraph(schema)
    g.add_node("fork", lambda s, c: {})
    g.add_node("a", lambda s, c: {})
    g.add_node("b", lambda s, c: {})
    g.add_node("join", lambda s, c: {"count": 1})
    g.set_entry_point("fork")
    for n in ("a", "b"):
        g.add_edge("fork", n)
        g.add_edge(n, "join")
    g.add_edge("join", END)

    assert g.compile().invoke()["count"] == 1


def test_recursion_limit():
    schema = StateSchema({"n": Channel(default=0)})
    g = StateGraph(schema)
    g.add_node("loop", lambda s, c: {"n": s["n"] + 1})
    g.set_entry_point("loop")
    # A router that never returns END - deliberately pathological.
    g.add_conditional_edges("loop", lambda s, c: "go", {"go": "loop", "stop": END})

    try:
        g.compile().invoke({}, RunConfig(recursion_limit=5))
    except RecursionLimitError as e:
        assert e.limit == 5
    else:
        raise AssertionError("expected RecursionLimitError")


def test_validation_catches_dangling_node():
    g = StateGraph(StateSchema({"x": Channel()}))
    g.add_node("a", lambda s, c: {})
    g.set_entry_point("a")
    try:
        g.compile()
    except GraphError as e:
        assert "no outgoing edges" in str(e)
    else:
        raise AssertionError("expected GraphError for dangling node")


def test_validation_catches_unreachable_end():
    g = StateGraph(StateSchema({"x": Channel()}))
    g.add_node("a", lambda s, c: {})
    g.add_node("b", lambda s, c: {})
    g.set_entry_point("a")
    g.add_edge("a", "b")
    g.add_edge("b", "a")
    try:
        g.compile()
    except GraphError as e:
        assert "END" in str(e)
    else:
        raise AssertionError("expected GraphError for unreachable END")


def test_node_retries():
    calls = {"n": 0}

    def flaky(s, c):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return {"x": "ok"}

    g = StateGraph(StateSchema({"x": Channel()}))
    g.add_node("flaky", flaky, retries=3, retry_delay=0.001)
    g.set_entry_point("flaky")
    g.add_edge("flaky", END)

    assert g.compile().invoke()["x"] == "ok" and calls["n"] == 3


# ===========================================================================
# agent loop
# ===========================================================================
def test_agent_runs_tool_then_answers():
    model = FakeModel(
        responses=[
            Message.assistant("", tool_calls=[ToolCall(name="multiply", args={"a": 6, "b": 7})]),
            Message.assistant("The answer is 42."),
        ]
    )
    agent = create_agent(model=model, tools=[multiply])
    out = agent.invoke({"messages": [Message.user("6 times 7?")]})

    roles = [m.role for m in out["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert out["messages"][-1].content == "The answer is 42."
    assert out["steps"] == 2


def test_agent_stops_without_tool_calls():
    agent = create_agent(model=FakeModel(responses=["Just answering directly."]), tools=[multiply])
    out = agent.invoke({"messages": [Message.user("hi")]})
    assert out["steps"] == 1 and len(out["messages"]) == 2


def test_max_iterations_stops_a_loop():
    """A model that always calls tools must still terminate."""
    def always_tools(history):
        return Message.assistant("", tool_calls=[ToolCall(name="multiply", args={"a": 1})])

    agent = create_agent(
        model=FakeModel(handler=always_tools), tools=[multiply], max_iterations=3
    )
    out = agent.invoke({"messages": [Message.user("go")]})
    assert out["steps"] == 3


def test_parallel_tool_calls_all_execute():
    model = FakeModel(
        responses=[
            Message.assistant(
                "",
                tool_calls=[
                    ToolCall(name="multiply", args={"a": 2}),
                    ToolCall(name="multiply", args={"a": 5}),
                ],
            ),
            Message.assistant("done"),
        ]
    )
    out = create_agent(model=model, tools=[multiply]).invoke(
        {"messages": [Message.user("x")]}
    )
    results = [m.content for m in out["messages"] if m.role == "tool"]
    assert results == ["4", "10"]


def test_system_prompt_injected_once():
    """Re-entering the model node must not stack duplicate system messages."""
    model = FakeModel(
        responses=[
            Message.assistant("", tool_calls=[ToolCall(name="multiply", args={"a": 1})]),
            Message.assistant("done"),
        ]
    )
    agent = create_agent(model=model, tools=[multiply], system_prompt="BE BRIEF")
    agent.invoke({"messages": [Message.user("x")]})
    second_call = model.calls[1]
    assert sum(1 for m in second_call if m.role == "system") == 1


# ===========================================================================
# checkpointing / human-in-the-loop
# ===========================================================================
def test_interrupt_and_resume():
    model = FakeModel(
        responses=[
            Message.assistant("", tool_calls=[ToolCall(name="multiply", args={"a": 3})]),
            Message.assistant("finished"),
        ]
    )
    agent = create_agent(
        model=model,
        tools=[multiply],
        checkpointer=InMemoryCheckpointer(),
        interrupt_before_tools=True,
    )
    cfg = RunConfig(thread_id="t1")

    events = list(agent.stream({"messages": [Message.user("x")]}, cfg))
    assert events[-1].interrupted

    # Resuming with the same thread id must move PAST the interrupt, not
    # re-trigger it. (Regression guard: this was a real bug.)
    events2 = list(agent.stream(None, cfg))
    assert not any(e.interrupted for e in events2)
    assert agent.get_state("t1")["messages"][-1].content == "finished"


def test_state_editing_before_resume():
    model = FakeModel(
        responses=[
            Message.assistant("", tool_calls=[ToolCall(name="multiply", args={"a": 1})]),
            Message.assistant("done"),
        ]
    )
    agent = create_agent(
        model=model,
        tools=[multiply],
        checkpointer=InMemoryCheckpointer(),
        interrupt_before_tools=True,
    )
    cfg = RunConfig(thread_id="t2")
    list(agent.stream({"messages": [Message.user("x")]}, cfg))

    pending = agent.get_state("t2")["messages"][-1]
    pending.tool_calls[0].args["a"] = 10  # human edits the arguments
    agent.update_state("t2", {"messages": [pending]})

    list(agent.stream(None, cfg))
    tool_out = [m for m in agent.get_state("t2")["messages"] if m.role == "tool"]
    assert tool_out[0].content == "20"


def test_multi_turn_memory():
    """Same thread id across separate invokes continues the conversation."""
    cp = InMemoryCheckpointer()
    agent = create_agent(model=FakeModel(responses=["first", "second"]), checkpointer=cp)
    cfg = RunConfig(thread_id="chat")

    agent.invoke({"messages": [Message.user("hello")]}, cfg)
    out = agent.invoke({"messages": [Message.user("again")]}, cfg)
    assert [m.content for m in out["messages"]] == ["hello", "first", "again", "second"]


# ===========================================================================
# memory helpers
# ===========================================================================
def test_trim_keeps_system_and_avoids_orphan_tool_messages():
    msgs = [Message.system("rules")]
    for i in range(10):
        msgs.append(Message.user(f"u{i}"))
        msgs.append(Message.assistant(f"a{i}", tool_calls=[ToolCall(name="t", args={})]))
        msgs.append(Message.tool("r", tool_call_id="x"))

    out = trim_messages(msgs, max_messages=6)
    assert out[0].role == "system"
    assert out[1].role != "tool"  # boundary repaired
    assert len(out) <= 6


def test_store_namespacing():
    store = InMemoryStore()
    store.put(("users", "u1"), "prefs", {"tone": "terse"})
    store.put(("users", "u2"), "prefs", {"tone": "chatty"})
    assert store.get(("users", "u1"), "prefs")["tone"] == "terse"
    assert len(store.search(("users",), "terse")) == 1
    assert len(store.search(("users",))) == 2


# ===========================================================================
# runner
# ===========================================================================
if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
