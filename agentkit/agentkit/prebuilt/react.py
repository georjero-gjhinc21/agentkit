"""
agentkit.prebuilt.react
=======================

The tool-calling agent loop, assembled from the primitives in the rest of the
package. If you only ever use one thing from this repo, it is `create_agent`.

THE LOOP
--------

        START -> model -> (tool calls?) -> tools -> model -> ...
                            |
                            no -> END

That is genuinely all a "ReAct" agent is: let the model choose a tool, run it,
show it the result, repeat until it stops asking for tools. Everything else —
planning, reflection, multi-agent — is a variation on this shape.

READ THIS FILE AS A WORKED EXAMPLE. It is ~120 lines of real code and it shows
how nodes, routers, state and reducers fit together. When you need something
this does not do (a planning step, a critic, a retrieval stage before the
model), copy it and edit the graph rather than adding flags here.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from ..graph import END, CompiledGraph, StateGraph
from ..memory import trim_messages
from ..models import BaseChatModel
from ..state import StateSchema, default_agent_state, last_message
from ..tools import Tool, ToolRegistry, execute_tool_call
from ..types import Message, RunConfig


def create_agent(
    model: BaseChatModel,
    tools: Sequence[Tool | Callable] | ToolRegistry | None = None,
    *,
    system_prompt: str | None = None,
    max_iterations: int = 10,
    schema: StateSchema | None = None,
    checkpointer: Any | None = None,
    middleware: Sequence[Any] = (),
    interrupt_before_tools: bool = False,
    max_history: int | None = None,
    tool_error_handling: str = "return",
    response_format: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> CompiledGraph:
    """Build a tool-calling agent.

    Args:
        model: Any `BaseChatModel`. Use `FakeModel` to test without a network.
        tools: Functions, `Tool`s, or a `ToolRegistry`. Bare functions are
            wrapped automatically via the `@tool` schema inference.
        system_prompt: Prepended once, only if the caller did not already
            supply a system message.
        max_iterations: Cap on model<->tool cycles. Distinct from the graph's
            `recursion_limit`: this one is about *reasoning* depth and, when
            hit, ends the run cleanly with whatever the agent has, rather than
            raising.
        checkpointer: Enables multi-turn memory and resumable runs.
        interrupt_before_tools: Pause before every tool execution so a human
            can approve. Combine with `Tool(requires_approval=True)` for
            per-tool gating.
        max_history: Trim to this many messages before each model call.
        tool_error_handling: "return" (feed the error back to the model, which
            can then recover) or "raise" (fail fast — better in tests).
        response_format: Optional final transform, e.g. to parse JSON out of
            the last message into a typed field.

    Returns:
        A compiled graph. Call `.invoke({"messages": [...]})` or `.stream(...)`.
    """
    registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry(list(tools or []))
    schema = schema or default_agent_state()
    tool_schemas = registry.schemas() if len(registry) else None

    # -----------------------------------------------------------------------
    # NODE 1: call the model
    # -----------------------------------------------------------------------
    def model_node(state: dict[str, Any], config: RunConfig) -> dict[str, Any]:
        messages: list[Message] = list(state["messages"])

        # Inject the system prompt once, and only if absent. Checking rather
        # than always-prepending is what makes this node safe to re-enter on
        # every loop iteration and on resumed threads.
        if system_prompt and not any(m.role == "system" for m in messages):
            messages.insert(0, Message.system(system_prompt))

        if max_history:
            messages = trim_messages(messages, max_messages=max_history)

        response = model.invoke(messages, tools=tool_schemas)

        # Stash usage on the message so tracing/budget middleware can find it
        # without the model layer needing to know those exist.
        reply = response.message
        reply.metadata["usage"] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        reply.metadata["model"] = model.model
        reply.metadata["finish_reason"] = response.finish_reason

        # `steps` uses the add_int reducer, so returning 1 means "increment".
        return {"messages": [reply], "steps": 1}

    # -----------------------------------------------------------------------
    # NODE 2: execute the requested tools
    # -----------------------------------------------------------------------
    def tools_node(state: dict[str, Any], config: RunConfig) -> dict[str, Any]:
        msg = last_message(state)
        if msg is None or not msg.has_tool_calls:
            return {}  # defensive: the router should never send us here

        results: list[Message] = []
        errors: list[str] = []
        for call in msg.tool_calls:
            result = execute_tool_call(registry, call, on_error=tool_error_handling)
            results.append(result)
            if result.metadata.get("ok") is False:
                errors.append(f"{call.name}: {result.metadata.get('error')}")

        # Note we return ALL results in one update. The add_messages reducer
        # appends them in order, and the provider adapters merge consecutive
        # tool results into a single request message — so parallel tool calls
        # work without any special handling here.
        update: dict[str, Any] = {"messages": results}
        if errors:
            update["errors"] = errors
        return update

    # -----------------------------------------------------------------------
    # ROUTER: the one decision this agent makes
    # -----------------------------------------------------------------------
    def should_continue(state: dict[str, Any], config: RunConfig) -> str:
        """After the model speaks: run tools, or stop?

        Three exits, in priority order:
          1. iteration cap reached  -> stop (protects against tool ping-pong)
          2. model requested tools  -> run them
          3. otherwise              -> the model produced a final answer
        """
        if state.get("steps", 0) >= max_iterations:
            return "finish"
        msg = last_message(state)
        if msg is not None and msg.has_tool_calls:
            return "tools"
        return "finish"

    # -----------------------------------------------------------------------
    # OPTIONAL NODE 3: post-process the final answer
    # -----------------------------------------------------------------------
    def finish_node(state: dict[str, Any], config: RunConfig) -> dict[str, Any]:
        return response_format(state) or {} if response_format else {}

    # -----------------------------------------------------------------------
    # Wire it together
    # -----------------------------------------------------------------------
    g = StateGraph(schema)
    g.add_node("model", model_node, description="Call the LLM with tool schemas")
    g.add_node("tools", tools_node, description="Execute requested tool calls")
    g.set_entry_point("model")

    if response_format:
        g.add_node("finish", finish_node, description="Shape the final output")
        g.add_conditional_edges("model", should_continue, {"tools": "tools", "finish": "finish"})
        g.add_edge("finish", END)
    else:
        g.add_conditional_edges("model", should_continue, {"tools": "tools", "finish": END})

    # The edge that closes the loop: tool results go straight back to the model.
    g.add_edge("tools", "model")

    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=("tools",) if interrupt_before_tools else (),
        middleware=middleware,
        # `steps` counts THIS turn's model<->tool cycles. Without the reset it
        # would accumulate over a long conversation and `max_iterations` would
        # eventually cut off every reply after one model call.
        reset_on_new_turn=("steps", "errors"),
    )
