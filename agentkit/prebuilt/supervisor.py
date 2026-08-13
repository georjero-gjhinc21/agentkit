"""
agentkit.prebuilt.supervisor
============================

Multi-agent orchestration: a supervisor routes work to specialist agents.

WHEN THIS IS WORTH IT
---------------------
Not as often as the hype suggests. Multi-agent costs you an extra model call
per hop, more places for context to get lost, and much harder debugging. Reach
for it when one of these is true:

  * The tool count is large enough that selection accuracy is degrading
    (roughly: past 15-20 tools). Splitting the toolbox by specialist fixes it.
  * Specialists need genuinely different system prompts, models or
    temperatures (a cheap fast model for triage, an expensive one for
    analysis).
  * You want separate teams to own separate agents behind a stable interface.

If none of those apply, a single agent with good tools will beat a swarm.

HOW IT WORKS
------------
Each worker is itself a `CompiledGraph`, so a worker can be a ReAct agent, a
hand-built workflow, or another supervisor — the composition nests. The
supervisor is just another node whose "tools" are handoffs.

CONTEXT ISOLATION
-----------------
`share_full_history` is the important knob. False (default) gives each worker
only the task description, keeping their contexts small, cheap and focused.
True lets them see everything, which helps on tasks needing continuity and
costs proportionally more. Start False.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from ..graph import END, CompiledGraph, StateGraph
from ..models import BaseChatModel
from ..state import Channel, StateSchema, add_messages, append, default_agent_state, last_message, last_value
from ..types import Message, RunConfig

FINISH = "FINISH"


def create_supervisor(
    model: BaseChatModel,
    workers: dict[str, CompiledGraph],
    *,
    system_prompt: str | None = None,
    max_handoffs: int = 8,
    checkpointer: Any | None = None,
    middleware: Sequence[Any] = (),
    share_full_history: bool = False,
    worker_descriptions: dict[str, str] | None = None,
) -> CompiledGraph:
    """Build a supervisor that delegates to named worker agents.

    Args:
        model: The routing model. Can be cheaper than the workers' — picking a
            name from a short list is an easy task.
        workers: name -> compiled agent. Names appear in the routing prompt,
            so make them descriptive ("researcher", not "agent_2").
        worker_descriptions: name -> one line on what it is for. Routing
            quality depends on these far more than on the model.
        max_handoffs: Cap on delegations before forcing a final answer.
        share_full_history: See module docstring.
    """
    if not workers:
        raise ValueError("A supervisor needs at least one worker.")

    descriptions = worker_descriptions or {}
    roster = "\n".join(f"  - {n}: {descriptions.get(n, 'a specialist agent')}" for n in workers)

    default_prompt = (
        "You are a supervisor coordinating specialist agents.\n\n"
        f"Available agents:\n{roster}\n\n"
        "Given the conversation so far, decide which agent should act next, "
        f"or reply {FINISH} if the task is complete.\n"
        f"Respond with ONLY the agent name or {FINISH} on the first line. "
        "If you reply " + FINISH + ", put the final answer to the user on the "
        "following lines."
    )

    # Supervisor state extends the default with routing bookkeeping.
    schema: StateSchema = default_agent_state()
    schema.add("next_worker", reducer=last_value, default=None,
               description="Which worker the supervisor selected this round.")
    schema.add("handoffs", reducer=append, default=list,
               description="Audit trail of delegations, for debugging routes.")

    # -----------------------------------------------------------------------
    def supervisor_node(state: dict[str, Any], config: RunConfig) -> dict[str, Any]:
        """Ask the routing model who goes next."""
        if len(state.get("handoffs") or []) >= max_handoffs:
            # Budget spent. Stop cleanly rather than looping.
            return {
                "next_worker": FINISH,
                "messages": [
                    Message.assistant(
                        "Reached the delegation limit; summarising with what we have.",
                        name="supervisor",
                    )
                ],
            }

        prompt = [Message.system(system_prompt or default_prompt), *state["messages"]]
        reply = model.invoke(prompt).message

        # Parse: first line is the decision, the rest is any final answer.
        lines = [ln.strip() for ln in reply.content.strip().splitlines() if ln.strip()]
        head = lines[0] if lines else FINISH
        # Tolerate decoration like "**researcher**" or "Next: researcher".
        choice = next((n for n in workers if n.lower() in head.lower()), None)
        if choice is None:
            choice = FINISH

        update: dict[str, Any] = {"next_worker": choice}
        if choice == FINISH:
            body = "\n".join(lines[1:]).strip() or reply.content.strip()
            update["messages"] = [Message.assistant(body, name="supervisor")]
        else:
            update["handoffs"] = [choice]
        return update

    # -----------------------------------------------------------------------
    def make_worker_node(name: str, agent: CompiledGraph) -> Callable[..., dict[str, Any]]:
        """Wrap a compiled agent so it looks like an ordinary node.

        This adapter is the whole trick behind composition: because a worker
        exposes the same `invoke(state) -> state` shape as a node, graphs nest
        arbitrarily without the supervisor knowing what is inside.
        """

        def worker_node(state: dict[str, Any], config: RunConfig) -> dict[str, Any]:
            if share_full_history:
                inbound = list(state["messages"])
            else:
                # Hand over only the task: the most recent user-facing content.
                task = last_message(state)
                inbound = [Message.user(task.content if task else "")]

            # Sub-runs get their own thread so their checkpoints do not collide
            # with the supervisor's.
            sub_config = RunConfig(
                thread_id=f"{config.thread_id}:{name}",
                recursion_limit=config.recursion_limit,
                tags=[*config.tags, f"worker:{name}"],
                context=config.context,
            )
            result = agent.invoke({"messages": inbound}, sub_config)

            # Return only what the worker *added*, tagged with its name so the
            # transcript stays readable and the supervisor can attribute work.
            produced = result.get("messages", [])[len(inbound):]
            out = [m for m in produced if m.role == "assistant" and m.content]
            for m in out:
                m.name = name
            return {"messages": out or [Message.assistant(f"({name} returned nothing)", name=name)]}

        return worker_node

    # -----------------------------------------------------------------------
    def route(state: dict[str, Any], config: RunConfig) -> str:
        choice = state.get("next_worker")
        return choice if choice in workers else FINISH

    g = StateGraph(schema)
    g.add_node("supervisor", supervisor_node, description="Decide who acts next")
    for name, agent in workers.items():
        g.add_node(name, make_worker_node(name, agent), description=descriptions.get(name, ""))
        # Workers always report back to the supervisor — a hub-and-spoke
        # topology. For a mesh, add worker->worker edges instead.
        g.add_edge(name, "supervisor")

    g.set_entry_point("supervisor")
    g.add_conditional_edges(
        "supervisor", route, {**{n: n for n in workers}, FINISH: END}
    )

    return g.compile(checkpointer=checkpointer, middleware=middleware, parallel=False)
