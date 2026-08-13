"""
Example 02 — a custom graph: parallel fan-out, custom state, conditional loop.

`create_agent` is one graph out of infinitely many. This example builds one by
hand to show the parts you actually compose with:

  * a custom state schema with domain channels and reducers
  * two nodes running in PARALLEL in the same superstep
  * a join node that sees both results merged
  * a conditional edge that loops back for another round if quality is low

Shape:

        START
          |
       plan
        /  \\            <- both run in one superstep
    research  analyze
        \\  /
        write
          |
      (good enough?) --no--> plan
          |yes
         END

    python examples/02_custom_workflow.py
"""

from agentkit import (
    END,
    Channel,
    Message,
    RunConfig,
    StateGraph,
    StateSchema,
    add_int,
    add_messages,
    append,
    merge_dict,
)

# ---------------------------------------------------------------------------
# 1. State schema.
#
# The reducer choice per channel IS the design. `findings` uses `append`, so
# the two parallel branches both contribute instead of one overwriting the
# other. `quality` uses the default overwrite, because only the latest score
# matters. Get these wrong and you get bugs that look like nondeterminism.
# ---------------------------------------------------------------------------
schema = StateSchema(
    {
        "messages": Channel(add_messages, list, "Conversation, for the final answer."),
        "topic": Channel(default=None, description="What we are working on."),
        "findings": Channel(append, list, "Accumulated notes from all branches."),
        "metrics": Channel(merge_dict, dict, "Numeric signals keyed by source."),
        "draft": Channel(default="", description="Latest written output."),
        "quality": Channel(default=0.0, description="Score of the latest draft, 0-1."),
        "rounds": Channel(add_int, 0, "Revision rounds completed."),
    }
)


# ---------------------------------------------------------------------------
# 2. Nodes. Each takes (state, config) and returns a PARTIAL update.
#    Never mutate `state` in place — return what changed and let the reducers
#    merge it. That discipline is what makes parallelism safe.
# ---------------------------------------------------------------------------
def plan(state: dict, config: RunConfig) -> dict:
    topic = state.get("topic") or "an unspecified topic"
    round_no = state.get("rounds", 0)
    note = f"Plan (round {round_no + 1}): outline key angles on {topic}."
    return {"findings": [note]}


def research(state: dict, config: RunConfig) -> dict:
    """Pretend to hit a search API. Runs concurrently with `analyze`."""
    topic = state.get("topic")
    return {
        "findings": [f"Research: three sources agree {topic} is trending upward."],
        "metrics": {"sources_found": 3},
    }


def analyze(state: dict, config: RunConfig) -> dict:
    """Pretend to crunch numbers. Runs concurrently with `research`.

    Note it CANNOT see research's output — both read the same pre-superstep
    snapshot. If a node needs another's result, it belongs downstream, not
    beside it.
    """
    return {
        "findings": ["Analysis: baseline metrics are within expected range."],
        "metrics": {"confidence": 0.62},
    }


def write(state: dict, config: RunConfig) -> dict:
    """Join point. By now both branches have merged into `findings`."""
    findings = state["findings"]
    draft = "\n".join(f"- {f}" for f in findings)

    # A stand-in for an LLM-as-judge or a rubric scorer. Improves each round so
    # the loop terminates; a real one would not be guaranteed to, which is why
    # the round cap below exists.
    quality = min(1.0, 0.45 + 0.3 * state.get("rounds", 0))

    return {
        "draft": draft,
        "quality": quality,
        "rounds": 1,
        "messages": [Message.assistant(f"Draft (quality={quality:.2f}):\n{draft}")],
    }


# ---------------------------------------------------------------------------
# 3. Router. Pure function of state -> a label in the path_map.
#
# Two exit conditions, and you need BOTH: quality met, or rounds exhausted.
# A quality-only condition is how agents end up burning a thousand dollars
# trying to satisfy a rubric they can never satisfy.
# ---------------------------------------------------------------------------
def good_enough(state: dict, config: RunConfig) -> str:
    if state.get("quality", 0) >= 0.75:
        return "done"
    if state.get("rounds", 0) >= 3:
        return "done"
    return "revise"


# ---------------------------------------------------------------------------
# 4. Wire it up.
# ---------------------------------------------------------------------------
g = StateGraph(schema)
g.add_node("plan", plan, description="Decide what to look into")
g.add_node("research", research, description="Gather external evidence")
g.add_node("analyze", analyze, description="Crunch internal numbers")
g.add_node("write", write, description="Merge findings into a draft")

g.set_entry_point("plan")

# TWO edges from one node = fan-out. Both land in the next superstep together.
g.add_edge("plan", "research")
g.add_edge("plan", "analyze")

# Both converge on `write`. It runs once, after both have merged.
g.add_edge("research", "write")
g.add_edge("analyze", "write")

g.add_conditional_edges("write", good_enough, {"revise": "plan", "done": END})

app = g.compile(parallel=True)


if __name__ == "__main__":
    print(app.builder.describe())
    print()

    # `stream` yields one event per superstep — the natural place to hook a
    # progress bar, a websocket push, or a debugger.
    for event in app.stream({"topic": "coastal wind energy"}):
        ran = " + ".join(event.nodes)
        print(f"step {event.step}: [{ran}]  {event.duration_ms:>6.1f}ms  "
              f"quality={event.state.get('quality', 0):.2f}")

    final = app.invoke({"topic": "coastal wind energy"})
    print("\nFinal draft:\n" + final["draft"])
    print("\nMetrics:", final["metrics"])
    print("Rounds:", final["rounds"])
