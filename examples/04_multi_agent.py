"""
Example 04 — a supervisor delegating to specialist agents.

Read the docstring in `agentkit/prebuilt/supervisor.py` first: multi-agent is
often the wrong answer. It earns its keep when the toolbox is too big for one
agent to choose from reliably, or when specialists need different prompts or
different models.

Structure:

    supervisor --> researcher (search tools)
        ^   \\
        |    -> calculator (math tools)
        |          |
        +----------+     <- workers always report back

Each worker is itself a full compiled agent, so it has its own loop, its own
tools and its own context. Because a worker is just a `CompiledGraph`, a
worker could equally be another supervisor — the composition nests.

    python examples/04_multi_agent.py
"""

from agentkit import FakeModel, Message, ToolCall, create_agent, create_supervisor, tool


# ---------------------------------------------------------------------------
# Specialist tools, deliberately split into two disjoint toolboxes.
# ---------------------------------------------------------------------------
@tool
def search_papers(query: str, limit: int = 3) -> list:
    """Search an academic paper index.

    Args:
        query: Free-text search terms.
        limit: Maximum results to return.
    """
    return [f"Paper {i + 1} on {query}" for i in range(limit)]


@tool
def compound_growth(principal: float, rate: float, years: int) -> float:
    """Compute compound growth.

    Args:
        principal: Starting amount.
        rate: Annual rate as a decimal, e.g. 0.07 for 7%.
        years: Number of years.
    """
    return round(principal * (1 + rate) ** years, 2)


# ---------------------------------------------------------------------------
# Worker 1: researcher. Its scripted model calls the search tool, then reports.
# ---------------------------------------------------------------------------
researcher = create_agent(
    model=FakeModel(
        responses=[
            Message.assistant(
                "Searching.",
                tool_calls=[ToolCall(name="search_papers", args={"query": "tidal energy yield"})],
            ),
            Message.assistant(
                "Found three papers; they report a mean annual yield growth of about 7%."
            ),
        ]
    ),
    tools=[search_papers],
    system_prompt="You are a research specialist. Cite what you find, briefly.",
)

# ---------------------------------------------------------------------------
# Worker 2: calculator.
# ---------------------------------------------------------------------------
calculator = create_agent(
    model=FakeModel(
        responses=[
            Message.assistant(
                "Computing.",
                tool_calls=[
                    ToolCall(
                        name="compound_growth",
                        args={"principal": 1000.0, "rate": 0.07, "years": 10},
                    )
                ],
            ),
            Message.assistant("At 7% annually, 1000 becomes 1967.15 after 10 years."),
        ]
    ),
    tools=[compound_growth],
    system_prompt="You are a quantitative specialist. Show the number.",
)

# ---------------------------------------------------------------------------
# The supervisor's model only has to emit a name. That is an easy task, so a
# small cheap model is usually the right choice here.
# ---------------------------------------------------------------------------
supervisor_model = FakeModel(
    responses=[
        Message.assistant("researcher"),
        Message.assistant("calculator"),
        Message.assistant(
            "FINISH\nResearch suggests ~7% annual growth, which turns 1000 into 1967.15 over 10 years."
        ),
    ]
)

app = create_supervisor(
    model=supervisor_model,
    workers={"researcher": researcher, "calculator": calculator},
    worker_descriptions={
        # These descriptions do most of the routing work. Vague ones ("helps
        # with stuff") produce bad routes no matter how good the model is.
        "researcher": "Searches literature and summarises findings. No arithmetic.",
        "calculator": "Performs numeric and financial calculations. No searching.",
    },
    max_handoffs=5,
)


if __name__ == "__main__":
    print(app.builder.describe())
    print()

    question = (
        "What growth rate do papers report for tidal energy, "
        "and what does that do to 1000 over 10 years?"
    )

    final: dict = {}
    for event in app.stream({"messages": [Message.user(question)]}):
        final = event.state
        print(f"step {event.step}: {event.nodes} -> next={event.state.get('next_worker')}")

    print("\nTranscript:")
    for m in final["messages"]:
        print(f"  {(m.name or m.role):>12}: {m.content[:90]}")

    print("\nDelegation path:", " -> ".join(final["handoffs"]))
    print("Each worker ran its own model+tool loop in an isolated context.")
