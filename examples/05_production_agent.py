"""
Example 05 — what a production wiring actually looks like.

Everything above this example was scripted so it would run offline. This one
talks to a real provider and turns on the things you want on before you ship:
durable checkpoints, a spend cap, redaction, guardrails and tracing.

    export ANTHROPIC_API_KEY=sk-...
    pip install anthropic
    python examples/05_production_agent.py "your question here"

Switching provider is one line — swap `AnthropicModel` for `OpenAIModel`, or
point `OpenAIModel(base_url=...)` at a local vLLM/Ollama server. Nothing else
in this file changes. That is the payoff of the neutral message format.
"""

from __future__ import annotations

import os
import sys

from agentkit import (
    BudgetMiddleware,
    ConsoleTracer,
    FileCheckpointer,
    GuardrailMiddleware,
    LoggingMiddleware,
    Message,
    RedactionMiddleware,
    RunConfig,
    UsageTrackingMiddleware,
    create_agent,
    tool,
)


# ---------------------------------------------------------------------------
# Tools
#
# Notice what makes a *good* tool: a narrow job, an unambiguous description,
# typed arguments, and a return value the model can act on. Tool quality is
# the single biggest lever on agent quality — far bigger than prompt wording.
# ---------------------------------------------------------------------------
@tool
def search_docs(query: str, top_k: int = 3) -> list:
    """Search the internal documentation index for passages matching a query.

    Use this before answering any question about internal systems, policies or
    APIs. Do not guess at internal details you have not looked up.

    Args:
        query: Natural-language search terms.
        top_k: How many passages to return (1-10).
    """
    # Stand-in for a vector store / BM25 index / whatever you actually run.
    corpus = {
        "deployment": "Deploys run via GitHub Actions on merge to main.",
        "oncall": "On-call rotates weekly; the pager schedule lives in PagerDuty.",
        "retention": "Logs are retained for 30 days, traces for 14.",
    }
    hits = [v for k, v in corpus.items() if any(w in k for w in query.lower().split())]
    return hits[:top_k] or ["No matching documentation found."]


@tool(requires_approval=True, tags=["write"])
def file_ticket(title: str, body: str, priority: str = "normal") -> str:
    """Open a ticket in the issue tracker. Creates a real, visible record.

    Args:
        title: One-line summary.
        body: Full description including reproduction steps.
        priority: One of "low", "normal", "high".
    """
    return f"Created ticket: [{priority}] {title}"


# ---------------------------------------------------------------------------
# Guardrails: cheap deterministic checks that run on every node's output.
# ---------------------------------------------------------------------------
def no_empty_answers(update: dict) -> str | None:
    """Catch the failure mode where a model returns nothing at all."""
    msgs = update.get("messages") or []
    for m in msgs:
        if getattr(m, "role", None) == "assistant" and not m.content and not m.tool_calls:
            return "assistant produced an empty message"
    return None


def bounded_length(update: dict, limit: int = 8000) -> str | None:
    msgs = update.get("messages") or []
    for m in msgs:
        if len(getattr(m, "content", "") or "") > limit:
            return f"message exceeded {limit} characters"
    return None


SYSTEM_PROMPT = """You are an internal engineering assistant.

Rules:
- Look things up with search_docs before stating any internal fact.
- If the docs do not cover something, say so plainly rather than guessing.
- Only file a ticket when the user explicitly asks for one.
- Be concise."""


def build_agent():
    from agentkit import AnthropicModel  # imported here so the file loads without the SDK

    return create_agent(
        model=AnthropicModel(model="claude-sonnet-4-5", temperature=0),
        tools=[search_docs, file_ticket],
        system_prompt=SYSTEM_PROMPT,
        max_iterations=8,

        # Durable state: resume a conversation across process restarts by
        # reusing the thread_id.
        checkpointer=FileCheckpointer(".agentkit/checkpoints"),

        # Any tool marked requires_approval should not fire unsupervised.
        interrupt_before_tools=True,

        # Keep the context window bounded on long sessions.
        max_history=40,

        # Order matters: outermost first. Tracing wraps everything so its
        # timings include the work the inner middleware does.
        middleware=[
            ConsoleTracer(),
            LoggingMiddleware(),
            UsageTrackingMiddleware(),
            BudgetMiddleware(max_steps=30, max_seconds=120, max_tokens=100_000),
            RedactionMiddleware(),
            GuardrailMiddleware([no_empty_answers, bounded_length], on_violation="annotate"),
        ],
    )


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(__doc__)
        print("ANTHROPIC_API_KEY is not set - nothing to run.")
        print("Try the offline examples instead: examples/01_minimal_agent.py")
        sys.exit(0)

    question = " ".join(sys.argv[1:]) or "How long are logs retained?"
    agent = build_agent()
    config = RunConfig(thread_id="prod-demo", tags=["example", "cli"])

    for event in agent.stream({"messages": [Message.user(question)]}, config):
        if event.interrupted:
            print(f"\n[paused before {event.nodes} - approve by re-running with the same thread_id]")
            break
        print(f"step {event.step}: {event.nodes}")

    state = agent.get_state("prod-demo") or {}
    if state.get("messages"):
        print("\nAnswer:", state["messages"][-1].content)
