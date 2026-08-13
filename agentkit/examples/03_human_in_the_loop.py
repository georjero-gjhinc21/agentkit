"""
Example 03 — pause for human approval, then resume.

This is the capability that justifies the whole graph architecture. An agent
about to send an email, move money, or delete a row should stop and ask. With
a while-loop you cannot stop: the state lives on the call stack, and a call
stack cannot be serialised and picked up ten minutes later by a different
process.

With a checkpointer plus `interrupt_before`, you get:

    run  -> agent decides to call `send_email` -> PAUSE, state written to disk
    (a human looks at it, maybe edits the arguments, maybe rejects)
    run  -> resumes at exactly that point with the edited state

The second `run` can be a different process, on a different machine, an hour
later. All it needs is the same `thread_id`.

    python examples/03_human_in_the_loop.py
"""

from agentkit import (
    FakeModel,
    InMemoryCheckpointer,
    Message,
    RunConfig,
    ToolCall,
    create_agent,
    tool,
)

SENT: list[dict] = []  # stands in for a real mail server


@tool(requires_approval=True, tags=["write", "external"])
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. This action is irreversible.

    Args:
        to: Recipient address.
        subject: Subject line.
        body: Plain-text body.
    """
    SENT.append({"to": to, "subject": subject, "body": body})
    return f"Email sent to {to}."


model = FakeModel(
    responses=[
        Message.assistant(
            "I'll send that email.",
            tool_calls=[
                ToolCall(
                    name="send_email",
                    args={
                        "to": "ops@exampl.com",  # note the typo - a human will fix it
                        "subject": "Weekly report",
                        "body": "All systems nominal.",
                    },
                )
            ],
        ),
        Message.assistant("Done - the weekly report has been sent."),
    ]
)

# The checkpointer is what makes pausing possible. Swap InMemoryCheckpointer
# for FileCheckpointer (or your own Postgres one) and the pause survives a
# process restart.
checkpointer = InMemoryCheckpointer()

agent = create_agent(
    model=model,
    tools=[send_email],
    system_prompt="You are an assistant that can send email.",
    checkpointer=checkpointer,
    interrupt_before_tools=True,  # <- stop before EVERY tool execution
)


if __name__ == "__main__":
    # One stable id ties the two runs together. In a web app this is your
    # conversation/session id.
    config = RunConfig(thread_id="demo-thread-1")

    # ---- PASS 1: runs until it wants to use a tool, then stops -------------
    print("=== pass 1: run until approval is needed ===")
    for event in agent.stream({"messages": [Message.user("Email ops the weekly report.")]}, config):
        status = "INTERRUPTED" if event.interrupted else "ok"
        print(f"  step {event.step}: {event.nodes} [{status}]")

    print(f"  emails actually sent so far: {len(SENT)}")  # 0 - nothing happened yet

    # ---- HUMAN REVIEW ------------------------------------------------------
    # The full state is on disk (or in memory here) and fully inspectable.
    state = agent.get_state("demo-thread-1")
    pending = state["messages"][-1]
    call = pending.tool_calls[0]
    print("\n=== awaiting approval ===")
    print(f"  tool: {call.name}")
    for k, v in call.args.items():
        print(f"    {k}: {v}")

    # The human spots the typo and corrects it. Because state is plain data,
    # editing it is just... editing it. No framework ceremony.
    print("\n  [human] fixing the recipient typo, then approving")
    call.args["to"] = "ops@example.com"
    agent.update_state("demo-thread-1", {"messages": [pending]})
    # ^ the add_messages reducer upserts by id, so this REPLACES the pending
    #   message rather than appending a duplicate.

    # A rejection path would instead inject a tool result saying "denied by
    # user" and let the model apologise and move on:
    #
    #   agent.update_state(thread_id, {"messages": [
    #       Message.tool("Denied by user.", tool_call_id=call.id, name=call.name)
    #   ]})

    # ---- PASS 2: resume from exactly where it stopped ----------------------
    print("\n=== pass 2: resume ===")
    for event in agent.stream(None, config):
        status = "INTERRUPTED" if event.interrupted else "ok"
        print(f"  step {event.step}: {event.nodes} [{status}]")

    final = agent.get_state("demo-thread-1")
    print("\nFinal reply:", final["messages"][-1].content)
    print("Emails sent:", SENT)
