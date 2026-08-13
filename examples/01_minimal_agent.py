"""
Example 01 — the smallest useful agent.

Runs with no API key and no network: `FakeModel` plays the part of the LLM by
following a script. Everything else — the graph, the tool execution, the loop
termination — is the real framework.

    python examples/01_minimal_agent.py
"""

from agentkit import ConsoleTracer, FakeModel, Message, ToolCall, create_agent, tool


# ---------------------------------------------------------------------------
# 1. Define tools as ordinary functions.
#    The docstring becomes the description the model sees, and the type hints
#    become the JSON Schema. Write both as if the reader is the model.
# ---------------------------------------------------------------------------
@tool
def get_weather(city: str, units: str = "celsius") -> dict:
    """Get the current weather for a city.

    Args:
        city: Name of the city, e.g. "Paris".
        units: Either "celsius" or "fahrenheit".
    """
    fake_db = {"paris": 18, "tokyo": 24, "port arthur": 31}
    c = fake_db.get(city.lower(), 20)
    temp = c if units == "celsius" else round(c * 9 / 5 + 32)
    return {"city": city, "temperature": temp, "units": units, "conditions": "clear"}


@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


# ---------------------------------------------------------------------------
# 2. A scripted "model". In production this is AnthropicModel(...) or
#    OpenAIModel(...) — the agent code below does not change.
# ---------------------------------------------------------------------------
model = FakeModel(
    responses=[
        # Turn 1: the model decides to call a tool.
        Message.assistant(
            "Let me check the weather.",
            tool_calls=[ToolCall(name="get_weather", args={"city": "Tokyo"})],
        ),
        # Turn 2: having seen the result, it answers.
        Message.assistant("It's currently 24C and clear in Tokyo."),
    ]
)

# ---------------------------------------------------------------------------
# 3. Assemble and run.
# ---------------------------------------------------------------------------
agent = create_agent(
    model=model,
    tools=[get_weather, add],
    system_prompt="You are a concise, helpful assistant.",
    middleware=[ConsoleTracer()],  # prints a timing tree at the end
)

if __name__ == "__main__":
    print("Tool schema the model receives:")
    print(" ", get_weather.name, "->", get_weather.parameters["properties"])

    final = agent.invoke({"messages": [Message.user("What's the weather in Tokyo?")]})

    print("\nTranscript:")
    for m in final["messages"]:
        label = m.name or m.role
        preview = m.content[:100] or f"[tool_calls: {[c.name for c in m.tool_calls]}]"
        print(f"  {label:>10}: {preview}")

    print(f"\nSteps taken: {final['steps']}")
    print("\nGraph structure (paste into any Mermaid renderer):")
    print(agent.to_mermaid())
