"""
Example 08 — Using LiteLLM for access to 100+ LLM providers

LiteLLMModel gives you a unified interface to OpenAI, Anthropic, Azure,
Bedrock, Cohere, Replicate, HuggingFace, Ollama, and 100+ more providers.

Just change the model name and set the appropriate API key.

USAGE:
    # For cloud providers, set their API key:
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...

    # Then run any example:
    python examples/08_litellm_multi_provider.py

    # For local models (Ollama), no API key needed:
    # 1. Install Ollama: https://ollama.ai
    # 2. Pull a model: ollama pull llama3
    # 3. Run: python examples/08_litellm_multi_provider.py ollama

For the full list of supported providers, see:
https://docs.litellm.ai/docs/providers
"""

from __future__ import annotations

import sys

from agentkit import (
    ConsoleTracer,
    FakeModel,
    LiteLLMModel,
    Message,
    ToolCall,
    create_agent,
    tool,
)


# Define some tools for our agent
@tool
def get_weather(city: str, units: str = "celsius") -> dict:
    """Get the current weather for a city.

    Args:
        city: Name of the city, e.g. "Paris".
        units: Either "celsius" or "fahrenheit".
    """
    # In a real app, this would call a weather API
    return {
        "city": city,
        "temperature": 24 if units == "celsius" else 75,
        "units": units,
        "conditions": "partly cloudy",
    }


@tool
def get_time(timezone: str = "UTC") -> str:
    """Get the current time in a specific timezone.

    Args:
        timezone: Timezone name, e.g. "America/New_York", "Europe/London", "UTC".
    """
    # In a real app, this would use datetime
    return f"The current time in {timezone} is 14:30"


def demo_with_fake_model():
    """Demo that works offline with no API key."""
    print("="*60)
    print("DEMO 1: Offline with FakeModel (no API key needed)")
    print("="*60)

    model = FakeModel(responses=[
        Message.assistant("", tool_calls=[
            ToolCall(name="get_weather", args={"city": "Tokyo", "units": "celsius"})
        ]),
        Message.assistant("", tool_calls=[
            ToolCall(name="get_time", args={"timezone": "Asia/Tokyo"})
        ]),
        Message.assistant("It's 14:30 in Tokyo with partly cloudy weather at 24°C."),
    ])

    agent = create_agent(
        model=model,
        tools=[get_weather, get_time],
        system_prompt="You are a helpful assistant. Be concise.",
        middleware=[ConsoleTracer()],
    )

    result = agent.invoke({
        "messages": [Message.user("What's the weather and time in Tokyo?")]
    })

    print("\nFinal answer:", result["messages"][-1].content)


def demo_with_openai():
    """Demo with OpenAI (requires OPENAI_API_KEY)."""
    print("\n" + "="*60)
    print("DEMO 2: OpenAI via LiteLLM")
    print("="*60)

    try:
        model = LiteLLMModel(
            model="gpt-4o-mini",
            temperature=0.7,
        )

        agent = create_agent(
            model=model,
            tools=[get_weather, get_time],
            system_prompt="You are a helpful assistant. Be concise.",
            middleware=[ConsoleTracer()],
        )

        result = agent.invoke({
            "messages": [Message.user("What's the weather in Paris?")]
        })

        print("\nFinal answer:", result["messages"][-1].content)

    except Exception as e:
        print(f"⚠️  Skipped (needs OPENAI_API_KEY): {e}")


def demo_with_anthropic():
    """Demo with Anthropic Claude (requires ANTHROPIC_API_KEY)."""
    print("\n" + "="*60)
    print("DEMO 3: Anthropic Claude via LiteLLM")
    print("="*60)

    try:
        model = LiteLLMModel(
            model="claude-sonnet-4-5",
            temperature=0.7,
            max_tokens=1024,
        )

        agent = create_agent(
            model=model,
            tools=[get_weather, get_time],
            system_prompt="You are a helpful assistant. Be concise.",
            middleware=[ConsoleTracer()],
        )

        result = agent.invoke({
            "messages": [Message.user("What's the time in London?")]
        })

        print("\nFinal answer:", result["messages"][-1].content)

    except Exception as e:
        print(f"⚠️  Skipped (needs ANTHROPIC_API_KEY): {e}")


def demo_with_ollama():
    """Demo with local Ollama (no API key needed, but needs Ollama installed)."""
    print("\n" + "="*60)
    print("DEMO 4: Local Ollama via LiteLLM")
    print("="*60)
    print("This requires Ollama to be installed and running:")
    print("  1. Install: https://ollama.ai")
    print("  2. Pull model: ollama pull llama3")
    print("  3. Ollama runs on http://localhost:11434")
    print()

    try:
        model = LiteLLMModel(
            model="ollama/llama3",
            api_base="http://localhost:11434",
            temperature=0.7,
        )

        agent = create_agent(
            model=model,
            tools=[get_weather, get_time],
            system_prompt="You are a helpful assistant. Be concise.",
            middleware=[ConsoleTracer()],
        )

        result = agent.invoke({
            "messages": [Message.user("What's the weather in Berlin?")]
        })

        print("\nFinal answer:", result["messages"][-1].content)

    except Exception as e:
        print(f"⚠️  Skipped (needs Ollama running): {e}")


def show_supported_providers():
    """Show examples of different provider model names."""
    print("\n" + "="*60)
    print("LITELLM SUPPORTED PROVIDERS (examples)")
    print("="*60)
    print("""
Provider             | Model Name Example           | API Key
---------------------|------------------------------|-------------------------
OpenAI               | gpt-4o, gpt-4o-mini         | OPENAI_API_KEY
Anthropic            | claude-sonnet-4-5           | ANTHROPIC_API_KEY
Azure OpenAI         | azure/gpt-4                 | AZURE_API_KEY
AWS Bedrock          | bedrock/anthropic.claude-v2 | AWS credentials
Google VertexAI      | vertex_ai/gemini-pro        | VERTEXAI_PROJECT
Cohere               | command-nightly             | COHERE_API_KEY
HuggingFace          | huggingface/meta-llama/...  | HUGGINGFACE_API_KEY
Replicate            | replicate/meta/llama-2-...  | REPLICATE_API_KEY
Together AI          | together_ai/togethercomputer| TOGETHERAI_API_KEY
Ollama (local)       | ollama/llama3               | (none - local)
Anyscale             | anyscale/meta-llama/...     | ANYSCALE_API_KEY
Perplexity           | perplexity/llama-3.1-...    | PERPLEXITYAI_API_KEY
Groq                 | groq/llama3-70b             | GROQ_API_KEY
OpenRouter           | openrouter/anthropic/...    | OPENROUTER_API_KEY

And 90+ more! See: https://docs.litellm.ai/docs/providers

Usage:
    model = LiteLLMModel(model="<provider>/<model-name>")

For local models (Ollama, vLLM), no API key is needed.
For cloud providers, set the appropriate environment variable.
""")


if __name__ == "__main__":
    print("╔" + "="*58 + "╗")
    print("║  AGENTKIT + LITELLM: 100+ LLM PROVIDERS IN ONE INTERFACE  ║")
    print("╚" + "="*58 + "╝")

    # Always run the offline demo
    demo_with_fake_model()

    # Run cloud provider demos if requested
    if len(sys.argv) > 1:
        demo_choice = sys.argv[1].lower()

        if demo_choice in ("openai", "all"):
            demo_with_openai()
        if demo_choice in ("anthropic", "claude", "all"):
            demo_with_anthropic()
        if demo_choice in ("ollama", "local", "all"):
            demo_with_ollama()
    else:
        # Show what's available
        print("\n" + "="*60)
        print("To test with real LLM providers, run:")
        print("="*60)
        print("  python examples/08_litellm_multi_provider.py openai")
        print("  python examples/08_litellm_multi_provider.py anthropic")
        print("  python examples/08_litellm_multi_provider.py ollama")
        print("  python examples/08_litellm_multi_provider.py all")

    # Show supported providers
    show_supported_providers()

    print("\n" + "="*60)
    print("KEY BENEFITS OF LITELLM")
    print("="*60)
    print("""
✓ One interface for 100+ providers
✓ Easy provider switching (change model name)
✓ Automatic retries and fallbacks
✓ Cost tracking across providers
✓ Support for local models (Ollama, vLLM)
✓ Drop-in replacement for OpenAI SDK

In agentkit, swap providers in one line:

    # From OpenAI
    model = LiteLLMModel("gpt-4o-mini")

    # To Anthropic
    model = LiteLLMModel("claude-sonnet-4-5")

    # To local Ollama
    model = LiteLLMModel("ollama/llama3")

Everything else in your agent code stays the same!
""")
