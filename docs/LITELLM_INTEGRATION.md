# LiteLLM Integration Guide

## Overview

Agentkit now includes **LiteLLMModel**, providing unified access to **100+ LLM providers** through a single interface. Switch between OpenAI, Anthropic, Azure, Bedrock, Cohere, Ollama, and many more by simply changing the model name.

## Quick Start

```python
from agentkit import create_agent, tool, LiteLLMModel, Message

@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    return {"city": city, "temp_c": 24}

# Use any provider - just change the model name
model = LiteLLMModel("gpt-4o-mini")  # OpenAI
# model = LiteLLMModel("claude-sonnet-4-5")  # Anthropic
# model = LiteLLMModel("ollama/llama3")  # Local Ollama

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
)

result = agent.invoke({"messages": [Message.user("Weather in Tokyo?")]})
print(result["messages"][-1].content)
```

## Installation

```bash
pip install litellm
```

Or install with agentkit's optional dependencies:

```bash
pip install -e ".[litellm]"
```

## Supported Providers

### Cloud Providers

| Provider | Model Name Example | Required API Key |
|----------|-------------------|------------------|
| OpenAI | `gpt-4o`, `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `claude-sonnet-4-5`, `claude-opus-4` | `ANTHROPIC_API_KEY` |
| Azure OpenAI | `azure/gpt-4` | `AZURE_API_KEY`, `AZURE_API_BASE` |
| AWS Bedrock | `bedrock/anthropic.claude-v2` | AWS credentials |
| Google Vertex AI | `vertex_ai/gemini-pro` | `VERTEXAI_PROJECT` |
| Cohere | `command-nightly` | `COHERE_API_KEY` |
| Groq | `groq/llama3-70b` | `GROQ_API_KEY` |
| Perplexity | `perplexity/llama-3.1-sonar-small` | `PERPLEXITYAI_API_KEY` |
| OpenRouter | `openrouter/anthropic/claude-3.5-sonnet` | `OPENROUTER_API_KEY` |
| Together AI | `together_ai/togethercomputer/llama-2-70b` | `TOGETHERAI_API_KEY` |
| Anyscale | `anyscale/meta-llama/Llama-2-70b-chat-hf` | `ANYSCALE_API_KEY` |

### Local Models (No API Key)

| Provider | Model Name Example | Requirements |
|----------|-------------------|--------------|
| Ollama | `ollama/llama3`, `ollama/mistral` | [Ollama installed](https://ollama.ai) |
| vLLM | `openai/model-name` (with `api_base`) | vLLM server running |
| LM Studio | `openai/local-model` (with `api_base`) | LM Studio running |

### Full Provider List

See [LiteLLM Providers Documentation](https://docs.litellm.ai/docs/providers) for the complete list of 100+ supported providers.

## Usage Examples

### Example 1: OpenAI

```python
from agentkit import LiteLLMModel

model = LiteLLMModel(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1024,
)
```

**Environment:**
```bash
export OPENAI_API_KEY=sk-...
```

### Example 2: Anthropic Claude

```python
model = LiteLLMModel(
    model="claude-sonnet-4-5",
    temperature=0.7,
    max_tokens=4096,
)
```

**Environment:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### Example 3: Azure OpenAI

```python
model = LiteLLMModel(
    model="azure/gpt-4",
    api_base="https://your-endpoint.openai.azure.com/",
)
```

**Environment:**
```bash
export AZURE_API_KEY=...
export AZURE_API_BASE=https://your-endpoint.openai.azure.com/
export AZURE_API_VERSION=2024-02-15-preview
```

### Example 4: AWS Bedrock

```python
model = LiteLLMModel(
    model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
)
```

**Environment:**
```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION_NAME=us-east-1
```

### Example 5: Local Ollama

```python
model = LiteLLMModel(
    model="ollama/llama3",
    api_base="http://localhost:11434",
)
```

**Setup:**
```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3

# Ollama runs automatically on localhost:11434
```

### Example 6: Multiple Providers with Fallback

```python
from agentkit import create_agent
from agentkit.runnables import FallbackRunnable

# Primary: OpenAI, Fallback: Anthropic
primary = LiteLLMModel("gpt-4o-mini")
fallback = LiteLLMModel("claude-sonnet-4-5")

# Automatically falls back if primary fails
chain = FallbackRunnable([primary, fallback])

agent = create_agent(model=primary, tools=[...])
```

## Configuration Options

### Basic Parameters

```python
LiteLLMModel(
    model="gpt-4o-mini",           # Model name (required)
    api_key="sk-...",              # Override env var (optional)
    api_base="https://...",        # Custom endpoint (optional)
    temperature=0.7,               # Sampling temperature (optional)
    max_tokens=1024,              # Max output tokens (optional)
    **litellm_kwargs               # Additional LiteLLM args
)
```

### Advanced Options

```python
model = LiteLLMModel(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1024,
    
    # LiteLLM-specific kwargs
    timeout=30,                    # Request timeout
    max_retries=3,                # Auto-retry on failure
    fallbacks=["gpt-3.5-turbo"],  # Fallback models
    metadata={                     # Custom metadata
        "user_id": "123",
        "project": "agentkit",
    },
)
```

## API Key Management

### Environment Variables (Recommended)

```bash
# Set in .env or shell
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GROQ_API_KEY=gsk_...
```

Then use without explicit key:
```python
model = LiteLLMModel("gpt-4o-mini")  # Reads OPENAI_API_KEY
```

### Explicit API Key

```python
model = LiteLLMModel(
    model="gpt-4o-mini",
    api_key="sk-explicit-key-here"
)
```

### .env File

```python
from dotenv import load_dotenv
load_dotenv()  # Loads from .env file

model = LiteLLMModel("gpt-4o-mini")  # Reads from environment
```

## Provider Switching

The beauty of LiteLLM is **zero code changes** when switching providers:

```python
# Development: Use cheap/fast model
model = LiteLLMModel("gpt-4o-mini")

# Production: Use more capable model
model = LiteLLMModel("claude-sonnet-4-5")

# Local testing: Use Ollama (no API cost)
model = LiteLLMModel("ollama/llama3")
```

Everything else in your agent code stays identical.

## Tool Calling Support

LiteLLM supports tool/function calling for compatible providers:

```python
from agentkit import create_agent, tool, LiteLLMModel

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# Works with any tool-capable provider
model = LiteLLMModel("gpt-4o-mini")  # ✓ Supports tools
# model = LiteLLMModel("claude-sonnet-4-5")  # ✓ Supports tools
# model = LiteLLMModel("groq/llama3-70b")  # ✓ Supports tools

agent = create_agent(model=model, tools=[multiply])
```

**Providers with tool support:**
- ✅ OpenAI (GPT-4, GPT-3.5)
- ✅ Anthropic (Claude 3+)
- ✅ Google Gemini
- ✅ Groq
- ✅ Together AI
- ⚠️ Ollama (limited, model-dependent)

## Cost Tracking

LiteLLM automatically tracks costs across providers:

```python
import litellm

# Enable cost tracking
litellm.success_callback = ["langfuse"]

model = LiteLLMModel("gpt-4o-mini")
result = agent.invoke({"messages": [...]})

# Check usage
print(result["usage"])
# Usage(input_tokens=100, output_tokens=50, model_calls=1)
```

## Error Handling

```python
from agentkit.errors import ModelError

try:
    result = agent.invoke({"messages": [...]})
except ModelError as e:
    print(f"Model call failed: {e}")
    # LiteLLM will retry automatically based on max_retries
```

## Best Practices

### 1. Start with Local Models

```python
# Develop/test with Ollama (free, fast iteration)
model = LiteLLMModel("ollama/llama3")

# Deploy with cloud provider
# model = LiteLLMModel("gpt-4o-mini")
```

### 2. Use Environment Variables

```python
# Good: Reads from environment
model = LiteLLMModel("gpt-4o-mini")

# Avoid: Hardcoded keys
# model = LiteLLMModel("gpt-4o-mini", api_key="sk-hardcoded")
```

### 3. Provider-Specific Settings

```python
# Store provider configs separately
PROVIDERS = {
    "openai": {"model": "gpt-4o-mini", "temperature": 0.7},
    "anthropic": {"model": "claude-sonnet-4-5", "temperature": 0.7},
    "local": {"model": "ollama/llama3", "api_base": "http://localhost:11434"},
}

# Select at runtime
config = PROVIDERS["openai"]
model = LiteLLMModel(**config)
```

### 4. Fallback Chains

```python
# Primary fast/cheap, fallback to capable/expensive
model = LiteLLMModel(
    model="gpt-4o-mini",
    fallbacks=["gpt-4o", "claude-sonnet-4-5"]
)
```

## Comparison: LiteLLM vs Direct Providers

| Feature | AnthropicModel | OpenAIModel | LiteLLMModel |
|---------|----------------|-------------|--------------|
| Providers | 1 (Anthropic) | 1 (OpenAI) | 100+ |
| API Key | ANTHROPIC_API_KEY | OPENAI_API_KEY | Provider-specific |
| Local models | ❌ | ⚠️ (via base_url) | ✅ (Ollama, vLLM) |
| Auto-retry | ✅ | ✅ | ✅ |
| Cost tracking | ❌ | ❌ | ✅ |
| Fallbacks | ❌ | ❌ | ✅ |
| Dependencies | anthropic | openai | litellm |

**When to use each:**

- **AnthropicModel**: You only use Anthropic Claude
- **OpenAIModel**: You only use OpenAI or OpenAI-compatible servers
- **LiteLLMModel**: You want flexibility, multi-provider support, or cost tracking

## Troubleshooting

### Provider Not Found

```
Error: litellm.exceptions.BadRequestError: Unknown provider
```

**Fix:** Check the [provider list](https://docs.litellm.ai/docs/providers) for correct model name format.

### API Key Not Set

```
Error: AuthenticationError: No API key provided
```

**Fix:** Set the appropriate environment variable:
```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

### Connection Refused (Ollama)

```
Error: Connection refused at http://localhost:11434
```

**Fix:** Ensure Ollama is running:
```bash
ollama serve  # If not running automatically
ollama pull llama3  # Download model
```

### Tool Calling Not Supported

```
Error: Provider does not support function calling
```

**Fix:** Switch to a provider that supports tools (OpenAI, Anthropic, Groq, etc.) or handle logic without tools.

## Example: Complete Agent with LiteLLM

```python
from agentkit import (
    create_agent,
    tool,
    LiteLLMModel,
    Message,
    ConsoleTracer,
    BudgetMiddleware,
)

@tool
def search_docs(query: str) -> str:
    """Search documentation."""
    return f"Results for: {query}"

@tool
def run_code(code: str) -> str:
    """Execute Python code safely."""
    return f"Executed: {code}"

# Use any provider
model = LiteLLMModel(
    model="gpt-4o-mini",  # or "claude-sonnet-4-5", "ollama/llama3", etc.
    temperature=0.7,
)

agent = create_agent(
    model=model,
    tools=[search_docs, run_code],
    system_prompt="You are a helpful coding assistant.",
    middleware=[
        ConsoleTracer(),
        BudgetMiddleware(max_tokens=10000),
    ],
)

result = agent.invoke({
    "messages": [Message.user("How do I read a CSV file in Python?")]
})

print(result["messages"][-1].content)
```

## Resources

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Supported Providers](https://docs.litellm.ai/docs/providers)
- [Example 08: Multi-Provider Demo](../examples/08_litellm_multi_provider.py)
- [Agentkit Models Documentation](../agentkit/models.py)

## Summary

**LiteLLMModel brings you:**
- ✅ Access to 100+ LLM providers
- ✅ One-line provider switching
- ✅ Local model support (Ollama, vLLM)
- ✅ Automatic retries and fallbacks
- ✅ Cost tracking across providers
- ✅ Drop-in compatibility with agentkit

**Get started:**
```bash
pip install litellm
python examples/08_litellm_multi_provider.py
```
