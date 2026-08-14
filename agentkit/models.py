"""
agentkit.models
===============

The provider boundary. Everything above this line is vendor-neutral;
everything below it knows about one specific API.

To support a new provider you implement exactly one method:

    class MyModel(BaseChatModel):
        def invoke(self, messages, tools=None, **kw) -> ModelResponse: ...

and nothing else in the framework needs to change. That is the whole point of
the neutral `Message` type in `agentkit.types`.

WHY THE SDKs ARE IMPORTED LAZILY
--------------------------------
`import anthropic` happens inside `__init__`, not at module top level. This
keeps the framework installable with zero dependencies and means a user who
only runs OpenAI never needs the Anthropic SDK on disk. The cost is that a
missing SDK surfaces at construction time instead of import time, so we raise
a message that says exactly what to `pip install`.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator, Sequence

from .errors import ConfigurationError, ModelError
from .types import Message, ModelResponse, ToolCall, Usage


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class BaseChatModel(ABC):
    """Interface every model adapter implements.

    Contract:
      * `invoke` takes neutral Messages and neutral tool schemas, and returns a
        neutral assistant Message (which may contain tool_calls).
      * It must not mutate the input list.
      * It should raise `ModelError` for provider failures so callers have one
        exception type to handle.
    """

    #: Human-readable identifier used in traces and logs.
    model: str = "unknown"

    @abstractmethod
    def invoke(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """One turn of generation."""

    def stream(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield text chunks.

        Default implementation is non-streaming: call `invoke` and emit the
        whole thing at once. Adapters that support real streaming override it.
        Callers can therefore always use `.stream()` without feature-detecting.
        """
        yield self.invoke(messages, tools, **kwargs).message.content

    # -- composition ---------------------------------------------------------
    def __or__(self, other: Any) -> Any:
        """Let a model sit in an LCEL-style chain: `prompt | model | parser`.

        The import is local because `runnables` imports `models` to recognise
        chat models during coercion — a module-level import here would be a
        cycle. Piping wraps `self` in a `RunnableModel`, which adapts this
        class's `(messages, tools) -> ModelResponse` signature to the
        Runnable's `input -> Message` one.
        """
        from .runnables import RunnableModel, coerce_runnable  # noqa: PLC0415

        return RunnableModel(self) | coerce_runnable(other)

    def __ror__(self, other: Any) -> Any:
        from .runnables import RunnableModel, coerce_runnable  # noqa: PLC0415

        return coerce_runnable(other) | RunnableModel(self)

    # -- shared retry helper -------------------------------------------------
    def _with_retries(
        self,
        fn: Callable[[], Any],
        *,
        attempts: int = 3,
        base_delay: float = 0.5,
        retry_on: tuple[type[BaseException], ...] = (Exception,),
    ) -> Any:
        """Exponential backoff with jitter.

        Rate limits and transient 5xx are the normal case in agent workloads,
        not the exception — a single agent run can issue dozens of model calls.
        Jitter matters when many agents retry in lockstep.
        """
        last: BaseException | None = None
        for i in range(attempts):
            try:
                return fn()
            except retry_on as exc:  # noqa: PERF203
                last = exc
                if i == attempts - 1:
                    break
                time.sleep(base_delay * (2**i) + random.uniform(0, 0.25))
        raise ModelError(f"{type(self).__name__} failed after {attempts} attempts: {last}") from last


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
class AnthropicModel(BaseChatModel):
    """Adapter for the Anthropic Messages API.

    Translation notes (this is where the vendor-specific weirdness lives):
      * system prompts are a top-level `system` parameter, NOT a message
      * content is a list of typed blocks; tool calls are `tool_use` blocks
      * tool results go in a *user* message as `tool_result` blocks
      * tool schemas use `input_schema`, not `parameters`
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_key: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        **client_kwargs: Any,
    ):
        try:
            import anthropic  # noqa: PLC0415 - lazy on purpose
        except ImportError as exc:  # pragma: no cover
            raise ConfigurationError(
                "AnthropicModel requires the SDK: pip install anthropic"
            ) from exc

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ConfigurationError("Set ANTHROPIC_API_KEY or pass api_key=...")

        self._client = anthropic.Anthropic(api_key=key, **client_kwargs)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    # -- neutral -> Anthropic wire format ------------------------------------
    @staticmethod
    def _convert(messages: Sequence[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        system: list[str] = []
        out: list[dict[str, Any]] = []

        for m in messages:
            if m.role == "system":
                system.append(m.content)
            elif m.role == "user":
                out.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for c in m.tool_calls:
                    blocks.append(
                        {"type": "tool_use", "id": c.id, "name": c.name, "input": c.args}
                    )
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
            elif m.role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content,
                }
                if m.metadata.get("ok") is False:
                    block["is_error"] = True
                # Consecutive tool results belong in ONE user message. Merging
                # them here avoids an API error when tools ran in parallel.
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})

        return ("\n\n".join(system) if system else None), out

    def invoke(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        system, converted = self._convert(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "messages": converted,
        }
        if system:
            payload["system"] = system
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = [
                {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                for t in tools
            ]
        payload.update(kwargs)

        resp = self._with_retries(lambda: self._client.messages.create(**payload))

        # -- Anthropic wire format -> neutral --------------------------------
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(name=block.name, args=dict(block.input), id=block.id))

        return ModelResponse(
            message=Message.assistant("".join(text_parts), tool_calls=calls),
            usage=Usage(
                input_tokens=getattr(resp.usage, "input_tokens", 0),
                output_tokens=getattr(resp.usage, "output_tokens", 0),
                model_calls=1,
            ),
            finish_reason=getattr(resp, "stop_reason", None),
            raw=resp,
        )


# ---------------------------------------------------------------------------
# OpenAI (and any OpenAI-compatible endpoint: vLLM, Ollama, Together, ...)
# ---------------------------------------------------------------------------
class OpenAIModel(BaseChatModel):
    """Adapter for the OpenAI Chat Completions API.

    Because so many servers copy this schema, pointing `base_url` at a local
    vLLM or Ollama instance gives you offline models for free.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        **client_kwargs: Any,
    ):
        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ConfigurationError("OpenAIModel requires: pip install openai") from exc

        key = api_key or os.getenv("OPENAI_API_KEY") or "not-needed-for-local"
        self._client = openai.OpenAI(api_key=key, base_url=base_url, **client_kwargs)
        self.model = model
        self.temperature = temperature

    @staticmethod
    def _convert(messages: Sequence[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role in ("system", "user"):
                out.append({"role": m.role, "content": m.content})
            elif m.role == "assistant":
                d: dict[str, Any] = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    d["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.args)},
                        }
                        for c in m.tool_calls
                    ]
                out.append(d)
            elif m.role == "tool":
                out.append(
                    {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
                )
        return out

    def invoke(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        payload: dict[str, Any] = {"model": self.model, "messages": self._convert(messages)}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["parameters"],
                    },
                }
                for t in tools
            ]
        payload.update(kwargs)

        resp = self._with_retries(lambda: self._client.chat.completions.create(**payload))
        choice = resp.choices[0]

        calls: list[ToolCall] = []
        for tc in choice.message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                # Models occasionally emit malformed JSON. Preserve it so the
                # tool layer can report a useful error instead of crashing here.
                args = {"__raw_arguments__": tc.function.arguments}
            calls.append(ToolCall(name=tc.function.name, args=args, id=tc.id))

        usage = getattr(resp, "usage", None)
        return ModelResponse(
            message=Message.assistant(choice.message.content or "", tool_calls=calls),
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                model_calls=1,
            ),
            finish_reason=choice.finish_reason,
            raw=resp,
        )


# ---------------------------------------------------------------------------
# LiteLLM — unified interface to 100+ providers
# ---------------------------------------------------------------------------
class LiteLLMModel(BaseChatModel):
    """Adapter for LiteLLM, providing access to 100+ LLM providers.

    LiteLLM gives you a unified interface to OpenAI, Anthropic, Azure, Bedrock,
    Cohere, Replicate, HuggingFace, Ollama, and many more. Just change the model
    name and optionally set the corresponding API key.

    Examples:
        # OpenAI
        LiteLLMModel("gpt-4o-mini")  # needs OPENAI_API_KEY

        # Anthropic
        LiteLLMModel("claude-sonnet-4-5")  # needs ANTHROPIC_API_KEY

        # Azure OpenAI
        LiteLLMModel("azure/gpt-4")  # needs AZURE_API_KEY, AZURE_API_BASE

        # Local Ollama
        LiteLLMModel("ollama/llama3")  # no key needed

        # AWS Bedrock
        LiteLLMModel("bedrock/anthropic.claude-3-sonnet")  # AWS credentials

    See: https://docs.litellm.ai/docs/providers for full provider list.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **litellm_kwargs: Any,
    ):
        try:
            import litellm  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ConfigurationError(
                "LiteLLMModel requires: pip install litellm"
            ) from exc

        # LiteLLM reads API keys from environment or you can pass them
        self._litellm = litellm
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.litellm_kwargs = litellm_kwargs

    @staticmethod
    def _convert(messages: Sequence[Message]) -> list[dict[str, Any]]:
        """Convert neutral Messages to LiteLLM format (OpenAI-compatible)."""
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role in ("system", "user"):
                out.append({"role": m.role, "content": m.content})
            elif m.role == "assistant":
                d: dict[str, Any] = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    d["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.args)},
                        }
                        for c in m.tool_calls
                    ]
                out.append(d)
            elif m.role == "tool":
                out.append(
                    {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
                )
        return out

    def invoke(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._convert(messages),
        }

        if self.api_key:
            payload["api_key"] = self.api_key
        if self.api_base:
            payload["api_base"] = self.api_base
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["parameters"],
                    },
                }
                for t in tools
            ]

        # Merge any additional kwargs
        payload.update(self.litellm_kwargs)
        payload.update(kwargs)

        resp = self._with_retries(lambda: self._litellm.completion(**payload))
        choice = resp.choices[0]

        # Parse tool calls
        calls: list[ToolCall] = []
        for tc in getattr(choice.message, "tool_calls", None) or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"__raw_arguments__": tc.function.arguments}
            calls.append(ToolCall(name=tc.function.name, args=args, id=tc.id))

        usage = getattr(resp, "usage", None)
        return ModelResponse(
            message=Message.assistant(getattr(choice.message, "content", None) or "", tool_calls=calls),
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                model_calls=1,
            ),
            finish_reason=choice.finish_reason,
            raw=resp,
        )


# ---------------------------------------------------------------------------
# Fake model — the most useful class in this file
# ---------------------------------------------------------------------------
class FakeModel(BaseChatModel):
    """A scripted model for tests, examples and CI.

    Agent frameworks are notoriously hard to test because the interesting
    logic (routing, retries, loop termination) is entangled with a
    non-deterministic network call. `FakeModel` cuts that knot: you hand it a
    list of responses, or a function of the message history, and the graph
    behaves deterministically.

    Every example in `examples/` runs against this with no API key, so a new
    contributor can `git clone && python examples/01_minimal_agent.py` and see
    the framework work in under ten seconds.
    """

    def __init__(
        self,
        responses: Sequence[Message | str] | None = None,
        handler: Callable[[list[Message]], Message] | None = None,
        model: str = "fake",
    ):
        if responses is None and handler is None:
            raise ConfigurationError("FakeModel needs either `responses` or `handler`.")
        self._responses = list(responses or [])
        self._handler = handler
        self._i = 0
        self.model = model
        # A lock because `batch()` calls models from several threads. The
        # script index stays consistent, though WHICH scripted response a
        # given input receives is then arbitrary - use `handler=` instead of
        # `responses=` when a batch test needs deterministic pairing.
        self._lock = threading.Lock()
        #: Every call recorded, so tests can assert on what the agent asked for.
        self.calls: list[list[Message]] = []

    def invoke(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        history = list(messages)

        with self._lock:
            self.calls.append(history)
            if self._handler is None:
                if self._i >= len(self._responses):
                    # Running off the end of the script is a test bug; say so.
                    raise ModelError(
                        f"FakeModel exhausted after {len(self._responses)} responses. "
                        "The agent looped more times than the script anticipated."
                    )
                nxt = self._responses[self._i]
                self._i += 1

        if self._handler is not None:
            msg = self._handler(history)
        else:
            msg = Message.assistant(nxt) if isinstance(nxt, str) else nxt

        return ModelResponse(
            message=msg,
            usage=Usage(input_tokens=len(history), output_tokens=1, model_calls=1),
            finish_reason="tool_use" if msg.has_tool_calls else "end_turn",
        )
