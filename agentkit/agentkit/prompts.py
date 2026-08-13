"""
agentkit.prompts
================

Prompt templates: parameterised, reusable, testable prompts.

WHY NOT JUST AN f-STRING?
-------------------------
An f-string interpolates and forgets. A template is an object, which means it
can be *inspected* (what variables does this need?), *validated* (you passed
`topic` but it wants `subject`), *versioned*, *reused* across chains, and
*composed* into a multi-message conversation. The failure it prevents most
often is the silent one: an f-string with a typo'd key raises at call time
deep inside a request; a template tells you at construction.

Three kinds, matching the three things people actually write:

    PromptTemplate         one string with {placeholders}
    ChatPromptTemplate     a system/user/assistant conversation, the real
                           shape of every modern chat API
    FewShotPromptTemplate  teach-by-example: a handful of input/output pairs
                           rendered into the prompt before the real input

Templates are Runnables, so they sit at the head of a chain:

    chain = ChatPromptTemplate.from_messages([...]) | model | StrOutputParser()
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from .errors import ConfigurationError
from .runnables import Runnable
from .types import Message

# Matches {var} but not {{escaped}} — so you can still write literal braces
# (JSON examples in prompts are common and must not be treated as variables).
_VAR_RE = re.compile(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})")


def extract_variables(template: str) -> list[str]:
    """Which placeholders does this template use? Order-preserving, deduped."""
    seen: list[str] = []
    for name in _VAR_RE.findall(template):
        if name not in seen:
            seen.append(name)
    return seen


def _render(template: str, values: dict[str, Any]) -> str:
    """Substitute placeholders and unescape doubled braces.

    Deliberately NOT `str.format`: format would choke on the literal braces in
    a JSON example and would happily accept extra keys. We want strict, quiet
    behaviour on braces and loud behaviour on missing variables.
    """
    def sub(match: re.Match) -> str:
        key = match.group(1)
        if key not in values:
            raise ConfigurationError(
                f"Missing prompt variable {key!r}. Template needs: "
                f"{', '.join(extract_variables(template))}"
            )
        return str(values[key])

    return _VAR_RE.sub(sub, template).replace("{{", "{").replace("}}", "}")


# ---------------------------------------------------------------------------
class PromptTemplate(Runnable):
    """A single templated string.

        t = PromptTemplate("Write a slogan for {product} highlighting {feature}.")
        t.format(product="agentkit", feature="readability")

    In a chain it takes a dict of variables and emits a string:

        chain = t | model | StrOutputParser()
        chain.invoke({"product": "agentkit", "feature": "readability"})
    """

    def __init__(self, template: str, partials: dict[str, Any] | None = None):
        self.template = template
        #: Values baked in now, overridable later. Useful for constants like a
        #: company name or today's date that every call would otherwise repeat.
        self.partials = dict(partials or {})
        self.input_variables = [v for v in extract_variables(template) if v not in self.partials]

    def partial(self, **kwargs: Any) -> "PromptTemplate":
        """Pre-fill some variables, returning a new template."""
        return PromptTemplate(self.template, {**self.partials, **kwargs})

    def format(self, **kwargs: Any) -> str:
        return _render(self.template, {**self.partials, **kwargs})

    def invoke(self, input: Any, config: dict[str, Any] | None = None) -> str:
        # A bare string is treated as the sole variable, so single-variable
        # chains can be called with `chain.invoke("hello")`.
        if isinstance(input, str):
            if len(self.input_variables) != 1:
                raise ConfigurationError(
                    f"Template needs {self.input_variables}; pass a dict, not a string."
                )
            return self.format(**{self.input_variables[0]: input})
        return self.format(**(input or {}))

    def __repr__(self) -> str:  # pragma: no cover
        return f"PromptTemplate(vars={self.input_variables})"


# ---------------------------------------------------------------------------
class MessagesPlaceholder:
    """A slot where a list of Messages gets spliced in.

    This is how conversation history enters a prompt. Without it you would
    have to flatten history into a string, which throws away role information
    the model actually uses.

        ChatPromptTemplate.from_messages([
            ("system", "You are helpful."),
            MessagesPlaceholder("history"),   # <- prior turns land here
            ("user", "{input}"),
        ])
    """

    def __init__(self, variable_name: str, optional: bool = True):
        self.variable_name = variable_name
        self.optional = optional


class ChatPromptTemplate(Runnable):
    """A templated multi-message conversation.

    The real shape of a chat request: a system message setting behaviour,
    optional history, then the user's turn. Keeping roles distinct matters —
    models weight system instructions differently from user text, and
    collapsing everything into one string loses that.
    """

    def __init__(self, messages: Sequence[Any], partials: dict[str, Any] | None = None):
        self.messages = list(messages)
        self.partials = dict(partials or {})

        variables: list[str] = []
        for m in self.messages:
            if isinstance(m, MessagesPlaceholder):
                variables.append(m.variable_name)
            else:
                _, content = self._unpack(m)
                variables.extend(v for v in extract_variables(content) if v not in variables)
        self.input_variables = [v for v in variables if v not in self.partials]

    @staticmethod
    def _unpack(item: Any) -> tuple[str, str]:
        """Accept ("system", "text"), a Message, or a bare string (= user)."""
        if isinstance(item, Message):
            return item.role, item.content
        if isinstance(item, tuple) and len(item) == 2:
            role, content = item
            role = {"human": "user", "ai": "assistant"}.get(role, role)
            if role not in ("system", "user", "assistant", "tool"):
                raise ConfigurationError(f"Unknown message role {role!r}.")
            return role, content
        if isinstance(item, str):
            return "user", item
        raise ConfigurationError(f"Cannot interpret prompt message: {item!r}")

    @classmethod
    def from_messages(cls, messages: Sequence[Any]) -> "ChatPromptTemplate":
        return cls(messages)

    @classmethod
    def from_template(cls, template: str, system: str | None = None) -> "ChatPromptTemplate":
        """Shorthand for the common system + user pair."""
        items: list[Any] = [("system", system)] if system else []
        items.append(("user", template))
        return cls(items)

    def partial(self, **kwargs: Any) -> "ChatPromptTemplate":
        return ChatPromptTemplate(self.messages, {**self.partials, **kwargs})

    def format_messages(self, **kwargs: Any) -> list[Message]:
        values = {**self.partials, **kwargs}
        out: list[Message] = []
        for item in self.messages:
            if isinstance(item, MessagesPlaceholder):
                history = values.get(item.variable_name)
                if history is None:
                    if not item.optional:
                        raise ConfigurationError(
                            f"Required history variable {item.variable_name!r} was not supplied."
                        )
                    continue
                out.extend(history)
                continue
            role, content = self._unpack(item)
            out.append(Message(role=role, content=_render(content, values)))
        return out

    def invoke(self, input: Any, config: dict[str, Any] | None = None) -> list[Message]:
        if isinstance(input, str):
            if len(self.input_variables) != 1:
                raise ConfigurationError(
                    f"Template needs {self.input_variables}; pass a dict, not a string."
                )
            return self.format_messages(**{self.input_variables[0]: input})
        return self.format_messages(**(input or {}))

    def __repr__(self) -> str:  # pragma: no cover
        return f"ChatPromptTemplate(vars={self.input_variables}, n={len(self.messages)})"


# ---------------------------------------------------------------------------
class FewShotPromptTemplate(Runnable):
    """Teach a pattern by showing worked examples before the real input.

        happy -> sad
        tall  -> short
        fast  -> ?

    Few-shot is the cheapest way to pin down an output *format* or a subtle
    labelling convention that prose instructions describe badly. Rules of
    thumb: 3-8 examples is the sweet spot; cover the edge cases you care
    about, including a negative one; keep them short, since every example is
    tokens on every single call.
    """

    def __init__(
        self,
        examples: Sequence[dict[str, Any]],
        example_template: str,
        suffix: str,
        prefix: str = "",
        separator: str = "\n\n",
    ):
        self.examples = list(examples)
        self.example_template = PromptTemplate(example_template)
        self.prefix = prefix
        self.suffix = suffix
        self.separator = separator
        self.input_variables = extract_variables(suffix) + extract_variables(prefix)

    def format(self, **kwargs: Any) -> str:
        rendered = [self.example_template.format(**ex) for ex in self.examples]
        parts = [p for p in [_render(self.prefix, kwargs) if self.prefix else ""] if p]
        parts.extend(rendered)
        parts.append(_render(self.suffix, kwargs))
        return self.separator.join(parts)

    def invoke(self, input: Any, config: dict[str, Any] | None = None) -> str:
        if isinstance(input, str):
            if len(self.input_variables) != 1:
                raise ConfigurationError(f"Template needs {self.input_variables}; pass a dict.")
            return self.format(**{self.input_variables[0]: input})
        return self.format(**(input or {}))
