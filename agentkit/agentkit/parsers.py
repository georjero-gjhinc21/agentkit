"""
agentkit.parsers
================

The last stage of a chain: turn model output into something your code can use.

WHY THIS DESERVES ITS OWN MODULE
--------------------------------
Models emit text. Programs need values. The gap between them is where a
surprising share of production incidents live, because models are *nearly*
compliant: they return valid JSON wrapped in a markdown fence, or a list with
one stray "Sure, here you go:" preamble, or the right shape with a key spelled
differently.

Parsers absorb that. Two principles run through this file:

  1. **Be liberal about surface, strict about shape.** Strip fences, find the
     first balanced JSON object, tolerate stray prose — then validate the
     result hard and fail loudly if a required field is missing.

  2. **Failed parses should be fixable, not fatal.** `get_format_instructions`
     produces text you inject into the prompt so the model knows the contract
     up front, and `RetryOutputParser` feeds the error back so it can correct
     itself. Prevention first, recovery second.
"""

from __future__ import annotations

import json
import re
from abc import abstractmethod
from typing import Any, Callable, Sequence

from .errors import AgentKitError
from .runnables import Runnable
from .types import Message


class OutputParserError(AgentKitError):
    """Model output could not be parsed. Carries the raw text for debugging —
    without it you are guessing at what the model actually said."""

    def __init__(self, message: str, raw: str = ""):
        self.raw = raw
        super().__init__(f"{message}\nRaw output: {raw[:500]}")


def _text_of(value: Any) -> str:
    """Accept a Message or a string, since parsers sit after model stages."""
    if isinstance(value, Message):
        return value.content
    return value if isinstance(value, str) else str(value)


# ---------------------------------------------------------------------------
class BaseOutputParser(Runnable):
    """Parsers are Runnables so they compose: `prompt | model | parser`."""

    @abstractmethod
    def parse(self, text: str) -> Any: ...

    def invoke(self, input: Any, config: dict[str, Any] | None = None) -> Any:
        return self.parse(_text_of(input))

    def get_format_instructions(self) -> str:
        """Text to paste into the prompt describing the expected format.

        Injecting this is the single highest-leverage thing you can do for
        parse reliability — far more effective than a cleverer parser.
        """
        return ""


class StrOutputParser(BaseOutputParser):
    """Message -> plain string. The default final stage of a chat chain."""

    def parse(self, text: str) -> str:
        return text

    def invoke(self, input: Any, config: dict[str, Any] | None = None) -> str:
        return _text_of(input)


class JSONOutputParser(BaseOutputParser):
    """Extract JSON, tolerating the ways models decorate it.

    Handles: bare JSON, ```json fenced blocks, and JSON embedded in prose. The
    brace-matching scan is there because a naive `text[first:last]` breaks the
    moment the model writes a sentence containing a `}` after the object.
    """

    def parse(self, text: str) -> Any:
        cleaned = text.strip()

        fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
        if fence:
            cleaned = fence.group(1).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        extracted = self._first_balanced(cleaned)
        if extracted is not None:
            try:
                return json.loads(extracted)
            except json.JSONDecodeError as exc:
                raise OutputParserError(f"Found JSON-like text but it is invalid: {exc}", text)
        raise OutputParserError("No JSON object or array found in the output.", text)

    @staticmethod
    def _first_balanced(text: str) -> str | None:
        """Scan for the first balanced {...} or [...], ignoring braces inside
        strings and honouring backslash escapes."""
        start = next((i for i, c in enumerate(text) if c in "{["), None)
        if start is None:
            return None
        opener = text[start]
        closer = "}" if opener == "{" else "]"
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if escaped:
                escaped = False
                continue
            if c == "\\":
                escaped = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    def get_format_instructions(self) -> str:
        return "Respond with a single valid JSON value and nothing else. No markdown fences, no commentary."


class StructuredOutputParser(BaseOutputParser):
    """JSON plus a required-field contract.

        parser = StructuredOutputParser({
            "sentiment": "one of: positive, negative, neutral",
            "confidence": "a number between 0 and 1",
        })

    The same dict generates the format instructions AND validates the result,
    so the prompt and the check can never drift apart — which is exactly how
    structured-output bugs usually happen.
    """

    def __init__(self, schema: dict[str, str], strict: bool = True):
        self.schema = schema
        #: strict=True rejects extra keys. Turn it off when a model habitually
        #: adds a harmless "reasoning" field you would rather ignore.
        self.strict = strict
        self._json = JSONOutputParser()

    def parse(self, text: str) -> dict[str, Any]:
        data = self._json.parse(text)
        if not isinstance(data, dict):
            raise OutputParserError(f"Expected a JSON object, got {type(data).__name__}.", text)

        missing = [k for k in self.schema if k not in data]
        if missing:
            raise OutputParserError(f"Missing required field(s): {', '.join(missing)}", text)
        if self.strict:
            extra = [k for k in data if k not in self.schema]
            if extra:
                raise OutputParserError(f"Unexpected field(s): {', '.join(extra)}", text)
        return data

    def get_format_instructions(self) -> str:
        fields = "\n".join(f'  "{k}": {v}' for k, v in self.schema.items())
        return f"Respond with a JSON object containing exactly these fields:\n{{\n{fields}\n}}\nNo other text."


class ListOutputParser(BaseOutputParser):
    """Parse a list, whether the model returns JSON, bullets, or numbers.

    Models ignore "return a comma-separated list" often enough that accepting
    all three common shapes is cheaper than fighting it.
    """

    def __init__(self, separator: str = ","):
        self.separator = separator

    def parse(self, text: str) -> list[str]:
        cleaned = text.strip()
        if cleaned.startswith("["):
            try:
                data = json.loads(cleaned)
                if isinstance(data, list):
                    return [str(x).strip() for x in data]
            except json.JSONDecodeError:
                pass

        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        bullets = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln) for ln in lines]
        if len(bullets) > 1 and any(re.match(r"^\s*(?:[-*•]|\d+[.)])\s", ln) for ln in lines):
            return bullets

        return [p.strip() for p in cleaned.split(self.separator) if p.strip()]

    def get_format_instructions(self) -> str:
        return "Respond with a JSON array of strings and nothing else."


class BooleanOutputParser(BaseOutputParser):
    """Yes/no answers, for routers and LLM-as-judge gates."""

    TRUE = {"yes", "true", "y", "1", "correct", "affirmative"}
    FALSE = {"no", "false", "n", "0", "incorrect", "negative"}

    def parse(self, text: str) -> bool:
        # Check the first token first — models love to add an explanation, and
        # the leading word is the answer.
        tokens = re.findall(r"[a-z0-9]+", text.strip().lower())
        for token in tokens[:3]:
            if token in self.TRUE:
                return True
            if token in self.FALSE:
                return False
        raise OutputParserError("Could not interpret the output as yes/no.", text)

    def get_format_instructions(self) -> str:
        return "Answer with exactly one word: yes or no."


# ---------------------------------------------------------------------------
class RetryOutputParser(BaseOutputParser):
    """On a parse failure, show the model its own bad output and ask again.

    This works far better than it has any right to: models are good at fixing
    a specific, quoted mistake. Keep `max_retries` small — if two attempts
    fail, the prompt is wrong, not the model, and burning more calls will not
    discover that for you.
    """

    def __init__(self, parser: BaseOutputParser, model: Any, max_retries: int = 2):
        self.parser = parser
        self.model = model
        self.max_retries = max_retries

    def parse(self, text: str) -> Any:
        current = text
        last: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                return self.parser.parse(current)
            except OutputParserError as exc:
                last = exc
                fix = [
                    Message.system(
                        "Your previous output could not be parsed. Return ONLY corrected "
                        "output in the required format, with no explanation."
                    ),
                    Message.user(
                        f"Required format:\n{self.parser.get_format_instructions()}\n\n"
                        f"Your output:\n{current}\n\nError: {exc}"
                    ),
                ]
                current = self.model.invoke(fix).message.content
        raise OutputParserError(f"Still unparseable after {self.max_retries} retries: {last}", current)

    def get_format_instructions(self) -> str:
        return self.parser.get_format_instructions()


class TransformOutputParser(BaseOutputParser):
    """Wrap any function as a parser. For the one-off case not worth a class."""

    def __init__(self, func: Callable[[str], Any]):
        self.func = func

    def parse(self, text: str) -> Any:
        return self.func(text)
