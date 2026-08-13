"""
agentkit.evals
==============

Stage 2 of the FDE loop: **evals** — turn non-determinism into evidence.

THE PROBLEM EVALS SOLVE
-----------------------
You changed a prompt. Is the agent better or worse? Without evals the honest
answer is "I ran it three times and it seemed fine", which is not an answer,
and it is the reason most AI pilots die between the demo and production. Nobody
will let an agent touch their invoices on the strength of a vibe.

An eval suite converts "seems fine" into "41 of 50 cases passed; of the nine
failures, five were missing data and four pulled the wrong record". That
sentence is what gets an agent deployed. It is also what lets you change
things later without fear, because you can prove you did not break anything.

THE HARD PART IS THE DATASET, NOT THE HARNESS
----------------------------------------------
The code here is maybe 200 lines. The work is assembling the golden dataset:
real inputs with known-correct outputs, drawn from what the business already
did by hand. Two rules that decide whether it is worth anything:

  1. **Use real historical data.** Ten thousand past emails already labelled
     by the person who did the job is a gift. Cases you invented test whether
     the agent matches your imagination.

  2. **Weight it toward the unhappy paths.** A dataset of clean inputs proves
     the agent handles the case that was never going to fail. Every exception
     you observed during the audit should become a case here.

For genuinely subjective outputs (does this presentation look good?) there is
no fully objective scorer. `LLMJudge` gets you a consistent, cheap proxy, and
`--needs-review` routing plus human feedback closes the rest of the gap. Do
not pretend a judge model is ground truth; it is a fast second opinion that
correlates with one.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from .types import Message


# ---------------------------------------------------------------------------
# Cases and datasets
# ---------------------------------------------------------------------------
@dataclass
class EvalCase:
    """One test: an input, the expected output, and how to judge it.

    `tags` earn their keep once the suite is large. Tagging cases by failure
    category ("missing-attachment", "duplicate-vendor") turns a bare pass rate
    into a diagnosis, because the report can then tell you *which kind* of
    input the agent is failing on.
    """

    input: Any
    expected: Any = None
    #: Free-form: which exception class this came from, which client, which
    #: severity. Drives the per-tag breakdown in the report.
    tags: list[str] = field(default_factory=list)
    #: Human-readable identifier. Defaults to a truncated input.
    name: str = ""
    #: Cases the business considers critical. A suite can pass overall while
    #: failing every case that actually matters, so these are reported apart.
    critical: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = str(self.input)[:60].replace("\n", " ")


class GoldenDataset:
    """A named collection of eval cases.

    "Golden" means the expected outputs are trusted — verified by whoever owns
    the workflow, not by you. Getting that sign-off is a political act as much
    as a technical one, and it is worth doing explicitly: the moment the client
    agrees these 50 answers are correct, arguments about agent quality become
    arguments about evidence.
    """

    def __init__(self, name: str, cases: Sequence[EvalCase] | None = None):
        self.name = name
        self.cases: list[EvalCase] = list(cases or [])

    def add(self, input: Any, expected: Any = None, **kwargs: Any) -> "GoldenDataset":
        self.cases.append(EvalCase(input=input, expected=expected, **kwargs))
        return self

    def filter(self, tag: str) -> "GoldenDataset":
        return GoldenDataset(f"{self.name}[{tag}]", [c for c in self.cases if tag in c.tags])

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    # -- persistence --------------------------------------------------------
    def save(self, path: str) -> None:
        """Datasets belong in version control next to the code they test."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"name": self.name, "cases": [asdict(c) for c in self.cases]},
                f,
                indent=2,
                default=str,
            )

    @classmethod
    def load(cls, path: str) -> "GoldenDataset":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["name"], [EvalCase(**c) for c in data["cases"]])

    @classmethod
    def from_history(
        cls,
        name: str,
        records: Sequence[dict[str, Any]],
        input_key: str = "input",
        output_key: str = "output",
        tag_key: str | None = None,
    ) -> "GoldenDataset":
        """Build a dataset from what the business already did by hand.

        This is the highest-leverage function in the module. Every workflow
        worth automating has a paper trail of past decisions, and that trail
        is a labelled dataset nobody has thought to use.
        """
        cases = [
            EvalCase(
                input=r[input_key],
                expected=r.get(output_key),
                tags=[str(r[tag_key])] if tag_key and r.get(tag_key) else [],
                metadata={k: v for k, v in r.items() if k not in (input_key, output_key)},
            )
            for r in records
        ]
        return cls(name, cases)


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------
@dataclass
class Score:
    """The verdict on one case.

    `reason` is not optional in spirit. A failing score with no explanation
    sends you back to re-run the case by hand, which is exactly the manual
    work the suite was supposed to eliminate.
    """

    passed: bool
    value: float = 0.0  # 0-1, for scorers that produce a gradient
    reason: str = ""


#: A scorer compares one actual output against one expected output.
Scorer = Callable[[Any, Any], Score]


def exact_match(actual: Any, expected: Any) -> Score:
    """Strict equality after string normalisation. Use for enums and IDs."""
    a = str(actual).strip().lower()
    e = str(expected).strip().lower()
    ok = a == e
    return Score(ok, 1.0 if ok else 0.0, "" if ok else f"expected {e!r}, got {a!r}")


def contains(actual: Any, expected: Any) -> Score:
    """Expected substring appears in the output. Case-insensitive.

    The right scorer for "did the answer mention the 30-day window", where the
    phrasing is free but the fact is not.
    """
    ok = str(expected).strip().lower() in str(actual).lower()
    return Score(ok, 1.0 if ok else 0.0, "" if ok else f"output did not contain {expected!r}")


def json_fields(*required: str) -> Scorer:
    """Check specific fields of a JSON output, ignoring the rest.

        scorer = json_fields("category", "urgency")

    Field-level comparison beats whole-object equality because it tells you
    *which* field drifted, and it does not fail the case over an extra key you
    do not care about.
    """

    def score(actual: Any, expected: Any) -> Score:
        a = _as_dict(actual)
        e = _as_dict(expected)
        if a is None:
            return Score(False, 0.0, "output was not valid JSON")
        wrong = [f for f in required if a.get(f) != (e or {}).get(f)]
        if wrong:
            detail = "; ".join(f"{f}: expected {(e or {}).get(f)!r}, got {a.get(f)!r}" for f in wrong)
            return Score(False, 1 - len(wrong) / len(required), detail)
        return Score(True, 1.0)

    return score


def numeric_within(tolerance: float = 0.01, relative: bool = False) -> Scorer:
    """Numeric comparison with a tolerance. For amounts, scores, counts."""

    def score(actual: Any, expected: Any) -> Score:
        try:
            a, e = float(_first_number(actual)), float(_first_number(expected))
        except (TypeError, ValueError):
            return Score(False, 0.0, f"could not read numbers from {actual!r} / {expected!r}")
        allowed = abs(e) * tolerance if relative else tolerance
        ok = abs(a - e) <= allowed
        return Score(ok, 1.0 if ok else 0.0, "" if ok else f"expected {e} ±{allowed}, got {a}")

    return score


def all_of(*scorers: Scorer) -> Scorer:
    """Every scorer must pass. Reports all failures, not just the first —
    fixing one at a time when three are broken wastes a lot of afternoons."""

    def score(actual: Any, expected: Any) -> Score:
        results = [s(actual, expected) for s in scorers]
        failures = [r.reason for r in results if not r.passed]
        mean = statistics.mean(r.value for r in results) if results else 0.0
        return Score(not failures, mean, "; ".join(failures))

    return score


def no_hallucinated_facts(actual: Any, expected: Any) -> Score:
    """Every bracketed citation in the answer must exist in the context.

    A narrow but genuinely useful guard for RAG: catches the specific failure
    where a model invents "[4]" because four sources felt more authoritative.
    `expected` is the context string the model was given.
    """
    cited = set(re.findall(r"\[(\d+)\]", str(actual)))
    available = set(re.findall(r"\[(\d+)\]", str(expected or "")))
    invented = sorted(cited - available)
    if invented:
        return Score(False, 0.0, f"cited non-existent source(s): {', '.join(invented)}")
    return Score(True, 1.0)


class LLMJudge:
    """Score subjective outputs with a model.

    For anything where correctness is a matter of degree — tone, completeness,
    whether a summary captured the point — there is no deterministic scorer.
    A judge model gives you something consistent and cheap enough to run on
    every change.

    Three things make judges usable rather than theatrical:
      * a **rubric** specific enough that two people would agree on it
      * a **low temperature**, because you want repeatability more than nuance
      * treating the score as a **proxy**, spot-checked against human
        judgement, not as ground truth

    A judge that agrees with your reviewers 85% of the time is useful. One you
    never validated is decoration.
    """

    def __init__(self, model: Any, rubric: str, threshold: float = 0.7):
        self.model = model
        self.rubric = rubric
        self.threshold = threshold

    def __call__(self, actual: Any, expected: Any) -> Score:
        prompt = [
            Message.system(
                "You grade AI outputs against a rubric. Reply with a JSON object: "
                '{"score": <0.0-1.0>, "reason": "<one sentence>"}. No other text.'
            ),
            Message.user(
                f"Rubric:\n{self.rubric}\n\n"
                f"Reference answer:\n{expected}\n\n"
                f"Output to grade:\n{actual}"
            ),
        ]
        raw = self.model.invoke(prompt).message.content
        try:
            data = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))  # type: ignore[union-attr]
            value = float(data.get("score", 0))
            reason = str(data.get("reason", ""))
        except (AttributeError, ValueError, TypeError, json.JSONDecodeError):
            return Score(False, 0.0, f"judge returned unparseable output: {raw[:120]}")
        return Score(value >= self.threshold, value, reason)


def _as_dict(value: Any) -> dict | None:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def _first_number(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    if not match:
        raise ValueError(f"no number in {value!r}")
    return float(match.group(0))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class CaseResult:
    case: EvalCase
    actual: Any
    score: Score
    duration_ms: float
    error: str | None = None


@dataclass
class EvalReport:
    """The evidence artifact. This is what you show the client.

    Deliberately blunt: a pass rate, a failure breakdown by category, and the
    critical-case rate reported separately. An overall 82% that hides a 40%
    rate on the cases the business cares about is worse than useless, because
    it is reassuring and wrong.
    """

    suite: str
    results: list[CaseResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.score.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return round(self.passed / self.total, 4) if self.total else 0.0

    @property
    def critical_pass_rate(self) -> float | None:
        crit = [r for r in self.results if r.case.critical]
        if not crit:
            return None
        return round(sum(1 for r in crit if r.score.passed) / len(crit), 4)

    @property
    def mean_latency_ms(self) -> float:
        return round(statistics.mean([r.duration_ms for r in self.results]), 1) if self.results else 0.0

    @property
    def p95_latency_ms(self) -> float:
        """Tail latency, because the mean hides the runs that time out."""
        if not self.results:
            return 0.0
        ordered = sorted(r.duration_ms for r in self.results)
        return round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 1)

    def failures_by_tag(self) -> dict[str, int]:
        """Which *kind* of input is failing. The diagnostic view."""
        counts: dict[str, int] = {}
        for r in self.results:
            if r.score.passed:
                continue
            for tag in r.case.tags or ["untagged"]:
                counts[tag] = counts.get(tag, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if not r.score.passed]

    def to_markdown(self, max_failures: int = 10) -> str:
        lines = [
            f"# Eval report: {self.suite}",
            "",
            f"**{self.passed}/{self.total} passed** ({self.pass_rate:.1%})  ",
        ]
        if self.critical_pass_rate is not None:
            lines.append(f"**Critical cases: {self.critical_pass_rate:.1%}**  ")
        lines += [
            f"Mean latency: {self.mean_latency_ms} ms · p95: {self.p95_latency_ms} ms",
            "",
        ]

        by_tag = self.failures_by_tag()
        if by_tag:
            lines += ["## Failures by category", ""]
            lines.extend(f"- **{tag}**: {n}" for tag, n in by_tag.items())
            lines.append("")

        fails = self.failures()
        if fails:
            lines += [f"## Failing cases (showing {min(len(fails), max_failures)} of {len(fails)})", ""]
            for r in fails[:max_failures]:
                lines.append(f"### {r.case.name}")
                lines.append(f"- **Why:** {r.error or r.score.reason or 'no reason recorded'}")
                lines.append(f"- **Expected:** `{str(r.case.expected)[:160]}`")
                lines.append(f"- **Got:** `{str(r.actual)[:160]}`")
                lines.append("")
        return "\n".join(lines)

    def summary(self) -> str:
        """One line, for CI output and standups."""
        crit = (
            f", critical {self.critical_pass_rate:.0%}"
            if self.critical_pass_rate is not None
            else ""
        )
        return (
            f"{self.suite}: {self.passed}/{self.total} passed "
            f"({self.pass_rate:.0%}{crit}), p95 {self.p95_latency_ms}ms"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_eval(
    target: Callable[[Any], Any] | Any,
    dataset: GoldenDataset,
    scorer: Scorer = exact_match,
    *,
    max_workers: int = 4,
) -> EvalReport:
    """Run every case and produce a report.

    `target` is anything callable, a Runnable, or a compiled graph — the
    adapter below means you can evaluate a one-line chain and a full agent
    with the same code, which is what makes it practical to actually keep the
    suite running.

    Exceptions are caught and recorded as failures rather than aborting the
    run. A suite that dies on case 7 tells you nothing about cases 8-50, and
    "it crashed" is itself a result worth reporting.
    """
    from concurrent.futures import ThreadPoolExecutor

    call = _make_callable(target)
    report = EvalReport(suite=dataset.name)

    def run_one(case: EvalCase) -> CaseResult:
        started = time.perf_counter()
        try:
            actual = call(case.input)
            score = scorer(actual, case.expected)
            error = None
        except Exception as exc:  # noqa: BLE001 - a crash is a failing case
            actual, error = None, f"{type(exc).__name__}: {exc}"
            score = Score(False, 0.0, error)
        return CaseResult(case, actual, score, round((time.perf_counter() - started) * 1000, 2), error)

    cases = list(dataset)
    if max_workers > 1 and len(cases) > 1:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(cases))) as pool:
            report.results = list(pool.map(run_one, cases))
    else:
        report.results = [run_one(c) for c in cases]
    return report


def _make_callable(target: Any) -> Callable[[Any], Any]:
    """Normalise chains, graphs and plain functions into one shape.

    Graphs are detected by their `builder` attribute and fed a message, since
    an agent's input is a conversation. The final assistant message is the
    output under test.
    """
    if hasattr(target, "builder") and hasattr(target, "invoke"):  # CompiledGraph
        def call_graph(x: Any) -> Any:
            msg = x if isinstance(x, Message) else Message.user(str(x))
            state = target.invoke({"messages": [msg]})
            msgs = state.get("messages", [])
            return msgs[-1].content if msgs else ""

        return call_graph

    if hasattr(target, "invoke"):  # Runnable
        return lambda x: target.invoke(x)

    if callable(target):
        return target

    raise TypeError(f"Cannot evaluate {type(target).__name__}: not callable and has no .invoke()")


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
@dataclass
class Regression:
    """What changed between two eval runs.

    The point of a baseline is not the headline pass rate. It is `newly_failing`
    — cases that used to work and now do not. A change that lifts the rate from
    82% to 85% while breaking four previously-passing cases is usually a bad
    change, and the aggregate number hides that completely.
    """

    baseline: EvalReport
    current: EvalReport

    @property
    def delta(self) -> float:
        return round(self.current.pass_rate - self.baseline.pass_rate, 4)

    @property
    def newly_failing(self) -> list[str]:
        was_ok = {r.case.name for r in self.baseline.results if r.score.passed}
        now_bad = {r.case.name for r in self.current.results if not r.score.passed}
        return sorted(was_ok & now_bad)

    @property
    def newly_passing(self) -> list[str]:
        was_bad = {r.case.name for r in self.baseline.results if not r.score.passed}
        now_ok = {r.case.name for r in self.current.results if r.score.passed}
        return sorted(was_bad & now_ok)

    @property
    def is_regression(self) -> bool:
        """Any previously-passing case that now fails counts as a regression,
        regardless of what the aggregate did."""
        return bool(self.newly_failing)

    def summary(self) -> str:
        arrow = "+" if self.delta >= 0 else ""
        parts = [
            f"pass rate {self.baseline.pass_rate:.0%} -> {self.current.pass_rate:.0%} "
            f"({arrow}{self.delta:.1%})"
        ]
        if self.newly_failing:
            parts.append(f"REGRESSION: {len(self.newly_failing)} case(s) broke: "
                         + ", ".join(self.newly_failing[:5]))
        if self.newly_passing:
            parts.append(f"fixed: {len(self.newly_passing)}")
        return " | ".join(parts)
