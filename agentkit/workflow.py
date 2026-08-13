"""
agentkit.workflow
=================

Stage 1 of the FDE loop: **audit** — map how the work actually happens, then
decide where intelligence belongs.

WHY THIS IS A MODULE AND NOT A GOOGLE DOC
------------------------------------------
The most expensive mistake in applied AI is not a bad model. It is pointing a
model at a step that never needed one. The commonly cited MIT figure is that
~95% of generative AI pilots fail to reach production, and the failures cluster
around workflows nobody actually observed before automating them.

Two things cause that:

  1. **The documented process is not the real process.** "An email arrives"
     is one box on a slide and forty sender formats in reality — PDFs,
     screenshots, forwarded threads, half of them exceptions that one person
     resolves from memory. Build for the slide and you have built for a
     workflow that does not exist.

  2. **Not every step wants an LLM.** In a ten-step workflow, typically two or
     three involve genuine judgment. The rest are lookups, validations and
     if/then branches that deterministic code does faster, cheaper and
     correctly every time. Putting a model on those steps buys you latency,
     cost and a new failure mode in exchange for nothing.

So this module makes the audit an *artifact*: a structured `Workflow` you can
diff, review with a client, and hand to the deployment stage. Writing it down
in a typed structure forces the questions people skip — what is the volume,
what does one failure cost, how does this step go wrong today.

The scoring here is a **structured prompt, not an oracle.** It encodes rules
of thumb that hold in most back-office workflows. It cannot know that your
"simple" validation step is load-bearing for a regulator. Override it, and
argue with it — that argument is the actual FDE work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Placement(str, Enum):
    """Where a step's logic should live.

    The ordering matters: prefer the cheapest, most predictable option that
    can do the job. Only escalate when the step genuinely needs what the next
    tier offers.
    """

    #: Rules, lookups, arithmetic, schema validation. Fast, free, auditable,
    #: correct every time. The default, and the right answer more often than
    #: people expect.
    DETERMINISTIC = "deterministic"

    #: Genuine ambiguity: classification of messy input, extraction from
    #: unstructured text, summarisation, drafting. Things where a rule would
    #: need a hundred branches and still miss cases.
    LLM = "llm"

    #: Irreversible, high-value, legally significant, or judgment the business
    #: is not willing to delegate. Money leaving the building. Anything a
    #: regulator would ask about.
    HUMAN = "human"

    #: Model drafts, human approves. The workhorse of real deployments: you
    #: get the speed of automation and the accountability of a signature.
    LLM_WITH_APPROVAL = "llm_with_approval"


@dataclass
class WorkflowStep:
    """One step as it is *actually* performed, not as it is documented.

    The fields that feel like busywork are the ones that pay off. `exceptions`
    is the single most valuable field in this class: there is one way a step
    goes right and a thousand ways it goes wrong, and an agent that handles
    only the happy path is a demo. `tribal_knowledge` is the second — anything
    that lives in one person's head is both the highest-value thing to encode
    and the thing that will silently break your agent when that person is on
    leave.
    """

    name: str
    description: str = ""

    #: Systems touched. Drives integration work, and reveals steps that are
    #: expensive purely because someone is rekeying between two tools.
    systems: list[str] = field(default_factory=list)

    #: How often this runs. Volume is half of ROI; a perfect automation of a
    #: monthly task is worth less than a rough one on an hourly task.
    volume_per_month: int = 0

    #: Human minutes per execution, including the waiting and the rework.
    minutes_per_run: float = 0.0

    #: Does this step require judgment a rule cannot express? Be honest. Most
    #: steps that feel judgmental are actually three rules and a lookup.
    requires_judgment: bool = False

    #: Can a wrong output be undone cheaply? Sending an email, issuing a
    #: refund and deleting a record are all irreversible in the ways that
    #: matter.
    reversible: bool = True

    #: Rough cost of one bad output, in currency. Used for risk weighting; an
    #: order of magnitude is enough, precision is not the point.
    error_cost: float = 0.0

    #: The unhappy paths. Observed, not imagined. If this list is empty you
    #: have not sat with the person who does the job.
    exceptions: list[str] = field(default_factory=list)

    #: Undocumented rules living in someone's head.
    tribal_knowledge: list[str] = field(default_factory=list)

    #: Set this to override the recommendation. The scorer is a starting
    #: point; you are the one who watched the work happen.
    placement_override: Placement | None = None

    notes: str = ""

    # -- derived ------------------------------------------------------------
    @property
    def hours_per_month(self) -> float:
        return (self.volume_per_month * self.minutes_per_run) / 60.0

    @property
    def risk_score(self) -> float:
        """0-1. Combines reversibility, blast radius and exception density.

        Deliberately crude. Its job is to sort steps into "safe to automate"
        and "needs a human on it", not to produce a number anyone should quote
        in a board deck.
        """
        score = 0.0
        if not self.reversible:
            score += 0.4
        # Blast radius is weighted comparably to reversibility. A reversible
        # mistake that costs $15,000 to unwind still deserves a human's eyes;
        # an earlier version under-weighted this and routed such steps to a
        # fully autonomous LLM.
        if self.error_cost >= 10_000:
            score += 0.4
        elif self.error_cost >= 1_000:
            score += 0.2
        # Many observed exceptions means the step is genuinely messy, which
        # raises the odds an agent meets a case it was never shown.
        score += min(0.2, len(self.exceptions) * 0.04)
        return round(min(1.0, score), 2)

    def recommend_placement(self) -> Placement:
        """Suggest where this step's logic belongs.

        The rules, in priority order:
          1. Irreversible and expensive -> a human signs, always.
          2. Needs judgment + risky     -> model drafts, human approves.
          3. Needs judgment, low risk   -> model runs it.
          4. Everything else            -> deterministic code.

        Rule 4 catching most steps is the expected outcome, not a bug.
        """
        if self.placement_override is not None:
            return self.placement_override

        risky = self.risk_score >= 0.5
        if not self.reversible and self.error_cost >= 1_000:
            return Placement.HUMAN
        if self.requires_judgment and risky:
            return Placement.LLM_WITH_APPROVAL
        if self.requires_judgment:
            return Placement.LLM
        return Placement.DETERMINISTIC


@dataclass
class Workflow:
    """A mapped workflow: the deliverable of an audit.

    Hand this to a client and it does three jobs at once — it proves you
    understood their business, it makes the automation plan concrete, and it
    surfaces the steps where they need to stay in control. That last one is
    what buys trust.
    """

    name: str
    owner: str = ""
    steps: list[WorkflowStep] = field(default_factory=list)
    #: What the business actually wants out of this. Write it in their words.
    business_goal: str = ""

    def add(self, step: WorkflowStep) -> "Workflow":
        self.steps.append(step)
        return self

    # -- aggregates ---------------------------------------------------------
    @property
    def total_hours_per_month(self) -> float:
        return round(sum(s.hours_per_month for s in self.steps), 1)

    def automatable_hours(self) -> float:
        """Hours in steps that do NOT require a human decision.

        Approval steps still cost human time, so they are excluded. Counting
        them is how automation projects end up promising savings that never
        show up in anyone's calendar.
        """
        return round(
            sum(
                s.hours_per_month
                for s in self.steps
                if s.recommend_placement() in (Placement.DETERMINISTIC, Placement.LLM)
            ),
            1,
        )

    def placement_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for step in self.steps:
            key = step.recommend_placement().value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def unmapped_exceptions(self) -> list[str]:
        """Steps with no recorded exceptions — i.e. steps you have not really
        audited yet. Every real step has unhappy paths."""
        return [s.name for s in self.steps if not s.exceptions]

    def priority_score(self, hourly_cost: float = 60.0) -> float:
        """Rough monthly value of automating this workflow, in currency.

        Time saved, discounted by the risk of the steps being automated. A
        comparison device for ranking workflows against each other — not a
        number to put in a contract.
        """
        value = self.automatable_hours() * hourly_cost
        avg_risk = (
            sum(s.risk_score for s in self.steps) / len(self.steps) if self.steps else 0.0
        )
        return round(value * (1 - 0.5 * avg_risk), 2)

    # -- output -------------------------------------------------------------
    def to_markdown(self) -> str:
        """Render the audit as a document you can put in front of a client.

        Client-readable output is not a nicety. The audit is often the first
        thing you deliver, and it is what earns the right to build anything.
        """
        lines = [
            f"# Workflow audit: {self.name}",
            "",
            f"**Owner:** {self.owner or 'unknown'}  ",
            f"**Goal:** {self.business_goal or 'not stated'}  ",
            f"**Current cost:** {self.total_hours_per_month} human hours/month  ",
            f"**Automatable:** {self.automatable_hours()} hours/month  ",
            f"**Estimated monthly value:** {self.priority_score():,.0f}",
            "",
            "## Steps",
            "",
            "| # | Step | Volume/mo | Min/run | Risk | Recommended placement |",
            "|---|------|-----------|---------|------|----------------------|",
        ]
        for i, s in enumerate(self.steps, 1):
            lines.append(
                f"| {i} | {s.name} | {s.volume_per_month} | {s.minutes_per_run} | "
                f"{s.risk_score} | `{s.recommend_placement().value}` |"
            )

        lines += ["", "## Exceptions observed", ""]
        for s in self.steps:
            if s.exceptions:
                lines.append(f"**{s.name}**")
                lines.extend(f"- {e}" for e in s.exceptions)
                lines.append("")

        tribal = [(s.name, k) for s in self.steps for k in s.tribal_knowledge]
        if tribal:
            lines += ["## Undocumented rules (tribal knowledge)", ""]
            lines.extend(f"- **{name}**: {rule}" for name, rule in tribal)
            lines.append("")

        gaps = self.unmapped_exceptions()
        if gaps:
            lines += [
                "## Audit gaps",
                "",
                "These steps have no recorded exceptions, which usually means "
                "they have not been observed in practice yet:",
                "",
            ]
            lines.extend(f"- {name}" for name in gaps)

        return "\n".join(lines)
