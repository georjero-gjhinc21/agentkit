"""
agentkit.deployment
===================

Stage 3 of the FDE loop: **deployment** — get the agent into the business
without anyone losing their job over it.

THREE THINGS DECIDE WHETHER AN AGENT SURVIVES CONTACT WITH A CLIENT

1. **An audit trail.** If you cannot show what the agent did and why, nobody
   will trust it, and correctly so. `AuditTrail` records every action in an
   append-only log with the reasoning attached. This is separate from
   `tracing.py`, which exists for *you* to debug — an audit trail exists for
   the *client* to review, and increasingly for their auditors and regulators.

2. **A ladder, not a switch.** Nobody flips an agent from nothing to full
   autonomy. `AutonomyLevel` walks it up in stages: shadow (runs on real
   inputs, changes nothing, and you compare its output to the human's),
   suggest, approve-each, then auto-with-exceptions. Each rung produces the
   evidence needed to climb the next one.

3. **Numbers in their language.** Nobody funds "94% accuracy". They fund
   "saves 120 hours a month, prevents ~8 costly errors, costs $340 to run".
   `ImpactMetrics` computes exactly the three buckets a business cares about:
   cost saved, risk reduced, revenue enabled.

A NOTE ON THE WORD "AUDIT"
--------------------------
It means two different things in this codebase, and conflating them causes
confusion:

  * `workflow.py` — the *audit* as discovery: mapping how work really happens.
  * this module — the *audit trail* as evidence: a log of what the agent did.

The first happens before you build. The second runs forever after.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Sequence

from .middleware import Middleware
from .types import Message, RunConfig


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------
@dataclass
class AuditEntry:
    """One recorded action. Append-only; never edited after the fact.

    `reasoning` is the field that turns a log into a trail. "Approved invoice
    INV-88" is a log line. "Approved invoice INV-88 because the PO number
    matched and the amount was under the $5,000 threshold" is something a
    controller can check, disagree with, and sign off on.
    """

    action: str
    actor: str  # "agent" | "human:alice@corp.com" | "system"
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    thread_id: str | None = None

    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    #: Which autonomy level was in force. Lets you answer "was a human
    #: supposed to check this?" months later, when it matters most.
    autonomy: str | None = None
    #: For human decisions: approved / rejected / edited.
    decision: str | None = None
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_line(self) -> str:
        """Human-readable single line, for review UIs and terminal output."""
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        mark = "OK " if self.success else "ERR"
        tail = f" — {self.reasoning}" if self.reasoning else ""
        return f"[{when}] {mark} {self.actor:>22} {self.action}{tail}"


class AuditTrail:
    """Append-only record of everything the agent and its humans did.

    Backed by JSONL on disk when given a path, because an audit trail that
    vanishes on restart is not one. Writes are line-appends under a lock, so
    concurrent runs cannot interleave a half-written record.

    What NOT to put in here: raw PII, secrets, full document contents. Compose
    with `RedactionMiddleware` if agent output may contain either — an audit
    trail is long-lived by design, which makes it the worst place for a leak.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[AuditEntry] = []
        self._lock = threading.Lock()

    def record(self, entry: AuditEntry) -> AuditEntry:
        with self._lock:
            self._entries.append(entry)
            if self.path:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry.to_dict(), default=str) + "\n")
        return entry

    def log(self, action: str, actor: str = "agent", **kwargs: Any) -> AuditEntry:
        """Convenience wrapper. `trail.log("issued_refund", amount=49.0)`."""
        known = {f for f in AuditEntry.__dataclass_fields__}
        fields = {k: v for k, v in kwargs.items() if k in known}
        extra = {k: v for k, v in kwargs.items() if k not in known}
        if extra:
            fields.setdefault("inputs", {}).update(extra)
        return self.record(AuditEntry(action=action, actor=actor, **fields))

    @property
    def entries(self) -> list[AuditEntry]:
        with self._lock:
            return list(self._entries)

    def for_thread(self, thread_id: str) -> list[AuditEntry]:
        """Everything that happened in one conversation. The view a client
        asks for when they want to know why a specific decision was made."""
        return [e for e in self.entries if e.thread_id == thread_id]

    def failures(self) -> list[AuditEntry]:
        return [e for e in self.entries if not e.success]

    def to_report(self, limit: int = 50) -> str:
        rows = self.entries[-limit:]
        header = f"Audit trail — {len(self.entries)} entries ({len(self.failures())} failed)"
        return "\n".join([header, "-" * len(header), *(e.to_line() for e in rows)])


class AuditMiddleware(Middleware):
    """Auto-record every node execution into an `AuditTrail`.

    Deliberately coarse: one entry per node, with the tool calls it made and
    whether it succeeded. Fine-grained per-token detail belongs in the tracer;
    an audit trail a human is expected to read must stay readable.
    """

    def __init__(self, trail: AuditTrail, actor: str = "agent"):
        self.trail = trail
        self.actor = actor

    def after_node(self, node, state, update, config: RunConfig):
        messages = update.get("messages") or []
        messages = messages if isinstance(messages, list) else [messages]

        tools_called: list[str] = []
        reasoning = ""
        for m in messages:
            if isinstance(m, Message):
                tools_called.extend(c.name for c in m.tool_calls)
                if m.role == "assistant" and m.content and not reasoning:
                    reasoning = m.content[:300]

        self.trail.log(
            action=f"node:{node}",
            actor=self.actor,
            thread_id=config.thread_id,
            reasoning=reasoning,
            outputs={"tools_called": tools_called, "keys": sorted(update)},
            success=not update.get("errors"),
        )
        return None

    def on_error(self, node, error, state, config: RunConfig):
        self.trail.log(
            action=f"node:{node}",
            actor=self.actor,
            thread_id=config.thread_id,
            success=False,
            error=f"{type(error).__name__}: {error}",
        )


# ---------------------------------------------------------------------------
# Autonomy ladder
# ---------------------------------------------------------------------------
class AutonomyLevel(IntEnum):
    """How much the agent is allowed to do without a human.

    Climb one rung at a time, and only on evidence. The usual mistake is
    jumping from SHADOW straight to AUTONOMOUS because the demo looked good.
    The usual consequence is one bad week that ends the project.
    """

    #: Runs on real inputs, writes nothing. Its output is logged beside what
    #: the human actually did, which gives you a free, real-world eval set and
    #: a disagreement rate to show the client. Stay here longer than feels
    #: necessary — it is the only rung with zero downside.
    SHADOW = 0

    #: Output is shown to the human as a suggestion they may ignore. Adoption
    #: rate here is the most honest quality signal you will ever get.
    SUGGEST = 1

    #: Agent acts, but every action needs an explicit human approval first.
    #: Wire with `interrupt_before_tools=True`.
    APPROVE_EACH = 2

    #: Agent acts alone on cases it is confident about, and escalates the
    #: rest. This is where most mature deployments settle — not full
    #: autonomy, but autonomy with a defined escape hatch.
    AUTO_WITH_EXCEPTIONS = 3

    #: No human in the path. Appropriate for reversible, low-value, high-volume
    #: work. Rarely appropriate for anything else.
    AUTONOMOUS = 4


@dataclass
class AutonomyPolicy:
    """Rules for what the agent may do at the current level.

    Per-tool overrides matter because autonomy is not one dial. An agent can
    reasonably be fully autonomous at reading a CRM and permanently gated at
    issuing refunds, in the same deployment, on the same day.
    """

    level: AutonomyLevel = AutonomyLevel.SHADOW
    #: Tools that always need approval regardless of level. Money, deletion,
    #: anything a customer sees.
    always_approve: list[str] = field(default_factory=list)
    #: Tools that never need approval regardless of level. Read-only lookups.
    never_approve: list[str] = field(default_factory=list)
    #: At AUTO_WITH_EXCEPTIONS, act alone only above this confidence.
    confidence_threshold: float = 0.85
    #: Currency ceiling above which a human always signs, whatever the level.
    value_ceiling: float | None = None

    def requires_approval(
        self, tool_name: str, confidence: float = 1.0, value: float | None = None
    ) -> bool:
        """The single decision this class exists to make.

        Order matters: the hard overrides are checked before the level, so
        raising the autonomy level can never silently un-gate a tool someone
        deliberately locked down.
        """
        if tool_name in self.always_approve:
            return True
        if self.value_ceiling is not None and value is not None and value > self.value_ceiling:
            return True
        if tool_name in self.never_approve:
            return False

        if self.level <= AutonomyLevel.APPROVE_EACH:
            return True
        if self.level == AutonomyLevel.AUTO_WITH_EXCEPTIONS:
            return confidence < self.confidence_threshold
        return False

    def may_execute(self) -> bool:
        """False in SHADOW and SUGGEST: the agent computes but must not act."""
        return self.level >= AutonomyLevel.APPROVE_EACH


@dataclass
class ShadowComparison:
    """One shadow-mode observation: what the agent said vs what the human did.

    Accumulating these is the cheapest, most credible evidence you can bring
    to a client, because it is measured on their real traffic and it cost them
    nothing. "Over 400 live invoices last month the agent agreed with your team
    96% of the time" ends a lot of arguments.
    """

    input_ref: str
    agent_output: Any
    human_output: Any
    agreed: bool
    timestamp: float = field(default_factory=time.time)
    note: str = ""


class ShadowLog:
    """Collects shadow comparisons and reports the agreement rate."""

    def __init__(self, trail: AuditTrail | None = None):
        self.comparisons: list[ShadowComparison] = []
        self.trail = trail

    def compare(
        self,
        input_ref: str,
        agent_output: Any,
        human_output: Any,
        *,
        match: Any = None,
    ) -> ShadowComparison:
        """Record one comparison.

        `match` lets you pass a custom equality function for outputs where
        string equality is too strict (different phrasing, same decision).
        """
        agreed = (
            bool(match(agent_output, human_output))
            if callable(match)
            else str(agent_output).strip().lower() == str(human_output).strip().lower()
        )
        c = ShadowComparison(input_ref, agent_output, human_output, agreed)
        self.comparisons.append(c)
        if self.trail:
            self.trail.log(
                action="shadow_comparison",
                actor="system",
                autonomy=AutonomyLevel.SHADOW.name,
                inputs={"ref": input_ref},
                outputs={"agent": str(agent_output)[:200], "human": str(human_output)[:200]},
                reasoning="agreed" if agreed else "disagreed",
                success=True,
            )
        return c

    @property
    def agreement_rate(self) -> float:
        if not self.comparisons:
            return 0.0
        return round(sum(1 for c in self.comparisons if c.agreed) / len(self.comparisons), 4)

    def disagreements(self) -> list[ShadowComparison]:
        """The interesting ones. Every disagreement is either an agent bug or
        a case where the human was inconsistent — and you want to know which."""
        return [c for c in self.comparisons if not c.agreed]

    def ready_to_promote(self, threshold: float = 0.95, min_samples: int = 100) -> bool:
        """Is there enough evidence to move up a rung?

        Both conditions matter. 100% agreement over six samples is noise, and
        promoting on it is how you end up explaining an incident.
        """
        return len(self.comparisons) >= min_samples and self.agreement_rate >= threshold


# ---------------------------------------------------------------------------
# Impact measurement
# ---------------------------------------------------------------------------
@dataclass
class ImpactMetrics:
    """The three buckets a business actually funds: cost, risk, revenue.

    Every number here should be defensible to a finance team, which mostly
    means being conservative and showing the arithmetic. An inflated estimate
    that gets caught costs you more credibility than a modest one earns.

    Note that `runs` and `escalations` are the denominator for everything else:
    an agent that handles 20% of volume and escalates the rest is still a
    success, but only if you say so up front instead of implying full coverage.
    """

    workflow: str = ""
    period_days: int = 30

    # -- volume ---------------------------------------------------------
    runs: int = 0
    successes: int = 0
    escalations: int = 0  # handed to a human by design, not a failure
    failures: int = 0     # genuine errors

    # -- cost saved -----------------------------------------------------
    minutes_saved_per_run: float = 0.0
    loaded_hourly_cost: float = 60.0  # salary + overhead, not salary

    # -- risk reduced ---------------------------------------------------
    baseline_error_rate: float = 0.0  # human error rate before the agent
    agent_error_rate: float = 0.0
    cost_per_error: float = 0.0

    # -- revenue enabled ------------------------------------------------
    revenue_attributed: float = 0.0
    revenue_note: str = ""  # how it was attributed; be honest about the model

    # -- what it costs to run -------------------------------------------
    model_cost: float = 0.0
    infra_cost: float = 0.0

    # -- derived --------------------------------------------------------
    @property
    def automation_rate(self) -> float:
        """Share of volume handled without a human. The honest coverage number."""
        return round((self.runs - self.escalations) / self.runs, 4) if self.runs else 0.0

    @property
    def success_rate(self) -> float:
        return round(self.successes / self.runs, 4) if self.runs else 0.0

    @property
    def hours_saved(self) -> float:
        completed = self.runs - self.escalations
        return round(max(0, completed) * self.minutes_saved_per_run / 60.0, 1)

    @property
    def cost_saved(self) -> float:
        return round(self.hours_saved * self.loaded_hourly_cost, 2)

    @property
    def errors_prevented(self) -> float:
        """Errors avoided versus the human baseline.

        Can be negative, and that is the point — an agent worse than the people
        it replaced should show up as a negative number rather than being
        quietly omitted from the deck.
        """
        return round((self.baseline_error_rate - self.agent_error_rate) * self.runs, 1)

    @property
    def risk_value(self) -> float:
        return round(self.errors_prevented * self.cost_per_error, 2)

    @property
    def total_cost(self) -> float:
        return round(self.model_cost + self.infra_cost, 2)

    @property
    def net_value(self) -> float:
        return round(self.cost_saved + self.risk_value + self.revenue_attributed - self.total_cost, 2)

    @property
    def roi(self) -> float | None:
        """Return per unit spent. None when nothing was spent, rather than a
        misleading infinity."""
        return round(self.net_value / self.total_cost, 2) if self.total_cost else None

    @property
    def cost_per_run(self) -> float:
        return round(self.total_cost / self.runs, 4) if self.runs else 0.0

    def to_markdown(self) -> str:
        """The slide. Numbers first, method underneath."""
        roi = f"{self.roi}x" if self.roi is not None else "n/a"
        lines = [
            f"# Impact: {self.workflow}",
            f"*Last {self.period_days} days*",
            "",
            f"**Net value: {self.net_value:,.0f}**  (ROI {roi})",
            "",
            "| Bucket | Value | Basis |",
            "|--------|-------|-------|",
            f"| Cost saved | {self.cost_saved:,.0f} | {self.hours_saved} hrs @ {self.loaded_hourly_cost:.0f}/hr |",
            f"| Risk reduced | {self.risk_value:,.0f} | {self.errors_prevented} errors prevented @ {self.cost_per_error:,.0f} |",
            f"| Revenue enabled | {self.revenue_attributed:,.0f} | {self.revenue_note or 'not attributed'} |",
            f"| Cost to run | ({self.total_cost:,.0f}) | {self.cost_per_run:.3f}/run |",
            "",
            "## Operations",
            "",
            f"- Runs: {self.runs:,}",
            f"- Automation rate: {self.automation_rate:.1%} ({self.escalations:,} escalated to a human)",
            f"- Success rate: {self.success_rate:.1%} ({self.failures:,} failures)",
            f"- Error rate: {self.agent_error_rate:.2%} vs {self.baseline_error_rate:.2%} human baseline",
        ]
        if self.errors_prevented < 0:
            lines += [
                "",
                "> **Note:** the agent's error rate is currently *above* the human "
                "baseline. Risk value is negative and this workflow should not be "
                "promoted to a higher autonomy level yet.",
            ]
        return "\n".join(lines)

    @classmethod
    def from_audit_trail(
        cls,
        trail: AuditTrail,
        workflow: str = "",
        **assumptions: Any,
    ) -> "ImpactMetrics":
        """Derive the volume figures from what actually happened.

        Only the counts come from the log. The economics — minutes saved per
        run, cost per error, hourly rate — are *assumptions*, and they must
        come from the client, agreed in writing, ideally during the audit. Made
        up by you, they are the fastest way to lose a room.
        """
        entries = trail.entries
        runs = sum(1 for e in entries if e.action.startswith("node:") or e.action == "run")
        failures = sum(1 for e in entries if not e.success)
        escalations = sum(1 for e in entries if e.decision in ("escalated", "rejected"))
        return cls(
            workflow=workflow,
            runs=runs,
            successes=runs - failures,
            failures=failures,
            escalations=escalations,
            **assumptions,
        )


# ---------------------------------------------------------------------------
# Rollout gate
# ---------------------------------------------------------------------------
@dataclass
class PromotionCriteria:
    """What must be true before the agent climbs a rung.

    Writing these down before you start is the whole trick. Agreed in advance,
    promotion is a checklist. Decided afterwards, it is a negotiation you will
    have while someone is annoyed about an incident.
    """

    min_runs: int = 100
    min_success_rate: float = 0.95
    max_failure_rate: float = 0.02
    min_shadow_agreement: float = 0.95
    #: Eval pass rate on the critical subset, not the overall suite.
    min_critical_eval_rate: float = 0.98
    #: Consecutive days without a serious incident.
    min_clean_days: int = 14

    def evaluate(
        self,
        metrics: ImpactMetrics,
        *,
        shadow: ShadowLog | None = None,
        critical_eval_rate: float | None = None,
        clean_days: int = 0,
    ) -> tuple[bool, list[str]]:
        """Return (ready, blockers). Blockers name the specific shortfall.

        Returning reasons rather than a bare bool is deliberate: "not ready"
        starts an argument, "not ready: 62 runs, need 100" starts a plan.
        """
        blockers: list[str] = []

        if metrics.runs < self.min_runs:
            blockers.append(f"only {metrics.runs} runs, need {self.min_runs}")
        if metrics.success_rate < self.min_success_rate:
            blockers.append(
                f"success rate {metrics.success_rate:.1%} below {self.min_success_rate:.0%}"
            )
        failure_rate = metrics.failures / metrics.runs if metrics.runs else 1.0
        if failure_rate > self.max_failure_rate:
            blockers.append(f"failure rate {failure_rate:.1%} above {self.max_failure_rate:.0%}")
        if shadow is not None and shadow.agreement_rate < self.min_shadow_agreement:
            blockers.append(
                f"shadow agreement {shadow.agreement_rate:.1%} below {self.min_shadow_agreement:.0%}"
            )
        if critical_eval_rate is not None and critical_eval_rate < self.min_critical_eval_rate:
            blockers.append(
                f"critical eval rate {critical_eval_rate:.1%} below {self.min_critical_eval_rate:.0%}"
            )
        if clean_days < self.min_clean_days:
            blockers.append(f"{clean_days} clean days, need {self.min_clean_days}")

        return (not blockers), blockers


def rollout_plan(
    workflow: str,
    current: AutonomyLevel = AutonomyLevel.SHADOW,
    criteria: PromotionCriteria | None = None,
) -> str:
    """A written rollout plan, for the client.

    Hand this over at kickoff. It converts "we're putting an AI in charge of
    your invoices", which sounds terrifying, into a staged plan with defined
    gates, which sounds like engineering.
    """
    criteria = criteria or PromotionCriteria()
    descriptions = {
        AutonomyLevel.SHADOW: (
            "Agent runs on live inputs but changes nothing. Its output is logged "
            "next to what your team actually did, producing an agreement rate."
        ),
        AutonomyLevel.SUGGEST: (
            "Agent output appears as a suggestion your team can accept or ignore. "
            "Acceptance rate becomes the quality signal."
        ),
        AutonomyLevel.APPROVE_EACH: (
            "Agent performs the work; every action waits for explicit human "
            "approval before it takes effect."
        ),
        AutonomyLevel.AUTO_WITH_EXCEPTIONS: (
            "Agent acts alone on high-confidence cases and escalates the rest to "
            "a human queue."
        ),
        AutonomyLevel.AUTONOMOUS: (
            "Agent acts without human review. Reserved for reversible, "
            "low-value, high-volume actions."
        ),
    }

    lines = [
        f"# Rollout plan: {workflow}",
        "",
        f"**Current stage:** {current.name}",
        "",
        "## Stages",
        "",
    ]
    for level in AutonomyLevel:
        marker = "->" if level == current else "  "
        state = "current" if level == current else ("done" if level < current else "planned")
        lines.append(f"{marker} **{level.name}** ({state}) — {descriptions[level]}")
    lines += [
        "",
        "## Promotion gate (must all pass to advance one stage)",
        "",
        f"- At least {criteria.min_runs} runs at the current stage",
        f"- Success rate at or above {criteria.min_success_rate:.0%}",
        f"- Failure rate at or below {criteria.max_failure_rate:.0%}",
        f"- Shadow/human agreement at or above {criteria.min_shadow_agreement:.0%}",
        f"- Critical eval cases at or above {criteria.min_critical_eval_rate:.0%}",
        f"- {criteria.min_clean_days} consecutive days without a serious incident",
        "",
        "Any single failed gate holds the current stage. Rollback to the previous "
        "stage is always available and requires no approval.",
    ]
    return "\n".join(lines)
