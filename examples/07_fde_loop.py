"""
Example 07 — the FDE loop: audit -> evals -> deployment.

The previous examples show how to *build* an agent. This one shows how to get
one into a business and prove it works, which is the harder half and the
reason most pilots never reach production.

The scenario: accounts-payable invoice intake at a mid-size company. Runs
offline.

    python examples/07_fde_loop.py

The loop:

    AUDIT       map the real workflow, decide where intelligence belongs
    EVALS       build a golden set from history, measure, catch regressions
    DEPLOYMENT  audit trail, shadow mode, promotion gate, impact in dollars

Then it runs again on the next workflow, and the next one is clearer because
you now understand the business.
"""

from agentkit import (
    AuditTrail,
    AutonomyLevel,
    AutonomyPolicy,
    GoldenDataset,
    ImpactMetrics,
    Placement,
    PromotionCriteria,
    Regression,
    ShadowLog,
    Workflow,
    WorkflowStep,
    exact_match,
    rollout_plan,
    run_eval,
)

# ===========================================================================
# STAGE 1 — AUDIT
#
# Map how the work ACTUALLY happens. Note the exceptions on every step: those
# come from sitting with the person who does the job, and they are the whole
# value of the audit. A step with no recorded exceptions has not been observed,
# it has been imagined.
# ===========================================================================
print("=" * 72)
print("STAGE 1 — AUDIT")
print("=" * 72)

ap = Workflow(
    name="Accounts payable — invoice intake",
    owner="Dana R., AP Manager",
    business_goal="Close the month faster without adding headcount.",
)

ap.add(
    WorkflowStep(
        "Receive invoice",
        description="Invoices arrive by email from ~40 vendors, no two formatted alike.",
        systems=["Outlook"],
        volume_per_month=1200,
        minutes_per_run=2,
        exceptions=[
            "PDF buried in a forwarded thread",
            "invoice pasted into the email body, no attachment",
            "vendor resends the same invoice weekly until paid",
            "scanned image, no extractable text",
        ],
        tribal_knowledge=[
            "Anything from Northwind goes straight to Sarah — she has a side agreement.",
        ],
    )
)
ap.add(
    WorkflowStep(
        "Extract line items",
        description="Rekey vendor, PO, amounts, tax into the ERP.",
        systems=["NetSuite"],
        volume_per_month=1200,
        minutes_per_run=7,
        requires_judgment=True,  # messy unstructured input: genuine LLM territory
        exceptions=[
            "multi-currency invoices",
            "line items split across pages",
            "credit notes that look like invoices",
        ],
    )
)
ap.add(
    WorkflowStep(
        "Match to purchase order",
        description="Look up the PO and confirm amounts within tolerance.",
        systems=["NetSuite"],
        volume_per_month=1200,
        minutes_per_run=3,
        # NOT judgment: this is a lookup and a numeric comparison. It feels
        # like thinking, but a rule does it faster and correctly every time.
        requires_judgment=False,
        exceptions=["partial deliveries", "PO closed early"],
    )
)
ap.add(
    WorkflowStep(
        "Route exceptions",
        description="Decide who resolves a mismatch and chase them.",
        systems=["Outlook", "Slack"],
        volume_per_month=180,
        minutes_per_run=12,
        requires_judgment=True,
        error_cost=1500,
        exceptions=["approver on leave", "disputed amount escalates to Legal"],
    )
)
ap.add(
    WorkflowStep(
        "Issue payment",
        description="Release funds to the vendor.",
        systems=["NetSuite", "Bank portal"],
        volume_per_month=1100,
        minutes_per_run=3,
        reversible=False,  # money leaving the building
        error_cost=25_000,
        exceptions=["wrong bank details", "duplicate payment"],
    )
)

print(f"\nWorkflow: {ap.name}")
print(f"Current cost:   {ap.total_hours_per_month} human hours/month")
print(f"Automatable:    {ap.automatable_hours()} hours/month")
print(f"Monthly value:  {ap.priority_score():,.0f}\n")

print("Where intelligence belongs:")
for step in ap.steps:
    placement = step.recommend_placement()
    print(f"  {step.name:<24} risk={step.risk_score:<5} -> {placement.value}")

# The distribution below is the point of the whole exercise. Two steps get an
# LLM. One is deterministic code. One is gated behind a human signature. An
# "AI invoice agent" that put a model on all five would be slower, costlier
# and less reliable than this.
print(f"\nPlacement summary: {ap.placement_summary()}")

gaps = ap.unmapped_exceptions()
print(f"Audit gaps (steps not yet observed): {gaps or 'none'}")

# The `to_markdown()` output is the client deliverable. It is often the first
# thing you hand over, and it earns the right to build anything at all.


# ===========================================================================
# STAGE 2 — EVALS
#
# Turn "it seems to work" into evidence. The dataset is built from decisions
# the AP team already made by hand — that history is a labelled dataset nobody
# thought to use.
# ===========================================================================
print("\n" + "=" * 72)
print("STAGE 2 — EVALS")
print("=" * 72)

# Weighted toward the unhappy paths. Every exception recorded in the audit
# should show up here as a case.
history = [
    {"invoice": "Acme Ltd, $412.00, PO-4471, amounts match", "decision": "auto_approve", "kind": "clean"},
    {"invoice": "Globex, $1,204.00, PO-4480, amounts match", "decision": "auto_approve", "kind": "clean"},
    {"invoice": "Initech, $88.50, PO-4491, amounts match", "decision": "auto_approve", "kind": "clean"},
    {"invoice": "Umbrella, $9,900.00, no PO number", "decision": "escalate", "kind": "missing-data"},
    {"invoice": "Northwind, $310.00, PO-4402, PO closed early", "decision": "escalate", "kind": "po-mismatch"},
    {"invoice": "Acme Ltd, $412.00, PO-4471, duplicate of INV-8871", "decision": "reject", "kind": "duplicate"},
    {"invoice": "Soylent, EUR 2,100.00, PO-4455, multi-currency", "decision": "escalate", "kind": "currency"},
    {"invoice": "Hooli, $47,000.00, PO-4499, amounts match", "decision": "escalate", "kind": "high-value"},
]

dataset = GoldenDataset.from_history(
    "ap-invoice-routing", history, input_key="invoice", output_key="decision", tag_key="kind"
)
# Mark the ones the business cannot afford to get wrong. An overall pass rate
# that hides failures here is reassuring and useless.
for case in dataset:
    if any(t in case.tags for t in ("duplicate", "high-value")):
        case.critical = True

print(f"\nGolden dataset: {len(dataset)} cases from historical decisions")


def agent_v1(invoice: str) -> str:
    """First attempt. Handles the happy path; misses several exceptions."""
    text = invoice.lower()
    if "duplicate" in text:
        return "reject"
    if "no po" in text:
        return "escalate"
    return "auto_approve"


report_v1 = run_eval(agent_v1, dataset, exact_match)
print(f"\nv1: {report_v1.summary()}")
print(f"    failures by category: {report_v1.failures_by_tag()}")
for failure in report_v1.failures():
    print(f"    - {failure.case.name[:46]:<46} {failure.score.reason}")


def agent_v2(invoice: str) -> str:
    """Second attempt after reading the failure breakdown.

    Note what happened: the report named the categories (currency, po-mismatch,
    high-value), so the fix was targeted rather than a general prompt rewrite
    and a hope.
    """
    text = invoice.lower()
    if "duplicate" in text:
        return "reject"
    if "no po" in text or "closed early" in text or "multi-currency" in text:
        return "escalate"
    amount = "".join(c for c in invoice.split("$")[-1].split(",")[0] if c.isdigit() or c == ".")
    if amount and float(amount or 0) > 10_000:
        return "escalate"
    return "auto_approve"


report_v2 = run_eval(agent_v2, dataset, exact_match)
print(f"\nv2: {report_v2.summary()}")
print(f"    failures by category: {report_v2.failures_by_tag()}")

# READ THAT LINE CAREFULLY. The headline went 62% -> 88%, which looks like a
# win worth shipping. But the CRITICAL pass rate is still 50%: the $47,000
# invoice slips through, because the amount parser above splits on the comma
# in "47,000" and reads it as $47.
#
# This is not a contrived example — comma-in-currency is one of the most
# common bugs in extraction code. The point is that the aggregate number hid
# it completely, and the critical-case rate did not. That is the entire reason
# `EvalReport` reports them separately, and it is why the promotion gate
# further down refuses to advance this agent.
print(f"    critical cases: {report_v2.critical_pass_rate:.0%} "
      f"<- the aggregate hid a failure on the cases that matter most")

regression = Regression(report_v1, report_v2)
print(f"\nvs baseline: {regression.summary()}")
print(f"regression detected: {regression.is_regression}")
# Only `newly_failing` blocks a release. A change that lifts the aggregate
# while breaking a case that used to work is usually a bad change, and the
# headline number hides that entirely.


# ===========================================================================
# STAGE 3 — DEPLOYMENT
#
# Nobody flips a switch from nothing to autonomy. Climb the ladder on evidence.
# ===========================================================================
print("\n" + "=" * 72)
print("STAGE 3 — DEPLOYMENT")
print("=" * 72)

trail = AuditTrail()  # pass a path to persist as JSONL

# --- Shadow mode: the agent runs on live invoices and changes nothing ------
# This rung has no downside and produces the most credible evidence you can
# bring a client, because it is measured on their real traffic.
shadow = ShadowLog(trail)
live_invoices = [
    ("INV-9001", "Acme Ltd, $220.00, PO-5001, amounts match", "auto_approve"),
    ("INV-9002", "Globex, $15,400.00, PO-5002, amounts match", "escalate"),
    ("INV-9003", "Initech, $75.00, PO-5003, amounts match", "auto_approve"),
    ("INV-9004", "Umbrella, $980.00, no PO number", "escalate"),
    ("INV-9005", "Northwind, $310.00, PO-5005, amounts match", "escalate"),  # tribal rule
]
for ref, invoice, what_the_human_did in live_invoices:
    shadow.compare(ref, agent_v2(invoice), what_the_human_did)

print(f"\nShadow mode: {len(shadow.comparisons)} live invoices observed")
print(f"Agreement with the AP team: {shadow.agreement_rate:.0%}")
for disagreement in shadow.disagreements():
    print(f"  disagreed on {disagreement.input_ref}: "
          f"agent said {disagreement.agent_output!r}, human did {disagreement.human_output!r}")
# That INV-9005 disagreement is the Northwind rule from the audit — knowledge
# that lives in one person's head and never made it into the agent. Shadow
# mode is how you find those before they cost anything.

# --- The autonomy policy ---------------------------------------------------
policy = AutonomyPolicy(
    level=AutonomyLevel.AUTO_WITH_EXCEPTIONS,
    always_approve=["issue_payment"],       # irreversible, always signed
    never_approve=["lookup_po", "read_invoice"],  # read-only, no gate needed
    confidence_threshold=0.9,
    value_ceiling=10_000,
)

print("\nAutonomy policy checks:")
for tool, confidence, value in [
    ("lookup_po", 0.99, None),
    ("classify_invoice", 0.95, 400.0),
    ("classify_invoice", 0.55, 400.0),
    ("classify_invoice", 0.99, 47_000.0),
    ("issue_payment", 0.99, 50.0),
]:
    gated = policy.requires_approval(tool, confidence, value)
    print(f"  {tool:<18} conf={confidence:<5} value={str(value):<8} "
          f"-> {'HUMAN APPROVES' if gated else 'agent acts alone'}")
# Autonomy is not one dial: the same agent is ungated on lookups and
# permanently gated on payments, on the same day.

# --- The audit trail -------------------------------------------------------
trail.log(
    "classify_invoice",
    thread_id="INV-9001",
    reasoning="PO-5001 matched, amount $220 under the $10k ceiling, vendor in good standing",
    outputs={"decision": "auto_approve"},
    autonomy=AutonomyLevel.AUTO_WITH_EXCEPTIONS.name,
)
trail.log(
    "issue_payment",
    actor="human:dana@example.com",
    thread_id="INV-9001",
    decision="approved",
    reasoning="Reviewed agent classification, confirmed vendor bank details on file",
)
trail.log(
    "extract_line_items",
    thread_id="INV-9006",
    success=False,
    error="ExtractionError: scanned image with no text layer",
    reasoning="Escalated to AP queue for manual entry",
    decision="escalated",
)

print("\n" + trail.to_report())
# The reasoning field is what makes this a trail rather than a log. A
# controller can read it, disagree, and sign off.

# --- Impact, in the three buckets a business funds ------------------------
# The counts come from operations. The economics — minutes saved, cost per
# error, hourly rate — come from the CLIENT and were agreed during the audit.
# Numbers you invented are the fastest way to lose the room.
impact = ImpactMetrics(
    workflow="AP invoice intake",
    period_days=30,
    runs=1200,
    successes=1170,
    escalations=180,
    failures=30,
    minutes_saved_per_run=9.0,       # extract + match, per the audit
    loaded_hourly_cost=62.0,         # AP clerk, fully loaded
    baseline_error_rate=0.035,       # measured from last year's corrections
    agent_error_rate=0.011,
    cost_per_error=850.0,            # average cost of a mispaid invoice
    model_cost=280.0,
    infra_cost=95.0,
)

print("\n" + impact.to_markdown())

# --- The promotion gate ----------------------------------------------------
# Agreed in advance, promotion is a checklist. Decided afterwards, it is a
# negotiation you have while someone is annoyed about an incident.
criteria = PromotionCriteria(min_runs=500, min_shadow_agreement=0.95, min_clean_days=14)
ready, blockers = criteria.evaluate(
    impact, shadow=shadow, critical_eval_rate=report_v2.critical_pass_rate, clean_days=21
)

print(f"\nReady to promote to {AutonomyLevel.AUTONOMOUS.name}? {ready}")
for blocker in blockers:
    print(f"  BLOCKED: {blocker}")
# "Not ready" starts an argument. "Not ready: shadow agreement 80%, need 95%"
# starts a plan.

print("\n" + "-" * 72)
print(rollout_plan("AP invoice intake", current=AutonomyLevel.AUTO_WITH_EXCEPTIONS, criteria=criteria))

print("\n" + "=" * 72)
print("The loop runs again on the next workflow — and the next one is clearer,")
print("because you now understand how this business actually works.")
print("=" * 72)
