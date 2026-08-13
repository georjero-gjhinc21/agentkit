# The FDE playbook

How to get an agent into a business and prove it works.

`ARCHITECTURE.md` explains how the framework is built and `PATTERNS.md` shows
what to build. This file is about the part that decides whether any of it
survives: deployment.

The premise is that frontier intelligence is now something every company can
buy. If everyone has access to the same models, the model is not the
advantage — where, how, and why it gets applied is. That work is a job, and
the job has a loop.

```
    AUDIT  ──────►  EVALS  ──────►  DEPLOYMENT
      ▲                                  │
      └──────────────────────────────────┘
        each workflow you fix makes the next one clearer
```

| Stage | Question | Module |
|---|---|---|
| Audit | How does the work actually happen, and where does intelligence belong? | `agentkit/workflow.py` |
| Evals | Does it work, and how do I know? | `agentkit/evals.py` |
| Deployment | Will they trust it, and what is it worth? | `agentkit/deployment.py` |

`examples/07_fde_loop.py` runs the whole thing end to end on an accounts-payable
workflow. Read this file, then run that.

---

## Stage 1 — Audit

### The documented process is not the real process

"An email arrives" is one box on a slide. In reality it is forty vendors, no
two formats alike, half of them exceptions, resolved by one person from memory.
Build for the slide and you have automated a workflow that does not exist.

This is why the audit happens where the work happens. A one-hour interview gets
you what someone *thinks* their job is. A day sitting beside them gets you what
it actually is — including the thing that breaks at 3pm and is not in any SOP.

The single most valuable field in `WorkflowStep` is `exceptions`. If it is
empty, you have not audited that step:

```python
WorkflowStep(
    "Receive invoice",
    volume_per_month=1200,
    minutes_per_run=2,
    exceptions=[
        "PDF buried in a forwarded thread",
        "invoice pasted into the body, no attachment",
        "vendor resends weekly until paid",
        "scanned image, no text layer",
    ],
    tribal_knowledge=["Anything from Northwind goes straight to Sarah"],
)
```

`Workflow.unmapped_exceptions()` lists the steps you have not really looked at
yet. It is usually longer than you expect.

### Where intelligence belongs

The commonly cited figure is that ~95% of generative AI pilots never reach
production. A large share of that is putting a model on a step that never
needed one.

`recommend_placement()` sorts each step into four buckets:

| Placement | When | Why not something else |
|---|---|---|
| `DETERMINISTIC` | lookups, arithmetic, validation, branching | A rule is faster, free, auditable, and right every time |
| `LLM` | messy input, extraction, classification, drafting | A rule would need a hundred branches and still miss cases |
| `LLM_WITH_APPROVAL` | judgment on something risky | You want the speed and the signature |
| `HUMAN` | irreversible and expensive | Nothing else is worth the tail risk |

In the AP example, five steps split 2 / 2 / 0 / 1. Most steps do not want a
model, and that is the expected result. An "AI invoice agent" that put a model
on all five would be slower, costlier and less reliable than the mix.

The scorer is a structured prompt, not an oracle. Override it. Arguing with it
is the actual work.

### The audit is a deliverable

`Workflow.to_markdown()` produces a client-readable document: the real process,
the exceptions, the undocumented rules, the recommended placements, the hours
and the estimated value.

Hand it over before you build anything. It proves you understood their business,
makes the plan concrete, and shows them exactly where they stay in control.
That last part is what buys the trust to build at all.

Two practical notes:

- **Run the first ones cheap or free.** Your first few clients teach you more
  than you teach them. That trade is worth making explicitly.
- **Some people hear "audit" and think "tax audit."** Calling it a *sprint* or a
  *discovery* lands better with no change to the work.

---

## Stage 2 — Evals

### The problem

You changed a prompt. Better or worse? Without evals the honest answer is "I
ran it three times and it seemed fine." Nobody lets that near their invoices.

An eval suite converts a vibe into a sentence you can put in front of a
controller: *41 of 50 cases passed; of the nine failures, five were missing
data and four pulled the wrong record.* That sentence gets agents deployed.

### The dataset is the work

The harness is 200 lines. The value is in the cases.

**Use real history.** Every workflow worth automating has a paper trail of past
decisions — a labelled dataset nobody thought to use:

```python
dataset = GoldenDataset.from_history(
    "ap-invoice-routing", past_decisions,
    input_key="invoice", output_key="decision", tag_key="exception_kind",
)
```

**Weight toward the unhappy paths.** There is one way a step goes right and a
thousand ways it goes wrong. Every exception from the audit becomes a case.

**Mark the critical ones.** An overall 88% that hides a 50% rate on high-value
invoices is reassuring and wrong. `EvalReport` reports `critical_pass_rate`
separately for exactly this reason — and example 07 demonstrates it with a real
bug, where a comma in `$47,000` gets parsed as `$47`.

**Get the expected outputs signed off** by whoever owns the workflow. Once the
client agrees those 50 answers are correct, arguments about quality become
arguments about evidence.

### Subjective outputs

Some things have no deterministic scorer — tone, completeness, whether a
summary caught the point. `LLMJudge` gives a consistent, cheap proxy:

```python
judge = LLMJudge(model, rubric="Does the summary state the decision and the owner?")
```

Three things make a judge usable rather than theatrical: a rubric specific
enough that two people would agree on it, a low temperature, and treating the
score as a proxy you spot-check against humans. A judge that agrees with your
reviewers 85% of the time is useful. One you never validated is decoration.

### Regression is not the headline number

```python
reg = Regression(baseline_report, new_report)
if reg.is_regression:
    print(reg.newly_failing)   # cases that used to work and now don't
```

A change that lifts the aggregate from 82% to 85% while breaking four
previously-passing cases is usually a bad change. `is_regression` keys off
`newly_failing`, not the delta, because a flat pass rate can hide a swap.

---

## Stage 3 — Deployment

### Build on what exists

A client who spent two years and several million dollars moving to NetSuite
will not move off it for your agent. Integrate on top: NetSuite plus Salesforce
plus SAP plus whatever else, with the agent in between. The migration pitch
loses to the augmentation pitch every time.

### The ladder, not the switch

```
SHADOW  →  SUGGEST  →  APPROVE_EACH  →  AUTO_WITH_EXCEPTIONS  →  AUTONOMOUS
```

Each rung produces the evidence needed to climb the next.

**Shadow mode has no downside and is the most under-used rung.** The agent runs
on live inputs and changes nothing; its output is logged next to what the human
actually did:

```python
shadow.compare("INV-9005", agent_decision, what_the_human_did)
shadow.agreement_rate           # 0.94
shadow.disagreements()          # the interesting ones
shadow.ready_to_promote(threshold=0.95, min_samples=100)
```

"Over 400 live invoices last month the agent agreed with your team 96% of the
time" is the most credible thing you can say, because it was measured on their
traffic and cost them nothing. Every disagreement is either an agent bug or an
inconsistent human — and you want to know which.

In example 07, shadow mode catches the Northwind rule: tribal knowledge from
the audit that never made it into the agent. That is what the rung is for.

**Autonomy is not one dial.** The same agent can be ungated on lookups and
permanently gated on payments:

```python
AutonomyPolicy(
    level=AutonomyLevel.AUTO_WITH_EXCEPTIONS,
    always_approve=["issue_payment"],        # irreversible: always signed
    never_approve=["lookup_po"],             # read-only: no gate
    confidence_threshold=0.9,
    value_ceiling=10_000,                    # a human signs above this
)
```

Hard overrides are checked before the level, so raising autonomy can never
silently un-gate something someone deliberately locked down.

Wire the gate to the runtime with `create_agent(..., interrupt_before_tools=True)`
and a checkpointer — see `examples/03_human_in_the_loop.py`.

### The audit trail

Two different things are called "audit" here, and it is worth keeping them
straight:

- `workflow.py` — the audit as **discovery**, before you build
- `deployment.py` — the audit trail as **evidence**, forever after

It is also distinct from `tracing.py`. A trace exists for you to debug. A trail
exists for the client, their controller, and eventually their regulator.

The `reasoning` field is what makes it a trail rather than a log:

```
approved invoice INV-88                                      ← a log line
approved INV-88: PO matched, amount under the $5,000 ceiling ← a trail
```

The second one can be checked, disagreed with, and signed off.

`AuditMiddleware` records one entry per node automatically. Keep it coarse —
a trail a human is expected to read has to stay readable. And keep PII out of
it: audit trails are long-lived by design, which makes them the worst place
for a leak. Compose with `RedactionMiddleware`.

### Numbers in their language

Nobody funds "94% accuracy." They fund three buckets:

```python
ImpactMetrics(
    runs=1200, escalations=180, failures=30,
    minutes_saved_per_run=9.0, loaded_hourly_cost=62.0,      # cost saved
    baseline_error_rate=0.035, agent_error_rate=0.011,
    cost_per_error=850.0,                                     # risk reduced
    revenue_attributed=0.0,                                   # revenue enabled
    model_cost=280.0, infra_cost=95.0,                        # cost to run
)
```

Rules that keep these defensible:

- **Counts come from the log. Economics come from the client.** Minutes saved
  and cost per error should be agreed during the audit, in writing. Numbers you
  invented are the fastest way to lose a room.
- **Report the automation rate honestly.** An agent handling 80% of volume and
  escalating the rest is a success — but only if you said so up front instead
  of implying full coverage.
- **Let it go negative.** If the agent's error rate is above the human baseline,
  `errors_prevented` is negative and `to_markdown()` prints a warning. An agent
  worse than the people it replaced should not be able to hide behind an
  aggregate.
- **Use loaded cost**, not salary.

### Write the promotion gate before you start

```python
criteria = PromotionCriteria(
    min_runs=500,
    min_success_rate=0.95,
    max_failure_rate=0.02,
    min_shadow_agreement=0.95,
    min_critical_eval_rate=0.98,
    min_clean_days=14,
)
ready, blockers = criteria.evaluate(impact, shadow=shadow, critical_eval_rate=..., clean_days=21)
```

Agreed in advance, promotion is a checklist. Decided afterwards, it is a
negotiation you have while someone is annoyed about an incident.

`evaluate` returns reasons, not a bare boolean, because "not ready" starts an
argument and "not ready: shadow agreement 80%, need 95%" starts a plan.

`rollout_plan()` renders the whole ladder and its gates as a document for the
client. It turns "we're putting an AI in charge of your invoices" into a staged
plan with defined checkpoints and a no-questions-asked rollback.

---

## Doing the job before you have the title

A condensed version of the 30-day plan from the source material, mapped to this
repo. The point is to have built the thing, not to have read about it.

### Week 1 — build an agent that completes a real loop

Pick one real back-office workflow (finance, HR, procurement, support). Ask a
model to describe it in granular detail if you do not have one to hand.

- `examples/01_minimal_agent.py` — the loop
- `agentkit/tools.py` — tool schemas from function signatures
- `agentkit/graph.py` — control flow, guardrails
- `agentkit/deployment.py` — the audit trail, from day one

The bar: an agent that solves a task end to end when prompted badly. If it only
works with a perfect prompt, it is a demo.

### Week 2 — harden it

This is where most of the value is, and where most tutorials stop.

- Typed outputs: `StructuredOutputParser`, not free-form text
- Exception handling for the unhappy paths you observed
- Node-level retries; `RunnableFallback` for provider outages
- `GuardrailMiddleware(..., on_violation="annotate")` so the agent sees its own
  violations and self-corrects

There is one way something goes right and a thousand ways it goes wrong. An
agent that only handles the happy path is worth approximately nothing; one that
handles the exceptions is worth a great deal.

### Week 3 — make it measurable

- Build the golden dataset, weighted toward exceptions
- `run_eval` in CI; keep a baseline; block on `Regression.is_regression`
- Route cheap subtasks to cheap models; measure with `UsageTrackingMiddleware`
- Fill in `ImpactMetrics` across all three buckets

### Week 4 — defend it

Rehearse two pitches for the same system.

*As an engineer:* the architecture, why each step got the placement it got,
what broke and how you fixed it, accuracy over iterations.

*As a VP:* the problem, the outcome, the evidence, the residual risk, the cost.

Then pitch it to actual businesses. They will tell you plainly what you got
wrong, which is the fastest feedback available and it is free.

---

## Where the framework maps

| FDE concern | Where |
|---|---|
| Map the real workflow | `workflow.Workflow`, `WorkflowStep` |
| Deterministic vs LLM vs human | `workflow.Placement`, `recommend_placement()` |
| Client-facing audit document | `Workflow.to_markdown()` |
| Golden dataset from history | `evals.GoldenDataset.from_history` |
| Scoring, including subjective | `evals` scorers, `LLMJudge` |
| Evidence report | `evals.EvalReport.to_markdown()` |
| Regression protection | `evals.Regression` |
| Audit trail | `deployment.AuditTrail`, `AuditMiddleware` |
| Shadow mode | `deployment.ShadowLog` |
| Progressive autonomy | `deployment.AutonomyLevel`, `AutonomyPolicy` |
| Human approval at runtime | `create_agent(interrupt_before_tools=True)` + checkpointer |
| Impact in dollars | `deployment.ImpactMetrics` |
| Promotion gates | `deployment.PromotionCriteria`, `rollout_plan()` |
| Model agnosticism | `models.py` — one neutral `Message` |
| Cheap model for subtasks | swap the adapter; `RunnableFallback` for redundancy |
