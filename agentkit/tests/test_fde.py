"""
Tests for the FDE loop: workflow mapping, evals, deployment.

Third test file, covering the third concern. `test_agentkit.py` tests the
graph runtime, `test_chains.py` tests composition, and this one tests the
machinery for getting an agent into a business and proving it works.

    python tests/test_fde.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentkit import (  # noqa: E402
    AuditTrail,
    AutonomyLevel,
    AutonomyPolicy,
    EvalCase,
    FakeModel,
    GoldenDataset,
    ImpactMetrics,
    Message,
    Placement,
    PromotionCriteria,
    Regression,
    ShadowLog,
    ToolCall,
    Workflow,
    WorkflowStep,
    all_of,
    contains,
    create_agent,
    exact_match,
    json_fields,
    no_hallucinated_facts,
    numeric_within,
    run_eval,
    tool,
)
from agentkit.evals import LLMJudge  # noqa: E402


# ===========================================================================
# workflow mapping
# ===========================================================================
def test_deterministic_is_the_default():
    """A plain step with no judgment and no risk should NOT get a model.
    This default is the whole point of the module."""
    step = WorkflowStep("Look up PO", volume_per_month=100, minutes_per_run=2)
    assert step.recommend_placement() == Placement.DETERMINISTIC


def test_judgment_gets_an_llm():
    step = WorkflowStep("Categorise complaint", requires_judgment=True)
    assert step.recommend_placement() == Placement.LLM


def test_irreversible_and_expensive_gets_a_human():
    step = WorkflowStep("Wire payment", reversible=False, error_cost=50_000, requires_judgment=True)
    assert step.recommend_placement() == Placement.HUMAN


def test_risky_judgment_gets_approval():
    """Judgment plus risk means draft-and-approve, not full autonomy."""
    step = WorkflowStep(
        "Approve discount",
        requires_judgment=True,
        error_cost=15_000,
        exceptions=["a", "b", "c"],
    )
    assert step.risk_score >= 0.5
    assert step.recommend_placement() == Placement.LLM_WITH_APPROVAL


def test_override_wins():
    step = WorkflowStep("Trivial step", placement_override=Placement.HUMAN)
    assert step.recommend_placement() == Placement.HUMAN


def test_approval_steps_are_not_counted_as_automatable():
    """Steps needing a signature still cost human time. Counting them is how
    projects promise savings that never appear."""
    wf = Workflow("test")
    wf.add(WorkflowStep("auto", volume_per_month=60, minutes_per_run=60))
    wf.add(WorkflowStep("signed", volume_per_month=60, minutes_per_run=60,
                        reversible=False, error_cost=20_000))
    assert wf.total_hours_per_month == 120.0
    assert wf.automatable_hours() == 60.0


def test_unmapped_exceptions_flags_unobserved_steps():
    wf = Workflow("test")
    wf.add(WorkflowStep("observed", exceptions=["something breaks"]))
    wf.add(WorkflowStep("not observed"))
    assert wf.unmapped_exceptions() == ["not observed"]


def test_markdown_report_includes_tribal_knowledge():
    wf = Workflow("AP", owner="Dana")
    wf.add(WorkflowStep("Intake", tribal_knowledge=["Northwind goes to Sarah"], exceptions=["x"]))
    md = wf.to_markdown()
    assert "Northwind goes to Sarah" in md and "Dana" in md


# ===========================================================================
# scorers
# ===========================================================================
def test_exact_match_normalises_case_and_space():
    assert exact_match("  Escalate ", "escalate").passed
    assert not exact_match("approve", "escalate").passed


def test_contains_scorer():
    assert contains("The window is 30 days from delivery", "30 days").passed
    assert not contains("No idea", "30 days").passed


def test_json_fields_reports_which_field_drifted():
    scorer = json_fields("category", "urgency")
    good = scorer('{"category": "refund", "urgency": "high", "extra": 1}',
                  {"category": "refund", "urgency": "high"})
    assert good.passed  # extra keys ignored

    bad = scorer('{"category": "refund", "urgency": "low"}',
                 {"category": "refund", "urgency": "high"})
    assert not bad.passed and "urgency" in bad.reason


def test_numeric_within_tolerance():
    assert numeric_within(0.5)("Total: $99.80", "$100.00").passed
    assert not numeric_within(0.01)("$99.00", "$100.00").passed
    assert numeric_within(0.05, relative=True)("$98", "$100").passed


def test_all_of_reports_every_failure():
    scorer = all_of(contains, numeric_within(0.01))
    result = scorer("nothing useful", "42")
    assert not result.passed
    assert result.reason.count(";") >= 1  # both scorers complained


def test_no_hallucinated_citations():
    context = "[1] first source [2] second source"
    assert no_hallucinated_facts("Per [1] and [2], yes.", context).passed
    bad = no_hallucinated_facts("Per [1] and [4], yes.", context)
    assert not bad.passed and "4" in bad.reason


def test_llm_judge_parses_score():
    judge = LLMJudge(
        FakeModel(responses=['{"score": 0.9, "reason": "covers the key points"}']),
        rubric="Does the summary capture the main decision?",
    )
    result = judge("some summary", "reference")
    assert result.passed and result.value == 0.9


def test_llm_judge_fails_gracefully_on_garbage():
    judge = LLMJudge(FakeModel(responses=["I think it's pretty good honestly"]), rubric="x")
    result = judge("a", "b")
    assert not result.passed and "unparseable" in result.reason


# ===========================================================================
# eval runner
# ===========================================================================
def test_run_eval_produces_pass_rate_and_tag_breakdown():
    ds = GoldenDataset("routing")
    ds.add("clean invoice", "approve", tags=["clean"])
    ds.add("no PO number", "escalate", tags=["missing-data"])
    ds.add("duplicate", "reject", tags=["duplicate"])

    report = run_eval(lambda x: "approve", ds, exact_match)
    assert report.total == 3 and report.passed == 1
    assert report.pass_rate == round(1 / 3, 4)
    assert report.failures_by_tag() == {"missing-data": 1, "duplicate": 1}


def test_critical_rate_is_reported_separately():
    """The headline can pass while every case that matters fails."""
    ds = GoldenDataset("t", [
        EvalCase("a", "x"), EvalCase("b", "x"), EvalCase("c", "x"),
        EvalCase("d", "y", critical=True),
    ])
    report = run_eval(lambda i: "x", ds, exact_match)
    assert report.pass_rate == 0.75
    assert report.critical_pass_rate == 0.0


def test_exceptions_become_failing_cases_not_crashes():
    """A suite that dies on case 2 tells you nothing about cases 3-50."""
    def explodes(x):
        if x == "bad":
            raise ValueError("kaboom")
        return "ok"

    ds = GoldenDataset("t", [EvalCase("good", "ok"), EvalCase("bad", "ok")])
    report = run_eval(explodes, ds, exact_match)
    assert report.total == 2 and report.passed == 1
    assert "kaboom" in report.failures()[0].error


def test_eval_runs_against_a_compiled_agent():
    """The runner must accept a graph, not just a function - otherwise the
    suite you write for a chain cannot be reused on the real agent."""
    @tool
    def double(n: int) -> int:
        """Double a number."""
        return n * 2

    model = FakeModel(
        responses=[
            Message.assistant("", tool_calls=[ToolCall(name="double", args={"n": 21})]),
            Message.assistant("The answer is 42."),
        ]
    )
    agent = create_agent(model=model, tools=[double])
    report = run_eval(agent, GoldenDataset("t", [EvalCase("double 21", "42")]), contains)
    assert report.passed == 1


def test_dataset_from_history_and_roundtrip():
    records = [
        {"q": "clean", "a": "approve", "kind": "clean"},
        {"q": "messy", "a": "escalate", "kind": "exception"},
    ]
    ds = GoldenDataset.from_history("hist", records, input_key="q", output_key="a", tag_key="kind")
    assert len(ds) == 2 and ds.cases[1].tags == ["exception"]

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "ds.json")
        ds.save(path)
        loaded = GoldenDataset.load(path)
    assert len(loaded) == 2 and loaded.cases[0].expected == "approve"


def test_dataset_filter_by_tag():
    ds = GoldenDataset("t")
    ds.add("a", "x", tags=["clean"])
    ds.add("b", "y", tags=["exception"])
    assert len(ds.filter("exception")) == 1


# ===========================================================================
# regression
# ===========================================================================
def test_flat_pass_rate_can_still_be_a_regression():
    """The headline number is not the safety check. A change that fixes one
    case and breaks another looks like no change at all."""
    ds = GoldenDataset("t", [EvalCase("a", "1"), EvalCase("b", "2")])
    baseline = run_eval(lambda x: "1", ds, exact_match)   # a passes, b fails
    current = run_eval(lambda x: "2", ds, exact_match)    # b passes, a fails

    reg = Regression(baseline, current)
    assert reg.delta == 0.0
    assert reg.is_regression
    assert reg.newly_failing == ["a"] and reg.newly_passing == ["b"]


def test_clean_improvement_is_not_a_regression():
    ds = GoldenDataset("t", [EvalCase("a", "1"), EvalCase("b", "1")])
    baseline = run_eval(lambda x: "0", ds, exact_match)
    current = run_eval(lambda x: "1", ds, exact_match)
    reg = Regression(baseline, current)
    assert not reg.is_regression and reg.delta == 1.0


# ===========================================================================
# audit trail
# ===========================================================================
def test_audit_trail_records_and_filters():
    trail = AuditTrail()
    trail.log("classify", thread_id="t1", reasoning="PO matched")
    trail.log("pay", actor="human:a@b.com", thread_id="t1", decision="approved")
    trail.log("extract", thread_id="t2", success=False, error="boom")

    assert len(trail.entries) == 3
    assert len(trail.for_thread("t1")) == 2
    assert len(trail.failures()) == 1
    assert "PO matched" in trail.to_report()


def test_audit_trail_persists_as_jsonl():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "audit.jsonl"
        trail = AuditTrail(path)
        trail.log("a")
        trail.log("b")
        lines = path.read_text().strip().splitlines()
    assert len(lines) == 2 and '"action": "a"' in lines[0]


def test_unknown_kwargs_land_in_inputs():
    trail = AuditTrail()
    entry = trail.log("pay", amount=49.0, vendor="Acme")
    assert entry.inputs["amount"] == 49.0 and entry.inputs["vendor"] == "Acme"


# ===========================================================================
# autonomy
# ===========================================================================
def test_low_levels_always_require_approval():
    for level in (AutonomyLevel.SHADOW, AutonomyLevel.SUGGEST, AutonomyLevel.APPROVE_EACH):
        assert AutonomyPolicy(level=level).requires_approval("anything")


def test_shadow_and_suggest_may_not_execute():
    assert not AutonomyPolicy(level=AutonomyLevel.SHADOW).may_execute()
    assert not AutonomyPolicy(level=AutonomyLevel.SUGGEST).may_execute()
    assert AutonomyPolicy(level=AutonomyLevel.APPROVE_EACH).may_execute()


def test_always_approve_survives_full_autonomy():
    """Raising the level must never silently un-gate a locked-down tool."""
    policy = AutonomyPolicy(level=AutonomyLevel.AUTONOMOUS, always_approve=["issue_refund"])
    assert policy.requires_approval("issue_refund", confidence=1.0)
    assert not policy.requires_approval("read_record", confidence=1.0)


def test_confidence_threshold_gates_exceptions():
    policy = AutonomyPolicy(level=AutonomyLevel.AUTO_WITH_EXCEPTIONS, confidence_threshold=0.9)
    assert not policy.requires_approval("classify", confidence=0.95)
    assert policy.requires_approval("classify", confidence=0.6)


def test_value_ceiling_overrides_high_confidence():
    policy = AutonomyPolicy(level=AutonomyLevel.AUTONOMOUS, value_ceiling=5_000)
    assert policy.requires_approval("pay", confidence=1.0, value=9_000)
    assert not policy.requires_approval("pay", confidence=1.0, value=100)


# ===========================================================================
# shadow mode
# ===========================================================================
def test_shadow_agreement_and_disagreements():
    shadow = ShadowLog()
    for i in range(10):
        shadow.compare(f"r{i}", "approve", "approve" if i < 8 else "escalate")
    assert shadow.agreement_rate == 0.8
    assert len(shadow.disagreements()) == 2


def test_promotion_needs_volume_not_just_a_good_rate():
    """100% agreement over six samples is noise."""
    shadow = ShadowLog()
    for i in range(6):
        shadow.compare(f"r{i}", "x", "x")
    assert shadow.agreement_rate == 1.0
    assert not shadow.ready_to_promote(min_samples=100)


def test_custom_match_function():
    shadow = ShadowLog()
    same_decision = lambda a, b: a.split(":")[0] == b.split(":")[0]  # noqa: E731
    c = shadow.compare("r1", "approve:fast", "approve:slow", match=same_decision)
    assert c.agreed


# ===========================================================================
# impact metrics
# ===========================================================================
def test_impact_arithmetic():
    m = ImpactMetrics(
        runs=1000, successes=980, escalations=200, failures=20,
        minutes_saved_per_run=6, loaded_hourly_cost=60,
        baseline_error_rate=0.05, agent_error_rate=0.01, cost_per_error=1000,
        model_cost=200, infra_cost=100,
    )
    assert m.automation_rate == 0.8          # 800 of 1000 handled without a human
    assert m.hours_saved == 80.0             # 800 runs * 6 min
    assert m.cost_saved == 4800.0
    assert m.errors_prevented == 40.0        # (0.05 - 0.01) * 1000
    assert m.risk_value == 40_000.0
    assert m.net_value == 44_500.0           # 4800 + 40000 - 300
    assert m.roi == round(44_500 / 300, 2)


def test_worse_than_human_shows_as_negative():
    """An agent worse than the people it replaced must not be able to hide
    behind an aggregate."""
    m = ImpactMetrics(runs=100, baseline_error_rate=0.01, agent_error_rate=0.05,
                      cost_per_error=500)
    assert m.errors_prevented < 0
    assert m.risk_value < 0
    assert "above" in m.to_markdown()  # the warning fires


def test_roi_is_none_when_nothing_was_spent():
    assert ImpactMetrics(runs=10).roi is None


# ===========================================================================
# promotion gate
# ===========================================================================
def test_gate_names_the_specific_shortfall():
    m = ImpactMetrics(runs=50, successes=45, failures=5)
    ready, blockers = PromotionCriteria(min_runs=100).evaluate(m, clean_days=0)
    assert not ready
    assert any("50 runs" in b for b in blockers)
    assert any("clean days" in b for b in blockers)


def test_gate_passes_when_everything_clears():
    m = ImpactMetrics(runs=500, successes=495, failures=5)
    shadow = ShadowLog()
    for i in range(200):
        shadow.compare(f"r{i}", "x", "x")
    ready, blockers = PromotionCriteria().evaluate(
        m, shadow=shadow, critical_eval_rate=1.0, clean_days=30
    )
    assert ready and blockers == []


# ===========================================================================
if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
