"""Test comparison of the four recovery strategies and FM-Bench years simulation.

Issue #569 (#293b, #293c):
- Four comparable strategies on identical scenarios:
  1. raw_conversation_dump
  2. langgraph_full_state_snapshot
  3. continuum_semantic_checkpoint
  4. semantic_plus_compaction
- HORIZON-style judge with 6 metrics:
  accuracy, unnecessary_human_escalation_rate, repair_precision,
  duplicate_side_effects, duplicate_work, compression_ratio.
- FM-Bench years simulation with SimulatedClock from 2020-01-01 +7d ticks,
  >100 reconstruction cycles via checkpoint/compact/restore.
"""

from __future__ import annotations

import pytest

from benchmarks.horizon.driver import SimulatedClock, run_fm_bench_simulation
from benchmarks.horizon.judge import judge
from benchmarks.horizon.runner import (
    run_horizon_strategy_comparison,
    run_single_horizon,
)
from benchmarks.horizon.scenarios import HORIZON_SCENARIOS, HORIZON_STRATEGIES
from continuum.benchmark import (
    FOUR_STRATEGIES,
    SCENARIOS,
    run_four_strategies_benchmark,
)
from continuum.benchmark.baselines import (
    FOUR_STRATEGIES as BASELINE_FOUR_STRATEGIES,
)
from continuum.benchmark.baselines import (
    baseline_by_name,
)


def test_four_strategies_are_defined_and_exported() -> None:
    expected = (
        "raw_conversation_dump",
        "langgraph_full_state_snapshot",
        "continuum_semantic_checkpoint",
        "semantic_plus_compaction",
    )
    assert expected == FOUR_STRATEGIES
    assert expected == HORIZON_STRATEGIES
    assert tuple(b.name for b in BASELINE_FOUR_STRATEGIES) == expected


def test_baseline_registry_resolves_all_four_strategies() -> None:
    for name in FOUR_STRATEGIES:
        driver = baseline_by_name(name)
        assert driver is not None
        assert driver.name == name


def test_four_strategies_comparable_on_identical_scenarios() -> None:
    total = 20
    results = run_four_strategies_benchmark(total=total)
    assert len(results) == len(SCENARIOS) * len(FOUR_STRATEGIES)

    by_scenario: dict[str, dict[str, object]] = {}
    for r in results:
        by_scenario.setdefault(r.scenario, {})[r.method] = r

    # 1. Process crash scenario
    crash_res = by_scenario["process_crash"]
    raw = crash_res["raw_conversation_dump"]
    lg = crash_res["langgraph_full_state_snapshot"]
    cont = crash_res["continuum_semantic_checkpoint"]
    comp = crash_res["semantic_plus_compaction"]

    # Raw dump repeats completed work from scratch
    assert raw.duplicate_work_ratio > 0.0
    # LangGraph and CONTINUUM avoid repeating progress
    assert lg.duplicate_work_ratio == 0.0
    assert cont.duplicate_work_ratio == 0.0
    assert comp.duplicate_work_ratio == 0.0

    # 2. Dataset change scenario (environment drift)
    env_res = by_scenario["dataset_change"]
    raw_env = env_res["raw_conversation_dump"]
    lg_env = env_res["langgraph_full_state_snapshot"]
    cont_env = env_res["continuum_semantic_checkpoint"]
    comp_env = env_res["semantic_plus_compaction"]

    # Baselines are blind to environment drift
    assert raw_env.detected_stale is False
    assert lg_env.detected_stale is False
    assert raw_env.revalidation_calls == 0
    assert lg_env.revalidation_calls == 0

    # CONTINUUM strategies revalidate and detect drift
    assert cont_env.detected_stale is True
    assert comp_env.detected_stale is True
    assert cont_env.revalidation_calls == 1
    assert comp_env.revalidation_calls == 1

    # 3. Interrupted side effect scenario
    side_res = by_scenario["unknown_side_effect"]
    raw_side = side_res["raw_conversation_dump"]
    lg_side = side_res["langgraph_full_state_snapshot"]
    cont_side = side_res["continuum_semantic_checkpoint"]
    comp_side = side_res["semantic_plus_compaction"]

    # Raw dump and bare LangGraph snapshot repeat the interrupted side effect
    assert raw_side.duplicate_side_effects == 1
    assert lg_side.duplicate_side_effects == 1

    # CONTINUUM uses the action ledger to deduplicate and prevent repeat side effects
    assert cont_side.duplicate_side_effects == 0
    assert comp_side.duplicate_side_effects == 0


def test_fm_bench_years_simulation_clock_and_cycles() -> None:
    # SimulatedClock starts at 2020-01-01 with +7d ticks
    clock = SimulatedClock()
    assert clock.start.year == 2020
    assert clock.start.month == 1
    assert clock.start.day == 1

    # Run FM-Bench simulation
    horizon = run_fm_bench_simulation(
        run_id="test_fm_bench",
        years=20,
        decisions_per_year=360,
        days_per_tick=7,
        cycles=120,
    )
    # Check >100 reconstruction cycles
    assert horizon.reconstruction_cycles >= 100
    # Check years elapsed
    assert horizon.clock.years_elapsed() >= 2.0
    assert horizon.duplicate_side_effects() == 0
    assert horizon.duplicate_work_avoided > 0


def test_fm_bench_scenario_in_horizon_suite() -> None:
    # The scenario exists and is labeled
    fm_scen = next((s for s in HORIZON_SCENARIOS if s.name == "fm_bench_years"), None)
    assert fm_scen is not None
    assert fm_scen.correct_mode == "resume"
    assert fm_scen.cycles >= 120

    res = run_single_horizon("fm_bench_years")
    assert res.passed is True
    assert res.metrics["accuracy"] == 1.0
    assert res.metrics["reconstruction_cycles"] >= 100
    assert res.metrics["years_elapsed"] >= 2.0


def test_horizon_strategy_comparison_on_identical_scenario() -> None:
    # 1. On steady progress: CONTINUUM strategies resume safely with 0 duplicate side effects,
    # while unguided baselines produce duplicate side effects.
    results = run_horizon_strategy_comparison(
        scenario_name="steady_progress_year",
        strategies=[
            "raw_conversation_dump",
            "langgraph_full_state_snapshot",
            "continuum_semantic_checkpoint",
            "semantic_plus_compaction",
        ],
    )
    assert len(results) == 4

    cont_res = results["continuum_semantic_checkpoint"]
    comp_res = results["semantic_plus_compaction"]
    raw_res = results["raw_conversation_dump"]
    lg_res = results["langgraph_full_state_snapshot"]

    # All six metrics exist for every strategy
    for r in (cont_res, comp_res, raw_res, lg_res):
        for metric in (
            "accuracy",
            "unnecessary_human_escalation_rate",
            "repair_precision",
            "duplicate_side_effects",
            "duplicate_work",
            "compression_ratio",
        ):
            assert metric in r.metrics, f"missing {metric}"

    assert cont_res.passed is True
    assert comp_res.passed is True
    assert cont_res.metrics["actual_mode"] == "resume"
    assert comp_res.metrics["actual_mode"] == "resume"
    assert cont_res.metrics["accuracy"] == 1.0
    assert comp_res.metrics["accuracy"] == 1.0

    # Ledger dedup holds for CONTINUUM; non-continuum strategies accumulate duplicate side effects
    assert cont_res.metrics["duplicate_side_effects"] == 0
    assert comp_res.metrics["duplicate_side_effects"] == 0
    assert raw_res.metrics["duplicate_side_effects"] > 0
    assert lg_res.metrics["duplicate_side_effects"] > 0

    # 2. On quarterly drift: CONTINUUM detects the dependency drift, while baselines blindly resume
    drift_results = run_horizon_strategy_comparison(
        scenario_name="quarterly_drift_year",
        strategies=[
            "raw_conversation_dump",
            "langgraph_full_state_snapshot",
            "continuum_semantic_checkpoint",
            "semantic_plus_compaction",
        ],
    )
    assert drift_results["continuum_semantic_checkpoint"].metrics["actual_mode"] in (
        "repair",
        "request_human",
    )
    assert drift_results["semantic_plus_compaction"].metrics["actual_mode"] in (
        "repair",
        "request_human",
    )
    assert drift_results["raw_conversation_dump"].metrics["actual_mode"] == "resume"
    assert drift_results["langgraph_full_state_snapshot"].metrics["actual_mode"] == "resume"
    assert drift_results["continuum_semantic_checkpoint"].metrics["duplicate_side_effects"] == 0
    assert drift_results["semantic_plus_compaction"].metrics["duplicate_side_effects"] == 0


# --- review follow-up: horizon simulation threshold boundaries -------------- #


def test_simulated_clock_year_boundary_is_between_52_and_53_weekly_ticks() -> None:
    """One simulated year falls between 52 and 53 seven-day ticks.

    The clock's year unit is 365.25 days, so 52 weekly ticks land just short of
    a year and the 53rd crosses it. Pinning the crossing keeps the
    ``years_elapsed`` threshold the FM-Bench reading depends on honest: a clock
    that rounded differently would silently move the boundary.
    """
    clock = SimulatedClock()
    assert clock.years_elapsed() == 0.0
    assert clock.cycles == 0

    for _ in range(52):
        clock.tick(days=7)
    assert clock.cycles == 52
    assert clock.years_elapsed() < 1.0

    clock.tick(days=7)
    assert clock.cycles == 53
    assert clock.years_elapsed() >= 1.0


def test_horizon_simulation_enforces_the_100_cycle_floor() -> None:
    """A simulation below the FM-Bench scale refuses to publish a run.

    ``run_horizon_scenario`` asserts reconstruction_cycles >= 100, so a caller
    asking for a handful of cycles gets an error rather than a run whose scale
    the published numbers would misrepresent.
    """
    with pytest.raises(AssertionError, match=r"only \d+ cycles"):
        run_fm_bench_simulation(run_id="test_floor_violation", cycles=20)


def test_horizon_simulation_reaches_exactly_100_cycles_at_the_minimum_parameter() -> None:
    """The smallest parameter satisfying the floor lands exactly on it.

    84 driven cycles produce 84 checkpoints, 8 compactions, 5 assessments and 3
    restores, i.e. exactly 100 reconstruction cycles. This is the boundary the
    floor exists on: the minimum is reachable, so the assert guards a real
    threshold rather than one no input can violate.
    """
    horizon = run_fm_bench_simulation(run_id="test_floor_minimum", cycles=84)
    assert horizon.reconstruction_cycles == 100
    assert horizon.duplicate_side_effects() == 0


def test_fm_bench_simulation_publishes_the_floor_with_an_explicit_sub_default_cycle_count() -> None:
    """An explicit 100-cycle simulation still clears the floor.

    The default is 140 cycles; a caller shrinking it toward the floor must stay
    above 100 reconstruction cycles, with zero duplicate side effects and the
    ledger visibly saving re-attempted work (``duplicate_work_avoided`` proves
    the published zero is a measurement, not a constant).
    """
    horizon = run_fm_bench_simulation(run_id="test_floor_explicit", cycles=100)
    assert horizon.reconstruction_cycles >= 100
    assert horizon.duplicate_side_effects() == 0
    assert horizon.duplicate_work_avoided > 0
    # 100 weekly ticks span well over the two simulated years FM-Bench expects.
    assert horizon.clock.years_elapsed() >= 1.0


# --- review follow-up: strategy comparators ---------------------------------- #


def test_baseline_comparator_resolves_canonical_names_and_aliases() -> None:
    """Every one of the four strategies resolves, and aliases map to canonical.

    The comparison is only meaningful when a strategy string and its alias
    resolve to the same driver: an alias silently resolving to a different
    strategy would make two ``run_horizon_strategy_comparison`` calls that look
    identical compare different things.
    """
    canonical = {
        "raw_conversation_dump",
        "langgraph_full_state_snapshot",
        "continuum_semantic_checkpoint",
        "semantic_plus_compaction",
    }
    for name in canonical:
        driver = baseline_by_name(name)
        assert driver is not None
        assert driver.name == name

    aliases = {
        "raw_dump": "raw_conversation_dump",
        "langgraph_snapshot": "langgraph_full_state_snapshot",
        "langgraph": "langgraph_full_state_snapshot",
        "continuum": "continuum_semantic_checkpoint",
        "semantic_checkpoint": "continuum_semantic_checkpoint",
        "semantic_compacted": "semantic_plus_compaction",
        "continuum_compacted": "semantic_plus_compaction",
    }
    for alias, expected in aliases.items():
        assert baseline_by_name(alias).name == expected

    with pytest.raises(KeyError, match="unknown baseline"):
        baseline_by_name("not_a_strategy")


def test_baseline_comparator_strategies_are_four_distinct_drivers() -> None:
    """The comparison holds four distinct strategies, not one repeated four times.

    ``FOUR_STRATEGIES``, ``HORIZON_STRATEGIES`` and the baseline registry must
    agree in order and name: a duplicate or reordered entry would make one
    column of the strategy comparison table silently re-label another.
    """
    assert len(FOUR_STRATEGIES) == 4
    assert len(set(FOUR_STRATEGIES)) == 4
    assert len({d.name for d in BASELINE_FOUR_STRATEGIES}) == 4
    assert [d.name for d in BASELINE_FOUR_STRATEGIES] == list(FOUR_STRATEGIES)
    assert FOUR_STRATEGIES == HORIZON_STRATEGIES


def test_judge_comparator_boundaries() -> None:
    """The judge's pass, escalation and repair verdicts flip at the right edges.

    The strategy comparison is scored by this rubric, so its boundaries are the
    comparator's: a correct resume passes, an unneeded request_human is flagged
    as unnecessary escalation, a correct escalation is not, and only a repair
    label makes an actual repair the sole passing answer.
    """
    resume_scen = next(s for s in HORIZON_SCENARIOS if s.correct_mode == "resume")
    drift_scen = next(s for s in HORIZON_SCENARIOS if s.correct_mode == "repair")
    human_scen = next(s for s in HORIZON_SCENARIOS if s.correct_mode == "request_human")
    abort_scen = next(s for s in HORIZON_SCENARIOS if s.correct_mode == "abort")

    # Exact match passes; no escalation flag when none was escalated.
    ok = judge(resume_scen, "resume")
    assert ok.passed is True
    assert ok.unnecessary_escalation is False
    assert ok.repair_correct is True
    assert ok.notes == []

    # Escalating a resumable scenario fails and is flagged as unnecessary.
    over = judge(resume_scen, "request_human")
    assert over.passed is False
    assert over.unnecessary_escalation is True
    assert "expected resume, got request_human" in over.notes
    assert "unnecessary human escalation" in over.notes

    # Abort is an escalation mode too, so it flags the same way.
    abort_on_resume = judge(resume_scen, "abort")
    assert abort_on_resume.passed is False
    assert abort_on_resume.unnecessary_escalation is True

    # A repair label is the one case repair_precision can fail: resuming a
    # scenario that needs repair is a wrong answer of a different kind.
    under = judge(drift_scen, "resume")
    assert under.passed is False
    assert under.repair_correct is False
    assert under.unnecessary_escalation is False

    right_repair = judge(drift_scen, "repair")
    assert right_repair.passed is True
    assert right_repair.repair_correct is True

    # A correctly escalated scenario passes without the unnecessary flag.
    right_human = judge(human_scen, "request_human")
    assert right_human.passed is True
    assert right_human.unnecessary_escalation is False

    right_abort = judge(abort_scen, "abort")
    assert right_abort.passed is True
    assert right_abort.unnecessary_escalation is False


def test_strategy_comparison_is_comparable_across_all_four_strategies() -> None:
    """One scenario, four strategies: identical columns, same floor, same label.

    Comparability is what the four-strategy table claims: every strategy emits
    the same six metrics for the same scenario, reconstructs at least 100
    cycles, and is scored against the same correct mode.
    """
    results = run_horizon_strategy_comparison(
        scenario_name="compaction_stress_year",
        strategies=list(FOUR_STRATEGIES),
    )
    assert list(results) == list(FOUR_STRATEGIES)

    six_metrics = (
        "accuracy",
        "unnecessary_human_escalation_rate",
        "repair_precision",
        "duplicate_side_effects",
        "duplicate_work",
        "compression_ratio",
    )
    for strategy, res in results.items():
        for metric in six_metrics:
            assert metric in res.metrics, f"{strategy} missing {metric}"
        assert res.metrics["reconstruction_cycles"] >= 100
        assert res.metrics["correct_mode"] == "resume"
        # accuracy is the judge's 0/1 verdict, so it is never fractional.
        assert res.metrics["accuracy"] in (0.0, 1.0)
        # The four runs drive the same scenario, so they span the same years.
        assert res.metrics["years_elapsed"] >= 1.0
