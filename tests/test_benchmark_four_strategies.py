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

from benchmarks.horizon.driver import SimulatedClock, run_fm_bench_simulation
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
