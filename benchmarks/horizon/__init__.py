"""Horizon-scale scenarios with simulated years and a recovery-decision judge.

Public API for the horizon benchmark. The suite drives hundreds of
compaction/archive/briefing cycles per scenario under a simulated clock
decoupled from wall clock, then scores the recovery decision against a
pre-labeled correct mode.
"""

from .driver import SimulatedClock, run_fm_bench_simulation, run_horizon_scenario
from .judge import JudgeLabel, judge
from .runner import (
    HORIZON_SCENARIOS,
    run_horizon_strategy_comparison,
    run_horizon_suite,
    run_single_horizon,
)
from .scenarios import HORIZON_STRATEGIES, HorizonScenario

__all__ = [
    "HORIZON_SCENARIOS",
    "HORIZON_STRATEGIES",
    "HorizonScenario",
    "JudgeLabel",
    "SimulatedClock",
    "judge",
    "run_fm_bench_simulation",
    "run_horizon_scenario",
    "run_horizon_strategy_comparison",
    "run_horizon_suite",
    "run_single_horizon",
]
