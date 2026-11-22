"""Baseline driver behaviour for CONTINUUM-Bench (issue 11).

The five drivers are the comparison's control group: they deliberately do not
use the recovery machinery, so a result attributed to semantic checkpointing
has to come from the CONTINUUM driver alone. These tests pin each driver's
exact output, the failure arm of the naive checkpointing baseline, and the
name registry the harness looks drivers up through (issue #2048).
"""

from __future__ import annotations

import pytest

from continuum.benchmark.baselines import (
    BASELINES,
    FOUR_STRATEGIES,
    ContinuumSemanticCheckpoint,
    FullTranscriptReplay,
    LangGraphFullStateSnapshot,
    NaiveCheckpointing,
    SemanticPlusCompaction,
    SimpleConversationSummarization,
    StructuredTaskSummary,
    baseline_by_name,
)
from continuum.events import EventType
from continuum.models import Run
from continuum.storage import SQLiteStorage


def _store_with_events() -> SQLiteStorage:
    storage = SQLiteStorage(":memory:")
    storage.create_run(Run(run_id="run_1", goal="g"))
    storage.append_event("run_1", EventType.RUN_STARTED, {"goal": "g", "total": 2})
    storage.append_event("run_1", EventType.TOOL_COMPLETED, {"tool": "t", "status": 200})
    storage.append_event("run_1", EventType.WORK_COMPLETED, {"task": "a"})
    storage.append_event("run_1", EventType.WORK_COMPLETED, {"task": "b"})
    return storage


def test_each_baseline_runs() -> None:
    for baseline in BASELINES:
        storage = SQLiteStorage(":memory:")
        storage.create_run(Run(run_id="run_1", goal="g"))
        storage.append_event("run_1", EventType.RUN_STARTED, {"goal": "g", "total": 1})
        result = baseline.run(storage, "run_1")
        assert "mode" in result


def test_baseline_lookup() -> None:
    assert baseline_by_name("full_transcript_replay").name == "full_transcript_replay"


def test_every_registered_baseline_is_found_by_name() -> None:
    """The harness resolves drivers by name; a driver that is instantiated in
    BASELINES but whose name does not match its registry entry is invisible."""
    for baseline in BASELINES:
        assert baseline_by_name(baseline.name) is baseline


def test_an_unknown_baseline_name_raises_key_error() -> None:
    with pytest.raises(KeyError, match="unknown baseline 'nope'"):
        baseline_by_name("nope")


def test_baselines_and_four_strategies_are_registered() -> None:
    assert {baseline.name for baseline in BASELINES} == {
        "full_transcript_replay",
        "raw_conversation_dump",
        "simple_conversation_summarization",
        "naive_checkpointing",
        "structured_task_summary",
        "continuum_semantic_checkpoint",
        "langgraph_full_state_snapshot",
        "semantic_plus_compaction",
    }
    assert {baseline.name for baseline in FOUR_STRATEGIES} == {
        "raw_conversation_dump",
        "langgraph_full_state_snapshot",
        "continuum_semantic_checkpoint",
        "semantic_plus_compaction",
    }
    assert (
        ContinuumSemanticCheckpoint().name == "continuum_semantic_checkpoint"
    )  # sanity: the registry covers the class, not just the instance
    assert LangGraphFullStateSnapshot().name == "langgraph_full_state_snapshot"
    assert SemanticPlusCompaction().name == "semantic_plus_compaction"


def test_baseline_aliases_resolve_properly() -> None:
    assert baseline_by_name("raw_conversation_dump").name == "raw_conversation_dump"
    assert baseline_by_name("raw_dump").name == "raw_conversation_dump"
    assert baseline_by_name("replay").name == "full_transcript_replay"
    assert baseline_by_name("replay").name == "full_transcript_replay"
    assert baseline_by_name("langgraph_snapshot").name == "langgraph_full_state_snapshot"
    assert baseline_by_name("langgraph").name == "langgraph_full_state_snapshot"
    assert baseline_by_name("continuum").name == "continuum_semantic_checkpoint"
    assert baseline_by_name("semantic_compacted").name == "semantic_plus_compaction"


def test_langgraph_full_state_snapshot_captures_all_events() -> None:
    storage = _store_with_events()
    res = LangGraphFullStateSnapshot().run(storage, "run_1")
    assert res["mode"] == "full_state_snapshot"
    assert res["snapshot_events"] == "4"
    assert int(res["snapshot_bytes"]) > 0


def test_semantic_plus_compaction_runs_checkpoint_and_compact() -> None:
    storage = _store_with_events()
    res = SemanticPlusCompaction().run(storage, "run_1")
    assert res["mode"] == "semantic_compacted"
    assert res["checkpoint_id"]


def test_full_transcript_replay_reports_the_event_count() -> None:
    storage = _store_with_events()
    assert FullTranscriptReplay().run(storage, "run_1") == {
        "replayed": "4",
        "mode": "replay_all",
    }


def test_simple_conversation_summarization_reports_the_summary() -> None:
    storage = _store_with_events()
    assert SimpleConversationSummarization().run(storage, "run_1") == {
        "summary": "summarized 4 events",
        "mode": "summarize",
    }


def test_structured_task_summary_counts_only_completed_work() -> None:
    """RUN_STARTED and TOOL_COMPLETED events are not work: only
    WORK_COMPLETED rows reach the count."""
    storage = _store_with_events()
    assert StructuredTaskSummary().run(storage, "run_1") == {
        "tasks": "2",
        "mode": "structured_summary",
    }


def test_structured_task_summary_with_no_work_reports_zero() -> None:
    storage = SQLiteStorage(":memory:")
    storage.create_run(Run(run_id="run_1", goal="g"))
    storage.append_event("run_1", EventType.RUN_STARTED, {"goal": "g", "total": 1})
    assert StructuredTaskSummary().run(storage, "run_1") == {
        "tasks": "0",
        "mode": "structured_summary",
    }


def test_naive_checkpointing_records_a_failure_instead_of_propagating_it() -> None:
    """The failure arm of the naive baseline: a run that cannot be checkpointed
    yields a ``naive_failed`` row carrying the error, never an exception that
    aborts the whole benchmark harness."""
    storage = SQLiteStorage(":memory:")
    storage.create_run(Run(run_id="run_1", goal="g"))
    result = NaiveCheckpointing().run(storage, "no_such_run")
    assert result["mode"] == "naive_failed"
    assert "error" in result
    assert result["error"]


def test_naive_checkpointing_reports_the_checkpoint_on_success() -> None:
    storage = _store_with_events()
    result = NaiveCheckpointing().run(storage, "run_1")
    assert result["mode"] == "naive"
    assert result["checkpoint_id"]
