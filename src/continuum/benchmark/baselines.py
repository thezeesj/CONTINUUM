"""Comparison baselines for CONTINUUM-Bench (issue 11, issue 569).

Drivers the harness can run to compare recovery approaches. The naive baselines
deliberately do not use the recovery machinery, so the comparison isolates what
semantic checkpointing adds. The CONTINUUM driver runs through the existing
adapters and checkpoint manager. No results are fabricated here, these are the
drivers only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from continuum.checkpoint import CheckpointManager
from continuum.events import EventType
from continuum.storage import SQLiteStorage

__all__ = [
    "BASELINES",
    "Baseline",
    "ContinuumSemanticCheckpoint",
    "FOUR_STRATEGIES",
    "FullTranscriptReplay",
    "LangGraphFullStateSnapshot",
    "NaiveCheckpointing",
    "RawConversationDump",
    "SemanticPlusCompaction",
    "SimpleConversationSummarization",
    "StructuredTaskSummary",
    "baseline_by_name",
]


class Baseline(Protocol):
    """Protocol defining the interface for benchmark baseline drivers."""

    name: str

    def run(self, storage: SQLiteStorage, run_id: str) -> dict[str, str]:
        """Execute the baseline strategy for a run and return result metrics."""
        ...


@dataclass(frozen=True, slots=True)
class FullTranscriptReplay:
    """Baseline strategy that replays the full event log without compression."""

    name: str = "full_transcript_replay"

    def run(self, storage: SQLiteStorage, run_id: str) -> dict[str, str]:
        """Read and replay all events for the given run."""
        events = storage.read_events(run_id)
        return {"replayed": str(len(events)), "mode": "replay_all"}


@dataclass(frozen=True, slots=True)
class RawConversationDump:
    """Baseline strategy that captures and replays raw conversation history."""

    name: str = "raw_conversation_dump"

    def run(self, storage: SQLiteStorage, run_id: str) -> dict[str, str]:
        """Replay all recorded events from the raw dump."""
        events = storage.read_events(run_id)
        return {"replayed": str(len(events)), "mode": "replay_all"}


@dataclass(frozen=True, slots=True)
class SimpleConversationSummarization:
    """Baseline strategy that condenses the event stream into a text summary."""

    name: str = "simple_conversation_summarization"

    def run(self, storage: SQLiteStorage, run_id: str) -> dict[str, str]:
        """Summarize all recorded events for the given run."""
        events = storage.read_events(run_id)
        summary = f"summarized {len(events)} events"
        return {"summary": summary, "mode": "summarize"}


@dataclass(frozen=True, slots=True)
class NaiveCheckpointing:
    """Baseline strategy that takes unguided checkpoints without recovery logic."""

    name: str = "naive_checkpointing"

    def run(self, storage: SQLiteStorage, run_id: str) -> dict[str, str]:
        """Attempt to checkpoint the run, catching and recording any failure."""
        manager = CheckpointManager(storage)
        try:
            checkpoint = manager.checkpoint(run_id)
            return {"checkpoint_id": checkpoint.checkpoint_id, "mode": "naive"}
        except Exception as exc:
            return {"error": str(exc), "mode": "naive_failed"}


@dataclass(frozen=True, slots=True)
class StructuredTaskSummary:
    """Baseline strategy that extracts structured summaries of completed work."""

    name: str = "structured_task_summary"

    def run(self, storage: SQLiteStorage, run_id: str) -> dict[str, str]:
        """Extract and count completed work events for the given run."""
        events = storage.read_events(run_id)
        tasks = [e for e in events if e.type == EventType.WORK_COMPLETED]
        return {"tasks": str(len(tasks)), "mode": "structured_summary"}


@dataclass(frozen=True, slots=True)
class ContinuumSemanticCheckpoint:
    """Baseline driver evaluating CONTINUUM semantic checkpointing."""

    name: str = "continuum_semantic_checkpoint"

    def run(self, storage: SQLiteStorage, run_id: str) -> dict[str, str]:
        """Create a semantic checkpoint for the run using CheckpointManager."""
        manager = CheckpointManager(storage)
        checkpoint = manager.checkpoint(run_id)
        return {"checkpoint_id": checkpoint.checkpoint_id, "mode": "continuum"}


@dataclass(frozen=True, slots=True)
class LangGraphFullStateSnapshot:
    """Baseline driver capturing full raw state snapshots without semantic filtering."""

    name: str = "langgraph_full_state_snapshot"

    def run(self, storage: SQLiteStorage, run_id: str) -> dict[str, str]:
        """Capture the full uncompacted event stream as a state snapshot."""
        events = list(storage.read_events(run_id))
        state_snapshot = {f"event_{i}": e.payload for i, e in enumerate(events)}
        snapshot_bytes = len(json.dumps(state_snapshot, sort_keys=True).encode("utf-8"))
        return {
            "snapshot_events": str(len(events)),
            "snapshot_bytes": str(snapshot_bytes),
            "mode": "full_state_snapshot",
        }


@dataclass(frozen=True, slots=True)
class SemanticPlusCompaction:
    """Baseline driver evaluating CONTINUUM semantic checkpointing with compaction."""

    name: str = "semantic_plus_compaction"

    def run(self, storage: SQLiteStorage, run_id: str) -> dict[str, str]:
        """Create a semantic checkpoint and compact the live log."""
        manager = CheckpointManager(storage)
        checkpoint = manager.checkpoint(run_id)
        storage.compact_run(run_id)
        return {
            "checkpoint_id": checkpoint.checkpoint_id,
            "mode": "semantic_compacted",
        }


FOUR_STRATEGIES: tuple[Baseline, ...] = (  # type: ignore[assignment]
    RawConversationDump(),
    LangGraphFullStateSnapshot(),
    ContinuumSemanticCheckpoint(),
    SemanticPlusCompaction(),
)

BASELINES: tuple[Baseline, ...] = (  # type: ignore[assignment]
    FullTranscriptReplay(),
    RawConversationDump(),
    SimpleConversationSummarization(),
    NaiveCheckpointing(),
    StructuredTaskSummary(),
    ContinuumSemanticCheckpoint(),
    LangGraphFullStateSnapshot(),
    SemanticPlusCompaction(),
)

_ALIASES: dict[str, str] = {
    "raw_dump": "raw_conversation_dump",
    "replay": "full_transcript_replay",
    "langgraph_snapshot": "langgraph_full_state_snapshot",
    "langgraph": "langgraph_full_state_snapshot",
    "continuum": "continuum_semantic_checkpoint",
    "semantic_checkpoint": "continuum_semantic_checkpoint",
    "semantic_compacted": "semantic_plus_compaction",
    "continuum_compacted": "semantic_plus_compaction",
    "semantic_compaction": "semantic_plus_compaction",
}


def baseline_by_name(name: str) -> Baseline:
    """Look up a registered baseline driver by its name or alias string."""
    canonical = _ALIASES.get(name, name)
    for baseline in BASELINES:
        if baseline.name == canonical:
            return baseline
    raise KeyError(f"unknown baseline {name!r}")
