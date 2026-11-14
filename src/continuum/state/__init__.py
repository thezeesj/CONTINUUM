"""Semantic state: projection, extraction, versioning and diffing."""

from continuum.state.diff import diff_states, render_diff
from continuum.state.extractor import (
    CompositeExtractor,
    DeterministicExtractor,
    ExtractionContext,
    LLMExtractor,
    LLMProposal,
    StateExtractor,
)
from continuum.state.semantic import (
    ProjectionError,
    ProjectionReport,
    project,
    project_incremental,
    weakest_origin_over,
)
from continuum.state.versioning import VersionChain, VersionEntry, state_fingerprint

__all__ = [
    "CompositeExtractor",
    "DeterministicExtractor",
    "ExtractionContext",
    "LLMExtractor",
    "LLMProposal",
    "ProjectionError",
    "ProjectionReport",
    "StateExtractor",
    "VersionChain",
    "VersionEntry",
    "diff_states",
    "project",
    "project_incremental",
    "render_diff",
    "state_fingerprint",
    "weakest_origin_over",
]
