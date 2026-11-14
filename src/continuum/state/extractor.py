"""Pluggable semantic state extraction.

An extractor turns a trajectory (the run's recorded events) plus an environment
into semantic state. The deterministic extractor is the default and the only
one required: it folds the event log and calls nothing external.

The optional LLM extractor exists for state that was never recorded
structurally: free-text reasoning a framework did not emit as events. It is
constrained by design:

* It runs only when explicitly enabled and given a callable; there is no
  provider SDK, no network default, no API key handling.
* It may only *add* components, never modify or delete what the deterministic
  fold produced. A model cannot overwrite a recorded fact.
* Everything it produces is tagged ``Origin.LLM`` and forced to
  ``REQUIRES_REVIEW``, so inferred state can never be silently trusted during
  recovery.

That asymmetry is the point: the deterministic layer is authoritative, the model
is an advisor.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from continuum.events import Event
from continuum.models import (
    Decision,
    EnvironmentSnapshot,
    Finding,
    Origin,
    PendingWork,
    Provenance,
    SemanticState,
    StateStatus,
)
from continuum.provenance_map import derived_origin
from continuum.state.semantic import (
    ProjectionReport,
    _as_str_list,
    _weakest_from_state,
    project_incremental,
)

__all__ = [
    "ExtractionContext",
    "StateExtractor",
    "DeterministicExtractor",
    "LLMExtractor",
    "CompositeExtractor",
    "LLMProposal",
]


@dataclass(frozen=True, slots=True)
class ExtractionContext:
    """Everything an extractor is allowed to look at.

    ``weakest_seed`` is the trust watermark the base's own fold held, which a
    base state cannot reconstruct on its own (#1933, #2039): the fold weakens
    the watermark from *every* consumed event, non-projecting ones included,
    and half the projecting components carry no provenance at all. A caller
    that holds the log should pass ``weakest_origin_over`` of the events the
    base folded, which reproduces a fresh read's watermark exactly; ``None``
    keeps the per-fact approximation, i.e. today's behaviour for callers with
    no log.
    """

    run_id: str
    trajectory: Sequence[Event]
    environment: EnvironmentSnapshot | None = None
    base: SemanticState | None = None
    weakest_seed: Any | None = None


def _accumulated_weakest(context: ExtractionContext) -> Any | None:
    """The watermark a deterministic fold of ``context`` ends at (#2039).

    ``project_incremental`` seeds the fold from ``weakest_seed`` when the
    caller supplied one and from the base's per-fact provenances otherwise,
    then weakens the watermark from every event it folds. Downstream stages
    of a composite must be seeded with that result rather than with the
    caller's seed, which the first stage's fold has already weakened further.
    """
    seed = context.weakest_seed
    if seed is None:
        seed = _weakest_from_state(context.base)
    candidates: list[Origin] = []
    if seed is not None:
        candidates.append(seed)
    candidates.extend(event.source for event in context.trajectory)
    return derived_origin(candidates) if candidates else None


@runtime_checkable
class StateExtractor(Protocol):
    """Turn a trajectory into semantic state.

    Implementations must be side-effect free and must not require network
    access unless explicitly configured by the caller.
    """

    name: str

    def extract(self, context: ExtractionContext) -> SemanticState:
        """Derive semantic state from an extraction context.

        Implementations must be side-effect free and should fold events onto
        any base state supplied in ``context``. Implementations that fold
        events onto ``context.base`` must honour ``context.weakest_seed`` when
        stamping derived provenance, so a base-plus-tail read never asserts a
        fact at higher trust than a fresh read of the same log (#2039).
        """
        ...


class DeterministicExtractor:
    """Folds recorded events into state. No model, no network, no clock."""

    name = "deterministic"

    def __init__(self) -> None:
        self.last_report: ProjectionReport | None = None

    def extract(self, context: ExtractionContext) -> SemanticState:
        """Fold recorded events into deterministic semantic state.

        Applies events from ``context.trajectory`` onto ``context.base`` (or
        from scratch when base is absent) using :func:`project_incremental`,
        seeding the trust watermark from ``context.weakest_seed`` so a
        base-plus-tail read stamps derived trust exactly like a fresh read of
        the same log (#2033). Without a seed the fold falls back to the base's
        per-fact provenances, which can only approximate the watermark a live
        fold held. Updates :attr:`last_report` with projection metrics and
        returns the resulting state.
        """
        state, report = project_incremental(
            context.run_id,
            context.trajectory,
            base=context.base,
            weakest_seed=context.weakest_seed,
        )
        self.last_report = report
        return state


@dataclass(frozen=True, slots=True)
class LLMProposal:
    """What a model is permitted to suggest."""

    decisions: Sequence[dict[str, Any]] = ()
    findings: Sequence[dict[str, Any]] = ()
    pending_work: Sequence[dict[str, Any]] = ()


#: A caller-supplied function. CONTINUUM never constructs one.
LLMCallable = Callable[[ExtractionContext, SemanticState], LLMProposal]


class LLMExtractor:
    """Optional enrichment layer over a base extractor.

    ``llm`` is supplied by the caller, since CONTINUUM has no provider
    dependency. If it raises, extraction falls back to the deterministic result
    rather than failing the run: losing an optional enrichment must never cost a
    recovery.
    """

    name = "llm"

    def __init__(
        self,
        llm: LLMCallable,
        *,
        base: StateExtractor | None = None,
        enabled: bool = True,
    ) -> None:
        self._llm = llm
        self._base: StateExtractor = base or DeterministicExtractor()
        self.enabled = enabled
        self.last_error: Exception | None = None

    def extract(self, context: ExtractionContext) -> SemanticState:
        """Enrich base state with candidate proposals from the configured model.

        Runs the underlying base extractor first. If disabled, returns the base
        state directly. When enabled, invokes the LLM callable and merges
        suggested decisions, findings, and pending work tagged as
        ``Origin.LLM`` and ``REQUIRES_REVIEW``. If the LLM callable raises,
        records the exception in :attr:`last_error` and falls back to base
        state so recovery is never blocked.
        """
        state = self._base.extract(context)
        if not self.enabled:
            return state

        try:
            proposal = self._llm(context, state)
            return self._merge(state, proposal)
        except Exception as exc:
            self.last_error = exc
            return state

    def _merge(self, state: SemanticState, proposal: LLMProposal) -> SemanticState:
        provenance = Provenance(origin=Origin.LLM, extractor=self.name)

        known_decisions = {d.decision_id for d in state.decisions}
        decisions = list(state.decisions)
        for raw in proposal.decisions:
            decision_id = str(raw.get("decision_id", ""))
            if not decision_id or decision_id in known_decisions:
                continue  # never overwrite a recorded fact
            # Also guards against a repeat later in this same proposal: the set is
            # seeded from the base state, so without this the second copy of an id
            # the LLM emitted twice would not be recognised as already handled.
            known_decisions.add(decision_id)
            decisions.append(
                Decision(
                    decision_id=decision_id,
                    decision=str(raw.get("decision", "")),
                    reason=str(raw.get("reason", "")),
                    evidence=_as_str_list(raw.get("evidence")),
                    status=StateStatus.REQUIRES_REVIEW,
                    provenance=provenance,
                )
            )

        known_findings = {f.finding_id for f in state.findings}
        findings = list(state.findings)
        for raw in proposal.findings:
            finding_id = str(raw.get("finding_id", ""))
            if not finding_id or finding_id in known_findings:
                continue
            known_findings.add(finding_id)
            confidence = float(raw.get("confidence", 0.5))
            findings.append(
                Finding(
                    finding_id=finding_id,
                    claim=str(raw.get("claim", "")),
                    evidence=_as_str_list(raw.get("evidence")),
                    confidence=min(max(confidence, 0.0), 1.0),
                    status=StateStatus.REQUIRES_REVIEW,
                    provenance=provenance,
                )
            )

        known_work = {w.task_id for w in state.pending_work}
        pending = list(state.pending_work)
        for raw in proposal.pending_work:
            task_id = str(raw.get("task_id", ""))
            if not task_id or task_id in known_work:
                continue
            known_work.add(task_id)
            pending.append(
                PendingWork(
                    task_id=task_id,
                    description=str(raw.get("description", "")),
                    status=StateStatus.REQUIRES_REVIEW,
                    provenance=provenance,
                )
            )

        return state.model_copy(
            update={
                "decisions": decisions,
                "findings": findings,
                "pending_work": pending,
            }
        )


class CompositeExtractor:
    """Chain extractors, feeding each result forward as the next one's base.

    Later extractors see an empty trajectory: the events were already folded by
    the first, and replaying them would double-apply progress counters.
    """

    name = "composite"

    def __init__(self, extractors: Iterable[StateExtractor]) -> None:
        self._extractors = tuple(extractors)
        if not self._extractors:
            raise ValueError("CompositeExtractor requires at least one extractor")

    def extract(self, context: ExtractionContext) -> SemanticState:
        """Run each configured extractor in sequence, chaining intermediate states.

        The first extractor receives the full context and folds recorded
        trajectory events. Downstream extractors receive the accumulated state
        as ``context.base`` with an empty trajectory, preventing double-counting
        of events while allowing successive enrichments, and are seeded with the
        watermark that accumulated state holds: the first stage's fold has
        weakened the caller's seed by every event it consumed, and a downstream
        extractor that stamps derived provenance must not launder that away
        (#2039). As the pipeline advances across multi-stage chains, the seed
        incorporates the evolving state's weakest origin to maintain
        monotonicity across intermediate stages.
        """
        first, *rest = self._extractors
        state = first.extract(context)
        weakest_seed = _accumulated_weakest(context)
        for extractor in rest:
            state_weakest = _weakest_from_state(state)
            if state_weakest is not None:
                weakest_seed = (
                    derived_origin([weakest_seed, state_weakest])
                    if weakest_seed is not None
                    else state_weakest
                )
            state = extractor.extract(
                ExtractionContext(
                    run_id=context.run_id,
                    trajectory=(),
                    environment=context.environment,
                    base=state,
                    weakest_seed=weakest_seed,
                )
            )
        return state
