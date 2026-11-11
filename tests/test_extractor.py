from __future__ import annotations

import pytest

from continuum.events import Event, EventLog, EventType
from continuum.models import Origin, SemanticState, StateStatus
from continuum.state.extractor import (
    CompositeExtractor,
    DeterministicExtractor,
    ExtractionContext,
    LLMExtractor,
    LLMProposal,
    StateExtractor,
)
from continuum.state.semantic import project, project_incremental, weakest_origin_over
from tests.test_provenance_compaction import _MIDDLE_EVENTS


def build_log(total: int = 10) -> EventLog:
    log = EventLog()
    log.append("run_1", EventType.RUN_STARTED, {"goal": "Analyze documents", "total": total})
    log.append("run_1", EventType.DECISION_CREATED, {"decision_id": "d1", "decision": "recorded"})
    log.append("run_1", EventType.WORK_COMPLETED, {})
    return log


def context(log: EventLog) -> ExtractionContext:
    return ExtractionContext(run_id="run_1", trajectory=log.events("run_1"))


# --- deterministic --------------------------------------------------------- #


def test_deterministic_extractor_matches_a_direct_projection() -> None:
    log = build_log()
    extractor = DeterministicExtractor()
    assert extractor.extract(context(log)) == project("run_1", log.events("run_1"))


def test_deterministic_extractor_reports_what_it_folded() -> None:
    log = build_log()
    extractor = DeterministicExtractor()
    extractor.extract(context(log))
    assert extractor.last_report is not None
    assert extractor.last_report.consumed == 3
    assert extractor.last_report.complete


def test_extractors_satisfy_the_protocol() -> None:
    assert isinstance(DeterministicExtractor(), StateExtractor)
    assert isinstance(LLMExtractor(lambda ctx, state: LLMProposal()), StateExtractor)


# --- the LLM layer is advisory, never authoritative ------------------------ #


def test_llm_additions_are_marked_for_review_and_tagged() -> None:
    log = build_log()
    extractor = LLMExtractor(
        lambda ctx, state: LLMProposal(
            findings=[{"finding_id": "f_llm", "claim": "inferred", "confidence": 0.7}]
        )
    )
    state = extractor.extract(context(log))
    finding = state.finding("f_llm")
    assert finding is not None
    assert finding.status is StateStatus.REQUIRES_REVIEW
    assert finding.provenance.origin is Origin.LLM
    assert not finding.provenance.reproducible


def test_the_llm_cannot_overwrite_a_recorded_decision() -> None:
    log = build_log()
    extractor = LLMExtractor(
        lambda ctx, state: LLMProposal(decisions=[{"decision_id": "d1", "decision": "hijacked"}])
    )
    state = extractor.extract(context(log))
    decision = state.decision("d1")
    assert decision is not None
    assert decision.decision == "recorded"
    assert decision.provenance.origin is Origin.DETERMINISTIC


def test_the_llm_cannot_delete_recorded_state() -> None:
    log = build_log()
    baseline = DeterministicExtractor().extract(context(log))
    state = LLMExtractor(lambda ctx, s: LLMProposal()).extract(context(log))
    assert state == baseline


def test_a_failing_llm_degrades_to_the_deterministic_state() -> None:
    log = build_log()

    def broken(ctx: ExtractionContext, state: object) -> LLMProposal:
        raise RuntimeError("provider unavailable")

    extractor = LLMExtractor(broken)
    state = extractor.extract(context(log))

    assert state == project("run_1", log.events("run_1"))
    assert isinstance(extractor.last_error, RuntimeError)


def test_a_malformed_proposal_degrades_to_the_deterministic_state() -> None:
    log = build_log()

    def malformed(ctx: ExtractionContext, state: object) -> LLMProposal:
        return LLMProposal(findings=[{"finding_id": "f1", "confidence": "high"}])

    extractor = LLMExtractor(malformed)
    state = extractor.extract(context(log))

    assert state == project("run_1", log.events("run_1"))
    assert isinstance(extractor.last_error, ValueError)


def test_a_disabled_llm_extractor_never_calls_out() -> None:
    calls: list[int] = []

    def counting(ctx: ExtractionContext, state: object) -> LLMProposal:
        calls.append(1)
        return LLMProposal()

    extractor = LLMExtractor(counting, enabled=False)
    extractor.extract(context(build_log()))
    assert calls == []


def test_llm_confidence_is_clamped_to_the_unit_interval() -> None:
    log = build_log()
    extractor = LLMExtractor(
        lambda ctx, state: LLMProposal(
            findings=[{"finding_id": "f_hi", "claim": "c", "confidence": 4.2}]
        )
    )
    finding = extractor.extract(context(log)).finding("f_hi")
    assert finding is not None and finding.confidence == 1.0


def test_proposals_without_ids_are_dropped() -> None:
    log = build_log()
    extractor = LLMExtractor(lambda ctx, state: LLMProposal(findings=[{"claim": "anonymous"}]))
    assert (
        extractor.extract(context(log)).findings
        == project("run_1", build_log().events("run_1")).findings
    )


# --- composition ----------------------------------------------------------- #


def test_a_repeated_decision_id_within_one_proposal_is_collapsed() -> None:
    """The known-id set is seeded from the base state; it must also grow as we append."""
    log = build_log()
    extractor = LLMExtractor(
        lambda ctx, state: LLMProposal(
            decisions=[
                {"decision_id": "X", "decision": "first"},
                {"decision_id": "X", "decision": "second"},
            ]
        )
    )
    state = extractor.extract(context(log))
    assert [d.decision_id for d in state.decisions].count("X") == 1


def test_a_repeated_finding_id_within_one_proposal_is_collapsed() -> None:
    log = build_log()
    extractor = LLMExtractor(
        lambda ctx, state: LLMProposal(
            findings=[
                {"finding_id": "Y", "claim": "a"},
                {"finding_id": "Y", "claim": "b"},
            ]
        )
    )
    state = extractor.extract(context(log))
    assert [f.finding_id for f in state.findings].count("Y") == 1


def test_a_repeated_task_id_within_one_proposal_is_collapsed() -> None:
    log = build_log()
    extractor = LLMExtractor(
        lambda ctx, state: LLMProposal(
            pending_work=[
                {"task_id": "T", "description": "a"},
                {"task_id": "T", "description": "b"},
            ]
        )
    )
    state = extractor.extract(context(log))
    assert [w.task_id for w in state.pending_work].count("T") == 1


def test_the_first_occurrence_wins_within_a_proposal() -> None:
    """Consistent with the base state winning over a proposal: earlier wins."""
    log = build_log()
    extractor = LLMExtractor(
        lambda ctx, state: LLMProposal(
            decisions=[
                {"decision_id": "X", "decision": "first"},
                {"decision_id": "X", "decision": "second"},
            ]
        )
    )
    decision = extractor.extract(context(log)).decision("X")
    assert decision is not None
    assert decision.decision == "first"


def test_distinct_ids_in_one_proposal_are_all_kept() -> None:
    """The de-dup guard must not collapse genuinely different components."""
    log = build_log()
    extractor = LLMExtractor(
        lambda ctx, state: LLMProposal(
            decisions=[
                {"decision_id": "X", "decision": "one"},
                {"decision_id": "Z", "decision": "two"},
            ],
            findings=[
                {"finding_id": "Y", "claim": "a"},
                {"finding_id": "W", "claim": "b"},
            ],
        )
    )
    state = extractor.extract(context(log))
    assert {"X", "Z"} <= {d.decision_id for d in state.decisions}
    assert {"Y", "W"} <= {f.finding_id for f in state.findings}


def test_a_proposal_repeating_a_recorded_id_still_cannot_overwrite_it() -> None:
    """d1 is recorded; repeating it twice must neither overwrite nor duplicate."""
    log = build_log()
    extractor = LLMExtractor(
        lambda ctx, state: LLMProposal(
            decisions=[
                {"decision_id": "d1", "decision": "hijacked"},
                {"decision_id": "d1", "decision": "hijacked again"},
            ]
        )
    )
    state = extractor.extract(context(log))
    assert [d.decision_id for d in state.decisions].count("d1") == 1
    decision = state.decision("d1")
    assert decision is not None
    assert decision.decision == "recorded"


def test_composite_chains_without_double_applying_events() -> None:
    log = build_log()
    composite = CompositeExtractor(
        [
            DeterministicExtractor(),
            LLMExtractor(
                lambda ctx, state: LLMProposal(
                    pending_work=[{"task_id": "t_llm", "description": "verify"}]
                )
            ),
        ]
    )
    state = composite.extract(context(log))

    assert state.progress.completed == 1  # not 2, events folded exactly once
    assert {w.task_id for w in state.pending_work} == {"t_llm"}
    assert state.decision("d1") is not None


def test_a_string_evidence_field_is_wrapped_not_split_into_characters() -> None:
    """LLM proposals are stringly by nature; a bare-string evidence field is
    the most common shape drift. Iterating it split the citation into
    per-character ids (['e','v','_','4','2']), corrupting the persisted state
    and blocking resume under strict_unknown via phantom dangling evidence."""
    log = build_log()
    extractor = LLMExtractor(
        lambda ctx, state: LLMProposal(
            decisions=[{"decision_id": "d_llm", "decision": "inferred", "evidence": "ev_42"}],
            findings=[{"finding_id": "f_llm", "claim": "inferred", "evidence": "ev_42"}],
        )
    )
    state = extractor.extract(context(log))

    decision = next(d for d in state.decisions if d.decision_id == "d_llm")
    finding = next(f for f in state.findings if f.finding_id == "f_llm")
    assert decision.evidence == ["ev_42"]
    assert finding.evidence == ["ev_42"]


# --- the seam must not launder derived trust (#2039) ----------------------- #


def _split_at_a_weak_middle_event(
    middle_type: EventType, middle_payload: dict[str, object]
) -> tuple[list[Event], list[Event], SemanticState]:
    """A log whose prefix holds a weak middle event, split around its base.

    The middle event's source weakens trust but what it writes carries no
    provenance the per-fact seed could read (the #1933 middle row): approvals,
    attempt lessons, trajectory reports and model state carry none, and
    non-projecting events leave no state trace at all.
    """
    log = EventLog()
    log.append("run_1", EventType.RUN_STARTED, {"goal": "g", "total": 1})
    log.append("run_1", middle_type, middle_payload, source=Origin.EXTERNAL_AGENT)
    prefix = list(log.events("run_1"))
    base = project("run_1", prefix)
    log.append(
        "run_1",
        EventType.EVIDENCE_ADDED,
        {
            "evidence_id": "ev1",
            "content": "c",
            "derived_origin": Origin.DETERMINISTIC.value,
            # Evidence.added_at defaults to now() at validation, so an unstamped
            # row re-dates itself on every fold and any two folds of the same
            # log would compare unequal for a reason unrelated to trust.
            "added_at": "2026-01-01T00:00:00+00:00",
        },
    )
    return prefix, list(log.events("run_1", after_sequence=base.source_sequence)), base


@pytest.mark.parametrize("middle_type, middle_payload", _MIDDLE_EVENTS)
def test_a_seeded_seam_matches_a_fresh_read_of_the_same_log(
    middle_type: EventType, middle_payload: dict[str, object]
) -> None:
    """The extractor seam must not launder derived trust (#2039).

    The fold clamps every derived fact's origin by a live watermark it weakens
    from *every* consumed event, non-projecting ones included, and a base
    state cannot reconstruct that watermark. Before the seam could carry a
    seed, a caller folding a tail onto a checkpoint base stamped appended
    derived evidence DETERMINISTIC while a fresh read of the same log stamped
    it EXTERNAL_AGENT, breaking the incremental==full invariant on the public
    extraction seam. Seeding from the prefix events reproduces the live fold
    exactly, so the derived evidence folds identical: id, content and the
    clamped provenance stamp.
    """
    prefix, tail, base = _split_at_a_weak_middle_event(middle_type, middle_payload)

    via_seam = DeterministicExtractor().extract(
        ExtractionContext(
            run_id="run_1",
            trajectory=tail,
            base=base,
            weakest_seed=weakest_origin_over(prefix),
        )
    )
    fresh = project("run_1", [*prefix, *tail])

    assert via_seam.evidence == fresh.evidence
    assert fresh.evidence[0].provenance.origin is Origin.EXTERNAL_AGENT
    assert via_seam.evidence[0].provenance.origin is Origin.EXTERNAL_AGENT


def test_callers_without_a_seed_keep_the_per_fact_fallback() -> None:
    """``weakest_seed`` is optional: no seed keeps today's fold (#2033, #2039).

    Not every caller holds the log, so the seed defaults to None and the fold
    falls back to the base's per-fact provenances, which can only approximate
    the watermark a live fold held. That approximation is the documented
    behaviour for log-less callers, pinned here so the default stays put.
    """
    prefix, tail, base = _split_at_a_weak_middle_event(*_MIDDLE_EVENTS[0])

    assert ExtractionContext(run_id="run_1", trajectory=()).weakest_seed is None
    via_seam = DeterministicExtractor().extract(
        ExtractionContext(run_id="run_1", trajectory=tail, base=base)
    )

    assert via_seam.evidence[0].provenance.origin is Origin.DETERMINISTIC


def test_an_llm_extractor_honours_the_seed_its_context_carries() -> None:
    """The enrichment layer folds through its own deterministic base (#2039).

    ``LLMExtractor`` runs its base extractor on the very context it was
    given, so the seed reaches the fold and wrapping the deterministic layer
    cannot launder derived trust.
    """
    prefix, tail, base = _split_at_a_weak_middle_event(*_MIDDLE_EVENTS[0])
    extractor = LLMExtractor(lambda ctx, state: LLMProposal())

    state = extractor.extract(
        ExtractionContext(
            run_id="run_1",
            trajectory=tail,
            base=base,
            weakest_seed=weakest_origin_over(prefix),
        )
    )

    assert state.evidence[0].provenance.origin is Origin.EXTERNAL_AGENT


class _DerivingExtractor:
    """A seam plugin that derives a fact and folds it, honouring the seed.

    The protocol's contract for implementations that fold onto ``context.base``
    is to honour ``context.weakest_seed``: this one derives an evidence event
    from what it saw and folds it through ``project_incremental`` with the
    seed the context carried, exactly as a well-behaved plugin would.
    """

    name = "deriving"

    def extract(self, context: ExtractionContext) -> SemanticState:
        assert context.base is not None
        derived = Event(
            run_id=context.run_id,
            sequence=context.base.source_sequence + 1,
            type=EventType.EVIDENCE_ADDED,
            payload={
                "evidence_id": "ev_derived",
                "content": "c",
                "derived_origin": Origin.DETERMINISTIC.value,
            },
        )
        state, _ = project_incremental(
            context.run_id,
            [derived],
            base=context.base,
            weakest_seed=context.weakest_seed,
        )
        return state


def test_composite_forwards_the_accumulated_watermark_downstream() -> None:
    """Downstream stages see the watermark the chain holds, not the caller's (#2039).

    The first stage weakens the caller's seed by every event it folds, so a
    downstream extractor that folds derived facts must be seeded with the
    accumulated watermark. Forwarding the caller's seed raw, or nothing at
    all, would leave the downstream per-fact fallback laundering the trust
    the first stage had already weakened.
    """
    prefix, tail, base = _split_at_a_weak_middle_event(*_MIDDLE_EVENTS[0])
    composite = CompositeExtractor([DeterministicExtractor(), _DerivingExtractor()])

    state = composite.extract(
        ExtractionContext(
            run_id="run_1",
            trajectory=tail,
            base=base,
            weakest_seed=weakest_origin_over(prefix),
        )
    )

    assert state.evidence[0].provenance.origin is Origin.EXTERNAL_AGENT
    assert state.evidence[-1].provenance.origin is Origin.EXTERNAL_AGENT
    assert state.evidence[-1].evidence_id == "ev_derived"


def test_nothing_folded_leaves_downstream_unseeded() -> None:
    """No events and no base mean no watermark to forward (#2039).

    ``derived_origin`` degrades an empty contributor list to
    ``EXTERNAL_AGENT``; forwarding that would over-weaken a downstream fold
    for no reason, so the composite forwards None instead and the downstream
    per-fact fallback applies unchanged.
    """
    from continuum.state.extractor import _accumulated_weakest

    assert _accumulated_weakest(ExtractionContext(run_id="run_1", trajectory=())) is None


def test_composite_accumulates_watermark_across_intermediate_stages() -> None:
    """Intermediate stages with weaker facts must constrain downstream stages (#2039, #2044).

    When a composite pipeline has three or more stages and an intermediate stage
    introduces facts with a weaker origin (such as an LLMExtractor introducing
    Origin.LLM candidate facts), subsequent stages in the pipeline must receive
    a watermark that incorporates the evolving state rather than remaining at the
    initial seed.
    """
    composite = CompositeExtractor(
        [
            DeterministicExtractor(),
            LLMExtractor(
                llm=lambda ctx, st: LLMProposal(
                    decisions=[{"decision_id": "d_llm", "decision": "proposal"}]
                )
            ),
            _DerivingExtractor(),
        ]
    )

    ctx = ExtractionContext(
        run_id="run_1",
        trajectory=(
            Event(
                run_id="run_1",
                sequence=1,
                type=EventType.RUN_STARTED,
                source=Origin.DETERMINISTIC,
                payload={"goal": "g"},
            ),
        ),
    )
    state = composite.extract(ctx)
    assert any(d.decision_id == "d_llm" for d in state.decisions)
    assert state.evidence[-1].evidence_id == "ev_derived"
    assert state.evidence[-1].provenance.origin is Origin.EXTERNAL_AGENT
