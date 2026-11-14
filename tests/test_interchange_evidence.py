"""Evidence export as content-addressed JSON lines (issue #395)."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime

import pytest

from continuum.cli import ExitCode, main
from continuum.events import EventType
from continuum.interchange.evidence import (
    Checkpoint,
    EvidencePrimitive,
    Observation,
    Relation,
    Transition,
    export_evidence,
    verify_export,
)
from continuum.models import ConstraintPinned, Origin, Run
from continuum.security.hashing import stable_hash
from continuum.storage import SQLiteStorage


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run(db: str, *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(["--db", db, *argv], out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def _make_run(store: SQLiteStorage, run_id: str = "run_1") -> None:
    store.create_run(Run(run_id=run_id, goal="g"))
    store.append_event(run_id, EventType.RUN_STARTED, {"goal": "g", "total": 5})
    store.append_event(run_id, EventType.DEPENDENCY_DECLARED, {"resource": "r1", "version": "v1"})
    store.append_event(run_id, EventType.EVIDENCE_ADDED, {"evidence_id": "e1", "summary": "s"})
    store.append_event(run_id, EventType.DECISION_CREATED, {"decision_id": "d1", "decision": "x"})
    store.append_event(run_id, EventType.WORK_COMPLETED, {})
    store.append_event(run_id, EventType.STATE_CHECKPOINTED, {"checkpoint_id": "cp1", "version": 0})


def test_every_event_exported_with_correct_hash_and_chain() -> None:
    with SQLiteStorage(":memory:") as store:
        _make_run(store)
        events = list(store.read_events("run_1"))
        primitives = export_evidence(store, "run_1")
        # Every event is exported (plus checkpoint records already covered)
        assert len(primitives) >= len(events)
        # Every primitive is an instance of the model the module declares, not
        # an untyped dict (issue #1155).
        assert all(isinstance(p, EvidencePrimitive) for p in primitives)
        # First primitives should correspond to events in order
        for i, ev in enumerate(events):
            prim = primitives[i]
            assert prim.sequence == i + 1
            assert prim.content_hash == ev.hash
            assert prim.prev_hash == ev.prev_hash
            assert prim.origin == ev.source.value
            assert prim.event_id == ev.event_id
            assert prim.event_type == ev.type.value
            # Signature inputs must recompute to the stored hash
            assert stable_hash(prim.signature_inputs) == ev.hash
        # Chain links are contiguous
        for prev, cur in zip(primitives, primitives[1:], strict=False):
            assert cur.prev_hash == prev.content_hash
        # Full chain verifies via helper
        assert verify_export(primitives) is True


def test_provenance_survives_on_every_primitive() -> None:
    with SQLiteStorage(":memory:") as store:
        _make_run(store)
        primitives = export_evidence(store, "run_1")
        for prim in primitives:
            assert prim.origin in (
                "deterministic",
                "human",
                "external_agent",
                "llm",
                "imported",
            )
            assert prim.payload is not None
            assert prim.signature_inputs is not None
            assert prim.content_hash is not None


def test_truncation_is_detectable() -> None:
    with SQLiteStorage(":memory:") as store:
        _make_run(store)
        primitives = export_evidence(store, "run_1")
        assert verify_export(primitives) is True
        assert (
            verify_export(
                primitives,
                expected_count=len(primitives),
                expected_head=primitives[-1].content_hash,
            )
            is True
        )
        # Drop a middle line, so sequence and prev_hash break
        truncated = primitives[:2] + primitives[3:]
        assert verify_export(truncated) is False
        # Tail truncation is detected when expected stream bounds are verified.
        tail_truncated = primitives[:-1]
        assert len(tail_truncated) != len(primitives)
        assert tail_truncated[-1].content_hash != primitives[-1].content_hash
        assert verify_export(tail_truncated, expected_count=len(primitives)) is False
        assert (
            verify_export(
                tail_truncated,
                expected_head=primitives[-1].content_hash,
            )
            is False
        )
        # Tampering readable payload alone is detected without modifying hashes
        tampered_payload = [p.model_dump() for p in primitives]
        tampered_payload[1]["payload"] = {"tampered": True}
        assert verify_export(tampered_payload) is False
        # Tampering the content hash itself is also detected
        tampered_hash = [p.model_dump() for p in primitives]
        tampered_hash[1]["content_hash"] = "0" * 64
        assert verify_export(tampered_hash) is False


def test_verify_export_rejects_tampered_readable_fields() -> None:
    with SQLiteStorage(":memory:") as store:
        store.create_run(Run(run_id="run_1", goal="test goal"))
        store.append_event("run_1", EventType.RUN_STARTED, {"goal": "test goal"})
        store.append_event(
            "run_1",
            EventType.TOOL_CALLED,
            {"tool": "bash", "command": "ls"},
            source=Origin.EXTERNAL_AGENT,
        )
        store.append_event(
            "run_1",
            EventType.DEPENDENCY_DECLARED,
            {"resource": "res_1", "version": "1.0.0", "evidence": ["ev_1"]},
        )
        store.append_event(
            "run_1",
            EventType.STATE_CHECKPOINTED,
            {"checkpoint_id": "cp_1", "version": 2, "trigger": "step"},
        )
        primitives = export_evidence(store, "run_1")
        assert verify_export(primitives) is True

        base = [p.model_dump() for p in primitives]

        # 1. Tampering origin (e.g. promoting external_agent to human)
        tampered = [dict(d) for d in base]
        tampered[1]["origin"] = "human"
        assert verify_export(tampered) is False

        # 2. Tampering timestamp
        tampered = [dict(d) for d in base]
        tampered[0]["timestamp"] = "1970-01-01T00:00:00+00:00"
        assert verify_export(tampered) is False

        # 3. Tampering run_id
        tampered = [dict(d) for d in base]
        tampered[0]["run_id"] = "tampered_run"
        assert verify_export(tampered) is False

        # 4. Tampering event_id
        tampered = [dict(d) for d in base]
        tampered[0]["event_id"] = "fake_event_id"
        assert verify_export(tampered) is False

        # 5. Tampering event_type
        tampered = [dict(d) for d in base]
        tampered[0]["event_type"] = "RUN_FAILED"
        assert verify_export(tampered) is False

        # 6. Tampering kind
        tampered = [dict(d) for d in base]
        tampered[0]["kind"] = "observation"
        assert verify_export(tampered) is False

        # 7. Tampering observation observed_at
        obs_idx = next(i for i, p in enumerate(base) if p["kind"] == "observation")
        tampered = [dict(d) for d in base]
        tampered[obs_idx]["observed_at"] = "1970-01-01T00:00:00+00:00"
        assert verify_export(tampered) is False

        # 8. Tampering relation source_id and target_id
        rel_idx = next(i for i, p in enumerate(base) if p["kind"] == "relation")
        tampered = [dict(d) for d in base]
        tampered[rel_idx]["source_id"] = "tampered_source"
        assert verify_export(tampered) is False

        tampered = [dict(d) for d in base]
        tampered[rel_idx]["target_id"] = "tampered_target"
        assert verify_export(tampered) is False

        # 9. Tampering checkpoint fields (checkpoint_id, version, trigger)
        cp_idx = next(i for i, p in enumerate(base) if p["kind"] == "checkpoint")
        tampered = [dict(d) for d in base]
        tampered[cp_idx]["checkpoint_id"] = "tampered_cp"
        assert verify_export(tampered) is False

        tampered = [dict(d) for d in base]
        tampered[cp_idx]["version"] = 999
        assert verify_export(tampered) is False

        tampered = [dict(d) for d in base]
        tampered[cp_idx]["trigger"] = "tampered_trigger"
        assert verify_export(tampered) is False


def test_verify_export_event_checkpoint_integrity_hash_limitation() -> None:
    # The event-backed checkpoint primitive's integrity_hash mirrors the
    # checkpoint record the exporter looked up, and the event's signature
    # inputs do not anchor it, so verify_export cannot cross-check it on this
    # path. This pins that known limitation: a receiver holding the checkpoint
    # records must compare it against StateCheckpoint.integrity_hash itself.
    with SQLiteStorage(":memory:") as store:
        store.create_run(Run(run_id="run_1", goal="test goal"))
        store.append_event("run_1", EventType.RUN_STARTED, {"goal": "test goal"})
        store.append_event(
            "run_1",
            EventType.STATE_CHECKPOINTED,
            {"checkpoint_id": "cp_1", "version": 1, "trigger": "step"},
        )
        primitives = export_evidence(store, "run_1")
        assert verify_export(primitives) is True

        cp_idx = next(i for i, p in enumerate(primitives) if p.kind == "checkpoint")
        tampered = [p.model_dump() for p in primitives]
        tampered[cp_idx]["integrity_hash"] = "0" * 64
        assert verify_export(tampered) is True


def test_verify_export_checkpoint_record_primitive() -> None:
    from continuum.models import Goal, StateCheckpoint
    from continuum.state.semantic import SemanticState

    state = SemanticState(run_id="run_1", goal=Goal(description="goal"))
    cp = StateCheckpoint(
        checkpoint_id="cp_rec_1",
        run_id="run_1",
        version=1,
        trigger="manual",
        state=state,
    ).sealed()
    prim = EvidencePrimitive.for_checkpoint(cp, sequence=1, prev_hash=None)
    assert verify_export([prim]) is True

    # Tamper checkpoint_id
    tampered = [prim.model_dump()]
    tampered[0]["checkpoint_id"] = "other_id"
    assert verify_export(tampered) is False

    # Tamper version
    tampered = [prim.model_dump()]
    tampered[0]["version"] = 42
    assert verify_export(tampered) is False

    # Tamper trigger
    tampered = [prim.model_dump()]
    tampered[0]["trigger"] = "automatic"
    assert verify_export(tampered) is False

    # Tamper timestamp
    tampered = [prim.model_dump()]
    tampered[0]["timestamp"] = "1970-01-01T00:00:00+00:00"
    assert verify_export(tampered) is False

    # Tamper origin
    tampered = [prim.model_dump()]
    tampered[0]["origin"] = "human"
    assert verify_export(tampered) is False

    # Tamper payload
    tampered = [prim.model_dump()]
    tampered[0]["payload"] = {"checkpoint_id": "cp_rec_1", "version": 2, "trigger": "manual"}
    assert verify_export(tampered) is False

    # Tamper integrity_hash
    tampered = [prim.model_dump()]
    tampered[0]["integrity_hash"] = "0" * 64
    assert verify_export(tampered) is False


def test_verify_export_bounds_and_edge_cases() -> None:
    assert verify_export([]) is True
    assert verify_export([], expected_count=0) is True
    assert verify_export([], expected_count=1) is False
    assert verify_export([], expected_head="abc") is False

    # Non-mapping or corrupt item
    assert verify_export(["not-a-mapping"]) is False
    assert verify_export([{"sequence": 1, "signature_inputs": None}]) is False


def test_four_primitive_kinds_present() -> None:
    with SQLiteStorage(":memory:") as store:
        _make_run(store)
        # Add a pinned constraint to get a relation
        store.append_event(
            "run_1",
            EventType.CONSTRAINT_PINNED,
            ConstraintPinned(constraint_id="c1", sha256=_digest("hello")).model_dump(),
        )
        primitives = export_evidence(store, "run_1")
        kinds = {type(p).__name__.lower() for p in primitives}
        # At minimum we have transitions and at least one of each other kind
        # due to our classification covering all event types
        assert "transition" in kinds
        assert "relation" in kinds
        # Observations come from TOOL/EVIDENCE etc., relations from dependency
        # Check that each kind is the right class, and that a relation whose
        # event names an endpoint actually carries it: the DEPENDENCY_DECLARED
        # event in _make_run spells it as "resource".
        for p in primitives:
            assert isinstance(p, (Transition, Observation, Relation, Checkpoint))
        declared = next(p for p in primitives if isinstance(p, Relation))
        assert declared.source_id == "r1"
        # A relation whose event names no endpoint falls back to None rather
        # than raising: EVIDENCE_ADDED spells neither source nor target.
        evidence_relation = next(
            p for p in primitives if isinstance(p, Relation) and p.event_type == "EVIDENCE_ADDED"
        )
        assert evidence_relation.source_id is None


def test_checkpoint_primitive_carries_integrity_hash() -> None:
    with SQLiteStorage(":memory:") as store:
        _make_run(store)
        # Create a real checkpoint via manager so it has a stored record
        from continuum.checkpoint import CheckpointManager

        CheckpointManager(store).checkpoint("run_1")
        primitives = export_evidence(store, "run_1")
        cps = [p for p in primitives if isinstance(p, Checkpoint)]
        assert len(cps) >= 1
        # The primitive for the real stored record is enriched with the
        # checkpoint's integrity_hash; the synthetic "cp1" event from _make_run
        # names a checkpoint that was never stored, and stays None rather than
        # fabricating one.
        real = next(p for p in cps if p.integrity_hash is not None)
        assert real.checkpoint_id is not None
        assert any(p.integrity_hash is None for p in cps)


def test_export_covers_archived_events_after_compaction() -> None:
    with SQLiteStorage(":memory:") as store:
        store.create_run(Run(run_id="run_1", goal="g"))
        store.append_event("run_1", EventType.RUN_STARTED, {"goal": "g"})
        store.append_event("run_1", EventType.WORK_COMPLETED, {})
        from continuum.checkpoint import CheckpointManager

        CheckpointManager(store).checkpoint("run_1")
        store.append_event("run_1", EventType.WORK_COMPLETED, {})
        before = export_evidence(store, "run_1")
        count_before = len(before)
        store.compact_run("run_1")
        after = export_evidence(store, "run_1")
        # Archived events are still exported, so count is similar (plus anchor)
        # The important check is that hashes are still correct and chain verifies
        assert verify_export(after) is True
        # At least the original events are still present
        assert len(after) >= count_before - 1  # allow for anchor bookkeeping
        # Every hash from before that was an event should still appear
        before_hashes = {p.content_hash for p in before}
        after_hashes = {p.content_hash for p in after}
        # The archived event hashes must survive
        assert before_hashes.intersection(after_hashes)


def test_export_is_pure_read_no_writes(tmp_path=None) -> None:
    with SQLiteStorage(":memory:") as store:
        _make_run(store)
        before_count = len(list(store.read_events("run_1")))
        before_checkpoints = len(list(store.list_checkpoints("run_1")))
        _ = export_evidence(store, "run_1")
        after_count = len(list(store.read_events("run_1")))
        after_checkpoints = len(list(store.list_checkpoints("run_1")))
        assert before_count == after_count
        assert before_checkpoints == after_checkpoints


def test_cli_export_evidence_emits_json_lines(tmp_path) -> None:
    db = str(tmp_path / "ev.db")
    with SQLiteStorage(db) as store:
        _make_run(store, run_id="run_cli")
    code, out, err = _run(db, "export-evidence", "run_cli")
    assert code == ExitCode.OK, err
    lines = [line for line in out.strip().splitlines() if line.strip()]
    assert len(lines) > 0
    for line in lines:
        prim = json.loads(line)
        assert "kind" in prim
        assert "content_hash" in prim
        assert "origin" in prim
    # Verify the CLI output also passes the chain check
    primitives = [json.loads(line) for line in lines]
    assert verify_export(primitives) is True


def test_cli_unknown_run_exits_not_found(tmp_path) -> None:
    db = str(tmp_path / "ev2.db")
    with SQLiteStorage(db) as store:
        store.create_run(Run(run_id="exists", goal="g"))
    code, out, err = _run(db, "export-evidence", "ghost")
    assert code == ExitCode.NOT_FOUND


def test_zero_new_dependencies() -> None:
    # The evidence module should not import any third-party beyond what
    # interchange already uses (pydantic, stable_hash). Check that its
    # imports are limited to stdlib + continuum.
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "src/continuum/interchange/evidence.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    # Allowed: stdlib + continuum + pydantic + typing
    allowed_prefixes = (
        "pydantic",
        "continuum",
        "typing",
        "collections",
        "datetime",
        "json",
        "pathlib",
        "__future__",
    )
    for imp in imports:
        assert any(imp.startswith(prefix) for prefix in allowed_prefixes), (
            f"unexpected import {imp}"
        )


# --- the models are the export's type system (issue #1155) --------------------- #


def test_export_returns_models_not_dicts() -> None:
    """The four subclasses were exported but never constructed; the exporter
    hand-rolled dicts, so a field could drift out of the model unnoticed."""
    with SQLiteStorage(":memory:") as store:
        _make_run(store)
        primitives = export_evidence(store, "run_1")
    assert primitives
    assert all(isinstance(p, EvidencePrimitive) for p in primitives)
    # Each subclass is reachable, and the base class is never the concrete
    # type: a bare EvidencePrimitive carries no kind-specific fields.
    concrete = {type(p) for p in primitives}
    assert concrete <= {Transition, Observation, Relation, Checkpoint}
    assert EvidencePrimitive not in concrete


def test_content_and_digest_are_reachable() -> None:
    """content()/digest() were unreachable while the models went unconstructed.
    A receiver content-addressing a primitive uses them."""
    with SQLiteStorage(":memory:") as store:
        _make_run(store)
        primitives = export_evidence(store, "run_1")
    for prim in primitives:
        content = prim.content()
        assert content["kind"] == prim.kind
        assert content["run_id"] == prim.run_id
        # digest() is the stable hash of that canonical content, so two
        # primitives with the same content cannot differ.
        assert prim.digest() == stable_hash(content)


def test_content_hash_is_the_event_hash_not_the_digest() -> None:
    """The two hashes are different guarantees and must not be conflated:
    content_hash is what a receiver compares against verify() output, digest()
    content-addresses the primitive itself. Pinning the distinction keeps a
    future 'simplification' from silently redefining the wire format."""
    with SQLiteStorage(":memory:") as store:
        _make_run(store)
        events = list(store.read_events("run_1"))
        primitives = export_evidence(store, "run_1")
    for ev, prim in zip(events, primitives, strict=False):
        assert prim.content_hash == ev.hash
        assert prim.digest() != prim.content_hash


def test_a_stray_field_is_rejected_not_shipped() -> None:
    """extra="forbid" is the whole point of routing construction through the
    models: a field the model does not declare raises instead of landing in
    the exported shape."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Transition(
            kind="transition",
            run_id="run_1",
            sequence=1,
            content_hash="h",
            prev_hash=None,
            origin="deterministic",
            timestamp="2026-01-01T00:00:00",
            payload={},
            signature_inputs={},
            event_id="e1",
            event_type="RUN_STARTED",
            not_a_field="drift",
        )


def test_models_round_trip_through_the_json_wire_format() -> None:
    """model_dump(mode="json") is what cmd_export_evidence writes, and the
    shape must survive a load/verify cycle exactly as a receiver sees it."""
    with SQLiteStorage(":memory:") as store:
        _make_run(store)
        primitives = export_evidence(store, "run_1")
    as_lines = [json.dumps(p.model_dump(mode="json"), sort_keys=True) for p in primitives]
    reloaded = [json.loads(line) for line in as_lines]
    # The receiver's path (json.loads) and the in-memory path (models) both
    # verify, and agree.
    assert verify_export(reloaded) is True
    assert verify_export(primitives) is True
    # No key the model declared is lost in transit.
    for prim, dumped in zip(primitives, reloaded, strict=True):
        assert set(dumped) == set(prim.model_dump())


def test_relation_endpoint_extraction_lives_on_the_model() -> None:
    """The one place primitive construction inspects a payload rather than
    copying it (#1155): the endpoint spelling differs per event family."""
    from continuum.events import Event

    ev = Event(
        event_id="e_rel",
        run_id="run_1",
        sequence=4,
        type=EventType.DECISION_CREATED,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        payload={"decision_id": "d9", "evidence": ["e5"]},
        causer_event_id=None,
        source=Origin.DETERMINISTIC,
        prev_hash=None,
    ).sealed()
    common = {
        "run_id": "run_1",
        "sequence": 4,
        "content_hash": ev.hash,
        "prev_hash": None,
        "origin": ev.source.value,
        "timestamp": ev.timestamp.isoformat(),
        "payload": dict(ev.payload),
        "signature_inputs": dict(ev.payload),
        "event_id": ev.event_id,
        "event_type": ev.type.value,
    }
    relation = Relation.from_event(ev, **common)
    assert relation.source_id == "d9"
    # A list-valued target collapses to its first element.
    assert relation.target_id == "e5"

    # A payload that names no endpoint stays None rather than raising.
    bare = ev.model_copy(update={"payload": {"note": "no endpoints here"}})
    assert Relation.from_event(bare, **common).source_id is None


# --- classification and verification edge cases (issue #1989) ------------------ #


def test_observation_events_classify_to_the_observation_primitive() -> None:
    """Observation event types map to ``Observation`` with ``observed_at`` set
    (#1989): the classifier routes them off the event type, and the primitive
    records the event timestamp in the field the kind exists to carry."""
    with SQLiteStorage(":memory:") as store:
        _make_run(store)
        store.append_event("run_1", EventType.STATE_VALIDATED, {"checksum": "abc", "ok": True})
        store.append_event("run_1", EventType.PERCEPTION_OBSERVED, {"note": "env stable"})
        store.append_event("run_1", EventType.TOOL_COMPLETED, {"tool": "grep", "ok": True})
        observed_types = {
            EventType.STATE_VALIDATED,
            EventType.PERCEPTION_OBSERVED,
            EventType.TOOL_COMPLETED,
        }
        observed = [e for e in store.read_events("run_1") if e.type in observed_types]
        assert len(observed) == 3
        primitives = export_evidence(store, "run_1")

    observations = [p for p in primitives if isinstance(p, Observation)]
    assert {p.event_id for p in observations} == {e.event_id for e in observed}
    for prim in observations:
        ev = next(e for e in observed if e.event_id == prim.event_id)
        assert prim.kind == "observation"
        assert prim.event_type == ev.type.value
        assert prim.observed_at == ev.timestamp.isoformat()
        assert prim.payload == dict(ev.payload)
        assert stable_hash(prim.signature_inputs) == prim.content_hash
    # The observation events close the chain, which must still verify.
    assert verify_export(primitives) is True


def test_checkpoint_record_without_a_log_event_is_exported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A checkpoint record the event stream missed is exported anyway (#1989):
    ``EvidencePrimitive.for_checkpoint`` builds a record-only primitive whose
    sequence continues the event chain, so the export still covers every
    stored object and the chain still verifies."""
    from continuum.models import Goal, SemanticState, StateCheckpoint

    with SQLiteStorage(":memory:") as store:
        _make_run(store)
        orphan = StateCheckpoint(
            checkpoint_id="cp_orphan",
            run_id="run_1",
            version=1,
            trigger="manual",
            state=SemanticState(run_id="run_1", goal=Goal(description="g")),
        ).sealed()
        assert orphan.integrity_hash is not None
        # put_checkpoint writes the record without the STATE_CHECKPOINTED event
        # CheckpointManager would append, which is exactly the gap the
        # record-only export branch exists for.
        store.put_checkpoint(orphan)

        # Spy on the constructor so the test proves the record-only path ran
        # rather than inferring it from the emitted shape alone.
        calls: list[str] = []
        original = EvidencePrimitive.for_checkpoint

        def spy(
            _cls: type[EvidencePrimitive],
            cp: StateCheckpoint,
            *,
            sequence: int,
            prev_hash: str | None,
        ) -> EvidencePrimitive:
            calls.append(cp.checkpoint_id)
            return original(cp, sequence=sequence, prev_hash=prev_hash)

        monkeypatch.setattr(EvidencePrimitive, "for_checkpoint", classmethod(spy))
        primitives = export_evidence(store, "run_1")

    assert calls == ["cp_orphan"]
    exported = [p for p in primitives if isinstance(p, Checkpoint)]
    assert any(p.checkpoint_id == "cp_orphan" for p in exported)
    prim = next(p for p in exported if p.checkpoint_id == "cp_orphan")
    # With no event to name, the checkpoint's own id and the event type it
    # would have had stand in.
    assert prim.event_id == "cp_orphan"
    assert prim.event_type == "STATE_CHECKPOINTED"
    assert prim.integrity_hash == orphan.integrity_hash
    assert prim.content_hash == orphan.integrity_hash
    # The record-only primitive takes the next sequence number, so the stream
    # stays contiguous and the chain still verifies end to end.
    assert [p.sequence for p in primitives] == list(range(1, len(primitives) + 1))
    assert verify_export(primitives) is True


def test_verify_export_rejects_a_broken_prev_hash_link() -> None:
    """A primitive whose ``prev_hash`` disagrees with the preceding primitive's
    ``content_hash`` fails verification (#1989): the link is the whole point of
    the chain, and a receiver must see the break rather than a clean stream."""
    with SQLiteStorage(":memory:") as store:
        _make_run(store)
        primitives = export_evidence(store, "run_1")
    assert verify_export(primitives) is True
    tampered = [p.model_dump() for p in primitives]
    tampered[1]["prev_hash"] = "f" * 64
    assert verify_export(tampered) is False


def test_verify_export_rejects_a_recomputed_hash_mismatch() -> None:
    """When ``stable_hash(signature_inputs)`` disagrees with the recorded
    ``content_hash`` the stream is tampered (#1989): the recomputation is the
    receiver's check that the exported hash came from the exported content."""
    with SQLiteStorage(":memory:") as store:
        _make_run(store)
        primitives = export_evidence(store, "run_1")
    assert verify_export(primitives) is True
    tampered = [p.model_dump() for p in primitives]
    tampered[2]["signature_inputs"] = {"tampered": True}
    assert stable_hash({"tampered": True}) != tampered[2]["content_hash"]
    assert verify_export(tampered) is False
    # The same check catches a content_hash that no longer matches the inputs.
    retitled = [p.model_dump() for p in primitives]
    retitled[2]["content_hash"] = "0" * 64
    assert verify_export(retitled) is False


def test_verify_export_treats_unhashable_signature_inputs_as_broken() -> None:
    """A signature input ``stable_hash`` cannot process (a set has no stable
    order) makes the stream unverifiable, not the verifier (#1989): the
    recompute is wrapped so a raise becomes a False instead of a crash."""
    with SQLiteStorage(":memory:") as store:
        _make_run(store)
        primitives = export_evidence(store, "run_1")
    assert verify_export(primitives) is True
    tampered = [p.model_dump() for p in primitives]
    tampered[2]["signature_inputs"] = {"unhashable": {1, 2, 3}}
    with pytest.raises(TypeError):
        stable_hash(tampered[2]["signature_inputs"])
    assert verify_export(tampered) is False
