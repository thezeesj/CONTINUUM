"""Content-addressed evidence export for external consumers.

The exporter maps a run's durable evidence to four neutral primitives:

* Transitions -- event-appended state movements (every event)
* Observations -- environment validations and diffs
* Relations -- dependency edges between components
* State Checkpoints -- checkpoint records

Each primitive is content-addressed (stable_hash of its content), carries
sequence, origin/provenance, and the signature inputs needed to re-verify
against the hash-chained log. The export is pure read, zero new
dependencies, and the receiver can detect truncation or tampering by
re-computing the chain exactly as ``verify()`` does.

Design constraints
------------------
* Native format stays authoritative; this is a read-only view.
* Hashes are the same as the storage layer's ``Event.hash`` and
  ``StateCheckpoint.integrity_hash``, so a receiver compares directly
  to ``verify()`` output.
* No third-party dependencies beyond pydantic and the existing hashing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from continuum.events import Event
from continuum.models import Origin, StateCheckpoint
from continuum.security.hashing import stable_hash, to_json
from continuum.storage.base import Storage

__all__ = [
    "EvidencePrimitive",
    "Transition",
    "Observation",
    "Relation",
    "Checkpoint",
    "export_evidence",
    "verify_export",
]

# ---------------------------------------------------------------------------
# Primitive models
# ---------------------------------------------------------------------------

Kind = Literal["transition", "observation", "relation", "checkpoint"]


class EvidencePrimitive(BaseModel):
    """Base for all exported primitives."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Kind
    run_id: str
    sequence: int
    content_hash: str
    prev_hash: str | None
    origin: str
    timestamp: str
    payload: dict[str, Any]
    signature_inputs: dict[str, Any]

    def content(self) -> dict[str, Any]:
        """Return the canonical fields represented by this primitive."""
        return {
            "kind": self.kind,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "prev_hash": self.prev_hash,
            "origin": self.origin,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "signature_inputs": self.signature_inputs,
        }

    def digest(self) -> str:
        """Return the stable hash of this primitive's canonical content."""
        return stable_hash(self.content())

    @classmethod
    def for_event(
        cls,
        ev: Event,
        *,
        sequence: int,
        prev_hash: str | None,
        checkpoints: Mapping[str, StateCheckpoint] | None = None,
    ) -> EvidencePrimitive:
        """Build the primitive ``ev`` classifies to (#1155).

        The four subclasses were exported but never constructed, so the
        exporter hand-rolled ``dict[str, Any]`` values and every subclass
        field could be added, renamed or dropped with nothing breaking --
        the dict keys were spelled separately in the function body. Routing
        construction through the models makes ``extra="forbid"`` a live
        check: a field that drifts now raises at export instead of silently
        shipping a different shape.
        """
        common = _common_fields(ev, sequence=sequence, prev_hash=prev_hash)
        known: Mapping[str, StateCheckpoint] = checkpoints or {}
        kind = _classify(ev)
        if kind == "observation":
            return Observation(observed_at=ev.timestamp.isoformat(), **common)
        if kind == "relation":
            return Relation.from_event(ev, **common)
        if kind == "checkpoint":
            raw_cid = ev.payload.get("checkpoint_id")
            record = known.get(raw_cid) if isinstance(raw_cid, str) else None
            cid, version, trigger = _checkpoint_meta(ev.payload, ev.event_id)
            return Checkpoint(
                checkpoint_id=cid,
                version=version,
                trigger=trigger,
                integrity_hash=record.integrity_hash if record is not None else None,
                **common,
            )
        return Transition(**common)

    @classmethod
    def for_checkpoint(
        cls, cp: StateCheckpoint, *, sequence: int, prev_hash: str | None
    ) -> EvidencePrimitive:
        """Build the primitive for a checkpoint record with no log event.

        Current storage writes a ``STATE_CHECKPOINTED`` event for every
        checkpoint, so this path covers a record the event stream missed
        rather than a whole class of object. With no event to name, the
        checkpoint's own id stands in as ``event_id`` and the kind names the
        event type it would have had.
        """
        return Checkpoint(
            run_id=cp.run_id,
            sequence=sequence,
            content_hash=cp.integrity_hash,
            prev_hash=prev_hash,
            origin=Origin.DETERMINISTIC.value,
            timestamp=cp.created_at.isoformat(),
            payload={
                "checkpoint_id": cp.checkpoint_id,
                "version": cp.version,
                "trigger": cp.trigger,
            },
            signature_inputs=json.loads(to_json(cp.content())),
            checkpoint_id=cp.checkpoint_id,
            version=cp.version,
            trigger=cp.trigger,
            integrity_hash=cp.integrity_hash,
            event_id=cp.checkpoint_id,
            event_type=_CHECKPOINT_EVENT_TYPE,
        )


class Transition(EvidencePrimitive):
    """Represent an event-backed state transition."""

    kind: Literal["transition"] = "transition"
    event_id: str
    event_type: str


class Observation(EvidencePrimitive):
    """Represent an environment or tool observation."""

    kind: Literal["observation"] = "observation"
    event_id: str
    event_type: str
    observed_at: str


class Relation(EvidencePrimitive):
    """Represent a dependency or derivation relation between items."""

    kind: Literal["relation"] = "relation"
    event_id: str
    event_type: str
    source_id: str | None = None
    target_id: str | None = None

    @classmethod
    def from_event(cls, ev: Event, **common: Any) -> Relation:
        """Build a relation primitive, guessing its endpoints from the payload.

        The guess lives in :func:`_relation_endpoints` so the readable fields
        and the values :func:`verify_export` cross-checks them against stay
        in one place (#1155).
        """
        source_id, target_id = _relation_endpoints(ev.payload)
        return cls(source_id=source_id, target_id=target_id, **common)


class Checkpoint(EvidencePrimitive):
    """Represent a persisted semantic-state checkpoint."""

    kind: Literal["checkpoint"] = "checkpoint"
    checkpoint_id: str
    version: int
    trigger: str
    integrity_hash: str | None
    # The exporter names the log event a checkpoint primitive came from, and
    # for a record-only checkpoint (no matching event) the checkpoint's own
    # id stands in. Declared now that the model is actually constructed
    # (#1155): the dict it replaces emitted both keys, and ``extra="forbid"``
    # would have rejected them had the model been built at the time.
    event_id: str
    event_type: str


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

# Observations are environment validations and diff results.
_OBSERVATION_TYPES = frozenset(
    {
        "STATE_VALIDATED",
        "ENVIRONMENT_CHANGED",
        "PERCEPTION_OBSERVED",
        "BRANCH_RESOLVED",
        "TOOL_COMPLETED",
        "TOOL_FAILED",
        "TOOL_CALLED",
        "REASONING_SUMMARY",
    }
)

# Relations are dependency edges.
_RELATION_TYPES = frozenset(
    {
        "DEPENDENCY_DECLARED",
        "FINDING_ADDED",
        "FINDING_INVALIDATED",
        "DECISION_CREATED",
        "DECISION_INVALIDATED",
        "EVIDENCE_ADDED",
        "WORK_ADDED",
        "CONSTRAINT_PINNED",
        "CONSTRAINT_RETRACTED",
    }
)

# Checkpoints are explicit checkpoint records; we also emit a checkpoint
# primitive for each STATE_CHECKPOINTED event, but the canonical checkpoint
# record comes from storage.list_checkpoints.
_CHECKPOINT_TYPES = frozenset({"STATE_CHECKPOINTED"})

#: The event type a checkpoint record would have been logged as. Named so the
#: event-backed and record-backed branches cannot spell it differently (#1155).
_CHECKPOINT_EVENT_TYPE = "STATE_CHECKPOINTED"


def _classify(event: Event) -> Kind:
    t = event.type.value
    if t in _CHECKPOINT_TYPES:
        return "checkpoint"
    if t in _OBSERVATION_TYPES:
        return "observation"
    if t in _RELATION_TYPES:
        return "relation"
    return "transition"


def _relation_endpoints(payload: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Guess a dependency edge's endpoints from the event payload.

    The endpoints are spelled differently per event family (``resource`` for
    a declaration, ``decision_id`` / ``finding_id`` for those), so this is the
    one place primitive construction inspects the payload instead of copying
    it. Sharing one helper between :meth:`Relation.from_event` and
    :func:`verify_export` keeps the readable fields and the values they are
    cross-checked against from drifting apart (#1155).
    """
    source = payload.get("resource") or payload.get("decision_id") or payload.get("finding_id")
    target = payload.get("evidence") or payload.get("depends_on")
    if isinstance(target, list) and target:
        target = target[0]
    return (str(source) if source else None, str(target) if target else None)


def _checkpoint_meta(payload: Mapping[str, Any], event_id: Any) -> tuple[str, int, str]:
    """Read a checkpoint primitive's (checkpoint_id, version, trigger) values.

    The event payload names the checkpoint it logged and falls back to the
    event's own id, and the version and trigger defaults match the record's
    defaults. Sharing one helper between :meth:`EvidencePrimitive.for_event`
    and :func:`verify_export` keeps construction and cross-checking in
    agreement (#1155).
    """
    cid = payload.get("checkpoint_id")
    if not isinstance(cid, str):
        cid = event_id
    return cid, payload.get("version", 0), payload.get("trigger", "unknown")


def _common_fields(ev: Event, *, sequence: int, prev_hash: str | None) -> dict[str, Any]:
    """Fields every event-backed primitive shares.

    ``content_hash`` is the event's own hash, not ``digest()``: it is what a
    receiver compares directly against ``verify()`` output, and the chain links
    are built from it. ``digest()`` content-addresses the primitive itself and
    is a separate guarantee.

    ``signature_inputs`` is the event's content dict, canonical-JSON round
    tripped so the timestamp lands as ISO-with-T: that matches
    ``stable_hash``'s canonicalisation and survives a JSON dump/load without
    ``default=str`` re-encoding (which would use a space separator and break
    the hash). This is what makes the exported hash identical to ``verify()``
    and lets a receiver recompute ``stable_hash(signature_inputs)`` after
    loading the JSON lines.
    """
    return {
        "run_id": ev.run_id,
        "sequence": sequence,
        "content_hash": ev.hash,
        "prev_hash": prev_hash,
        "origin": ev.source.value,
        "timestamp": ev.timestamp.isoformat(),
        "payload": dict(ev.payload),
        "signature_inputs": json.loads(to_json(ev.content())),
        "event_id": ev.event_id,
        "event_type": ev.type.value,
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_evidence(storage: Storage, run_id: str) -> list[EvidencePrimitive]:
    """Export a run's evidence as typed, JSON-serialisable primitives.

    Reads ``events`` plus ``archived_events`` (so compacted runs are fully
    covered) and ``checkpoints``. Each primitive carries ``content_hash``
    (the event's own hash, so it matches ``verify()`` directly), ``prev_hash``
    (previous primitive's hash for chain verification), ``origin`` and
    signature inputs.

    The result is the module's own models, not untyped dicts (#1155): the
    four subclasses were declared and exported but never constructed, so a
    field could drift out of the model and nothing would notice. Serialize
    with ``model_dump()``::

        for prim in export_evidence(storage, run_id):
            print(json.dumps(prim.model_dump(mode="json"), sort_keys=True))

    Truncation or tampering is detectable by the receiver::

        prev = None
        for i, prim in enumerate(exported, start=1):
            assert prim.sequence == i
            assert prim.prev_hash == prev
            assert prim.content_hash == stable_hash(prim.signature_inputs)
            prev = prim.content_hash
        assert len(exported) == expected_count
        assert exported[-1].content_hash == expected_head

    Or via :func:`verify_export`::

        ok = verify_export(
            exported,
            expected_count=expected_count,
            expected_head=expected_head,
        )

    Pure read, no writes, zero new dependencies.
    """
    # Validate run exists early, fail closed.
    storage.get_run(run_id)

    # Gather full history: archived + live, sorted by sequence.
    archived = list(storage.read_archived_events(run_id))
    live = list(storage.read_events(run_id))
    events = sorted([*archived, *live], key=lambda e: e.sequence)

    # Map checkpoint_id -> checkpoint for enrichment of event-backed
    # checkpoint primitives with the stored integrity_hash.
    checkpoints = {c.checkpoint_id: c for c in storage.list_checkpoints(run_id)}

    primitives: list[EvidencePrimitive] = []
    prev_hash: str | None = None
    seq = 0

    for ev in events:
        seq += 1
        primitives.append(
            EvidencePrimitive.for_event(
                ev, sequence=seq, prev_hash=prev_hash, checkpoints=checkpoints
            )
        )
        prev_hash = ev.hash

    # In current storage, each checkpoint is also an event of type
    # STATE_CHECKPOINTED, so the loop above has already covered it. To avoid
    # duplication, only records with no matching event are emitted here. This
    # keeps the export covering every event exactly once for the truncation
    # check, while still surfacing the checkpoint's integrity_hash for
    # external verification.
    emitted_cids = {p.checkpoint_id for p in primitives if isinstance(p, Checkpoint)}
    for cp in checkpoints.values():
        if cp.checkpoint_id in emitted_cids:
            continue
        seq += 1
        primitives.append(EvidencePrimitive.for_checkpoint(cp, sequence=seq, prev_hash=prev_hash))
        prev_hash = cp.integrity_hash

    return primitives


def _same_timestamp(t1: Any, t2: Any) -> bool:
    """Compare two timestamps for semantic equality across ISO variations."""
    if t1 == t2:
        return True
    if isinstance(t1, str) and isinstance(t2, str):
        try:
            return datetime.fromisoformat(t1) == datetime.fromisoformat(t2)
        except Exception:
            return False
    return False


def verify_export(
    primitives: Sequence[EvidencePrimitive | Mapping[str, Any]],
    *,
    expected_count: int | None = None,
    expected_head: str | None = None,
) -> bool:
    """Verify an exported stream exactly as a receiver would.

    Accepts the models or the mappings a JSON-lines consumer produces after
    ``json.loads``, so a receiver reading a file and a caller holding the
    export in memory run the same check.

    Returns True if the chain is intact, sequences are contiguous starting at
    1, prev_hash links are correct, each content_hash matches the recomputed
    digest of signature_inputs, readable duplicated fields match their verified
    counterparts in signature_inputs, and stream boundaries match expected_count
    and expected_head if provided.

    Known limitation: the event-backed checkpoint primitive's ``integrity_hash``
    mirrors the checkpoint record the exporter looked up, and the event's
    signature inputs do not anchor it, so it is only cross-checked on the
    record-backed path (where it must equal ``content_hash``). A receiver that
    also holds the checkpoint records should compare it against
    ``StateCheckpoint.integrity_hash`` directly.
    """
    if expected_count is not None and len(primitives) != expected_count:
        return False
    if expected_head is not None:
        if not primitives:
            return False
        last = primitives[-1]
        last_hash = (
            last.content_hash
            if isinstance(last, EvidencePrimitive)
            else last.get("content_hash")
            if isinstance(last, Mapping)
            else None
        )
        if last_hash != expected_head:
            return False

    prev: str | None = None
    for i, prim in enumerate(primitives, start=1):
        if isinstance(prim, EvidencePrimitive):
            p = prim.model_dump()
        elif isinstance(prim, Mapping):
            p = dict(prim)
        else:
            return False

        if p.get("sequence") != i:
            return False
        if p.get("prev_hash") != prev:
            return False

        sig = p.get("signature_inputs")
        if not isinstance(sig, Mapping):
            return False

        try:
            recomputed = stable_hash(sig)
        except Exception:
            return False
        if p.get("content_hash") != recomputed:
            return False

        # Cross-check readable fields against verified signature_inputs.
        if "event_id" in sig:
            # Event-backed primitive
            if p.get("run_id") != sig.get("run_id"):
                return False
            if p.get("event_id") != sig.get("event_id"):
                return False
            if p.get("event_type") != sig.get("type"):
                return False
            if p.get("origin") != sig.get("source"):
                return False
            if not _same_timestamp(p.get("timestamp"), sig.get("timestamp")):
                return False
            if p.get("payload") != sig.get("payload"):
                return False

            ev_type = sig.get("type")
            if ev_type in _CHECKPOINT_TYPES:
                expected_kind = "checkpoint"
            elif ev_type in _OBSERVATION_TYPES:
                expected_kind = "observation"
            elif ev_type in _RELATION_TYPES:
                expected_kind = "relation"
            else:
                expected_kind = "transition"

            if p.get("kind") != expected_kind:
                return False

            if expected_kind == "observation":
                if not _same_timestamp(p.get("observed_at"), sig.get("timestamp")):
                    return False
            elif expected_kind == "relation":
                ev_payload = sig.get("payload")
                if not isinstance(ev_payload, Mapping):
                    return False
                expected_source, expected_target = _relation_endpoints(ev_payload)
                if p.get("source_id") != expected_source or p.get("target_id") != expected_target:
                    return False
            elif expected_kind == "checkpoint":
                ev_payload = sig.get("payload")
                if not isinstance(ev_payload, Mapping):
                    return False
                expected_cid, expected_version, expected_trigger = _checkpoint_meta(
                    ev_payload, sig.get("event_id")
                )
                if p.get("checkpoint_id") != expected_cid:
                    return False
                if p.get("version") != expected_version:
                    return False
                if p.get("trigger") != expected_trigger:
                    return False
        else:
            # Standalone checkpoint record primitive (built via for_checkpoint)
            if p.get("kind") != "checkpoint":
                return False
            if p.get("run_id") != sig.get("run_id"):
                return False
            if p.get("checkpoint_id") != sig.get("checkpoint_id"):
                return False
            if p.get("version") != sig.get("version"):
                return False
            if p.get("trigger") != sig.get("trigger"):
                return False
            if not _same_timestamp(p.get("timestamp"), sig.get("created_at")):
                return False
            if p.get("event_id") != sig.get("checkpoint_id"):
                return False
            if p.get("event_type") != _CHECKPOINT_EVENT_TYPE:
                return False
            if p.get("origin") != Origin.DETERMINISTIC.value:
                return False
            expected_payload = {
                "checkpoint_id": sig.get("checkpoint_id"),
                "version": sig.get("version"),
                "trigger": sig.get("trigger"),
            }
            if p.get("payload") != expected_payload:
                return False
            if p.get("integrity_hash") != p.get("content_hash"):
                return False

        prev = p.get("content_hash")
    return True
