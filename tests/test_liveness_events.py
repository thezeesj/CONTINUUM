"""LIVENESS events, WAIT mapping and continuum watch (issue #562)."""

from __future__ import annotations

import http.server
import io
import json
import sqlite3
import threading
import unittest.mock
from datetime import UTC, datetime, timedelta
from pathlib import Path

from continuum.cli.main import main
from continuum.events import EventType
from continuum.models import Run
from continuum.recovery import RecoveryEngine
from continuum.storage import SQLiteStorage


class _ArchiveDeadStorage(SQLiteStorage):
    """A store whose archived prefix cannot be read back.

    Every method except the archive read still works, so this isolates exactly
    the failure the fallback exists for: the live tail is served, the archive
    is not. ``served_live_tail`` records that the fallback path actually ran,
    so the test cannot pass vacuously.
    """

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.served_live_tail = False

    def read_archived_events(self, run_id: str):  # type: ignore[override]
        raise sqlite3.DatabaseError("archive page is unreadable")

    def read_all_events(self, run_id: str):  # type: ignore[override]
        raise sqlite3.DatabaseError("archive page is unreadable")

    def read_events(self, run_id: str):  # type: ignore[override]
        self.served_live_tail = True
        return super().read_events(run_id)


def test_liveness_events_are_hash_chained(tmp_path: Path) -> None:
    db = str(tmp_path / "liveness_chain.db")
    with SQLiteStorage(db) as store:
        run_id = "run_liveness_chain"
        store.create_run_started(Run(run_id=run_id, goal="test chain"))
        e1 = store.append_event(
            run_id,
            EventType.LIVENESS_SILENCE_DETECTED,
            {"silence_seconds": 4000, "threshold_seconds": 3600, "phase": "otherwise"},
        )
        e2 = store.append_event(run_id, EventType.LIVENESS_RECOVERED, {"silence_seconds": 10})
        assert e1.hash is not None
        assert e2.prev_hash == e1.hash
        report = store.verify_events(run_id)
        assert report.ok is True
        assert report.trusted_through[run_id] == 3


def test_engine_maps_breach_to_wait(tmp_path: Path) -> None:
    db = str(tmp_path / "wait_map.db")
    with SQLiteStorage(db) as store:
        run_id = "run_wait_map"
        store.create_run_started(Run(run_id=run_id, goal="wait test"))
        from continuum.events import Event

        old_ts = datetime.now(UTC) - timedelta(seconds=7200)
        last_seq = store.last_sequence(run_id)
        ev = Event(
            run_id=run_id,
            sequence=last_seq + 1,
            type=EventType.TASK_UPDATED,
            timestamp=old_ts,
            payload={"completed": 1},
            prev_hash=store.read_events(run_id)[-1].hash,
        ).sealed()
        store.append_sealed(ev)
        engine = RecoveryEngine(store)
        decision = engine.assess(run_id)
        assert decision.mode.value == "wait"
        assert decision.mode.value != "rollback"
        assert decision.contract.liveness is not None
        assert decision.contract.liveness["breached"] is True
        assert decision.contract.liveness["breaches"] >= 0
        # A breach is advisory in the sense of #302 (never a rollback), but it
        # is not inert: WAIT reaches `continuum resume` as exit 20, which is the
        # link the health.py docstring and the liveness-watch guide describe.
        from continuum.cli.exitcodes import exit_code_for

        assert exit_code_for(decision.mode) == 20


def test_breach_rationale_states_the_reason_once(tmp_path: Path) -> None:
    # Regression for #1042: the liveness-breach WAIT block was pasted twice
    # into _decide, so a breached run reported the identical sentence in the
    # rationale twice, and the sealed contract's reason read
    # "...; ..." with the duplicate. One breach, one proposal, one sentence.
    db = str(tmp_path / "rationale_once.db")
    with SQLiteStorage(db) as store:
        run_id = "run_rationale_once"
        store.create_run_started(Run(run_id=run_id, goal="rationale once"))
        from continuum.events import Event

        old_ts = datetime.now(UTC) - timedelta(seconds=7200)
        last_seq = store.last_sequence(run_id)
        ev = Event(
            run_id=run_id,
            sequence=last_seq + 1,
            type=EventType.TASK_UPDATED,
            timestamp=old_ts,
            payload={"completed": 1},
            prev_hash=store.read_events(run_id)[-1].hash,
        ).sealed()
        store.append_sealed(ev)
        decision = RecoveryEngine(store).assess(run_id)
        assert decision.mode.value == "wait"
        assert len(decision.rationale) == len(set(decision.rationale)), decision.rationale
        assert len(decision.rationale) == 1, decision.rationale
        assert decision.rationale[0].startswith("liveness breach:")
        # The sealed reason joins the rationale with "; " and must not
        # repeat the sentence either.
        assert decision.contract.reason.count("liveness breach:") == 1


def test_watch_appends_detected_and_recovered(tmp_path: Path) -> None:
    db = str(tmp_path / "watch_events.db")
    with SQLiteStorage(db) as store:
        run_id = "run_watch_events"
        store.create_run_started(Run(run_id=run_id, goal="watch test"))
        from continuum.events import Event

        old_ts = datetime.now(UTC) - timedelta(seconds=7200)
        last_seq = store.last_sequence(run_id)
        ev = Event(
            run_id=run_id,
            sequence=last_seq + 1,
            type=EventType.TASK_UPDATED,
            timestamp=old_ts,
            payload={"completed": 1},
            prev_hash=store.read_events(run_id)[-1].hash,
        ).sealed()
        store.append_sealed(ev)

    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["--db", db, "watch", run_id, "--max-silence", "3600", "--on-breach", "exit"],
        out=out,
        err=err,
    )
    assert code == 20
    with SQLiteStorage(db) as store:
        events = store.read_events(run_id)
        types = [e.type for e in events]
        assert EventType.LIVENESS_SILENCE_DETECTED in types
        store.append_event(run_id, EventType.TASK_UPDATED, {"completed": 2})
        out2, err2 = io.StringIO(), io.StringIO()
        code2 = main(
            ["--db", db, "watch", run_id, "--max-silence", "3600", "--on-breach", "exit"],
            out=out2,
            err=err2,
        )
        assert code2 == 0
        events2 = store.read_events(run_id)
        assert events2[-1].type == EventType.LIVENESS_RECOVERED


def test_watch_webhook_fail_open(tmp_path: Path) -> None:
    db = str(tmp_path / "watch_webhook.db")
    with SQLiteStorage(db) as store:
        run_id = "run_watch_webhook"
        store.create_run_started(Run(run_id=run_id, goal="webhook test"))

    out, err = io.StringIO(), io.StringIO()
    code = main(
        [
            "--db",
            db,
            "watch",
            run_id,
            "--max-silence",
            "1",
            "--on-breach",
            "webhook",
            "--webhook-url",
            "http://127.0.0.1:1/nonexistent",
        ],
        out=out,
        err=err,
    )
    assert code in (0, 20)
    assert "warning: webhook delivery failed" in err.getvalue() or code in (0, 20)


def test_watch_webhook_delivers_on_breach(tmp_path: Path) -> None:
    received: list[bytes] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            received.append(body)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format, *args):
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        db = str(tmp_path / "watch_webhook_deliver.db")
        with SQLiteStorage(db) as store:
            run_id = "run_watch_deliver"
            store.create_run_started(Run(run_id=run_id, goal="deliver test"))
            from continuum.events import Event

            old_ts = datetime.now(UTC) - timedelta(seconds=7200)
            last_seq = store.last_sequence(run_id)
            ev = Event(
                run_id=run_id,
                sequence=last_seq + 1,
                type=EventType.TASK_UPDATED,
                timestamp=old_ts,
                payload={"completed": 1},
                prev_hash=store.read_events(run_id)[-1].hash,
            ).sealed()
            store.append_sealed(ev)
        out, err = io.StringIO(), io.StringIO()
        url = f"http://127.0.0.1:{port}/hook"
        code = main(
            [
                "--db",
                db,
                "watch",
                run_id,
                "--max-silence",
                "3600",
                "--on-breach",
                "webhook",
                "--webhook-url",
                url,
            ],
            out=out,
            err=err,
        )
        assert code == 20
        import time

        time.sleep(0.2)
        assert len(received) == 1
        payload = json.loads(received[0].decode("utf-8"))
        assert payload["breached"] is True
        assert payload["run_id"] == run_id
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_contract_liveness_section(tmp_path: Path) -> None:
    db = str(tmp_path / "contract_liveness.db")
    with SQLiteStorage(db) as store:
        run_id = "run_contract_liveness"
        store.create_run_started(Run(run_id=run_id, goal="contract liveness"))
        engine = RecoveryEngine(store)
        decision = engine.assess(run_id)
        assert decision.contract.liveness is not None
        assert "last_append_age" in decision.contract.liveness
        assert "breaches" in decision.contract.liveness
        assert "breached" in decision.contract.liveness


def test_max_silence_override_wins_without_open_claim(tmp_path: Path) -> None:
    """An explicit --max-silence beats the default otherwise scope (#670)."""
    db = str(tmp_path / "watch_override.db")
    with SQLiteStorage(db) as store:
        run_id = "run_watch_override"
        store.create_run_started(Run(run_id=run_id, goal="watch test"))
        from continuum.events import Event

        old_ts = datetime.now(UTC) - timedelta(seconds=30)
        last_seq = store.last_sequence(run_id)
        ev = Event(
            run_id=run_id,
            sequence=last_seq + 1,
            type=EventType.TASK_UPDATED,
            timestamp=old_ts,
            payload={"completed": 1},
            prev_hash=store.read_events(run_id)[-1].hash,
        ).sealed()
        store.append_sealed(ev)

    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["--db", db, "watch", run_id, "--max-silence", "5", "--on-breach", "exit"],
        out=out,
        err=err,
    )
    assert code == 20, out.getvalue()
    with SQLiteStorage(db) as store:
        types = [e.type for e in store.read_events(run_id)]
        assert EventType.LIVENESS_SILENCE_DETECTED in types


def _append_timed(
    store: SQLiteStorage, run_id: str, etype: EventType, payload: dict, ts: datetime
) -> None:
    """Append a hash-chained event with an explicit timestamp (for backdating)."""
    from continuum.events import Event

    prev_hash = store.read_events(run_id)[-1].hash
    ev = Event(
        run_id=run_id,
        sequence=store.last_sequence(run_id) + 1,
        type=etype,
        timestamp=ts,
        payload=payload,
        prev_hash=prev_hash,
    ).sealed()
    store.append_sealed(ev)


def test_watch_does_not_duplicate_detected_after_compaction(tmp_path: Path) -> None:
    """An archived DETECTED is still the open episode: watch must not re-mint it (#1072).

    The episode scan in cmd_watch read the live tail only, so after
    compaction moved the DETECTED into events_archive the same breach episode
    looked unrecorded and every watch invocation appended another
    LIVENESS_SILENCE_DETECTED for it.
    """
    db = str(tmp_path / "watch_compact_dup.db")
    run_id = "run_watch_compact_dup"
    old_ts = datetime.now(UTC) - timedelta(hours=3)
    with SQLiteStorage(db) as store:
        store.create_run_started(Run(run_id=run_id, goal="compact dup"))
        _append_timed(
            store,
            run_id,
            EventType.WORK_COMPLETED,
            {"doc": 0},
            old_ts,
        )
        _append_timed(
            store,
            run_id,
            EventType.LIVENESS_SILENCE_DETECTED,
            {"silence_seconds": 10800, "threshold_seconds": 3600, "phase": "otherwise"},
            old_ts,
        )
        store.compact_run(run_id)
        # The compacted prefix (DETECTED included) is archived; only the anchor
        # is live. Append an old-timestamp live tail so the run is still silent.
        assert store.read_events(run_id)[-1].type is not EventType.LIVENESS_SILENCE_DETECTED, (
            "setup error: DETECTED must land in the archive, not stay live"
        )
        _append_timed(store, run_id, EventType.WORK_COMPLETED, {"doc": 1}, old_ts)

    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["--db", db, "watch", run_id, "--max-silence", "3600", "--on-breach", "exit"],
        out=out,
        err=err,
    )
    assert code == 20, out.getvalue()
    with SQLiteStorage(db) as store:
        detected = [
            e
            for e in store.read_all_events(run_id)
            if e.type is EventType.LIVENESS_SILENCE_DETECTED
        ]
        assert len(detected) == 1, "compaction must not cause a duplicate DETECTED"


def test_watch_mints_recovered_after_compaction(tmp_path: Path) -> None:
    """A compacted run that resumed must still mint LIVENESS_RECOVERED (#1072).

    With the episode scan reading the live tail only, an archived DETECTED was
    invisible, so the not-breached branch never fired and the episode never
    terminated in the audit trail.
    """
    db = str(tmp_path / "watch_compact_recovered.db")
    run_id = "run_watch_compact_recovered"
    old_ts = datetime.now(UTC) - timedelta(hours=3)
    with SQLiteStorage(db) as store:
        store.create_run_started(Run(run_id=run_id, goal="compact recovered"))
        _append_timed(store, run_id, EventType.WORK_COMPLETED, {"doc": 0}, old_ts)
        _append_timed(
            store,
            run_id,
            EventType.LIVENESS_SILENCE_DETECTED,
            {"silence_seconds": 10800, "threshold_seconds": 3600, "phase": "otherwise"},
            old_ts,
        )
        store.compact_run(run_id)
        # Fresh live tail: the run has resumed, so it is no longer breached.
        _append_timed(store, run_id, EventType.WORK_COMPLETED, {"doc": 1}, datetime.now(UTC))

    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["--db", db, "watch", run_id, "--max-silence", "3600", "--on-breach", "exit"],
        out=out,
        err=err,
    )
    assert code == 0, out.getvalue()
    with SQLiteStorage(db) as store:
        recovered = [
            e for e in store.read_all_events(run_id) if e.type is EventType.LIVENESS_RECOVERED
        ]
        assert len(recovered) == 1, "recovery of an archived episode must be recorded"


def test_compaction_does_not_restart_the_silence_clock(tmp_path: Path) -> None:
    """A compaction mid-silence must not make a still-silent run read healthy (#2088).

    ``compact_run`` appends fresh anchor markers (``STATE_CHECKPOINTED`` with
    trigger=compaction, then ``EVENT_LOG_ANCHORED``) and moves the older
    prefix into ``events_archive``. The silence clock used to read the live
    tail only, so after compaction the newest live event was the marker and
    the clock restarted from compaction time: a run silent for hours read
    ``silence ~ 0, breached=False``, the engine dropped its WAIT proposal
    for ``resume``, and the archive-aware episode scan mints
    ``LIVENESS_RECOVERED`` for a run that never resumed.
    """
    db = str(tmp_path / "watch_silence_compact.db")
    run_id = "run_watch_silence_compact"
    old_ts = datetime.now(UTC) - timedelta(hours=3)
    with SQLiteStorage(db) as store:
        store.create_run_started(Run(run_id=run_id, goal="silence survives compaction"))
        _append_timed(store, run_id, EventType.WORK_COMPLETED, {"doc": 0}, old_ts)

    # watch records the breach, minting DETECTED at wall-clock now.
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["--db", db, "watch", run_id, "--max-silence", "3600", "--on-breach", "exit"],
        out=out,
        err=err,
    )
    assert code == 20, out.getvalue()

    with SQLiteStorage(db) as store:
        store.compact_run(run_id)
        # The setup must put the clock's old enemies where the fix expects
        # them: the live tail ends on the anchor marker, the real work (and
        # the DETECTED episode) lives in the archive.
        assert store.read_events(run_id)[-1].type is EventType.EVENT_LOG_ANCHORED
        archived_types = {e.type for e in store.read_archived_events(run_id)}
        assert EventType.WORK_COMPLETED in archived_types

    from continuum.recovery.health import advisory_for_storage

    with SQLiteStorage(db) as store:
        advisory = advisory_for_storage(store, run_id)
        assert advisory["breached"] is True
        assert advisory["silence_seconds"] > 3600, advisory
        assert RecoveryEngine(store).assess(run_id).mode.value == "wait"

    # The archived DETECTED is still the open episode: the still-silent run
    # must not mint LIVENESS_RECOVERED, nor a duplicate DETECTED.
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["--db", db, "watch", run_id, "--max-silence", "3600", "--on-breach", "exit"],
        out=out,
        err=err,
    )
    assert code == 20, out.getvalue()
    with SQLiteStorage(db) as store:
        detected = [
            e
            for e in store.read_all_events(run_id)
            if e.type is EventType.LIVENESS_SILENCE_DETECTED
        ]
        recovered = [
            e for e in store.read_all_events(run_id) if e.type is EventType.LIVENESS_RECOVERED
        ]
        assert len(detected) == 1, "the archived episode must not be duplicated"
        assert not recovered, "a still-silent run must not be recorded as recovered"


def test_watch_sustains_a_breach_across_invocations(tmp_path: Path) -> None:
    """A breach episode must outlive the watch call that opened it (#2088).

    watch appends ``LIVENESS_SILENCE_DETECTED`` as its own audit record.
    When the silence clock counts that marker the observer resets the clock
    it measures, so the very next invocation sees silence of ~0 and mints
    ``LIVENESS_RECOVERED`` for a run that never resumed. The liveness
    markers are observations about the cadence, not evidence of it.
    """
    db = str(tmp_path / "watch_sustained.db")
    run_id = "run_watch_sustained"
    old_ts = datetime.now(UTC) - timedelta(hours=3)
    with SQLiteStorage(db) as store:
        store.create_run_started(Run(run_id=run_id, goal="sustained breach"))
        _append_timed(store, run_id, EventType.WORK_COMPLETED, {"doc": 0}, old_ts)

    codes: list[int] = []
    for _ in range(2):
        out, err = io.StringIO(), io.StringIO()
        codes.append(
            main(
                ["--db", db, "watch", run_id, "--max-silence", "3600", "--on-breach", "exit"],
                out=out,
                err=err,
            )
        )
    assert codes == [20, 20], codes
    with SQLiteStorage(db) as store:
        detected = [
            e
            for e in store.read_all_events(run_id)
            if e.type is EventType.LIVENESS_SILENCE_DETECTED
        ]
        recovered = [
            e for e in store.read_all_events(run_id) if e.type is EventType.LIVENESS_RECOVERED
        ]
        assert len(detected) == 1, "a sustained episode needs no second DETECTED"
        assert not recovered, "a run that never resumed is not recovered"


def test_a_manual_checkpoint_still_anchors_the_silence_clock(tmp_path: Path) -> None:
    """Only the compaction-anchor checkpoint is bookkeeping, not every one (#2088).

    The clock skips ``STATE_CHECKPOINTED`` by trigger, not by type: a manual
    checkpoint is operator interaction with the run and must count as
    activity, or every freshly checkpointed run would read as silent.
    """
    from continuum.checkpoint import CheckpointManager

    db = str(tmp_path / "watch_manual_checkpoint.db")
    run_id = "run_watch_manual_checkpoint"
    old_ts = datetime.now(UTC) - timedelta(hours=3)
    with SQLiteStorage(db) as store:
        store.create_run_started(Run(run_id=run_id, goal="manual checkpoint"))
        _append_timed(store, run_id, EventType.WORK_COMPLETED, {"doc": 0}, old_ts)
        CheckpointManager(store).checkpoint(run_id)
        assert store.read_events(run_id)[-1].type is EventType.STATE_CHECKPOINTED

    from continuum.recovery.health import advisory_for_storage

    with SQLiteStorage(db) as store:
        advisory = advisory_for_storage(store, run_id)
        assert advisory["breached"] is False
        assert advisory["silence_seconds"] < 3600, advisory


def test_watch_falls_back_to_the_live_tail_when_the_archive_is_unreadable(
    tmp_path: Path,
) -> None:
    """An unreadable archive degrades to the live tail, not to a crash (#1072).

    The archive-aware scan is the fix, but a store that cannot serve its
    archived prefix must still answer: watch is a long-running loop, so one
    unreadable archive page must not kill the whole episode state machine.
    """
    db = str(tmp_path / "watch_archive_dead.db")
    run_id = "run_watch_archive_dead"
    with SQLiteStorage(db) as store:
        store.create_run_started(Run(run_id=run_id, goal="archive unreadable"))
        store.append_event(run_id, EventType.WORK_COMPLETED, {"doc": 0})

    dead_archive = _ArchiveDeadStorage(db)
    # ``open_storage`` is imported into cli.main, so patch it there: the CLI
    # builds the store, cmd_watch only receives it.
    with unittest.mock.patch("continuum.cli.main.open_storage", return_value=dead_archive):
        out, err = io.StringIO(), io.StringIO()
        code = main(
            ["--db", db, "watch", run_id, "--max-silence", "3600", "--on-breach", "exit"],
            out=out,
            err=err,
        )
    assert code == 0, out.getvalue()
    # The fallback must actually have run: the archive read raised, so the
    # live tail is the only place the episode scan could have looked.
    assert dead_archive.served_live_tail, "watch did not fall back to the live tail"
    with SQLiteStorage(db) as store:
        types = [e.type for e in store.read_events(run_id)]
    assert EventType.LIVENESS_SILENCE_DETECTED not in types
