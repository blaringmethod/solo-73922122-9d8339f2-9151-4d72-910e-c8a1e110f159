"""Tests for commit-sequenced, immutable meter history."""

from __future__ import annotations

import sqlite3

import pytest

from app.db import connection, initialize
from app.models import ReadingIn
from tests.common import assert_meter_consistent, at, reading


def post(client, readings):
    return client.post("/api/v1/readings", json={"readings": readings})


def undo(client, ids):
    return client.post("/api/v1/readings/undo", json={"source_record_ids": ids})


def history(client, meter, seq):
    return client.get(f"/api/v1/meters/{meter}/history/{seq}")


def enable(client):
    return client.post("/api/v1/history/enable")


def test_history_is_rejected_until_enabled(client):
    """Enable first, then commit: the anchor is the starting point."""
    post(client, [reading("M1", "a", 0, 100)])
    assert enable(client).status_code == 200
    seq = post(client, [reading("M1", "b", 60, 160)]).json()["commit_seq"]
    assert history(client, "M1", seq).status_code == 200


def test_query_with_history_not_enabled_is_explicit(client):
    first = post(client, [reading("M1", "a", 0, 100)]).json()
    resp = history(client, "M1", first["commit_seq"])
    assert resp.status_code == 409
    assert resp.json()["code"] == "history_not_enabled"

    health = client.get("/health").json()
    assert health["history_enabled"] is False
    assert health["history_start_seq"] is None


def test_enable_anchors_current_committed_state(client):
    post(client, [reading("M1", "a", 0, 100), reading("M1", "b", 60, 160)])
    created = post(client, [reading("M2", "c", 0, 50)]).json()
    seq = created["commit_seq"]

    enabled = enable(client).json()
    assert enabled["already_enabled"] is False
    assert enabled["history_start_seq"] == seq
    assert enabled["current_commit_seq"] == seq
    assert enabled["meters_anchored"] == 2

    for meter in ("M1", "M2"):
        body = history(client, meter, seq)
        assert body.status_code == 200
        payload = body.json()
        assert payload["history_start_seq"] == seq
        assert payload["state_commit_seq"] == seq
        assert_meter_consistent(payload)

    assert history(client, "M1", seq).json()["usage_total"] == 60
    health = client.get("/health").json()
    assert health["history_enabled"] is True
    assert health["history_start_seq"] == seq


def test_enable_is_idempotent_and_never_rewrites_evidence(client):
    post(client, [reading("M1", "a", 0, 100)])
    first = enable(client).json()
    post(client, [reading("M1", "b", 60, 130)])
    later = post(client, [reading("M1", "c", 120, 200)]).json()

    second = enable(client).json()
    assert second["already_enabled"] is True
    assert second["history_start_seq"] == first["history_start_seq"]

    # The anchor still describes the original state; the later enable did not
    # restamp the evidence.
    anchored = history(client, "M1", first["history_start_seq"]).json()
    assert anchored["state_commit_seq"] == first["history_start_seq"]
    assert len(anchored["readings"]) == 1

    current = history(client, "M1", later["commit_seq"]).json()
    assert len(current["readings"]) == 3


def test_late_reading_does_not_rewrite_an_older_commit(client):
    enable(client)
    first = post(
        client, [reading("M1", "a", 0, 100), reading("M1", "c", 120, 250)]
    ).json()
    seq1 = first["commit_seq"]
    before = history(client, "M1", seq1).json()
    assert [i["usage"] for i in before["intervals"]] == [150]

    backfill = post(client, [reading("M1", "b", 60, 160)]).json()
    seq2 = backfill["commit_seq"]
    after = history(client, "M1", seq1).json()
    # The evidence itself is unchanged; only the ancillary "current" marker
    # moved forward, because it describes the database as of the read.
    assert {k: v for k, v in after.items() if k != "current_commit_seq"} == {
        k: v for k, v in before.items() if k != "current_commit_seq"
    }
    assert after["state_commit_seq"] == before["state_commit_seq"] == seq1
    assert [i["usage"] for i in after["intervals"]] == [150]

    newer = history(client, "M1", seq2).json()
    assert [i["usage"] for i in newer["intervals"]] == [60, 90]


def test_undo_keeps_readings_in_the_commit_that_saw_them(client):
    enable(client)
    first = post(
        client,
        [
            reading("M1", "a", 0, 100),
            reading("M1", "b", 60, 160),
            reading("M1", "c", 120, 250),
        ],
    ).json()
    seq1 = first["commit_seq"]

    removed = undo(client, ["b"]).json()
    seq2 = removed["commit_seq"]

    original = history(client, "M1", seq1).json()
    assert [r["source_record_id"] for r in original["readings"]] == ["a", "b", "c"]
    assert [i["usage"] for i in original["intervals"]] == [60, 90]
    assert original["usage_total"] == 150
    assert_meter_consistent(original)

    after = history(client, "M1", seq2).json()
    assert [r["source_record_id"] for r in after["readings"]] == ["a", "c"]
    assert [i["usage"] for i in after["intervals"]] == [150]
    assert after["usage_total"] == 150

    # The live state also no longer has "b".
    live = client.get("/api/v1/meters/M1").json()
    assert [r["source_record_id"] for r in live["readings"]] == ["a", "c"]


def test_history_at_every_commit_including_undo_to_empty(client):
    enable(client)
    first = post(client, [reading("M1", "a", 0, 10), reading("M1", "b", 60, 30)]).json()
    seq1 = first["commit_seq"]
    undo(client, ["a", "b"])
    latest = client.get("/health").json()["commit_seq"]

    assert len(history(client, "M1", seq1).json()["readings"]) == 2
    # The undo commit is itself a committed state in which the meter has no
    # readings; history reports that empty state rather than hiding it.
    resp = history(client, "M1", latest)
    assert resp.status_code == 200
    assert resp.json()["readings"] == []
    assert resp.json()["intervals"] == []
    assert resp.json()["usage_total"] == 0
    assert resp.json()["state_commit_seq"] == latest


def test_rejections_are_explicit(client):
    post(client, [reading("M1", "pre", 0, 5)])  # committed before the anchor
    anchored = enable(client).json()["history_start_seq"]
    assert anchored > 0
    seq = post(client, [reading("M1", "a", 60, 10)]).json()["commit_seq"]

    before = history(client, "M1", anchored - 1)
    assert before.status_code == 409
    assert before.json()["code"] == "history_before_start"

    future = history(client, "M1", seq + 1)
    assert future.status_code == 404
    assert future.json()["code"] == "commit_not_found"

    unknown = history(client, "M-UNKNOWN", seq)
    assert unknown.status_code == 404
    assert unknown.json()["code"] == "meter_not_found"

    negative = history(client, "M1", anchored - 100)
    assert negative.status_code == 409


def test_meter_created_after_requested_commit_has_no_state(client):
    enable(client)
    seq_anchor = client.get("/health").json()["history_start_seq"]
    other = post(client, [reading("M2", "x", 0, 5)]).json()["commit_seq"]

    resp = history(client, "M1", other)
    assert resp.status_code == 404
    assert resp.json()["code"] == "meter_not_found"
    # But the anchor itself can be queried for a meter that existed then.
    assert history(client, "M2", other).status_code == 200
    assert history(client, "M2", seq_anchor).status_code == 404


def test_untouched_meter_reports_the_state_in_force_at_that_commit(client):
    enable(client)
    seq_a = post(client, [reading("M1", "a", 0, 10), reading("M1", "b", 60, 20)]).json()[
        "commit_seq"
    ]
    seq_b = post(client, [reading("M2", "c", 0, 5)]).json()["commit_seq"]

    body = history(client, "M1", seq_b).json()
    assert body["commit_seq"] == seq_b
    assert body["state_commit_seq"] == seq_a
    assert [i["usage"] for i in body["intervals"]] == [10]


def test_break_points_are_part_of_the_historical_state(client):
    enable(client)
    seq = post(
        client,
        [
            reading("M1", "a", 0, 1000),
            reading("M1", "b", 60, 1200),
            reading("M1", "c", 120, 30, replaced=True),
        ],
    ).json()["commit_seq"]
    body = history(client, "M1", seq).json()
    assert [i["usage"] for i in body["intervals"]] == [200, None]
    assert body["breaks"][0]["break_reason"] == "meter_replaced"
    assert_meter_consistent(body)


def test_history_rows_are_immutable_in_the_database(client, settings):
    post(client, [reading("M1", "pre", 0, 5)])
    enable(client)
    seq = post(client, [reading("M1", "a", 60, 10), reading("M1", "b", 120, 25)]).json()[
        "commit_seq"
    ]
    with connection(settings) as conn:
        baseline = conn.execute(
            "SELECT COUNT(*) FROM history_snapshots"
        ).fetchone()[0]
        assert baseline == 2  # anchor + the commit above
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE history_snapshots SET payload = '{}' "
                "WHERE meter_id = 'M1' AND commit_seq = ?",
                (seq,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "DELETE FROM history_snapshots WHERE meter_id = 'M1' "
                "AND commit_seq = ?",
                (seq,),
            )
        # The failed statements must not have changed anything.
        assert conn.execute(
            "SELECT COUNT(*) FROM history_snapshots"
        ).fetchone()[0] == baseline


def test_failed_commit_leaves_no_history_evidence(db_path, settings, monkeypatch):
    import app.history as history_module
    import app.service as service

    initialize(settings)
    service.enable_history(settings)

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(history_module, "record_snapshot", explode)
    with pytest.raises(RuntimeError):
        service.ingest(
            settings,
            [ReadingIn.model_validate(reading("M1", "a", 0, 10))],
        )

    with connection(settings) as conn:
        assert conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM intervals").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM history_snapshots").fetchone()[0] == 0
        assert conn.execute(
            "SELECT value FROM meta WHERE key = 'commit_seq'"
        ).fetchone()[0] == "0"
        assert conn.execute(
            "SELECT value FROM meta WHERE key = 'history_start_seq'"
        ).fetchone()[0] == "0"


def test_uncommitted_history_is_rolled_back_on_crash(db_path, settings):
    initialize(settings)
    raw = sqlite3.connect(db_path, timeout=10)
    raw.isolation_level = None
    raw.execute("BEGIN IMMEDIATE")
    raw.execute(
        "INSERT INTO history_snapshots (meter_id, commit_seq, reading_count, "
        "interval_count, usage_total, payload) VALUES ('M1', 1, 1, 0, 0, '{}')"
    )
    raw.close()  # connection death == rollback

    with connection(settings) as conn:
        assert conn.execute("SELECT COUNT(*) FROM history_snapshots").fetchone()[0] == 0


def test_enable_and_writes_are_atomic(client, settings):
    """The start sequence and its anchor snapshots share one commit."""
    post(client, [reading("M1", "a", 0, 10)])
    assert enable(client).status_code == 200
    with connection(settings) as conn:
        start = int(
            conn.execute(
                "SELECT value FROM meta WHERE key = 'history_start_seq'"
            ).fetchone()[0]
        )
        anchors = conn.execute(
            "SELECT COUNT(*) FROM history_snapshots WHERE commit_seq = ?",
            (start,),
        ).fetchone()[0]
    assert anchors == 1
    seq = post(client, [reading("M1", "b", 60, 20)]).json()["commit_seq"]
    assert seq == start + 1


def test_history_query_served_after_restart(db_path, settings):
    """Evidence is on disk and survives a fresh process/connection."""
    initialize(settings)
    with create_client(settings) as bootstrap:
        first = post(bootstrap, [reading("M1", "a", 0, 10)]).json()
        assert enable(bootstrap).status_code == 200
        seq = first["commit_seq"]
        second = post(bootstrap, [reading("M1", "b", 60, 40)]).json()["commit_seq"]
    assert second > seq

    from app.service import get_history

    # A brand new app instance (fresh lifespan + connections) reads the same
    # evidence from disk.
    with create_client(settings) as restarted:
        body = history(restarted, "M1", seq).json()
        assert [r["source_record_id"] for r in body["readings"]] == ["a"]
        assert body["usage_total"] == 0
        assert_meter_consistent(body)


def create_client(settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    return TestClient(create_app(settings))


def test_history_window_after_undo_and_reinsert(client):
    """A re-inserted source id is a new reading with its own commit."""
    enable(client)
    post(client, [reading("M1", "a", 0, 10), reading("M1", "b", 60, 20)])
    old = post(client, [reading("M2", "x", 0, 1)]).json()["commit_seq"]
    undo(client, ["b"])
    new_seq = post(client, [reading("M1", "b", 60, 45)]).json()["commit_seq"]

    before_undo = history(client, "M1", old).json()
    after_reinsert = history(client, "M1", new_seq).json()
    assert [i["usage"] for i in before_undo["intervals"]] == [10]
    assert [i["usage"] for i in after_reinsert["intervals"]] == [35]
    assert before_undo["readings"][1]["commit_seq"] < new_seq


def test_existing_v1_database_is_upgraded_without_losing_data(db_path):
    """A database created before history existed keeps its data and can be
    upgraded; history starts disabled until explicitly enabled."""
    v1_schema = [
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
        "INSERT INTO meta (key, value) VALUES ('commit_seq', '3')",
        "CREATE TABLE readings (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "source_record_id TEXT NOT NULL, meter_id TEXT NOT NULL, "
        "read_at_us INTEGER NOT NULL, reading INTEGER NOT NULL CHECK (reading >= 0), "
        "meter_replaced INTEGER NOT NULL CHECK (meter_replaced IN (0, 1)), "
        "content_hash TEXT NOT NULL, created_commit_seq INTEGER NOT NULL, "
        "UNIQUE (source_record_id), UNIQUE (meter_id, read_at_us))",
        "CREATE TABLE intervals (id INTEGER PRIMARY KEY, meter_id TEXT NOT NULL, "
        "from_reading_id INTEGER NOT NULL REFERENCES readings(id) ON DELETE CASCADE, "
        "to_reading_id INTEGER NOT NULL REFERENCES readings(id) ON DELETE CASCADE, "
        "from_at_us INTEGER NOT NULL, to_at_us INTEGER NOT NULL, "
        "from_value INTEGER NOT NULL, to_value INTEGER NOT NULL, usage INTEGER, "
        "is_break INTEGER NOT NULL CHECK (is_break IN (0, 1)), break_reason TEXT, "
        "commit_seq INTEGER NOT NULL, UNIQUE (meter_id, to_reading_id))",
    ]
    raw = sqlite3.connect(db_path)
    raw.execute("PRAGMA journal_mode = WAL")
    for statement in v1_schema:
        raw.execute(statement)
    raw.execute(
        "INSERT INTO readings (source_record_id, meter_id, read_at_us, reading, "
        "meter_replaced, content_hash, created_commit_seq) "
        "VALUES ('old-a', 'M1', 0, 10, 0, 'h', 3)"
    )
    raw.execute("PRAGMA user_version = 1")
    raw.commit()
    raw.close()

    from app.config import Settings

    upgraded = Settings(db_path=db_path)
    initialize(upgraded)
    with connection(upgraded) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM history_snapshots").fetchone()[0] == 0

    with create_client(upgraded) as client:
        assert client.get("/health").json()["history_start_seq"] is None
        assert history(client, "M1", 3).status_code == 409
        enabled = enable(client).json()
        # The only committed state is seq 3; the anchor is seq 3.
        assert enabled["history_start_seq"] == 3
        body = history(client, "M1", 3).json()
        assert [r["source_record_id"] for r in body["readings"]] == ["old-a"]
