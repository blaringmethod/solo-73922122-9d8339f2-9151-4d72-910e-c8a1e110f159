"""Commit-sequenced, immutable per-meter history evidence.

Every state-changing commit records a complete snapshot of each meter it
touched.  A snapshot is addressed by ``(meter_id, commit_seq)`` and is written
in the *same* transaction as the live readings and intervals, so a failure or a
crash can never persist a snapshot that disagrees with the committed data.

Snapshots are append-only.  Later events (backfilled readings, undo) create a
new snapshot at a new commit sequence; the rows describing earlier commits are
never updated or deleted (enforced by triggers on ``history_snapshots`` too).
As a result the history endpoint can answer "what did downstream see after
commit N?" even for readings that have since been undone.

History is *enabled* by recording a starting commit sequence.  Everything that
happened before that point has no evidence and is rejected explicitly; the
starting sequence is established from the readings that are committed at the
moment history is enabled.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Iterable

from .db import (
    current_commit_seq,
    history_start_seq,
    set_history_start_seq,
)
from .models import (
    HistoryStateResponse,
    IntervalOut,
    ReadingOut,
)
from .timeutil import microseconds_to_iso


class HistoryError(Exception):
    """A history query that cannot be answered for the requested commit."""

    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _reading_dict(row: sqlite3.Row) -> dict:
    return {
        "source_record_id": row["source_record_id"],
        "read_at": microseconds_to_iso(row["read_at_us"]),
        "reading": int(row["reading"]),
        "meter_replaced": bool(row["meter_replaced"]),
        "commit_seq": int(row["created_commit_seq"]),
    }


def _interval_dict(row: sqlite3.Row) -> dict:
    return {
        "from_source_record_id": row["from_source"],
        "to_source_record_id": row["to_source"],
        "from_read_at": microseconds_to_iso(row["from_at_us"]),
        "to_read_at": microseconds_to_iso(row["to_at_us"]),
        "from_reading": int(row["from_value"]),
        "to_reading": int(row["to_value"]),
        "usage": None if row["usage"] is None else int(row["usage"]),
        "is_break": bool(row["is_break"]),
        "break_reason": row["break_reason"],
        "commit_seq": int(row["commit_seq"]),
    }


def record_snapshot(
    conn: sqlite3.Connection, meter_id: str, commit_seq: int
) -> None:
    """Persist the current committed state of one meter as evidence.

    Must be called inside the write transaction that established that state.
    Re-recording the same ``(meter_id, commit_seq)`` is a no-op, never an
    overwrite.
    """
    reading_rows = conn.execute(
        "SELECT source_record_id, read_at_us, reading, meter_replaced, "
        "created_commit_seq FROM readings WHERE meter_id = ? "
        "ORDER BY read_at_us ASC",
        (meter_id,),
    ).fetchall()
    interval_rows = conn.execute(
        "SELECT i.from_at_us, i.to_at_us, i.from_value, i.to_value, i.usage, "
        "i.is_break, i.break_reason, i.commit_seq, "
        "f.source_record_id AS from_source, "
        "t.source_record_id AS to_source "
        "FROM intervals i "
        "JOIN readings f ON f.id = i.from_reading_id "
        "JOIN readings t ON t.id = i.to_reading_id "
        "WHERE i.meter_id = ? ORDER BY i.to_at_us ASC",
        (meter_id,),
    ).fetchall()

    readings = [_reading_dict(row) for row in reading_rows]
    intervals = [_interval_dict(row) for row in interval_rows]
    usage_total = sum(
        item["usage"]
        for item in intervals
        if not item["is_break"] and item["usage"] is not None
    )
    payload = json.dumps(
        {
            "readings": readings,
            "intervals": intervals,
            "usage_total": usage_total,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    # Plain INSERT: one snapshot per (meter, commit).  A second insert means a
    # caller tried to record the same commit twice, which would be a bug and
    # must fail loudly rather than silently overwrite evidence.
    conn.execute(
        "INSERT INTO history_snapshots "
        "(meter_id, commit_seq, reading_count, interval_count, usage_total, "
        "payload) VALUES (?, ?, ?, ?, ?, ?)",
        (
            meter_id,
            commit_seq,
            len(readings),
            len(intervals),
            usage_total,
            payload,
        ),
    )


def record_commit_snapshots(
    conn: sqlite3.Connection, meter_ids: Iterable[str], commit_seq: int
) -> None:
    """Record evidence for every meter touched by a new commit.

    No-op while history has never been enabled, so pre-existing behaviour is
    unchanged until an operator explicitly anchors a start sequence.
    """
    if history_start_seq(conn) is None:
        return
    for meter_id in sorted(set(meter_ids)):
        record_snapshot(conn, meter_id, commit_seq)


def start_history(conn: sqlite3.Connection) -> int:
    """Enable history, anchored at the current committed state.

    Idempotent: once a start sequence exists it is returned unchanged and no
    snapshots are rewritten.  Runs inside the caller's write transaction.
    """
    existing = history_start_seq(conn)
    if existing is not None:
        return existing
    seq = current_commit_seq(conn)
    set_history_start_seq(conn, seq)
    meter_ids = [
        row["meter_id"]
        for row in conn.execute(
            "SELECT DISTINCT meter_id FROM readings ORDER BY meter_id"
        ).fetchall()
    ]
    _record_snapshots(conn, meter_ids, seq)
    return seq


def _record_snapshots(
    conn: sqlite3.Connection, meter_ids: Iterable[str], commit_seq: int
) -> None:
    for meter_id in sorted(set(meter_ids)):
        record_snapshot(conn, meter_id, commit_seq)


def _snapshot_at(
    conn: sqlite3.Connection, meter_id: str, commit_seq: int
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT meter_id, commit_seq, reading_count, interval_count, "
        "usage_total, payload FROM history_snapshots "
        "WHERE meter_id = ? AND commit_seq <= ? "
        "ORDER BY commit_seq DESC LIMIT 1",
        (meter_id, commit_seq),
    ).fetchone()


def query_history(
    conn: sqlite3.Connection, meter_id: str, commit_seq: int
) -> HistoryStateResponse:
    """Return the meter state as of ``commit_seq`` from stored evidence.

    The caller supplies a read transaction, so the start sequence, the current
    sequence and the snapshot all come from one committed database snapshot.
    """
    start_seq = history_start_seq(conn)
    if start_seq is None:
        raise HistoryError(
            409,
            "history_not_enabled",
            "历史查询尚未启用，请先调用 POST /api/v1/history/enable",
        )
    current_seq = current_commit_seq(conn)
    if commit_seq < start_seq:
        raise HistoryError(
            409,
            "history_before_start",
            f"提交序号 {commit_seq} 早于历史起点 {start_seq}，没有可重建的依据",
        )
    if commit_seq > current_seq:
        raise HistoryError(
            404,
            "commit_not_found",
            f"提交序号 {commit_seq} 尚未产生（当前为 {current_seq}）",
        )

    row = _snapshot_at(conn, meter_id, commit_seq)
    if row is None:
        # The meter had not been created yet at that commit, so there is no
        # evidence describing it.  (A meter whose readings were all undone is
        # different: it has an explicit empty snapshot, returned below.)
        raise HistoryError(
            404,
            "meter_not_found",
            f"电表 {meter_id} 在提交 {commit_seq} 尚无读数记录",
        )

    payload = json.loads(row["payload"])
    readings = [ReadingOut(**item) for item in payload["readings"]]
    intervals = [IntervalOut(**item) for item in payload["intervals"]]
    return HistoryStateResponse(
        meter_id=meter_id,
        commit_seq=commit_seq,
        state_commit_seq=int(row["commit_seq"]),
        history_start_seq=int(start_seq),
        current_commit_seq=int(current_seq),
        readings=readings,
        intervals=intervals,
        breaks=[item for item in intervals if item.is_break],
        usage_total=int(payload["usage_total"]),
    )
