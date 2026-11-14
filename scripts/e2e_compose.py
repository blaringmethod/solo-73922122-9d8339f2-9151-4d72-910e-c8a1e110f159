"""End-to-end smoke test against a running Compose deployment.

Usage:
    python scripts/e2e_compose.py http://localhost:8000
"""

from __future__ import annotations

import json
import os
import sys

import httpx


def main(base_url: str, second_url: str | None = None) -> int:
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        health = client.get("/health")
        assert health.status_code == 200, health.text
        print("health:", health.json())

        batch = {
            "readings": [
                {
                    "meter_id": "E2E-M1",
                    "source_record_id": "e2e-00",
                    "read_at": "2026-02-01T00:00:00Z",
                    "reading": 1000,
                },
                {
                    "meter_id": "E2E-M1",
                    "source_record_id": "e2e-02",
                    "read_at": "2026-02-01T02:00:00Z",
                    "reading": 1300,
                },
                {
                    "meter_id": "E2E-M1",
                    "source_record_id": "e2e-03",
                    "read_at": "2026-02-01T03:00:00Z",
                    "reading": 20,
                    "meter_replaced": True,
                },
                {
                    "meter_id": "E2E-M1",
                    "source_record_id": "e2e-04",
                    "read_at": "2026-02-01T04:00:00Z",
                    "reading": 75,
                },
            ]
        }
        # First submit with a gap (02:00 missing), then backfill it.
        initial = [r for r in batch["readings"] if r["source_record_id"] != "e2e-02"]
        backfill = [r for r in batch["readings"] if r["source_record_id"] == "e2e-02"]
        assert client.post("/api/v1/readings", json={"readings": initial}).status_code == 200
        assert client.post("/api/v1/readings", json={"readings": backfill}).status_code == 200

        retry = client.post("/api/v1/readings", json=batch)
        assert retry.status_code == 200 and retry.json()["changed"] is False
        print("idempotent retry:", retry.json()["counts"])

        conflict = client.post(
            "/api/v1/readings",
            json={
                "readings": [
                    {
                        "meter_id": "E2E-M1",
                        "source_record_id": "e2e-00",
                        "read_at": "2026-02-01T00:00:00Z",
                        "reading": 1001,
                    }
                ]
            },
        )
        assert conflict.status_code == 409, conflict.text
        print("conflict rejected:", conflict.json()["conflicts"][0]["type"])

        undo = client.post(
            "/api/v1/readings/undo", json={"source_record_ids": ["e2e-02"]}
        )
        assert undo.status_code == 200
        print("undo:", undo.json()["counts"])

        state = client.get("/api/v1/meters/E2E-M1").json()
        print("final state:")
        print(json.dumps(state, ensure_ascii=False, indent=2))
        # e2e-02 is gone: 1000 -> 20 is a meter-replacement break (no
        # usage), 20 -> 75 is 55.
        assert state["usage_total"] == 55
        assert state["breaks"][0]["break_reason"] == "meter_replaced"

        # ---- history ---------------------------------------------------
        enabled = client.post("/api/v1/history/enable")
        assert enabled.status_code == 200, enabled.text
        start_seq = enabled.json()["history_start_seq"]
        print("history enabled at commit:", start_seq)

        # Anchor the state, then backfill a reading that sorts before it and
        # verify the anchor's evidence is not rewritten.
        anchor = client.get(
            f"/api/v1/meters/E2E-M1/history/{start_seq}"
        ).json()
        backfill_late = client.post(
            "/api/v1/readings",
            json={
                "readings": [
                    {
                        "meter_id": "E2E-M1",
                        "source_record_id": "e2e-late",
                        "read_at": "2026-02-01T03:30:00Z",
                        "reading": 40,
                    }
                ]
            },
        )
        assert backfill_late.status_code == 200, backfill_late.text
        late_seq = backfill_late.json()["commit_seq"]
        anchor_again = client.get(
            f"/api/v1/meters/E2E-M1/history/{start_seq}"
        ).json()
        assert [r["source_record_id"] for r in anchor_again["readings"]] == [
            r["source_record_id"] for r in anchor["readings"]
        ], "late reading rewrote an older commit"
        assert anchor_again["state_commit_seq"] == start_seq

        late_state = client.get(
            f"/api/v1/meters/E2E-M1/history/{late_seq}"
        ).json()
        assert late_state["state_commit_seq"] == late_seq
        assert any(
            r["source_record_id"] == "e2e-late"
            for r in late_state["readings"]
        )
        print("history backfill kept the anchor intact")

        # Undo a reading that the anchor commit contained; the anchor must
        # still show it, while a later commit must not.
        undo_anchor = client.post(
            "/api/v1/readings/undo", json={"source_record_ids": ["e2e-late"]}
        )
        assert undo_anchor.status_code == 200
        after_undo_seq = undo_anchor.json()["commit_seq"]
        assert any(
            r["source_record_id"] == "e2e-late"
            for r in client.get(
                f"/api/v1/meters/E2E-M1/history/{late_seq}"
            ).json()["readings"]
        ), "undone reading disappeared from its original commit"
        assert not any(
            r["source_record_id"] == "e2e-late"
            for r in client.get(
                f"/api/v1/meters/E2E-M1/history/{after_undo_seq}"
            ).json()["readings"]
        )
        print("history keeps undone readings in their original commit")

        # Explicit rejections.
        before = client.get(
            f"/api/v1/meters/E2E-M1/history/{start_seq - 1}"
        )
        assert before.status_code == 409, before.text
        assert before.json()["code"] == "history_before_start"
        future = client.get(
            f"/api/v1/meters/E2E-M1/history/{after_undo_seq + 1000}"
        )
        assert future.status_code == 404, future.text
        assert future.json()["code"] == "commit_not_found"
        print("history rejects out-of-range sequences")

        listing = client.get("/api/v1/meters").json()
        assert any(m["meter_id"] == "E2E-M1" for m in listing)

    # Optional: prove both processes serve the same data, including history
    # evidence committed through the first process.
    if second_url:
        with httpx.Client(base_url=second_url, timeout=30.0) as other:
            other_state = other.get("/api/v1/meters/E2E-M1").json()
            other_anchor = other.get(
                f"/api/v1/meters/E2E-M1/history/{start_seq}"
            ).json()
            other_late = other.get(
                f"/api/v1/meters/E2E-M1/history/{late_seq}"
            ).json()
        current = other_state["commit_seq"]
        assert current == after_undo_seq
        assert other_anchor["readings"] == anchor["readings"]
        assert other_anchor["intervals"] == anchor["intervals"]
        assert other_late["state_commit_seq"] == late_seq
        assert other_state["usage_total"] == 55
        print("second process sees the same snapshot and history:", current)

    # /docs must be usable.
    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        docs = client.get("/docs")
        assert docs.status_code == 200
        spec = client.get("/openapi.json").json()
        paths = spec["paths"]
        assert "/api/v1/readings" in paths
        assert "/api/v1/readings/undo" in paths
        assert "/api/v1/meters/{meter_id}" in paths
        assert "/api/v1/history/enable" in paths
        assert "/api/v1/meters/{meter_id}/history/{commit_seq}" in paths
    print("openapi paths ok")
    return 0


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "BASE_URL", "http://localhost:8000"
    )
    second = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("SECOND_URL")
    raise SystemExit(main(base, second))
