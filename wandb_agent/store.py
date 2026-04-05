"""SQLite persistence for snapshots, diagnoses, and relaunches."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from wandb_agent.poller import Diagnosis, RunSnapshot

_DEFAULT_DB = Path.home() / ".wandb-agent" / "store.db"

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    project     TEXT NOT NULL,
    entity      TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    snapshot_at TEXT NOT NULL,
    metrics_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS diagnoses (
    diagnosis_id     TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL,
    timestamp        TEXT NOT NULL,
    diagnosis_json   TEXT NOT NULL,
    approved         INTEGER,
    rejection_reason TEXT
);

CREATE TABLE IF NOT EXISTS relaunches (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    diagnosis_id TEXT NOT NULL,
    run_id       TEXT NOT NULL,
    launched_at  TEXT NOT NULL,
    pid          INTEGER
);
"""


class RunStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or _DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def save_snapshot(self, snapshot: RunSnapshot) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO runs (run_id, project, entity, config_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    snapshot.run_id,
                    snapshot.project,
                    snapshot.entity,
                    json.dumps(snapshot.config),
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.execute(
                "INSERT INTO snapshots (run_id, snapshot_at, metrics_json) VALUES (?, ?, ?)",
                (
                    snapshot.run_id,
                    snapshot.snapshot_at.isoformat(),
                    json.dumps(snapshot.history),
                ),
            )

    def get_run_info(self, run_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_all_run_ids(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT run_id FROM diagnoses ORDER BY run_id"
            ).fetchall()
        return [row["run_id"] for row in rows]

    # ------------------------------------------------------------------
    # Diagnoses
    # ------------------------------------------------------------------

    def save_diagnosis(self, diagnosis: Diagnosis) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO diagnoses "
                "(diagnosis_id, run_id, timestamp, diagnosis_json, approved, rejection_reason) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    diagnosis.diagnosis_id,
                    diagnosis.run_id,
                    diagnosis.timestamp.isoformat(),
                    diagnosis.model_dump_json(),
                    None if diagnosis.approved is None else int(diagnosis.approved),
                    diagnosis.rejection_reason,
                ),
            )

    def get_diagnosis(self, diagnosis_id: str) -> Diagnosis | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT diagnosis_json FROM diagnoses WHERE diagnosis_id = ?",
                (diagnosis_id,),
            ).fetchone()
        return Diagnosis.model_validate_json(row["diagnosis_json"]) if row else None

    def get_past_diagnoses(self, run_id: str, limit: int = 10) -> list[Diagnosis]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT diagnosis_json FROM diagnoses WHERE run_id = ? "
                "ORDER BY timestamp ASC LIMIT ?",
                (run_id, limit),
            ).fetchall()
        return [Diagnosis.model_validate_json(row["diagnosis_json"]) for row in rows]

    def update_approval(
        self, diagnosis_id: str, approved: bool, reason: str | None
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE diagnoses SET approved = ?, rejection_reason = ? "
                "WHERE diagnosis_id = ?",
                (int(approved), reason, diagnosis_id),
            )

    def get_pending_diagnoses(self) -> list[Diagnosis]:
        """Return diagnoses awaiting approval for stop_and_relaunch."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT diagnosis_json FROM diagnoses WHERE approved IS NULL "
                "ORDER BY timestamp DESC"
            ).fetchall()
        return [
            d
            for row in rows
            if (d := Diagnosis.model_validate_json(row["diagnosis_json"])).suggested_action
            == "stop_and_relaunch"
        ]

    def get_approved_stop_and_relaunch(self) -> list[Diagnosis]:
        """Return approved stop_and_relaunch diagnoses that have not yet been relaunched."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT d.diagnosis_json FROM diagnoses d "
                "LEFT JOIN relaunches r ON d.diagnosis_id = r.diagnosis_id "
                "WHERE d.approved = 1 AND r.id IS NULL "
                "ORDER BY d.timestamp ASC"
            ).fetchall()
        return [
            d
            for row in rows
            if (d := Diagnosis.model_validate_json(row["diagnosis_json"])).suggested_action
            == "stop_and_relaunch"
        ]

    # ------------------------------------------------------------------
    # Relaunches
    # ------------------------------------------------------------------

    def save_relaunch(self, diagnosis_id: str, run_id: str, pid: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO relaunches (diagnosis_id, run_id, launched_at, pid) "
                "VALUES (?, ?, ?, ?)",
                (diagnosis_id, run_id, datetime.utcnow().isoformat(), pid),
            )

    def get_daily_relaunch_count(self, run_id: str | None = None) -> int:
        cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        with self._connect() as conn:
            if run_id:
                row = conn.execute(
                    "SELECT COUNT(*) FROM relaunches WHERE launched_at > ? AND run_id = ?",
                    (cutoff, run_id),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM relaunches WHERE launched_at > ?",
                    (cutoff,),
                ).fetchone()
        return row[0]

    def get_total_relaunch_count(self, run_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM relaunches WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return row[0]
