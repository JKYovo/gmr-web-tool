import json
import sqlite3
import threading
from pathlib import Path

from gmr_web.common import ensure_dir


class SQLiteJobStore:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        ensure_dir(self.db_path.parent)
        self._lock = threading.Lock()
        self._init_db()
        self.fail_stale_running_jobs()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

    def fail_stale_running_jobs(self):
        for job in self.list_jobs(limit=1000):
            if job["status"] in {"queued", "running"}:
                job["status"] = "failed"
                job["finished_at"] = job.get("finished_at") or job.get("updated_at")
                job["error_summary"] = "Service restarted before the job finished."
                self.save_job(job)

    def save_job(self, job):
        payload = json.dumps(job, ensure_ascii=False)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (id, status, updated_at, payload_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    payload_json=excluded.payload_json
                """,
                (job["job_id"], job["status"], job["updated_at"], payload),
            )

    def create_job(self, job):
        self.save_job(job)

    def get_job(self, job_id):
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT payload_json FROM jobs WHERE id=?", (job_id,)).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def list_jobs(self, limit=50):
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM jobs ORDER BY updated_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

