import shutil
import threading
from pathlib import Path
from queue import Empty, Queue

from gmr_web.common import (
    JOB_ROOT,
    SUPPORTED_ROBOTS,
    TERMINAL_STATUSES,
    ensure_dir,
    infer_source_type,
    make_job_id,
    safe_stem,
    source_config_path,
    short_id,
    stage_input,
    utc_now_iso,
    validate_source_file,
    write_json,
    zip_artifacts,
)
from gmr_web.runner import convert_bvh, convert_fbx_offline, convert_gvhmr_pt, convert_smplx_npz


class JobManager:
    def __init__(self, store, job_root=JOB_ROOT):
        self.store = store
        self.job_root = ensure_dir(job_root)
        self._queue = Queue()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="gmr-web-worker")
            self._thread.start()

    def shutdown(self):
        self._stop_event.set()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def submit_job(
        self,
        *,
        input_file,
        source_type="auto",
        robot="elf3",
        ground_clearance=0.03,
        generate_video=True,
        smoothing_alpha=0.35,
    ):
        input_path = Path(input_file).expanduser().resolve()
        resolved_source_type = infer_source_type(input_path, source_type)
        if robot not in SUPPORTED_ROBOTS:
            raise ValueError(f"Unsupported robot for this internal GMR Web: {robot}")
        validate_source_file(input_path, resolved_source_type)

        job_id = make_job_id()
        output_dir = ensure_dir(self.job_root / f"{safe_stem(input_path)}_{short_id(job_id)}")
        staged_input = stage_input(input_path, output_dir)
        job = {
            "job_id": job_id,
            "status": "queued",
            "source_type": resolved_source_type,
            "robot": robot,
            "ik_config": str(source_config_path(resolved_source_type)),
            "source_input_file": str(input_path),
            "input_file": str(staged_input),
            "output_dir": str(output_dir),
            "ground_clearance": float(ground_clearance),
            "smoothing_alpha": float(smoothing_alpha),
            "generate_video": bool(generate_video),
            "submitted_at": utc_now_iso(),
            "started_at": None,
            "finished_at": None,
            "updated_at": utc_now_iso(),
            "artifacts": {},
            "error_summary": None,
            "logs": [],
            "cancel_requested": False,
        }
        self.store.create_job(job)
        write_json(output_dir / "job.json", job)
        self._queue.put(job_id)
        return job

    def list_jobs(self, limit=50):
        return self.store.list_jobs(limit=limit)

    def get_job(self, job_id):
        return self.store.get_job(job_id)

    def cancel_job(self, job_id):
        job = self.get_job(job_id)
        if job is None:
            return None
        if job["status"] in TERMINAL_STATUSES:
            return job
        job["cancel_requested"] = True
        if job["status"] == "queued":
            job["status"] = "cancelled"
            job["finished_at"] = utc_now_iso()
            job["error_summary"] = "Cancelled before execution."
        self._append_log(job, "[Control] Cancellation requested.")
        self._save_job(job)
        return job

    def retry_job(self, job_id):
        job = self.get_job(job_id)
        if job is None:
            return None
        if job["status"] not in TERMINAL_STATUSES:
            raise RuntimeError("Only terminal jobs can be retried.")
        output_dir = Path(job["output_dir"])
        for name in ("robot_motion.pkl", "robot_preview.mp4", "artifacts.zip"):
            path = output_dir / name
            if path.exists():
                path.unlink()
        job["status"] = "queued"
        job["started_at"] = None
        job["finished_at"] = None
        job["updated_at"] = utc_now_iso()
        job["artifacts"] = {}
        job["error_summary"] = None
        job["cancel_requested"] = False
        self._append_log(job, "[Control] Retry requested.")
        self._save_job(job)
        self._queue.put(job_id)
        return job

    def _append_log(self, job, message):
        job.setdefault("logs", []).append(message)
        job["updated_at"] = utc_now_iso()

    def _save_job(self, job):
        job["updated_at"] = utc_now_iso()
        self._build_artifacts(job)
        self.store.save_job(job)
        write_json(Path(job["output_dir"]) / "job.json", job)

    def _log_callback(self, job_id):
        def callback(message):
            job = self.get_job(job_id)
            if job is None:
                return
            self._append_log(job, message)
            self.store.save_job(job)
            write_json(Path(job["output_dir"]) / "job.json", job)

        return callback

    def _build_artifacts(self, job):
        output_dir = Path(job["output_dir"])
        candidates = {
            "job_json_path": output_dir / "job.json",
            "motion_path": output_dir / "robot_motion.pkl",
            "video_path": output_dir / "robot_preview.mp4",
            "artifacts_zip_path": output_dir / "artifacts.zip",
        }
        job["artifacts"] = {key: str(path) for key, path in candidates.items() if path.exists()}
        return job["artifacts"]

    def _finalize_success(self, job):
        output_dir = Path(job["output_dir"])
        artifacts = self._build_artifacts(job)
        zip_path = zip_artifacts(output_dir, list(artifacts.values()))
        artifacts["artifacts_zip_path"] = str(zip_path)
        job["artifacts"] = artifacts
        job["status"] = "succeeded"
        job["finished_at"] = utc_now_iso()
        self._save_job(job)

    def _run_job(self, job):
        output_dir = Path(job["output_dir"])
        if job["source_type"] == "gvhmr_smplx":
            artifacts = convert_gvhmr_pt(
                job["input_file"],
                output_dir,
                robot=job["robot"],
                ground_clearance=job["ground_clearance"],
                generate_video=job["generate_video"],
                logger=self._log_callback(job["job_id"]),
            )
        elif job["source_type"] == "smplx_npz":
            artifacts = convert_smplx_npz(
                job["input_file"],
                output_dir,
                robot=job["robot"],
                ground_clearance=job["ground_clearance"],
                generate_video=job["generate_video"],
                logger=self._log_callback(job["job_id"]),
            )
        elif job["source_type"] in {"bvh_xsens", "bvh_lafan1", "bvh_nokov"}:
            artifacts = convert_bvh(
                job["input_file"],
                output_dir,
                source_type=job["source_type"],
                robot=job["robot"],
                ground_clearance=job["ground_clearance"],
                smoothing_alpha=job["smoothing_alpha"],
                generate_video=job["generate_video"],
                logger=self._log_callback(job["job_id"]),
            )
        elif job["source_type"] == "fbx_offline":
            artifacts = convert_fbx_offline(
                job["input_file"],
                output_dir,
                robot=job["robot"],
                ground_clearance=job["ground_clearance"],
                generate_video=job["generate_video"],
                logger=self._log_callback(job["job_id"]),
            )
        else:
            raise ValueError(f"Unsupported source type: {job['source_type']}")
        job["artifacts"].update(artifacts)

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except Empty:
                continue
            if job_id is None:
                continue

            job = self.get_job(job_id)
            if job is None or job.get("cancel_requested"):
                continue

            job["status"] = "running"
            job["started_at"] = utc_now_iso()
            self._append_log(job, "[GMR] Job started.")
            self._save_job(job)
            try:
                self._run_job(job)
                latest_job = self.get_job(job_id) or job
                latest_job["artifacts"].update(job.get("artifacts", {}))
                job = latest_job
                if job.get("cancel_requested"):
                    job["status"] = "cancelled"
                    job["finished_at"] = utc_now_iso()
                    job["error_summary"] = "Cancelled after the current stage finished."
                    self._save_job(job)
                else:
                    self._append_log(job, "[GMR] Job completed.")
                    self._finalize_success(job)
            except Exception as exc:
                job["status"] = "failed"
                job["finished_at"] = utc_now_iso()
                job["error_summary"] = str(exc)
                self._append_log(job, f"[Error] {exc}")
                self._save_job(job)
