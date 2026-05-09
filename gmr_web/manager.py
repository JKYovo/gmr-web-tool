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
from gmr_web.external_backend import (
    default_optimize_output,
    default_optimize_quality_output,
    default_video_output,
    external_backend_enabled,
    external_postprocess_available,
    run_external_postprocess,
    run_external_retarget,
)
from gmr_web.runner import convert_bvh, convert_fbx_offline, convert_gvhmr_pt, convert_smplx_npz


POSTPROCESS_PIPELINES = {"v2", "v2_foot"}
POSTPROCESS_PROFILES = {"preview", "soft", "strict"}


class JobManager:
    def __init__(self, store, job_root=JOB_ROOT):
        self.store = store
        self.job_root = ensure_dir(job_root)
        self._queue = Queue()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
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
        display_name=None,
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
            "display_name": display_name or input_path.name,
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
        self.start()
        self._queue.put(("convert", job_id))
        return job

    def request_postprocess(self, job_id, profile="soft", pipeline="v2_foot", render=True):
        job = self.get_job(job_id)
        if job is None:
            return None
        if job.get("status") != "succeeded":
            raise RuntimeError("Only succeeded GMR jobs can be post-processed.")
        if job.get("postprocess_status") in {"queued", "running"}:
            raise RuntimeError("Postprocess is already queued or running for this job.")
        if not external_postprocess_available():
            raise RuntimeError("Postprocess is provided by gmr-optimizer. Please set GMR_POSTPROCESS_CMD.")
        if profile not in POSTPROCESS_PROFILES:
            raise ValueError(f"Unsupported postprocess profile: {profile}")
        if pipeline not in POSTPROCESS_PIPELINES:
            raise ValueError(f"Unsupported postprocess pipeline: {pipeline}")

        motion_path = Path(job.get("artifacts", {}).get("motion_path") or Path(job["output_dir"]) / "robot_motion.pkl")
        if not motion_path.exists():
            raise FileNotFoundError(f"robot_motion.pkl is required before postprocess: {motion_path}")

        job["postprocess_status"] = "queued"
        job["postprocess_profile"] = profile
        job["postprocess_pipeline"] = pipeline
        job["postprocess_render"] = bool(render)
        job["postprocess_error"] = None
        job["postprocess_started_at"] = None
        job["postprocess_finished_at"] = None
        job["postprocess_artifacts"] = {}
        self._append_log(job, f"[Postprocess] Queued profile={profile}, pipeline={pipeline}, render={bool(render)}.")
        self._save_job(job)
        self.start()
        self._queue.put(("postprocess", job_id))
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
        for pattern in ("motion_*.pkl", "preview_*.mp4", "quality_*.json"):
            for path in output_dir.glob(pattern):
                if path.exists():
                    path.unlink()
        job["status"] = "queued"
        job["started_at"] = None
        job["finished_at"] = None
        job["updated_at"] = utc_now_iso()
        job["artifacts"] = {}
        job["error_summary"] = None
        job["cancel_requested"] = False
        for key in (
            "postprocess_status",
            "postprocess_profile",
            "postprocess_pipeline",
            "postprocess_render",
            "postprocess_error",
            "postprocess_started_at",
            "postprocess_finished_at",
            "postprocess_artifacts",
        ):
            job.pop(key, None)
        self._append_log(job, "[Control] Retry requested.")
        self._save_job(job)
        self.start()
        self._queue.put(("convert", job_id))
        return job

    def _append_log(self, job, message):
        job.setdefault("logs", []).append(message)
        job["updated_at"] = utc_now_iso()

    def _save_job(self, job):
        job["updated_at"] = utc_now_iso()
        self._build_artifacts(job)
        self.store.save_job(job)
        output_dir = ensure_dir(Path(job["output_dir"]))
        write_json(output_dir / "job.json", job)

    def _log_callback(self, job_id):
        def callback(message):
            job = self.get_job(job_id)
            if job is None:
                return
            self._append_log(job, message)
            self.store.save_job(job)
            output_dir = ensure_dir(Path(job["output_dir"]))
            write_json(output_dir / "job.json", job)

        return callback

    def _build_artifacts(self, job):
        output_dir = Path(job["output_dir"])
        motion_path = output_dir / "robot_motion.pkl"
        profile = job.get("postprocess_profile", "soft")
        pipeline = job.get("postprocess_pipeline", "v2_foot")
        candidates = {
            "job_json_path": output_dir / "job.json",
            "motion_path": motion_path,
            "video_path": output_dir / "robot_preview.mp4",
            "optimized_motion_path": default_optimize_output(motion_path, profile, pipeline),
            "optimized_video_path": default_video_output(motion_path, profile, pipeline),
            "quality_report_path": default_optimize_quality_output(motion_path, profile, pipeline),
            "artifacts_zip_path": output_dir / "artifacts.zip",
        }
        job["artifacts"] = {key: str(path) for key, path in candidates.items() if path.exists()}
        postprocess_artifacts = {
            key: job["artifacts"][key]
            for key in ("optimized_motion_path", "optimized_video_path", "quality_report_path")
            if key in job["artifacts"]
        }
        if postprocess_artifacts or job.get("postprocess_status"):
            job["postprocess_artifacts"] = postprocess_artifacts
        return job["artifacts"]

    def _refresh_artifact_zip(self, job):
        output_dir = Path(job["output_dir"])
        artifacts = self._build_artifacts(job)
        artifact_values = [
            path
            for key, path in artifacts.items()
            if key != "artifacts_zip_path" and Path(path).exists()
        ]
        zip_path = zip_artifacts(output_dir, artifact_values)
        artifacts["artifacts_zip_path"] = str(zip_path)
        job["artifacts"] = artifacts
        return artifacts

    def _finalize_success(self, job):
        self._refresh_artifact_zip(job)
        job["status"] = "succeeded"
        job["finished_at"] = utc_now_iso()
        self._save_job(job)

    def _run_job(self, job):
        output_dir = Path(job["output_dir"])
        if external_backend_enabled():
            artifacts = run_external_retarget(job, logger=self._log_callback(job["job_id"]))
        elif job["source_type"] == "gvhmr_smplx":
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

    def _run_postprocess(self, job):
        return run_external_postprocess(job, logger=self._log_callback(job["job_id"]))

    def _run_convert_queue_item(self, job_id):
        job = self.get_job(job_id)
        if job is None or job.get("cancel_requested"):
            return

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

    def _run_postprocess_queue_item(self, job_id):
        job = self.get_job(job_id)
        if job is None:
            return
        if job.get("status") != "succeeded":
            job["postprocess_status"] = "failed"
            job["postprocess_error"] = "Only succeeded GMR jobs can be post-processed."
            self._append_log(job, f"[Postprocess] Failed: {job['postprocess_error']}")
            self._save_job(job)
            return

        job["postprocess_status"] = "running"
        job["postprocess_started_at"] = utc_now_iso()
        job["postprocess_error"] = None
        self._append_log(job, "[Postprocess] Started.")
        self._save_job(job)
        try:
            artifacts = self._run_postprocess(job)
            latest_job = self.get_job(job_id) or job
            latest_job["postprocess_artifacts"] = artifacts
            latest_job["postprocess_status"] = "succeeded"
            latest_job["postprocess_finished_at"] = utc_now_iso()
            latest_job["postprocess_error"] = None
            latest_job["artifacts"].update(artifacts)
            self._append_log(latest_job, "[Postprocess] Completed.")
            self._refresh_artifact_zip(latest_job)
            self._save_job(latest_job)
        except Exception as exc:
            job["postprocess_status"] = "failed"
            job["postprocess_finished_at"] = utc_now_iso()
            job["postprocess_error"] = str(exc)
            self._append_log(job, f"[Postprocess] Failed: {exc}")
            self._save_job(job)

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except Empty:
                continue
            if item is None:
                continue
            if isinstance(item, tuple):
                task_kind, job_id = item
            else:
                task_kind, job_id = "convert", item

            try:
                if task_kind == "postprocess":
                    self._run_postprocess_queue_item(job_id)
                else:
                    self._run_convert_queue_item(job_id)
            except Exception as exc:
                self._mark_worker_error(task_kind, job_id, exc)

    def _mark_worker_error(self, task_kind, job_id, exc):
        job = self.get_job(job_id)
        if job is None:
            return
        if task_kind == "postprocess":
            job["postprocess_status"] = "failed"
            job["postprocess_finished_at"] = utc_now_iso()
            job["postprocess_error"] = str(exc)
            self._append_log(job, f"[Worker] Unhandled postprocess error: {exc}")
        else:
            job["status"] = "failed"
            job["finished_at"] = utc_now_iso()
            job["error_summary"] = str(exc)
            self._append_log(job, f"[Worker] Unhandled GMR error: {exc}")
        try:
            self._save_job(job)
        except Exception:
            self.store.save_job(job)
