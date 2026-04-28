import argparse
from contextlib import asynccontextmanager
from pathlib import Path

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

from gmr_web.common import DB_PATH, DEFAULT_HOST, DEFAULT_PORT, JOB_ROOT, ensure_dir
from gmr_web.manager import JobManager
from gmr_web.store import SQLiteJobStore
from gmr_web.ui import build_ui


class JobCreateRequest(BaseModel):
    input_file: str
    source_type: str = "auto"
    robot: str = "elf3"
    ground_clearance: float = 0.03
    generate_video: bool = True
    smoothing_alpha: float = 0.35


def create_components():
    ensure_dir(JOB_ROOT)
    store = SQLiteJobStore(DB_PATH)
    manager = JobManager(store, job_root=JOB_ROOT)
    return store, manager


def create_app():
    store, manager = create_components()

    @asynccontextmanager
    async def lifespan(app):
        manager.start()
        try:
            yield
        finally:
            manager.shutdown()

    app = FastAPI(title="GMR Web Service", lifespan=lifespan)
    app.state.store = store
    app.state.manager = manager

    @app.get("/health")
    def health():
        return {"status": "ok", "job_root": str(JOB_ROOT), "db_path": str(DB_PATH)}

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/ui")

    @app.post("/jobs")
    def create_job(request: JobCreateRequest):
        try:
            return manager.submit_job(**request.model_dump())
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/jobs")
    def list_jobs(limit: int = 50):
        return manager.list_jobs(limit=limit)

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        job = manager.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return job

    @app.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        job = manager.cancel_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return job

    @app.post("/jobs/{job_id}/retry")
    def retry_job(job_id: str):
        try:
            job = manager.retry_job(job_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return job

    @app.get("/jobs/{job_id}/artifacts")
    def download_artifacts(job_id: str):
        job = manager.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        artifact_path = job.get("artifacts", {}).get("artifacts_zip_path")
        if not artifact_path or not Path(artifact_path).exists():
            raise HTTPException(status_code=404, detail=f"No artifact bundle for job: {job_id}")
        return FileResponse(artifact_path, filename=f"{job_id}_artifacts.zip")

    app = gr.mount_gradio_app(app, build_ui(manager), path="/ui")
    return app


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


def main():
    args = parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

