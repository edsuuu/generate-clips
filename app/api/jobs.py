"""Gerenciamento de jobs: execução em background + pub/sub de eventos."""

from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Optional

import httpx

from app.api.schemas import JobCreate
from app.db.models import Cut as CutModel, Job, JobStatus
from app.db.session import SessionLocal
from app.pipeline.runner import PipelineOptions, PipelineRunner, ProgressEvent
from app.support.logger import logger


class JobEventBus:
    """Pub/sub em memória para distribuir progresso pra SSE e WebSocket."""

    def __init__(self):
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._lock = threading.Lock()

    def subscribe(self, job_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        with self._lock:
            self._subscribers.setdefault(job_id, []).append(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            subs = self._subscribers.get(job_id, [])
            if queue in subs:
                subs.remove(queue)
            if not subs:
                self._subscribers.pop(job_id, None)

    def publish(self, job_id: str, event: dict) -> None:
        with self._lock:
            subs = list(self._subscribers.get(job_id, []))
        for q in subs:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass


bus = JobEventBus()


def create_job(payload: JobCreate) -> str:
    """Cria o registro no banco e dispara worker em thread separada."""
    job_id = str(uuid.uuid4())

    options_dict = {
        "min_cuts": payload.min_cuts,
        "max_cuts": payload.max_cuts,
        "min_gap": payload.min_gap,
        "no_subtitles": payload.no_subtitles,
        "no_vertical": payload.no_vertical,
        "no_metadata": payload.no_metadata,
        "no_face_tracking": payload.no_face_tracking,
        "no_validate": payload.no_validate,
        "subtitle_only": payload.subtitle_only,
    }

    with SessionLocal() as db:
        job = Job(
            id=job_id,
            url=payload.url,
            status=JobStatus.PENDING,
            progress=0.0,
            stage="pending",
            message="Aguardando início",
            webhook_url=str(payload.webhook_url) if payload.webhook_url else None,
            webhook_token=payload.webhook_token,
            webhook_header=payload.webhook_header or "Authorization",
            llm_provider=payload.llm,
            options=options_dict,
        )
        db.add(job)
        db.commit()

    thread = threading.Thread(target=_run_job, args=(job_id, payload), daemon=True)
    thread.start()

    return job_id


def _run_job(job_id: str, payload: JobCreate) -> None:
    """Executa o pipeline em uma thread separada."""
    logger.info(f"Iniciando job {job_id}: {payload.url}")

    def on_progress(ev: ProgressEvent) -> None:
        payload_dict = ev.to_dict()
        payload_dict["job_id"] = job_id
        payload_dict["ts"] = datetime.utcnow().isoformat() + "Z"
        bus.publish(job_id, payload_dict)
        _save_progress(job_id, ev)

    _update_status(job_id, JobStatus.RUNNING)

    try:
        runner = PipelineRunner(on_progress=on_progress)
        result = runner.run(
            url=payload.url,
            options=PipelineOptions(
                llm=payload.llm,
                min_cuts=payload.min_cuts,
                max_cuts=payload.max_cuts,
                min_gap=payload.min_gap,
                no_subtitles=payload.no_subtitles,
                no_vertical=payload.no_vertical,
                no_metadata=payload.no_metadata,
                no_face_tracking=payload.no_face_tracking,
                no_validate=payload.no_validate,
                subtitle_only=payload.subtitle_only,
            ),
        )
        _save_result(job_id, result)
        _update_status(job_id, JobStatus.COMPLETED, finished=True)
        bus.publish(job_id, {
            "job_id": job_id, "stage": "done", "percent": 100.0,
            "message": "Concluído", "result": result,
            "ts": datetime.utcnow().isoformat() + "Z",
        })
        _fire_webhook(job_id, success=True, result=result)
    except Exception as e:
        logger.exception(f"Job {job_id} falhou")
        _save_error(job_id, str(e))
        _update_status(job_id, JobStatus.FAILED, finished=True)
        bus.publish(job_id, {
            "job_id": job_id, "stage": "error", "percent": 0.0,
            "message": str(e),
            "ts": datetime.utcnow().isoformat() + "Z",
        })
        _fire_webhook(job_id, success=False, error=str(e))


def _save_progress(job_id: str, ev: ProgressEvent) -> None:
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None:
            return
        job.progress = ev.percent
        job.stage = ev.stage
        job.message = ev.message[:500]
        db.commit()


def _save_result(job_id: str, result: dict) -> None:
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None:
            return
        video = result.get("video", {})
        job.video_youtube_id = video.get("youtube_id")
        job.video_title = video.get("title")
        job.video_duration = video.get("duration_seconds")
        job.transcript_text = (
            result.get("transcript", {}).get("text") if "transcript" in result else None
        )
        job.result = result

        for c in result.get("cuts", []):
            db.add(CutModel(
                job_id=job_id,
                index=c["index"], name=c["name"],
                start_seconds=c["start_seconds"], end_seconds=c["end_seconds"],
                duration_seconds=c["duration_seconds"],
                score=c["score"], reason=c.get("reason"),
                video_path=c["video_path"],
                title=c.get("title"), description=c.get("description"),
                hashtags=c.get("hashtags"),
            ))
        db.commit()


def _save_error(job_id: str, message: str) -> None:
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None:
            return
        job.error_message = message
        db.commit()


def _update_status(job_id: str, status: str, finished: bool = False) -> None:
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None:
            return
        job.status = status
        if finished:
            job.finished_at = datetime.utcnow()
        db.commit()


def _fire_webhook(job_id: str, success: bool, result: Optional[dict] = None, error: Optional[str] = None) -> None:
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None or not job.webhook_url:
            return
        url = job.webhook_url
        token = job.webhook_token
        header = job.webhook_header or "Authorization"

    payload = {
        "job_id": job_id,
        "success": success,
        "result": result,
        "error": error,
    }
    headers = {"Content-Type": "application/json"}
    if token:
        headers[header] = token

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            status_text = f"{resp.status_code}"
    except Exception as e:
        status_text = f"error: {e}"

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None:
            return
        job.webhook_status = status_text[:40]
        job.webhook_attempt_at = datetime.utcnow()
        db.commit()
    logger.info(f"Webhook {job_id} -> {url}: {status_text}")
