"""FastAPI app — endpoints REST + SSE + WebSocket."""

from __future__ import annotations

import asyncio
import json

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session

from app.api.jobs import bus, create_job
from app.api.schemas import JobCreate, JobOut
from app.db.models import Job
from app.db.session import get_session, init_db

app = FastAPI(
    title="auto-post API",
    version="0.3.0",
    description="API para enfileirar processamento de vídeos e acompanhar progresso.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/jobs", response_model=JobOut, status_code=202)
def post_job(payload: JobCreate, db: Session = Depends(get_session)) -> JobOut:
    """Cria um job e retorna imediatamente (202). O pipeline roda em background.

    Use GET /jobs/{id} para status pontual,
    GET /jobs/{id}/events para SSE,
    WS /jobs/{id}/ws para WebSocket.

    Se webhook_url for fornecido, o servidor faz POST com o resultado ao concluir,
    enviando webhook_token no header webhook_header (default: Authorization).
    """
    job_id = create_job(payload)
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=500, detail="Falha ao criar job")
    return JobOut.model_validate(job)


@app.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: Session = Depends(get_session)) -> JobOut:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    return JobOut.model_validate(job)


@app.get("/jobs", response_model=list[JobOut])
def list_jobs(limit: int = 50, db: Session = Depends(get_session)) -> list[JobOut]:
    jobs = db.query(Job).order_by(Job.created_at.desc()).limit(limit).all()
    return [JobOut.model_validate(j) for j in jobs]


@app.get("/jobs/{job_id}/events")
async def job_events(job_id: str, db: Session = Depends(get_session)):
    """Server-Sent Events — emite cada evento de progresso conforme acontece.

    O cliente pode usar EventSource(url) (JS) ou `curl -N <url>`.
    Fecha o stream automaticamente quando o job atinge estado terminal.
    """
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job não encontrado")

    queue = bus.subscribe(job_id)

    async def event_generator():
        # Snapshot inicial
        yield {
            "event": "snapshot",
            "data": json.dumps({
                "job_id": job_id,
                "stage": job.stage,
                "percent": job.progress,
                "message": job.message,
                "status": job.status,
            }),
        }
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": ""}
                    continue
                yield {"event": event.get("stage", "progress"), "data": json.dumps(event)}
                if event.get("stage") in {"done", "error"}:
                    break
        finally:
            bus.unsubscribe(job_id, queue)

    return EventSourceResponse(event_generator())


@app.websocket("/jobs/{job_id}/ws")
async def job_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    with next(get_session()) as db:
        job = db.get(Job, job_id)

    if job is None:
        await websocket.send_json({"error": "Job não encontrado"})
        await websocket.close(code=1008)
        return

    await websocket.send_json({
        "type": "snapshot",
        "job_id": job_id,
        "stage": job.stage,
        "percent": job.progress,
        "message": job.message,
        "status": job.status,
    })

    queue = bus.subscribe(job_id)
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
                continue

            event["type"] = "progress"
            await websocket.send_json(event)
            if event.get("stage") in {"done", "error"}:
                break
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe(job_id, queue)
        try:
            await websocket.close()
        except Exception:
            pass
