"""FastAPI app — microsserviço stateless de processamento de vídeo.

O Python NÃO persiste nada em banco. O Laravel é o dono do banco e da
orquestração. Aqui só processamos arquivos, salvamos no MinIO e devolvemos
resultados via webhook/callback. Progresso em tempo real é relay em memória
(SSE/WebSocket), sem persistência.
"""

from __future__ import annotations

import asyncio
import json
import threading

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from app.api.events import bus
from app.api.schemas import (
    AcceptedJobOut,
    IngestVideoRequest,
    RecommendCutsRequest,
    RenderCutsRequest,
    SubtitleFullRequest,
)
from app.pipeline.workflows import (
    ingest_video,
    recommend_cuts,
    render_cuts,
    subtitle_full_video,
)
from app.pipeline.workflows.video import new_job_id
from app.support.config import settings
from app.support.logger import logger

app = FastAPI(
    title="auto-post API",
    version="0.4.0",
    description="Microsserviço stateless de processamento de vídeo (orquestrado pelo Laravel).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


def _require_python_token(authorization: str | None = Header(default=None)) -> None:
    if not settings.python_api_token:
        return
    accepted = {settings.python_api_token, f"Bearer {settings.python_api_token}"}
    if authorization not in accepted:
        raise HTTPException(status_code=401, detail="Token invalido")


def _run_background(target, *args) -> str:
    job_id = new_job_id()

    def runner() -> None:
        try:
            target(job_id, *args)
        except Exception:
            logger.exception(f"Workflow {job_id} falhou")

    threading.Thread(target=runner, daemon=True).start()
    return job_id


@app.post(
    "/videos/ingest",
    response_model=AcceptedJobOut,
    status_code=202,
    dependencies=[Depends(_require_python_token)],
)
def post_video_ingest(payload: IngestVideoRequest) -> AcceptedJobOut:
    job_id = _run_background(ingest_video, payload)
    return AcceptedJobOut(job_id=job_id)


@app.post(
    "/videos/{video_id}/subtitle-full",
    response_model=AcceptedJobOut,
    status_code=202,
    dependencies=[Depends(_require_python_token)],
)
def post_subtitle_full(video_id: str, payload: SubtitleFullRequest) -> AcceptedJobOut:
    job_id = _run_background(subtitle_full_video, video_id, payload)
    return AcceptedJobOut(job_id=job_id)


@app.post(
    "/videos/{video_id}/recommend-cuts",
    dependencies=[Depends(_require_python_token)],
)
def post_recommend_cuts(video_id: str, payload: RecommendCutsRequest):
    # Síncrono: a recomendação via LLM é rápida e o Laravel quer a resposta na hora.
    return recommend_cuts(video_id, payload)


@app.post(
    "/videos/{video_id}/render-cuts",
    response_model=AcceptedJobOut,
    status_code=202,
    dependencies=[Depends(_require_python_token)],
)
def post_render_cuts(video_id: str, payload: RenderCutsRequest) -> AcceptedJobOut:
    job_id = _run_background(render_cuts, video_id, payload)
    return AcceptedJobOut(job_id=job_id)


@app.get("/jobs/{job_id}/events")
async def job_events(job_id: str):
    """SSE — progresso em tempo real do job (relay em memória, sem DB).

    Cliente: EventSource(url) no browser ou `curl -N <url>`.
    """
    queue = bus.subscribe(job_id)

    async def event_generator():
        snap = bus.snapshot(job_id)
        if snap is not None:
            yield {"event": "snapshot", "data": json.dumps(snap)}
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

    snap = bus.snapshot(job_id)
    if snap is not None:
        await websocket.send_json({"type": "snapshot", **snap})

    queue = bus.subscribe(job_id)
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
                continue
            await websocket.send_json({"type": "progress", **event})
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
