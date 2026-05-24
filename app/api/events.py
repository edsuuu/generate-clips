"""Pub/sub em memória para progresso em tempo real (SSE + WebSocket).

Sem banco de dados: o Python é stateless. O estado de cada job vive só em
memória durante o processamento. O Laravel é o dono da persistência e recebe
os resultados via webhook/callback.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from typing import Any

from app.support.logger import logger


class JobEventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._snapshots: dict[str, dict[str, Any]] = {}
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

    def snapshot(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._snapshots.get(job_id)

    def publish(self, job_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            self._snapshots[job_id] = event
            subs = list(self._subscribers.get(job_id, []))
        for q in subs:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass


bus = JobEventBus()


def emit(job_id: str, stage: str, percent: float, message: str = "", **detail: Any) -> dict[str, Any]:
    """Publica um evento de progresso no bus (SSE/WebSocket) e loga no terminal."""
    pct = round(float(percent), 1)
    event = {
        "job_id": job_id,
        "stage": stage,
        "percent": pct,
        "message": message,
        "detail": detail,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    bus.publish(job_id, event)

    bar_len = 24
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    short_job = job_id.split("-")[0]
    logger.info(f"[{short_job}] {bar} {pct:5.1f}% · {stage:<10} {message}")

    return event
