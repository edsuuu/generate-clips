"""Entry point da API HTTP (FastAPI + uvicorn).

Microsserviço stateless de processamento de vídeo, orquestrado pelo Laravel.
Não há mais CLI: a única forma de rodar é subindo a API.

    python main.py                 # sobe a API em settings.api_host:settings.api_port
    AUTO_POST_RELOAD=1 python main.py   # com auto-reload em mudança de código

Host/porta vêm do .env via Settings (API_HOST / API_PORT).
"""

from __future__ import annotations

import os

import uvicorn

from app.support.config import settings


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "sim"}


def run() -> None:
    uvicorn.run(
        "app.api.main:app",
        host=os.environ.get("AUTO_POST_HOST", settings.api_host),
        port=int(os.environ.get("AUTO_POST_PORT", settings.api_port)),
        reload=_env_bool("AUTO_POST_RELOAD", False),
    )


if __name__ == "__main__":
    run()
