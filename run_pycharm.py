"""Launcher simples para rodar o projeto pelo botão Play do PyCharm.

Default:
- Sobe a API HTTP (`serve`)

Opcional:
- Defina AUTO_POST_RUN_MODE=process para rodar um processamento único.
- Para process, informe também AUTO_POST_URL.
"""

from __future__ import annotations

import os
from pathlib import Path

from main import process, serve, version
from app.support.config import settings


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "sim"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


def _env_path(name: str, default: Path | None) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return default
    return Path(value)


def main() -> None:
    mode = os.environ.get("AUTO_POST_RUN_MODE", "serve").strip().lower()

    if mode == "serve":
        serve(
            host=os.environ.get("AUTO_POST_HOST", settings.api_host),
            port=_env_int("AUTO_POST_PORT", settings.api_port),
            reload=_env_bool("AUTO_POST_RELOAD", False),
        )
        return

    if mode == "process":
        url = os.environ.get("AUTO_POST_URL", "").strip()
        if not url:
            raise SystemExit(
                "AUTO_POST_URL nao definido. "
                "No PyCharm, configure a variavel de ambiente AUTO_POST_URL com a URL do YouTube."
            )

        process(
            url=url,
            llm=os.environ.get("AUTO_POST_LLM", settings.llm_provider),
            output_dir=_env_path("AUTO_POST_OUTPUT_DIR", settings.output_dir) or settings.output_dir,
            min_cuts=_env_int("AUTO_POST_MIN_CUTS", settings.min_cuts),
            max_cuts=_env_int("AUTO_POST_MAX_CUTS", settings.max_cuts),
            min_gap=_env_float("AUTO_POST_MIN_GAP", settings.min_gap_between_cuts),
            no_subtitles=_env_bool("AUTO_POST_NO_SUBTITLES", False),
            no_vertical=_env_bool("AUTO_POST_NO_VERTICAL", False),
            no_metadata=_env_bool("AUTO_POST_NO_METADATA", False),
            no_face_tracking=_env_bool("AUTO_POST_NO_FACE_TRACKING", False),
            no_validate=_env_bool("AUTO_POST_NO_VALIDATE", False),
            subtitle_only=_env_bool("AUTO_POST_SUBTITLE_ONLY", False),
            json_result=_env_path("AUTO_POST_JSON_RESULT", None),
        )
        return

    if mode == "version":
        version()
        return

    raise SystemExit(
        f"AUTO_POST_RUN_MODE={mode!r} invalido. Use: serve, process ou version."
    )


if __name__ == "__main__":
    main()
