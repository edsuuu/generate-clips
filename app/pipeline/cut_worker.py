from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.pipeline.cutter import Cutter
from app.pipeline.face_tracker import FaceTracker
from app.pipeline.subtitler import Subtitler
from app.pipeline.workflows.video import _transcript_from_payload
from app.support.config import settings
from app.support.logger import logger
from app.support.types import Highlight
from app.api.schemas import RenderCutRequest


def render_cut(
    source_path: Path,
    transcript_payload: dict[str, Any],
    cut_payload: dict[str, Any],
    out_path: Path,
) -> None:
    cut = RenderCutRequest(**cut_payload)
    transcript = _transcript_from_payload(transcript_payload)
    highlight = Highlight(start=cut.start_seconds, end=cut.end_seconds)

    face_tracker = None
    if cut.vertical and cut.face_tracking and settings.face_tracking_enabled:
        try:
            face_tracker = FaceTracker()
        except Exception as exc:
            logger.warning(f"FaceTracker indisponivel ({exc}); usando crop estatico")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_path.with_suffix(".raw.mp4")
    subtitled_path = out_path.with_suffix(".subtitled.mp4")
    raw_path.unlink(missing_ok=True)
    subtitled_path.unlink(missing_ok=True)
    out_path.unlink(missing_ok=True)

    cutter = Cutter(
        output_dir=out_path.parent,
        vertical=cut.vertical,
        face_tracker=face_tracker if cut.face_tracking else None,
    )
    cutter._cut_one(source_path, highlight, raw_path)

    Subtitler().burn_subtitles(raw_path, transcript, highlight, subtitled_path)
    raw_path.unlink(missing_ok=True)
    subtitled_path.replace(out_path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Renderiza um corte em processo isolado.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--transcript-json", required=True, type=Path)
    parser.add_argument("--cut-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        render_cut(
            args.source,
            _load_json(args.transcript_json),
            _load_json(args.cut_json),
            args.output,
        )
        return 0
    except Exception as exc:
        logger.exception(f"Falha ao renderizar corte em worker: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
