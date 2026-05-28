from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

import httpx

from app.api.events import emit
from app.api.schemas import (
    IngestVideoRequest,
    RecommendCutsRequest,
    RenderCutsRequest,
    SubtitleFullRequest,
)
from app.llm import get_provider
from app.pipeline.analyzer import Analyzer
from app.pipeline.cutter import Cutter
from app.pipeline.downloader import Downloader
from app.pipeline.hls import HlsPackager
from app.pipeline.metadata import MetadataGenerator
from app.pipeline.subtitler import Subtitler
from app.pipeline.transcriber import Transcriber
from app.storage import MinioStorageProvider
from app.support.config import settings
from app.support.logger import logger
from app.support.types import Highlight, Segment, Transcript, Word


def _job_temp_dir(job_id: str) -> Path:
    path = settings.temp_dir / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cleanup(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _object_path(video_id: str, suffix: str) -> str:
    return f"videos/{video_id}/{suffix.lstrip('/')}"


def _transcript_to_payload(transcript: Transcript) -> dict[str, Any]:
    return {
        "language": transcript.language,
        "duration_seconds": transcript.duration,
        "text": transcript.full_text,
        "segments": [
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": [
                    {"start": word.start, "end": word.end, "text": word.text}
                    for word in segment.words
                ],
            }
            for segment in transcript.segments
        ],
    }


def _transcript_from_payload(
    payload: dict[str, Any], fallback_text: str | None = None
) -> Transcript:
    segments = [
        Segment(
            text=str(segment.get("text", "")),
            start=float(segment.get("start", 0)),
            end=float(segment.get("end", 0)),
            words=[
                Word(
                    text=str(word.get("text", "")),
                    start=float(word.get("start", 0)),
                    end=float(word.get("end", 0)),
                )
                for word in segment.get("words", [])
            ],
        )
        for segment in payload.get("segments", [])
    ]

    if not segments and fallback_text:
        duration = float(payload.get("duration_seconds", 0))
        segments = [Segment(text=fallback_text, start=0.0, end=duration, words=[])]

    return Transcript(
        language=str(payload.get("language", settings.whisper_language)),
        duration=float(payload.get("duration_seconds", payload.get("duration", 0))),
        segments=segments,
    )


def _upload_hls_package(
    storage: MinioStorageProvider,
    video_id: str,
    package_dir: Path,
    prefix: str = "hls",
) -> dict[str, Any]:
    root_prefix = _object_path(video_id, prefix)
    master_file: dict[str, Any] | None = None

    for file_path in sorted(package_dir.rglob("*")):
        if not file_path.is_file():
            continue

        relative = file_path.relative_to(package_dir).as_posix()
        object_path = f"{root_prefix}/{relative}"
        content_type = (
            "application/vnd.apple.mpegurl" if file_path.suffix == ".m3u8" else "video/mp2t"
        )
        stored = storage.upload_file(
            file_path,
            object_path,
            file_type="hls_master" if relative == "master.m3u8" else "hls_asset",
            content_type=content_type,
        ).to_dict()
        if relative == "master.m3u8":
            master_file = stored

    if master_file is None:
        raise RuntimeError("Pacote HLS gerado sem master.m3u8")

    return master_file


def _post_webhook(
    callback_url: Any,
    token: str | None,
    header: str | None,
    payload: dict[str, Any],
    *,
    raise_on_error: bool | None = None,
) -> bool:
    if callback_url is None:
        return False

    headers = {"Content-Type": "application/json"}
    if token:
        headers[header or "Authorization"] = token

    should_raise = settings.webhook_fail_job_on_error if raise_on_error is None else raise_on_error
    event_name = str(payload.get("event", "unknown"))

    try:
        logger.info(f"[webhook] POST {callback_url} event={event_name}")
        with httpx.Client(timeout=settings.webhook_timeout_seconds) as client:
            response = client.post(str(callback_url), json=payload, headers=headers)
            response.raise_for_status()
        logger.info(
            f"[webhook] entregue com sucesso event={event_name} status={response.status_code}"
        )
        return True
    except Exception as exc:
        logger.warning(f"[webhook] falhou event={event_name} url={callback_url}: {exc}")
        if should_raise:
            raise
        return False


def _failure_payload(job_id: str, video_id: str, event: str, error: Exception) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "video_id": video_id,
        "event": event,
        "status": "failed",
        "message": str(error),
        "payloads": [
            {
                "type": "python_error",
                "payload": {"error": str(error), "event": event},
            }
        ],
    }


def _validate_transcript(audio_path: Path, transcript: Transcript) -> Transcript:
    from app.pipeline.validator import TranscriptValidator

    validator = TranscriptValidator()
    return validator.validate(audio_path, transcript)


def ingest_video(job_id: str, payload: IngestVideoRequest) -> dict[str, Any]:
    logger.info(f"[ingest {job_id}] video_id={payload.video_id} url={payload.url}")
    temp_dir = _job_temp_dir(job_id)
    storage = MinioStorageProvider()
    try:
        emit(job_id, "download", 2, "Baixando vídeo...")
        downloader = Downloader(output_dir=temp_dir)
        # Download ocupa a banda 2%-25%
        video = downloader.download(
            payload.url,
            on_progress=lambda p: emit(job_id, "download", 2 + p * 0.23, "Baixando vídeo..."),
        )

        files = []
        if payload.options.upload_original_to_minio:
            emit(job_id, "upload", 27, "Enviando original ao MinIO...")
            original_path = _object_path(
                payload.video_id, f"original/source{video.file_path.suffix}"
            )
            files.append(
                storage.upload_file(video.file_path, original_path, file_type="original").to_dict()
            )

            emit(job_id, "hls", 32, "Empacotando vídeo em HLS...")
            hls_dir = HlsPackager().package(
                video.file_path,
                temp_dir / "hls",
                total_seconds=video.duration,
                on_progress=lambda p: emit(job_id, "hls", 32 + p * 0.13, "Gerando HLS..."),
            )
            emit(job_id, "upload", 46, "Enviando HLS ao MinIO...")
            files.append(_upload_hls_package(storage, payload.video_id, hls_dir))

        transcript = None
        validated = None
        if payload.options.transcribe:
            emit(job_id, "transcribe", 48, "Transcrevendo áudio...")
            # Transcrição ocupa a banda 48%-88%
            transcript = Transcriber().transcribe(
                video.file_path,
                on_progress=lambda p: emit(
                    job_id, "transcribe", 48 + p * 0.40, "Transcrevendo áudio..."
                ),
            )
            audio_path = video.file_path.with_suffix(".wav")
            if audio_path.exists():
                files.append(
                    storage.upload_file(
                        audio_path,
                        _object_path(payload.video_id, "audio/source.wav"),
                        file_type="audio",
                        content_type="audio/wav",
                    ).to_dict()
                )

            if payload.options.validate_transcript:
                emit(job_id, "validate", 90, "Validando transcrição com áudio...")
                try:
                    validated = _validate_transcript(audio_path, transcript)
                except Exception as exc:
                    logger.warning(f"Validacao de transcricao falhou: {exc}")

        raw_payload = _transcript_to_payload(transcript) if transcript else None
        validated_payload = _transcript_to_payload(validated) if validated else None
        payloads: list[dict[str, Any]] = [
            {
                "type": "ingest_result",
                "payload": {
                    "video": {
                        "external_video_id": video.video_id,
                        "url": video.url,
                        "title": video.title,
                        "duration_seconds": video.duration,
                    }
                },
            },
        ]
        if raw_payload:
            payloads.append({"type": "transcript_raw", "payload": raw_payload})
        if validated_payload:
            payloads.append({"type": "transcript_validated", "payload": validated_payload})
        result = {
            "job_id": job_id,
            "video_id": payload.video_id,
            "event": "ingest.completed",
            "status": "waiting_transcript_review" if transcript else "downloaded",
            "message": "Transcricao finalizada aguardando revisao"
            if transcript
            else "Download finalizado",
            "video": {
                "external_video_id": video.video_id,
                "url": video.url,
                "title": video.title,
                "duration_seconds": video.duration,
            },
            "files": files,
            "transcript": {
                "language": transcript.language if transcript else None,
                "duration_seconds": transcript.duration if transcript else None,
                "raw_text": transcript.full_text if transcript else None,
                "validated_text": validated.full_text if validated else None,
                "is_validated_by_ai": validated is not None,
            },
            "payloads": payloads,
        }

        emit(job_id, "done", 100, "Ingestão concluída")
        _post_webhook(
            payload.callback_url,
            payload.callback_token,
            payload.callback_header,
            result,
        )
        return result
    except Exception as exc:
        emit(job_id, "error", 0, str(exc))
        failure = _failure_payload(job_id, payload.video_id, "ingest.failed", exc)
        _post_webhook(
            payload.callback_url,
            payload.callback_token,
            payload.callback_header,
            failure,
            raise_on_error=False,
        )
        raise
    finally:
        _cleanup(temp_dir)


def subtitle_full_video(job_id: str, video_id: str, payload: SubtitleFullRequest) -> dict[str, Any]:
    logger.info(f"[subtitle_full {job_id}] video_id={video_id}")
    temp_dir = _job_temp_dir(job_id)
    storage = MinioStorageProvider()
    try:
        emit(job_id, "download", 10, "Baixando vídeo do MinIO...")
        source_ext = Path(payload.source_file.path).suffix or ".mp4"
        source_path = storage.download_file(
            payload.source_file.path, temp_dir / f"source{source_ext}"
        )
        transcript = _transcript_from_payload(payload.transcript_json, payload.transcript_text)
        out_path = temp_dir / "legendado.mp4"

        emit(job_id, "subtitle", 40, "Queimando legendas no vídeo completo...")
        # Burn de legenda ocupa a banda 40%-95%
        Subtitler().burn_subtitles(
            source_path,
            transcript,
            Highlight(start=0.0, end=transcript.duration),
            out_path,
            on_progress=lambda p: emit(job_id, "subtitle", 40 + p * 0.55, "Queimando legendas..."),
        )
        emit(job_id, "hls", 95, "Empacotando vídeo legendado em HLS...")
        hls_dir = HlsPackager().package(
            out_path,
            temp_dir / "hls",
            total_seconds=transcript.duration,
            on_progress=lambda p: emit(job_id, "hls", 95 + p * 0.03, "Gerando HLS legendado..."),
        )
        emit(job_id, "upload", 98, "Enviando vídeo legendado ao MinIO...")

        stored = storage.upload_file(
            out_path,
            payload.output.path,
            file_type="legendado",
            content_type="video/mp4",
        ).to_dict()
        hls_master = _upload_hls_package(storage, video_id, hls_dir, prefix="hls-subtitled")
        result = {
            "job_id": job_id,
            "video_id": video_id,
            "event": "subtitle_full.completed",
            "status": "full_subtitled",
            "message": "Video completo legendado",
            "file": stored,
            "files": [stored, hls_master],
            "payloads": [{"type": "subtitle_result", "payload": {"file": stored}}],
        }
        emit(job_id, "done", 100, "Vídeo legendado")
        _post_webhook(
            payload.callback_url,
            payload.callback_token,
            payload.callback_header,
            result,
        )
        return result
    except Exception as exc:
        emit(job_id, "error", 0, str(exc))
        failure = _failure_payload(job_id, video_id, "subtitle_full.failed", exc)
        _post_webhook(
            payload.callback_url,
            payload.callback_token,
            payload.callback_header,
            failure,
            raise_on_error=False,
        )
        raise
    finally:
        _cleanup(temp_dir)


def recommend_cuts(video_id: str, payload: RecommendCutsRequest) -> dict[str, Any]:
    transcript = _transcript_from_payload(payload.transcript_json, payload.transcript_text)
    provider = get_provider(payload.llm)
    analyzer = Analyzer(
        provider,
        min_cuts=payload.constraints.min_cuts,
        max_cuts=payload.constraints.max_cuts,
        min_gap=payload.constraints.min_gap,
    )
    highlights = analyzer.select_highlights(transcript)
    cuts = [
        {
            "index": index,
            "name": f"PT{index}",
            "type": f"pt{index}",
            "source": "ai",
            "start_seconds": highlight.start,
            "end_seconds": highlight.end,
            "duration_seconds": highlight.duration,
            "score": highlight.score,
            "reason": highlight.reason,
        }
        for index, highlight in enumerate(highlights, start=1)
    ]
    result = {
        "video_id": video_id,
        "event": "recommend_cuts.completed",
        "status": "waiting_cuts",
        "cuts": cuts,
        "payloads": [
            {
                "type": "cuts_recommendation_result",
                "payload": {"video": payload.video or {}, "cuts": cuts},
            }
        ],
    }
    _post_webhook(
        payload.callback_url,
        payload.callback_token,
        payload.callback_header,
        result,
    )
    return result


def render_cuts(job_id: str, video_id: str, payload: RenderCutsRequest) -> dict[str, Any]:
    temp_dir = _job_temp_dir(job_id)
    storage = MinioStorageProvider()
    try:
        emit(job_id, "start", 1, "Preparando renderização...")
        source_ext = Path(payload.source_file.path).suffix or ".mp4"
        emit(job_id, "download", 3, "Baixando vídeo do MinIO...")
        source_path = storage.download_file(
            payload.source_file.path, temp_dir / f"source{source_ext}"
        )
        transcript = _transcript_from_payload(payload.transcript_json)
        out_dir = temp_dir / "cuts"
        out_dir.mkdir(parents=True, exist_ok=True)
        files = []

        # Gera título/descrição/hashtags por corte (alimenta a tela de agendamento no Laravel).
        metadata_gen = None
        video_title = ""
        if payload.video and isinstance(payload.video, dict):
            video_title = str(payload.video.get("title") or "")
        if payload.generate_metadata:
            try:
                metadata_gen = MetadataGenerator(get_provider(payload.llm))
            except Exception as exc:
                logger.warning(f"MetadataGenerator indisponível ({exc}); cortes sem metadados")

        # Reaproveita um único FaceTracker entre cortes (carregar modelos é caro).
        face_tracker = None
        if (
            any(c.vertical and c.face_tracking for c in payload.cuts)
            and settings.face_tracking_enabled
        ):
            try:
                from app.pipeline.face_tracker import FaceTracker

                face_tracker = FaceTracker()
            except Exception as exc:
                logger.warning(f"FaceTracker indisponivel ({exc}); usando crop estatico")

        total = len(payload.cuts)
        for idx, cut in enumerate(payload.cuts, start=1):
            logger.info(
                f"Renderizando corte {cut.name} ({cut.start_seconds:.1f}s-{cut.end_seconds:.1f}s)"
            )
            emit(
                job_id,
                "cut",
                round(idx / max(total, 1) * 90, 1),
                f"Renderizando {cut.name} ({idx}/{total})",
                index=idx,
                total=total,
            )
            out_path = out_dir / f"{cut.type}.mp4"
            cutter = Cutter(
                output_dir=out_dir,
                vertical=cut.vertical,
                face_tracker=face_tracker if cut.face_tracking else None,
            )
            cutter._cut_one(
                source_path,
                Highlight(start=cut.start_seconds, end=cut.end_seconds),
                out_path,
            )

            subtitled_path = out_dir / f"{cut.type}.subtitled.mp4"
            Subtitler().burn_subtitles(
                out_path,
                transcript,
                Highlight(start=cut.start_seconds, end=cut.end_seconds),
                subtitled_path,
            )
            out_path.unlink(missing_ok=True)
            subtitled_path.rename(out_path)

            stored = storage.upload_file(
                out_path,
                cut.output_path,
                file_type=cut.type,
                content_type="video/mp4",
            ).to_dict()
            stored["cut_id"] = cut.cut_id

            if metadata_gen is not None:
                try:
                    stored.update(
                        metadata_gen.generate(
                            video_title,
                            transcript,
                            Highlight(start=cut.start_seconds, end=cut.end_seconds),
                        )
                    )
                except Exception as exc:
                    logger.warning(f"Metadados do corte {cut.name} falharam: {exc}")

            files.append(stored)

        result = {
            "job_id": job_id,
            "video_id": video_id,
            "event": "render_cuts.completed",
            "status": "completed",
            "message": "Cortes renderizados",
            "files": files,
            "payloads": [{"type": "render_result", "payload": {"files": files}}],
        }
        emit(job_id, "done", 100, "Cortes renderizados")
        _post_webhook(
            payload.callback_url,
            payload.callback_token,
            payload.callback_header,
            result,
        )
        return result
    except Exception as exc:
        emit(job_id, "error", 0, str(exc))
        failure = _failure_payload(job_id, video_id, "render_cuts.failed", exc)
        _post_webhook(
            payload.callback_url,
            payload.callback_token,
            payload.callback_header,
            failure,
            raise_on_error=False,
        )
        raise
    finally:
        _cleanup(temp_dir)


def new_job_id() -> str:
    return str(uuid.uuid4())
