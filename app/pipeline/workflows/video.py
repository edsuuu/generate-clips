from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
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
from app.pipeline.downloader import Downloader
from app.pipeline.hls import HlsPackager
from app.pipeline.metadata import MetadataGenerator
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


def _cut_worker_env(overrides: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONFAULTHANDLER", "1")
    if overrides:
        env.update(overrides)
    return env


def _render_cut_in_worker(
    source_path: Path,
    transcript_path: Path,
    cut_path: Path,
    out_path: Path,
    *,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "app.pipeline.cut_worker",
        "--source",
        str(source_path),
        "--transcript-json",
        str(transcript_path),
        "--cut-json",
        str(cut_path),
        "--output",
        str(out_path),
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=_cut_worker_env(env_overrides),
    )


def _retry_cut_env() -> dict[str, str]:
    return {
        "FACE_TRACKING_ENABLED": "false",
        "FACE_TRACKING_DELEGATE": "cpu",
        "FFMPEG_HWACCEL": "none",
        "FFMPEG_ENCODER": "libx264",
    }


def _render_cut_with_retry(
    source_path: Path,
    transcript_path: Path,
    cut_data: dict[str, Any],
    out_path: Path,
    *,
    force_safe_mode: bool = False,
) -> bool:
    cut_path = out_path.with_suffix(".cut.json")
    cut_path.write_text(json.dumps(cut_data), encoding="utf-8")

    attempts: list[tuple[str, dict[str, str] | None]]
    if force_safe_mode:
        attempts = [("safe", _retry_cut_env())]
    else:
        attempts = [("normal", None), ("safe", _retry_cut_env())]

    last_result: subprocess.CompletedProcess[str] | None = None
    try:
        for attempt_index, (label, overrides) in enumerate(attempts, start=1):
            out_path.unlink(missing_ok=True)
            result = _render_cut_in_worker(
                source_path,
                transcript_path,
                cut_path,
                out_path,
                env_overrides=overrides,
            )
            last_result = result

            if result.returncode == 0 and out_path.exists():
                return force_safe_mode or attempt_index > 1

            crash_info = f"code {result.returncode}"
            if result.returncode < 0:
                crash_info = f"sinal {-result.returncode}"
            logger.warning(
                f"Worker do corte falhou ({label}, {crash_info}). "
                f"{'Reiniciando em modo seguro...' if attempt_index == 1 else 'Sem novas tentativas.'}"
            )
            if result.stdout.strip():
                logger.warning(f"[cut-worker stdout] {result.stdout.strip()[-2000:]}")
            if result.stderr.strip():
                logger.warning(f"[cut-worker stderr] {result.stderr.strip()[-2000:]}")
            if attempt_index == 1 and not force_safe_mode:
                logger.warning(
                    "O restante dos cortes deste job será renderizado em modo seguro."
                )

        stderr = ""
        if last_result is not None and last_result.stderr:
            stderr = last_result.stderr.strip()
        raise RuntimeError(
            f"Falha ao renderizar corte {cut_data.get('name')!r} após 2 tentativas. "
            f"{stderr[-1500:]}"
        )
    finally:
        cut_path.unlink(missing_ok=True)


def _generate_thumbnail(
    video_path: Path,
    video_id: str,
    duration: float,
    storage: MinioStorageProvider,
    temp_dir: Path,
) -> dict[str, Any] | None:
    """Extrai um frame do vídeo e sobe para o MinIO como thumbnail JPEG."""
    seek = min(2.0, max(0.0, duration * 0.05))
    thumb_local = temp_dir / "thumbnail_cover.jpg"
    object_path = _object_path(video_id, "thumbnail/cover.jpg")
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-ss", f"{seek:.3f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-vf", "scale=960:-2",
            "-q:v", "2",
            str(thumb_local),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0 or not thumb_local.exists():
            logger.warning(f"Thumbnail: ffmpeg falhou — {result.stderr.strip()[:300]}")
            return None
        stored = storage.upload_file(
            thumb_local, object_path, file_type="thumbnail", content_type="image/jpeg"
        )
        return stored.to_dict()
    except Exception as exc:
        logger.warning(f"Thumbnail: falha ignorada — {exc}")
        return None
    finally:
        thumb_local.unlink(missing_ok=True)


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

            emit(job_id, "thumbnail", 29, "Gerando thumbnail...")
            thumb = _generate_thumbnail(
                video.file_path, payload.video_id, video.duration, storage, temp_dir
            )
            if thumb:
                files.append(thumb)

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
            "status": "waiting_cuts" if transcript else "downloaded",
            "message": "Transcricao finalizada, pronto para cortes"
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

        total = len(payload.cuts)
        transcript_payload = _transcript_to_payload(transcript)
        transcript_path = temp_dir / "transcript.json"
        transcript_path.write_text(json.dumps(transcript_payload), encoding="utf-8")
        force_safe_mode = False
        failed_cuts: list[str] = []
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
            try:
                force_safe_mode = _render_cut_with_retry(
                    source_path,
                    transcript_path,
                    cut.model_dump(),
                    out_path,
                    force_safe_mode=force_safe_mode,
                )
            except Exception as cut_exc:
                logger.error(
                    f"Corte {cut.name} falhou após todas as tentativas e será ignorado: {cut_exc}"
                )
                failed_cuts.append(cut.name)
                force_safe_mode = True  # cortes restantes em modo seguro
                continue

            stored = storage.upload_file(
                out_path,
                cut.output_path,
                file_type=cut.type,
                content_type="video/mp4",
            ).to_dict()
            stored["cut_id"] = cut.cut_id
            out_path.unlink(missing_ok=True)

            # Metadados: propaga os existentes (re-render) sem chamar a IA;
            # só gera via IA se o corte ainda não tiver título nem descrição.
            has_existing_meta = bool((cut.title or "").strip() or (cut.description or "").strip())
            if has_existing_meta:
                stored["title"] = cut.title or ""
                stored["description"] = cut.description or ""
                stored["hashtags"] = cut.hashtags or []
                logger.info(f"Metadados do corte {cut.name} reaproveitados (re-render).")
            elif metadata_gen is not None:
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

        if failed_cuts:
            logger.warning(f"Cortes que falharam e foram ignorados: {', '.join(failed_cuts)}")

        status = "completed" if not failed_cuts else "completed_partial"
        message = "Cortes renderizados" if not failed_cuts else (
            f"Cortes renderizados com falhas: {', '.join(failed_cuts)} não foram gerados"
        )
        result = {
            "job_id": job_id,
            "video_id": video_id,
            "event": "render_cuts.completed",
            "status": status,
            "message": message,
            "files": files,
            "failed_cuts": failed_cuts,
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
