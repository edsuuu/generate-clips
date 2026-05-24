"""Orquestrador do pipeline com callback de progresso.

Uso típico (CLI ou API):

    runner = PipelineRunner()
    result = runner.run(
        url="https://...",
        on_progress=lambda ev: print(ev),
    )
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from app.llm.factory import get_provider
from app.pipeline.analyzer import Analyzer
from app.pipeline.cutter import Cutter
from app.pipeline.downloader import Downloader
from app.pipeline.metadata import MetadataGenerator
from app.pipeline.subtitler import Subtitler
from app.pipeline.transcriber import Transcriber
from app.support.config import settings
from app.support.logger import logger
from app.support.types import Cut, Highlight


# Pesos de cada etapa no progresso global (somam 100).
WEIGHTS = {
    "download": 5,
    "transcribe": 22,
    "validate": 5,
    "analyze": 3,
    "face_track": 15,
    "cut": 15,
    "subtitle": 20,
    "metadata": 15,
}


@dataclass
class ProgressEvent:
    stage: str
    percent: float
    message: str = ""
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "percent": round(self.percent, 1),
            "message": self.message,
            "detail": self.detail,
        }


ProgressCallback = Callable[[ProgressEvent], None]


@dataclass
class PipelineOptions:
    llm: str | None = None
    output_dir: Path | None = None
    min_cuts: int | None = None
    max_cuts: int | None = None
    min_gap: float | None = None
    no_subtitles: bool = False
    no_vertical: bool = False
    no_metadata: bool = False
    no_face_tracking: bool = False
    no_validate: bool = False
    subtitle_only: bool = False


class PipelineRunner:
    def __init__(self, on_progress: ProgressCallback | None = None):
        self._cb = on_progress or (lambda ev: None)
        self._total = 0.0  # acumulador 0-100

    def _emit(self, stage: str, message: str = "", **detail) -> None:
        ev = ProgressEvent(
            stage=stage,
            percent=min(100.0, self._total),
            message=message,
            detail=detail,
        )
        self._cb(ev)

    def _advance(self, stage: str, fraction: float, message: str = "", **detail) -> None:
        """Avança o progresso global pelo peso da stage * fraction (0-1)."""
        weight = WEIGHTS.get(stage, 0)
        self._total = min(100.0, self._total + weight * fraction)
        self._emit(stage, message=message, **detail)

    def run(self, url: str, options: PipelineOptions | None = None) -> dict:
        opts = options or PipelineOptions()

        self._total = 0.0
        self._emit("start", message=f"Iniciando processamento de {url}")

        output_dir = opts.output_dir or settings.output_dir

        # 1. Download
        downloader = Downloader(output_dir=output_dir)
        self._emit("download", message="Baixando vídeo (melhor qualidade disponível)...")
        video = downloader.download(url)
        self._advance("download", 1.0, message=f"Vídeo baixado: {video.title}")

        # 2. Transcrição
        transcriber = Transcriber()
        self._emit("transcribe", message="Transcrevendo áudio com Whisper large-v3...")
        transcript = transcriber.transcribe(video.file_path)
        self._advance(
            "transcribe", 1.0,
            message=f"Transcrição concluída ({len(transcript.segments)} segmentos)",
        )

        # Exporta transcript.txt cru (pré-validação)
        audio_path = video.file_path.with_suffix(".wav")
        transcript_path = video.file_path.parent / "transcript.txt"
        transcript_path.write_text(transcript.full_text, encoding="utf-8")

        # 2.5. Validação cruzando áudio × texto via Gemini multimodal
        if not opts.no_validate:
            transcript = self._validate_transcript(audio_path, transcript, video.file_path.parent)
        else:
            self._advance("validate", 1.0, message="Validação pulada")

        # Modo subtitle-only: pula tudo e legenda o vídeo inteiro
        if opts.subtitle_only:
            return self._run_subtitle_only(video, transcript)

        # 3. Análise
        provider = get_provider(opts.llm)
        self._emit("analyze", message=f"Selecionando melhores momentos via {provider.name}...")
        analyzer = Analyzer(
            provider,
            min_cuts=opts.min_cuts,
            max_cuts=opts.max_cuts,
            min_gap=opts.min_gap,
        )
        highlights = analyzer.select_highlights(transcript)
        self._advance(
            "analyze", 1.0,
            message=f"{len(highlights)} momentos selecionados",
            highlights=[asdict(h) for h in highlights],
        )

        if not highlights:
            self._emit("error", message="Nenhum momento válido encontrado.")
            raise RuntimeError("Nenhum highlight válido após filtragem")

        # 4. Cortes (face tracking embutido)
        cuts_dir = video.file_path.parent / "cuts"
        cuts_dir.mkdir(parents=True, exist_ok=True)

        face_tracker = None
        if not opts.no_vertical and not opts.no_face_tracking and settings.face_tracking_enabled:
            from app.pipeline.face_tracker import FaceTracker
            face_tracker = FaceTracker()

        cutter = Cutter(
            output_dir=cuts_dir,
            vertical=not opts.no_vertical,
            face_tracker=face_tracker,
        )

        cuts: list[Cut] = []
        # Face tracking + cut juntos: emitimos progresso por highlight
        total = len(highlights)
        for i, h in enumerate(highlights, start=1):
            name = f"PT{i}"
            self._emit(
                "face_track",
                message=f"Face tracking em {name} ({i}/{total})",
                index=i, total=total,
            )
            self._emit(
                "cut",
                message=f"Cortando {name} ({i}/{total})",
                index=i, total=total,
            )
            cut_path = cuts_dir / f"{name}.mp4"
            cutter._cut_one(video.file_path, h, cut_path)
            cuts.append(Cut(index=i, name=name, highlight=h, video_path=cut_path))

            self._advance(
                "face_track", 1.0 / total,
                message=f"Face tracking {name} OK",
                index=i, total=total,
            )
            self._advance(
                "cut", 1.0 / total,
                message=f"Corte {name} OK",
                index=i, total=total,
            )

        # 5. Legendas
        if not opts.no_subtitles:
            subtitler = Subtitler()
            for i, cut in enumerate(cuts, start=1):
                self._emit(
                    "subtitle",
                    message=f"Queimando legendas em {cut.name} ({i}/{total})",
                    index=i, total=total,
                )
                subtitled = cut.video_path.with_name(cut.video_path.stem + "_sub.mp4")
                subtitler.burn_subtitles(cut.video_path, transcript, cut.highlight, subtitled)
                cut.video_path.unlink()
                subtitled.rename(cut.video_path)
                self._advance(
                    "subtitle", 1.0 / total,
                    message=f"Legenda {cut.name} OK",
                    index=i, total=total,
                )
        else:
            self._advance("subtitle", 1.0, message="Legendas puladas")

        # 6. Metadados
        if not opts.no_metadata:
            meta_gen = MetadataGenerator(provider)
            for i, cut in enumerate(cuts, start=1):
                self._emit(
                    "metadata",
                    message=f"Gerando metadados de {cut.name} ({i}/{total})",
                    index=i, total=total,
                )
                cut.metadata = meta_gen.generate(video.title, transcript, cut.highlight)
                meta_path = cut.video_path.with_suffix(".json")
                meta_path.write_text(
                    json.dumps(
                        {
                            "name": cut.name,
                            "start": cut.highlight.start,
                            "end": cut.highlight.end,
                            "duration": cut.highlight.duration,
                            "score": cut.highlight.score,
                            "reason": cut.highlight.reason,
                            **asdict(cut.metadata),
                        },
                        ensure_ascii=False, indent=2,
                    ),
                    encoding="utf-8",
                )
                self._advance(
                    "metadata", 1.0 / total,
                    message=f"Metadados {cut.name}: {cut.metadata.title}",
                    index=i, total=total,
                )
        else:
            self._advance("metadata", 1.0, message="Metadados pulados")

        self._total = 100.0
        self._emit("done", message=f"Concluído: {len(cuts)} cortes gerados")

        return self._build_result(video, transcript, cuts)

    def _validate_transcript(self, audio_path, transcript, output_dir: Path):
        try:
            from app.pipeline.validator import TranscriptValidator
            validator = TranscriptValidator()
        except Exception as e:
            logger.warning(f"Validator indisponível ({e}). Pulando validação.")
            self._advance("validate", 1.0, message=f"Validação indisponível: {e}")
            return transcript

        self._emit("validate", message="Validando transcrição com Gemini (áudio × texto)...")
        validated = validator.validate(audio_path, transcript)
        # Exporta versão validada
        (output_dir / "transcript_validated.txt").write_text(
            validated.full_text, encoding="utf-8"
        )
        self._advance(
            "validate", 1.0,
            message="Transcrição validada e revisada com áudio",
        )
        return validated

    def _run_subtitle_only(self, video, transcript) -> dict:
        # Modo subtitle-only redistribui o peso restante para legenda
        full_out = video.file_path.parent / "full_subtitled.mp4"
        subtitler = Subtitler()
        self._emit("subtitle", message="Queimando legendas no vídeo completo...")
        subtitler.burn_subtitles(
            video.file_path,
            transcript,
            Highlight(start=0.0, end=transcript.duration),
            full_out,
        )
        self._total = 100.0
        self._emit("done", message=f"Vídeo completo legendado: {full_out}")

        return {
            "mode": "subtitle_only",
            "video": {
                "youtube_id": video.video_id,
                "url": video.url,
                "title": video.title,
                "duration_seconds": video.duration,
                "source_path": str(video.file_path),
            },
            "transcript_text": transcript.full_text,
            "output_path": str(full_out),
        }

    def _build_result(self, video, transcript, cuts: list[Cut]) -> dict:
        return {
            "mode": "cuts",
            "video": {
                "youtube_id": video.video_id,
                "url": video.url,
                "title": video.title,
                "duration_seconds": video.duration,
                "source_path": str(video.file_path),
            },
            "transcript": {
                "language": transcript.language,
                "duration_seconds": transcript.duration,
                "text": transcript.full_text,
            },
            "cuts": [
                {
                    "index": c.index,
                    "name": c.name,
                    "start_seconds": c.highlight.start,
                    "end_seconds": c.highlight.end,
                    "duration_seconds": c.highlight.duration,
                    "score": c.highlight.score,
                    "reason": c.highlight.reason,
                    "video_path": str(c.video_path),
                    "title": c.metadata.title if c.metadata else None,
                    "description": c.metadata.description if c.metadata else None,
                    "hashtags": c.metadata.hashtags if c.metadata else None,
                }
                for c in cuts
            ],
        }
