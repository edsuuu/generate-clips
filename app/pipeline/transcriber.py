import subprocess
from collections.abc import Callable
from pathlib import Path

from app.support.config import settings
from app.support.logger import logger
from app.support.types import Segment, Transcript, Word


def _resolve_device_and_compute() -> tuple[str, str]:
    device = settings.whisper_device
    compute_type = settings.whisper_compute_type

    if device == "auto":
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"

    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    return device, compute_type


class Transcriber:
    def __init__(self, model_name: str | None = None):
        from faster_whisper import WhisperModel

        self.model_name = model_name or settings.whisper_model
        device, compute_type = _resolve_device_and_compute()
        logger.info(
            f"Carregando Whisper {self.model_name} (device={device}, compute={compute_type})"
        )
        self.model = WhisperModel(self.model_name, device=device, compute_type=compute_type)

    def _extract_audio(self, video_path: Path) -> Path:
        audio_path = video_path.with_suffix(".wav")
        if audio_path.exists():
            return audio_path

        logger.info("Extraindo áudio do vídeo...")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg falhou: {result.stderr}")
        return audio_path

    def transcribe(
        self, video_path: Path, on_progress: Callable[[float], object] | None = None
    ) -> Transcript:
        audio_path = self._extract_audio(video_path)

        logger.info(
            f"Transcrevendo {audio_path.name} "
            f"(beam={settings.whisper_beam_size}, lang={settings.whisper_language})..."
        )
        # initial_prompt vazio = None (evita "prompt leak", bug clássico em que
        # o Whisper passa a transcrever o próprio prompt como se fosse o áudio).
        initial_prompt = settings.whisper_initial_prompt.strip() or None

        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=settings.whisper_language,
            beam_size=settings.whisper_beam_size,
            best_of=settings.whisper_beam_size,
            temperature=0.0,
            initial_prompt=initial_prompt,
            condition_on_previous_text=False,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            no_speech_threshold=0.5,
            log_prob_threshold=-1.0,
        )

        total = float(info.duration) or 0.0
        segments: list[Segment] = []
        for seg in segments_iter:
            words = [
                Word(text=w.word.strip(), start=float(w.start), end=float(w.end))
                for w in (seg.words or [])
            ]
            segments.append(
                Segment(
                    text=seg.text.strip(),
                    start=float(seg.start),
                    end=float(seg.end),
                    words=words,
                )
            )
            if on_progress is not None and total > 0:
                on_progress(min(99.0, float(seg.end) / total * 100.0))

        transcript = Transcript(
            language=info.language,
            duration=float(info.duration),
            segments=segments,
        )

        logger.info(
            f"Transcrição concluída: {len(segments)} segmentos, "
            f"idioma={info.language}, duração={info.duration:.0f}s"
        )
        return transcript
