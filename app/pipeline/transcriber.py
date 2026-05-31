import platform
import subprocess
from collections.abc import Callable
from pathlib import Path

from app.support.config import settings
from app.support.logger import logger
from app.support.types import Segment, Transcript, Word

# Mapa de nomes de modelo OpenAI -> repos MLX (mlx-community). Override completo
# via WHISPER_MLX_MODEL (aceita qualquer repo HF, ex.: "mlx-community/whisper-large-v3-turbo").
_MLX_MODEL_REPOS = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def _resolve_backend() -> str:
    """Escolhe o engine de transcrição.

    auto = MLX (GPU Metal) no macOS Apple Silicon quando `mlx-whisper` estiver
    instalado; senão faster-whisper (CPU/CUDA). faster-whisper não tem backend
    Metal, então no Mac o ganho de GPU vem do MLX.
    """
    backend = settings.whisper_backend.strip().lower() or "auto"
    if backend != "auto":
        return backend
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx_whisper  # noqa: F401
        except ImportError:
            logger.info("mlx-whisper indisponível; transcrição via faster-whisper (CPU).")
            return "faster"
        return "mlx"
    return "faster"


def _resolve_mlx_repo() -> str:
    override = settings.whisper_mlx_model.strip()
    if override:
        return override
    name = settings.whisper_model.strip()
    if name in _MLX_MODEL_REPOS:
        return _MLX_MODEL_REPOS[name]
    if "/" in name:  # já é um repo HF completo
        return name
    return f"mlx-community/whisper-{name}"


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except ImportError:
        pass
    try:
        import ctranslate2  # type: ignore
        return bool(ctranslate2.get_supported_compute_types("cuda"))
    except Exception:
        pass
    return False


def _resolve_device_and_compute() -> tuple[str, str]:
    device = settings.whisper_device
    compute_type = settings.whisper_compute_type

    if device == "auto":
        device = "cuda" if _cuda_available() else "cpu"

    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    return device, compute_type


class Transcriber:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.whisper_model
        self.backend = _resolve_backend()

        if self.backend == "mlx":
            self._mlx_repo = _resolve_mlx_repo()
            logger.info(f"Whisper via MLX (GPU Metal): {self._mlx_repo}")
            return

        from faster_whisper import WhisperModel

        device, compute_type = _resolve_device_and_compute()
        logger.info(
            f"Carregando Whisper {self.model_name} "
            f"(faster-whisper, device={device}, compute={compute_type})"
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
        # initial_prompt vazio = None (evita "prompt leak", bug clássico em que
        # o Whisper passa a transcrever o próprio prompt como se fosse o áudio).
        initial_prompt = settings.whisper_initial_prompt.strip() or None

        if self.backend == "mlx":
            return self._transcribe_mlx(audio_path, initial_prompt, on_progress)
        return self._transcribe_faster(audio_path, initial_prompt, on_progress)

    def _transcribe_mlx(
        self,
        audio_path: Path,
        initial_prompt: str | None,
        on_progress: Callable[[float], object] | None,
    ) -> Transcript:
        import mlx_whisper

        logger.info(
            f"Transcrevendo {audio_path.name} via MLX (greedy, lang={settings.whisper_language})..."
        )
        # MLX 0.4.x não tem beam search; greedy com temperature=0 é o padrão.
        # vad_filter do faster-whisper não existe aqui — os thresholds de
        # no_speech/logprob/compression cumprem o papel de suprimir alucinação.
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=self._mlx_repo,
            language=settings.whisper_language or None,
            temperature=0.0,
            initial_prompt=initial_prompt,
            condition_on_previous_text=False,
            word_timestamps=True,
            no_speech_threshold=0.5,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            verbose=None,  # None = sem barra tqdm nem print por segmento (log limpo)
        )

        segments: list[Segment] = []
        for seg in result.get("segments", []):
            words = [
                Word(text=str(w["word"]).strip(), start=float(w["start"]), end=float(w["end"]))
                for w in (seg.get("words") or [])
            ]
            segments.append(
                Segment(
                    text=str(seg.get("text", "")).strip(),
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    words=words,
                )
            )

        # MLX não devolve duração total; usa o fim do último segmento.
        duration = max((s.end for s in segments), default=0.0)
        language = str(result.get("language") or settings.whisper_language)
        if on_progress is not None:
            on_progress(100.0)

        logger.info(
            f"Transcrição concluída (MLX): {len(segments)} segmentos, "
            f"idioma={language}, duração={duration:.0f}s"
        )
        return Transcript(language=language, duration=duration, segments=segments)

    def _transcribe_faster(
        self,
        audio_path: Path,
        initial_prompt: str | None,
        on_progress: Callable[[float], object] | None,
    ) -> Transcript:
        logger.info(
            f"Transcrevendo {audio_path.name} "
            f"(beam={settings.whisper_beam_size}, lang={settings.whisper_language})..."
        )

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
