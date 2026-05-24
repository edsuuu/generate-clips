from collections.abc import Callable
from pathlib import Path

from app.support.ffmpeg import build_video_encode_profile, run_with_progress
from app.support.logger import logger
from app.support.types import Highlight, Transcript, Word


# Estilo TikTok: branco com contorno preto grosso (Default), amarelo brilhante
# escalado quando ativo (Highlight). Fonte Arial Black — disponível em macOS,
# Windows e quase todas distros Linux modernas; libass resolve via fontconfig.
# Cores ASS = AABBGGRR (alpha invertido). Posicionamento na metade inferior.
ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,86,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,2,0,1,8,4,2,80,80,420,1
Style: Highlight,Arial Black,100,&H0000F2FF,&H0000F2FF,&H00000000,&H80000000,1,0,0,0,108,108,2,0,1,9,4,2,80,80,420,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


WORDS_PER_FRAME = 3  # 3 palavras visíveis por vez
SHORT_GAP_SECONDS = 0.18
LONG_GAP_SECONDS = 0.45
WORD_END_PADDING_SECONDS = 0.05


def _format_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def _collect_words_in_range(transcript: Transcript, start: float, end: float) -> list[Word]:
    words: list[Word] = []
    for seg in transcript.segments:
        if seg.end < start or seg.start > end:
            continue
        for w in seg.words:
            if w.end < start or w.start > end:
                continue
            words.append(
                Word(
                    text=w.text.strip(),
                    start=max(w.start, start) - start,
                    end=min(w.end, end) - start,
                )
            )
    return words


def _build_ass(words: list[Word]) -> str:
    events: list[str] = []

    for index, active in enumerate(words):
        window_start = max(0, index - WORDS_PER_FRAME + 1)
        window = list(words[window_start:index + 1])

        # Quebra a janela se houve silencio perceptivel antes da fala atual.
        while len(window) > 1 and active.start - window[0].end > LONG_GAP_SECONDS:
            window.pop(0)

        parts = []
        for w in window[:-1]:
            parts.append(_escape(w.text.upper()))
        parts.append("{\\rHighlight}" + _escape(active.text.upper()) + "{\\rDefault}")
        line = " ".join(parts)

        next_word = words[index + 1] if index + 1 < len(words) else None
        seg_start = active.start
        seg_end = active.end + WORD_END_PADDING_SECONDS
        if next_word is not None:
            if next_word.start - active.end <= SHORT_GAP_SECONDS:
                seg_end = next_word.start
            else:
                seg_end = min(seg_end, next_word.start)
        seg_end = max(seg_start + 0.01, seg_end)

        events.append(
            f"Dialogue: 0,{_format_ts(seg_start)},{_format_ts(seg_end)},"
            f"Default,,0,0,0,,{line}"
        )

    return ASS_HEADER + "\n".join(events) + "\n"


class Subtitler:
    def __init__(self):
        pass

    def burn_subtitles(
        self,
        video_path: Path,
        transcript: Transcript,
        highlight: Highlight,
        out_path: Path,
        on_progress: Callable[[float], None] | None = None,
    ) -> Path:
        words = _collect_words_in_range(transcript, highlight.start, highlight.end)
        if not words:
            logger.warning(f"Sem palavras para legendar em {video_path.name}, copiando original")
            out_path.write_bytes(video_path.read_bytes())
            if on_progress:
                on_progress(100.0)
            return out_path

        ass_path = video_path.with_suffix(".ass")
        ass_path.write_text(_build_ass(words), encoding="utf-8")

        logger.info(f"Queimando legendas em {out_path.name}...")
        encode_profile = build_video_encode_profile()

        # Roda no diretório do .ass com nome simples para evitar problemas
        # de escape no filtro subtitles (que usa ':' como separador interno).
        cwd = ass_path.parent
        cmd = [
            "ffmpeg", "-y", "-i", video_path.name,
            "-vf", f"subtitles=filename={ass_path.name}",
            *encode_profile.args,
            "-c:a", "copy",
            "-movflags", "+faststart",
            out_path.name,
        ]
        run_with_progress(
            cmd,
            total_seconds=max(0.0, highlight.end - highlight.start),
            on_progress=on_progress,
            cwd=cwd,
            encoder=encode_profile.encoder,
            stage="subtitle",
        )
        return out_path
