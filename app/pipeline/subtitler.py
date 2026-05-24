import subprocess
from pathlib import Path

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
    for i in range(0, len(words), WORDS_PER_FRAME):
        group = words[i:i + WORDS_PER_FRAME]
        group_end = group[-1].end

        for j, active in enumerate(group):
            parts = []
            for k, w in enumerate(group):
                txt = _escape(w.text.upper())
                if k == j:
                    parts.append("{\\rHighlight}" + txt + "{\\rDefault}")
                else:
                    parts.append(txt)
            line = " ".join(parts)

            seg_start = active.start
            seg_end = active.end if j < len(group) - 1 else group_end
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
    ) -> Path:
        words = _collect_words_in_range(transcript, highlight.start, highlight.end)
        if not words:
            logger.warning(f"Sem palavras para legendar em {video_path.name}, copiando original")
            out_path.write_bytes(video_path.read_bytes())
            return out_path

        ass_path = video_path.with_suffix(".ass")
        ass_path.write_text(_build_ass(words), encoding="utf-8")

        logger.info(f"Queimando legendas em {out_path.name}...")

        # Roda no diretório do .ass com nome simples para evitar problemas
        # de escape no filtro subtitles (que usa ':' como separador interno).
        cwd = ass_path.parent
        cmd = [
            "ffmpeg", "-y", "-i", video_path.name,
            "-vf", f"subtitles=filename={ass_path.name}",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-c:a", "copy",
            "-movflags", "+faststart",
            out_path.name,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg falhou legendando {out_path.name}:\n{result.stderr}")

        return out_path
