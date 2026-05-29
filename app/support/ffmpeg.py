"""Helpers para renderizar com ffmpeg com encoder, logs e throttling local."""

from __future__ import annotations

import os
import platform
import re
import shlex
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from app.support.config import settings
from app.support.logger import logger

_OUT_TIME_RE = re.compile(r"out_time=(\d+):(\d+):(\d+(?:\.\d+)?)")
_RENDER_SLOT_DIR = settings.temp_dir / "ffmpeg-render-slots"


@dataclass(frozen=True)
class VideoEncodeProfile:
    encoder: str
    args: tuple[str, ...]


def build_video_encode_profile() -> VideoEncodeProfile:
    requested = settings.ffmpeg_encoder.strip().lower() or "auto"

    if requested == "auto":
        if platform.system() == "Darwin" and _supports_encoder("h264_videotoolbox"):
            return VideoEncodeProfile(
                encoder="h264_videotoolbox",
                args=("-c:v", "h264_videotoolbox", "-b:v", settings.ffmpeg_video_bitrate),
            )
        if platform.system() == "Windows" and _supports_encoder("h264_nvenc"):
            return _nvenc_profile()
        return _libx264_profile()

    if requested == "h264_videotoolbox":
        if _supports_encoder("h264_videotoolbox"):
            return VideoEncodeProfile(
                encoder="h264_videotoolbox",
                args=("-c:v", "h264_videotoolbox", "-b:v", settings.ffmpeg_video_bitrate),
            )
        logger.warning(
            "FFMPEG_ENCODER=h264_videotoolbox, mas o encoder nao esta disponivel. "
            "Caindo para libx264."
        )
        return _libx264_profile()

    if requested == "h264_nvenc":
        if _supports_encoder("h264_nvenc"):
            return _nvenc_profile()
        logger.warning(
            "FFMPEG_ENCODER=h264_nvenc, mas o encoder nao esta disponivel. Caindo para libx264."
        )
        return _libx264_profile()

    if requested == "libx264":
        return _libx264_profile()

    logger.warning(
        f"FFMPEG_ENCODER={settings.ffmpeg_encoder!r} nao reconhecido. Usando fallback libx264."
    )
    return _libx264_profile()


def _libx264_profile() -> VideoEncodeProfile:
    return VideoEncodeProfile(
        encoder="libx264",
        args=(
            "-c:v",
            "libx264",
            "-preset",
            settings.ffmpeg_preset,
            "-crf",
            str(settings.ffmpeg_crf),
        ),
    )


def _nvenc_profile() -> VideoEncodeProfile:
    return VideoEncodeProfile(
        encoder="h264_nvenc",
        args=(
            "-c:v",
            "h264_nvenc",
            "-preset",
            settings.ffmpeg_nvenc_preset,
            "-b:v",
            settings.ffmpeg_video_bitrate,
        ),
    )


@lru_cache(maxsize=1)
def _available_encoders() -> set[str]:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        logger.warning("ffmpeg nao encontrado ao detectar encoders; usando fallback libx264.")
        return set()

    output = "\n".join(part for part in [result.stdout, result.stderr] if part)
    encoders: set[str] = set()
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("Encoders:", "--")):
            continue
        match = re.match(r"^[A-Z\\.]{6}\s+([^\s]+)", stripped)
        if match:
            encoders.add(match.group(1))
    return encoders


def _supports_encoder(name: str) -> bool:
    return name in _available_encoders()


@lru_cache(maxsize=1)
def _available_hwaccels() -> set[str]:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return set()

    output = "\n".join(part for part in [result.stdout, result.stderr] if part)
    accels: set[str] = set()
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("hardware acceleration"):
            continue
        accels.add(stripped)
    return accels


@lru_cache(maxsize=1)
def build_decode_args() -> tuple[str, ...]:
    """Args de decode por hardware, inseridos antes do `-i` de entrada.

    Offload do decode para a GPU (VideoToolbox no macOS). Sem
    `-hwaccel_output_format`, os frames voltam para a memória do sistema, o que
    mantém compatibilidade com filtros de software (ex.: `subtitles`/libass).
    """
    requested = settings.ffmpeg_hwaccel.strip().lower() or "auto"

    if requested in ("none", "off", "cpu", "false"):
        return ()

    if requested == "auto":
        if platform.system() == "Darwin" and "videotoolbox" in _available_hwaccels():
            return ("-hwaccel", "videotoolbox")
        return ()

    if requested in _available_hwaccels():
        return ("-hwaccel", requested)

    logger.warning(
        f"FFMPEG_HWACCEL={settings.ffmpeg_hwaccel!r} indisponivel; decode seguira em CPU."
    )
    return ()


def _io_summary(cmd: list[str]) -> tuple[list[str], str]:
    inputs: list[str] = []
    for idx, token in enumerate(cmd[:-1]):
        if token == "-i" and idx + 1 < len(cmd):
            inputs.append(cmd[idx + 1])
    output = cmd[-1] if cmd else ""
    return inputs, output


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _clear_stale_slot(slot_path: Path) -> None:
    try:
        payload = slot_path.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return

    if not payload:
        slot_path.unlink(missing_ok=True)
        return

    try:
        pid = int(payload[0])
    except ValueError:
        slot_path.unlink(missing_ok=True)
        return

    if not _is_pid_alive(pid):
        slot_path.unlink(missing_ok=True)


def _acquire_render_slot() -> tuple[Path, int, float]:
    _RENDER_SLOT_DIR.mkdir(parents=True, exist_ok=True)
    max_slots = max(1, settings.ffmpeg_max_concurrent_renders)
    wait_start = time.monotonic()

    while True:
        for slot_index in range(max_slots):
            slot_path = _RENDER_SLOT_DIR / f"slot-{slot_index}.lock"
            try:
                fd = os.open(slot_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                _clear_stale_slot(slot_path)
                continue

            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(f"{os.getpid()}\n{datetime.now().isoformat(timespec='seconds')}\n")

            wait_seconds = time.monotonic() - wait_start
            return slot_path, slot_index, wait_seconds

        time.sleep(0.25)


def _start_frame_writer(proc: subprocess.Popen, frames: Iterator[bytes]) -> threading.Thread:
    """Escreve frames crus no stdin do ffmpeg numa thread e fecha o pipe ao fim."""

    def _pump() -> None:
        try:
            for chunk in frames:
                proc.stdin.write(chunk)  # type: ignore[union-attr]
        except (BrokenPipeError, ValueError):
            pass
        finally:
            try:
                proc.stdin.close()  # type: ignore[union-attr]
            except OSError:
                pass

    thread = threading.Thread(target=_pump, daemon=True)
    thread.start()
    return thread


def _start_stderr_drainer(proc: subprocess.Popen) -> tuple[threading.Thread, list[bytes]]:
    """Drena stderr numa thread para evitar deadlock quando o buffer do pipe enche.

    Sem isso, ffmpeg trava esperando alguem ler stderr (ex.: HLS muxer com 60+
    segmentos enche os 64KB do pipe), enquanto a gente fica preso lendo stdout.
    """
    chunks: list[bytes] = []

    def _drain() -> None:
        if proc.stderr is None:
            return
        try:
            while True:
                chunk = proc.stderr.read(4096)
                if not chunk:
                    break
                chunks.append(
                    chunk if isinstance(chunk, bytes) else chunk.encode("utf-8", "ignore")
                )
        except (OSError, ValueError):
            pass

    thread = threading.Thread(target=_drain, daemon=True)
    thread.start()
    return thread, chunks


def run_with_progress(
    cmd: list[str],
    total_seconds: float,
    on_progress: Callable[[float], object] | None = None,
    cwd: Path | str | None = None,
    encoder: str | None = None,
    stage: str = "render",
    stdin_frames: Iterator[bytes] | None = None,
) -> None:
    """Roda ffmpeg e chama on_progress(percent 0-100) conforme avanca.

    Se `stdin_frames` for fornecido, ffmpeg le video cru de `pipe:0`: os frames
    sao escritos em uma thread enquanto o progresso e lido de `pipe:1`. Use para
    enviar frames OpenCV direto ao encoder de hardware, sem arquivo intermediario
    nem re-encode em software.
    """

    full = [cmd[0], "-progress", "pipe:1", "-nostats", *cmd[1:]]
    inputs, output = _io_summary(cmd)
    selected_encoder = encoder or "desconhecido"
    piping = stdin_frames is not None
    slot_path, slot_index, wait_seconds = _acquire_render_slot()
    try:
        started_at = datetime.now().isoformat(timespec="seconds")
        render_start = time.monotonic()
        logger.info(
            f"[ffmpeg:{stage}] inicio={started_at} encoder={selected_encoder} "
            f"slot={slot_index} fila={wait_seconds:.2f}s input={inputs or ['?']} "
            f"output={output} cmd={shlex.join(cmd)}"
        )

        # Em modo pipe, stdin recebe bytes crus -> processo em modo binario e
        # decodificamos as linhas de progresso manualmente.
        proc = subprocess.Popen(
            full,
            stdin=subprocess.PIPE if piping else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=not piping,
            bufsize=0 if piping else 1,
            cwd=str(cwd) if cwd else None,
        )

        writer = _start_frame_writer(proc, stdin_frames) if stdin_frames is not None else None
        stderr_thread, stderr_chunks = _start_stderr_drainer(proc)

        last_pct = -1.0
        last_logged_pct = -1.0
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.decode("utf-8", "ignore") if piping else raw
            match = _OUT_TIME_RE.search(line)
            if match and total_seconds > 0:
                h, m, s = match.groups()
                current = int(h) * 3600 + int(m) * 60 + float(s)
                pct = min(99.0, current / total_seconds * 100.0)
                if pct - last_pct >= 1.0:
                    last_pct = pct
                    if on_progress is not None:
                        on_progress(pct)
                if pct - last_logged_pct >= 10.0:
                    last_logged_pct = pct
                    logger.info(
                        f"[ffmpeg:{stage}] progresso={pct:5.1f}% "
                        f"({current:.1f}s/{total_seconds:.1f}s)"
                    )
            elif line.startswith("progress=end"):
                if on_progress is not None:
                    on_progress(100.0)
                logger.info(f"[ffmpeg:{stage}] progresso=100.0%")

        if writer is not None:
            writer.join()
        returncode = proc.wait()
        stderr_thread.join(timeout=5.0)
        stderr = b"".join(stderr_chunks).decode("utf-8", "ignore")
        finished_at = datetime.now().isoformat(timespec="seconds")
        elapsed = time.monotonic() - render_start

        logger.info(
            f"[ffmpeg:{stage}] fim={finished_at} duracao={elapsed:.2f}s "
            f"encoder={selected_encoder} slot={slot_index} output={output} exit={returncode}"
        )

        if returncode != 0:
            raise RuntimeError(f"ffmpeg falhou (exit {returncode}):\n{stderr[-1500:]}")
    finally:
        slot_path.unlink(missing_ok=True)
