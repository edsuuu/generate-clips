"""Corta segmentos do vídeo e gera saída vertical 9:16.

Suporta dois modos:
- Crop centralizado estático (rápido, via ffmpeg)
- Crop dinâmico seguindo trajetória do speaker (OpenCV frame-a-frame + mux ffmpeg)
"""

from __future__ import annotations

from pathlib import Path

import cv2  # type: ignore
import numpy as np

from app.support.ffmpeg import build_video_encode_profile, run_with_progress
from app.support.logger import logger
from app.support.types import CropTrajectory, Cut, Highlight, VideoInfo

TARGET_W = 1080
TARGET_H = 1920


class Cutter:
    def __init__(
        self,
        output_dir: Path,
        vertical: bool = True,
        face_tracker=None,
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vertical = vertical
        self.face_tracker = face_tracker
        self.encode_profile = build_video_encode_profile()

    def cut_all(self, video: VideoInfo, highlights: list[Highlight]) -> list[Cut]:
        cuts: list[Cut] = []
        for i, h in enumerate(highlights, start=1):
            name = f"PT{i}"
            out_path = self.output_dir / f"{name}.mp4"
            self._cut_one(video.file_path, h, out_path)
            cuts.append(Cut(index=i, name=name, highlight=h, video_path=out_path))
        return cuts

    def _cut_one(self, source: Path, highlight: Highlight, out_path: Path) -> None:
        logger.info(
            f"Cortando {out_path.name}: {highlight.start:.1f}s -> "
            f"{highlight.end:.1f}s ({highlight.duration:.1f}s)"
        )

        if not self.vertical:
            self._cut_simple(source, highlight, out_path)
            return

        if self.face_tracker is not None:
            try:
                trajectory = self.face_tracker.track_segment(source, highlight.start, highlight.end)
                self._cut_dynamic(source, highlight, out_path, trajectory)
                return
            except Exception as e:
                logger.warning(
                    f"Face tracking falhou em {out_path.name}: {e}. Caindo para crop centralizado."
                )

        self._cut_vertical_static(source, highlight, out_path)

    def _cut_simple(self, source: Path, highlight: Highlight, out_path: Path) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{highlight.start:.3f}",
            "-i",
            str(source),
            "-t",
            f"{highlight.duration:.3f}",
            *self.encode_profile.args,
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        run_with_progress(
            cmd,
            total_seconds=highlight.duration,
            encoder=self.encode_profile.encoder,
            stage="cut-simple",
        )

    def _cut_vertical_static(self, source: Path, highlight: Highlight, out_path: Path) -> None:
        vf = (
            f"scale=w={TARGET_W}:h={TARGET_H}:force_original_aspect_ratio=increase,"
            f"crop={TARGET_W}:{TARGET_H}"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{highlight.start:.3f}",
            "-i",
            str(source),
            "-t",
            f"{highlight.duration:.3f}",
            "-vf",
            vf,
            *self.encode_profile.args,
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        run_with_progress(
            cmd,
            total_seconds=highlight.duration,
            encoder=self.encode_profile.encoder,
            stage="cut-static",
        )

    def _cut_dynamic(
        self,
        source: Path,
        highlight: Highlight,
        out_path: Path,
        trajectory: CropTrajectory,
    ) -> None:
        """Renderiza frame-a-frame com crop seguindo a trajetória do speaker."""
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir {source}")

        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Decide a janela de crop no espaço do vídeo original.
        # Mantemos a proporção 9:16. A altura do crop = altura do vídeo (máximo),
        # a largura = altura * 9/16. Se o vídeo for muito largo, isso já garante
        # vertical sem perder verticalmente.
        crop_h = src_h
        crop_w = int(round(crop_h * TARGET_W / TARGET_H))
        if crop_w > src_w:
            # Vídeo mais quadrado que 9:16 — limita largura ao max e ajusta altura
            crop_w = src_w
            crop_h = int(round(crop_w * TARGET_H / TARGET_W))

        # Escreve frames em arquivo temporário (sem áudio); depois faz mux com ffmpeg.
        tmp_video = out_path.with_suffix(".novideo.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_video), fourcc, fps, (TARGET_W, TARGET_H))

        start_frame = int(highlight.start * fps)
        end_frame = int(highlight.end * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        while frame_idx < end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / fps  # tempo absoluto no vídeo
            cx, cy = trajectory.value_at(t)

            x = int(np.clip(cx - crop_w // 2, 0, src_w - crop_w))
            y = int(np.clip(cy - crop_h // 2, 0, src_h - crop_h))

            cropped = frame[y : y + crop_h, x : x + crop_w]
            resized = cv2.resize(cropped, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)
            writer.write(resized)
            frame_idx += 1

        writer.release()
        cap.release()

        # Mux do áudio original com o vídeo croppado e re-encode para H.264.
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(tmp_video),
            "-ss",
            f"{highlight.start:.3f}",
            "-i",
            str(source),
            "-t",
            f"{highlight.duration:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            *self.encode_profile.args,
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        try:
            run_with_progress(
                cmd,
                total_seconds=highlight.duration,
                encoder=self.encode_profile.encoder,
                stage="cut-dynamic-mux",
            )
        finally:
            tmp_video.unlink(missing_ok=True)
