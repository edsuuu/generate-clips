from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.support.ffmpeg import build_decode_args, build_video_encode_profile, run_with_progress


@dataclass(frozen=True)
class HlsPackage:
    master_playlist: Path
    output_dir: Path


class HlsPackager:
    def package(
        self,
        source_path: Path,
        output_dir: Path,
        total_seconds: float,
        on_progress: Callable[[float], object] | None = None,
    ) -> HlsPackage:
        output_dir.mkdir(parents=True, exist_ok=True)
        master_playlist = output_dir / "master.m3u8"
        profile = build_video_encode_profile()

        cmd = [
            "ffmpeg",
            "-y",
            *build_decode_args(),
            "-i",
            str(source_path),
            *profile.args,
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-ac",
            "2",
            "-ar",
            "48000",
            "-hls_time",
            "4",
            "-hls_playlist_type",
            "vod",
            "-hls_segment_filename",
            "segment_%03d.ts",
            "master.m3u8",
        ]

        run_with_progress(
            cmd,
            total_seconds=max(total_seconds, 1.0),
            on_progress=on_progress,
            cwd=output_dir,
            encoder=profile.encoder,
            stage="hls",
        )

        return HlsPackage(
            master_playlist=master_playlist,
            output_dir=output_dir,
        )
