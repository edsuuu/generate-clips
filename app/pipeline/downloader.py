import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yt_dlp

from app.support.logger import logger
from app.support.types import VideoInfo


def _extract_video_id(url: str) -> str | None:
    """Tenta extrair o video_id sem bater no YouTube — economiza request e
    permite reaproveitar cache mesmo se a metadata estiver rate-limited."""
    parsed = urlparse(url)
    if parsed.hostname in {"youtu.be"}:
        return parsed.path.lstrip("/").split("/")[0] or None
    if parsed.hostname and "youtube" in parsed.hostname:
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
        # shorts/abc, embed/abc, live/abc
        m = re.match(r"^/(?:shorts|embed|live)/([^/?]+)", parsed.path)
        if m:
            return m.group(1)
    return None


# Sempre baixa na melhor qualidade possível disponível.
# Preferência: bestvideo+bestaudio mesclados em mp4. Fallback: melhor stream único.
BEST_QUALITY_FORMAT = (
    "bestvideo[ext=mp4]+bestaudio[ext=m4a]/"
    "bestvideo+bestaudio/"
    "best[ext=mp4]/best"
)


class Downloader:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(
        self,
        url: str,
        video_format: str = BEST_QUALITY_FORMAT,
        on_progress=None,
    ) -> VideoInfo:
        # Atalho de cache: se conseguimos extrair o video_id da URL e já temos
        # o arquivo + metadata salvos, retorna sem bater no YouTube.
        guessed_id = _extract_video_id(url)
        if guessed_id is not None:
            cached_meta = self._load_cached_meta(self.output_dir / guessed_id)
            cached_file = self._find_existing(self.output_dir / guessed_id)
            if cached_meta is not None and cached_file is not None:
                logger.info(
                    f"Vídeo já em cache: {cached_meta.get('title', guessed_id)} "
                    f"(pulando download)"
                )
                return VideoInfo(
                    url=url, video_id=guessed_id,
                    title=cached_meta.get("title", guessed_id),
                    duration=float(cached_meta.get("duration", 0)),
                    file_path=cached_file,
                )

        # Sem cache utilizável — vamos buscar metadados no YouTube.
        info_opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
        with yt_dlp.YoutubeDL(info_opts) as ydl:
            meta = ydl.extract_info(url, download=False)

        video_id = meta["id"]
        title = meta.get("title", video_id)
        duration = float(meta.get("duration", 0))

        video_dir = self.output_dir / video_id
        cached = self._find_existing(video_dir)
        if cached is not None:
            logger.info(f"Vídeo já em cache: {cached.name} (pulando download)")
            self._save_meta(video_dir, title, duration)
            return VideoInfo(
                url=url, video_id=video_id, title=title,
                duration=duration, file_path=cached,
            )

        logger.info(f"Baixando vídeo: {url}")
        ydl_opts = {
            "format": video_format,
            "merge_output_format": "mp4",
            "outtmpl": str(video_dir / "source.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "postprocessors": [
                {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}
            ],
        }
        if on_progress is not None:
            ydl_opts["progress_hooks"] = [self._make_hook(on_progress)]

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)

        file_path = self._find_existing(video_dir)
        if file_path is None:
            raise FileNotFoundError(f"Arquivo baixado não encontrado em {video_dir}")

        self._save_meta(video_dir, title, duration)
        logger.info(f"Vídeo baixado: {title} ({duration:.0f}s) -> {file_path.name}")
        return VideoInfo(
            url=url, video_id=video_id, title=title,
            duration=duration, file_path=file_path,
        )

    @staticmethod
    def _make_hook(on_progress):
        def hook(d: dict) -> None:
            if d.get("status") != "downloading":
                return
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            done = d.get("downloaded_bytes", 0)
            if total:
                on_progress(min(99.0, done / total * 100.0))
        return hook

    @staticmethod
    def _find_existing(video_dir: Path) -> Path | None:
        if not video_dir.exists():
            return None
        for ext in ("mp4", "mkv", "webm", "mov"):
            for f in video_dir.glob(f"source.{ext}"):
                if f.is_file() and f.stat().st_size > 0:
                    return f
        return None

    @staticmethod
    def _load_cached_meta(video_dir: Path) -> dict | None:
        meta_file = video_dir / "meta.json"
        if not meta_file.is_file():
            return None
        import json
        try:
            return json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _save_meta(video_dir: Path, title: str, duration: float) -> None:
        import json
        (video_dir / "meta.json").write_text(
            json.dumps({"title": title, "duration": duration}, ensure_ascii=False),
            encoding="utf-8",
        )
