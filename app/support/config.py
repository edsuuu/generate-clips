from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = "auto"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma2:9b"

    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-flash-latest"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    whisper_model: str = "large-v3"
    # Engine: auto | mlx | faster. auto = MLX (GPU Metal) no macOS Apple Silicon
    # com mlx-whisper instalado; senão faster-whisper (CPU/CUDA).
    whisper_backend: str = "auto"
    # Override do repo MLX (HF). Vazio = deriva de whisper_model.
    whisper_mlx_model: str = ""
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"
    whisper_language: str = "pt"
    whisper_beam_size: int = 10
    whisper_initial_prompt: str = ""

    temp_dir: Path = Path("/tmp/auto-post")
    storage_disk: str = "minio"
    minio_endpoint: str = "http://127.0.0.1:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "auto-post"
    minio_secure: bool = False
    python_api_token: str = ""

    max_cut_duration: int = 80
    min_cut_duration: int = 60
    min_cuts: int = 6
    max_cuts: int = 20
    min_gap_between_cuts: float = 1.0

    face_tracking_enabled: bool = True
    face_tracking_sample_fps: int = 6
    # Delegate de inferência do MediaPipe: auto | gpu | cpu.
    # auto = GPU (OpenGL/EGL) no Linux/Windows com NVIDIA; CPU no macOS (Metal
    # aborta o processo no FaceLandmarker — bug não corrigido no wheel do pip).
    face_tracking_delegate: str = "auto"

    api_host: str = "0.0.0.0"
    api_port: int = 8765
    webhook_timeout_seconds: float = 30.0
    webhook_fail_job_on_error: bool = False

    ffmpeg_encoder: str = "auto"
    # Aceleração de decode por hardware: auto | none | videotoolbox | cuda | <nome ffmpeg>.
    # auto = videotoolbox no macOS; cuda no Linux/Windows com NVIDIA.
    ffmpeg_hwaccel: str = "auto"
    ffmpeg_crf: int = 23
    ffmpeg_preset: str = "veryfast"
    ffmpeg_video_bitrate: str = "5M"
    ffmpeg_nvenc_preset: str = "p4"
    # h264_nvenc: quality control em modo VBR (0-51, menor = melhor qualidade; 20 ≈ CRF 23 libx264)
    ffmpeg_nvenc_cq: int = 20
    # h264_nvenc: bitrate máximo no modo VBR (evita picos excessivos)
    ffmpeg_nvenc_maxrate: str = "8M"
    ffmpeg_max_concurrent_renders: int = 1


settings = Settings()
