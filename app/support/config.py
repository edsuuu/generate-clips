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

    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    whisper_model: str = "large-v3"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"
    whisper_language: str = "pt"
    whisper_beam_size: int = 10
    whisper_initial_prompt: str = ""

    output_dir: Path = Path("./output")
    max_cut_duration: int = 80
    min_cut_duration: int = 60
    min_cuts: int = 6
    max_cuts: int = 20
    min_gap_between_cuts: float = 1.0

    face_tracking_enabled: bool = True
    face_tracking_sample_fps: int = 6

    db_host: str = "127.0.0.1"
    db_port: int = 3306
    db_database: str = "auto_post"
    db_user: str = "root"
    db_password: str = "root"

    api_host: str = "0.0.0.0"
    api_port: int = 8765

    @property
    def database_url(self) -> str:
        return (
            f"mysql+pymysql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_database}?charset=utf8mb4"
        )


settings = Settings()
