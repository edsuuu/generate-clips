from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class JobCreate(BaseModel):
    url: str = Field(..., description="URL do vídeo do YouTube")

    # Opcionais — webhook chamado ao concluir (success ou failure)
    webhook_url: Optional[HttpUrl] = Field(
        None, description="URL para POST do payload de conclusão"
    )
    webhook_token: Optional[str] = Field(
        None, description="Valor enviado no header de autenticação"
    )
    webhook_header: Optional[str] = Field(
        "Authorization",
        description="Nome do header que receberá o token (default: Authorization)",
    )

    # Opções do pipeline
    llm: Optional[str] = Field(None, description="auto | local | gemini | claude | gpt")
    min_cuts: Optional[int] = None
    max_cuts: Optional[int] = None
    min_gap: Optional[float] = None
    no_subtitles: bool = False
    no_vertical: bool = False
    no_metadata: bool = False
    no_face_tracking: bool = False
    no_validate: bool = False
    subtitle_only: bool = False


class CutOut(BaseModel):
    index: int
    name: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    score: float
    reason: Optional[str] = None
    video_path: str
    title: Optional[str] = None
    description: Optional[str] = None
    hashtags: Optional[list[str]] = None

    model_config = {"from_attributes": True}


class JobOut(BaseModel):
    id: str
    url: str
    status: str
    progress: float
    stage: str
    message: str

    video_youtube_id: Optional[str] = None
    video_title: Optional[str] = None
    video_duration: Optional[float] = None

    error_message: Optional[str] = None
    webhook_status: Optional[str] = None

    created_at: datetime
    updated_at: datetime
    finished_at: Optional[datetime] = None

    cuts: list[CutOut] = []

    model_config = {"from_attributes": True}
