from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, HttpUrl


class CallbackMixin(BaseModel):
    callback_url: Optional[HttpUrl] = Field(
        None, description="Webhook Laravel chamado quando a etapa terminar"
    )
    callback_token: Optional[str] = Field(None, description="Token enviado no callback")
    callback_header: Optional[str] = Field(
        "Authorization", description="Header que recebe o token"
    )


class AcceptedJobOut(BaseModel):
    job_id: str
    status: str = "accepted"


class MinioObject(BaseModel):
    bucket: Optional[str] = None
    path: str


class IngestOptions(BaseModel):
    transcribe: bool = True
    validate_transcript: bool = True
    upload_original_to_minio: bool = True
    llm: Optional[str] = None


class IngestVideoRequest(CallbackMixin):
    video_id: str = Field(..., description="UUID do video no Laravel")
    url: str
    options: IngestOptions = Field(default_factory=IngestOptions)


class SubtitleFullRequest(CallbackMixin):
    transcript_text: Optional[str] = None
    transcript_json: dict[str, Any]
    source_file: MinioObject
    output: MinioObject


class CutConstraints(BaseModel):
    min_cuts: Optional[int] = None
    max_cuts: Optional[int] = None
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None
    min_gap: Optional[float] = None


class RecommendCutsRequest(CallbackMixin):
    transcript_json: dict[str, Any]
    video: dict[str, Any] = Field(default_factory=dict)
    constraints: CutConstraints = Field(default_factory=CutConstraints)
    user_prompt: Optional[str] = None
    llm: Optional[str] = None


class RenderCutRequest(BaseModel):
    cut_id: str
    name: str
    type: str
    start_seconds: float
    end_seconds: float
    vertical: bool = True
    face_tracking: bool = True
    output_path: str


class RenderCutsRequest(CallbackMixin):
    source_file: MinioObject
    transcript_json: dict[str, Any] = Field(default_factory=dict)
    cuts: list[RenderCutRequest]
