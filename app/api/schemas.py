from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, HttpUrl, model_validator


class CallbackMixin(BaseModel):
    callback_url: HttpUrl | None = Field(
        None, description="Webhook Laravel chamado quando a etapa terminar"
    )
    callback_token: str | None = Field(None, description="Token enviado no callback")
    callback_header: str | None = Field("Authorization", description="Header que recebe o token")


class AcceptedJobOut(BaseModel):
    job_id: str
    status: str = "accepted"


class MinioObject(BaseModel):
    bucket: str | None = None
    path: str


class IngestOptions(BaseModel):
    transcribe: bool = True
    validate_transcript: bool = True
    upload_original_to_minio: bool = True
    llm: str | None = None


class IngestVideoRequest(CallbackMixin):
    video_id: str = Field(..., description="UUID do video no Laravel")
    url: str
    options: IngestOptions = Field(default_factory=IngestOptions)


class SubtitleFullRequest(CallbackMixin):
    transcript_text: str | None = None
    transcript_json: dict[str, Any]
    source_file: MinioObject
    output: MinioObject


class CutConstraints(BaseModel):
    min_cuts: int | None = None
    max_cuts: int | None = None
    min_duration: float | None = None
    max_duration: float | None = None
    min_gap: float | None = None


class RecommendCutsRequest(CallbackMixin):
    transcript_json: dict[str, Any] = Field(default_factory=dict)
    transcript_text: str | None = None
    video: dict[str, Any] | None = None
    constraints: CutConstraints = Field(default_factory=CutConstraints)
    user_prompt: str | None = None
    llm: str | None = None

    @model_validator(mode="after")
    def _require_transcript_source(self) -> RecommendCutsRequest:
        if not self.transcript_json and not (self.transcript_text or "").strip():
            raise ValueError("Informe transcript_json ou transcript_text")
        return self


class RenderCutRequest(BaseModel):
    cut_id: str
    name: str
    type: str
    start_seconds: float
    end_seconds: float
    vertical: bool = True
    face_tracking: bool = True
    output_path: str
    # Metadados já gerados em render anterior — se presentes, a IA não é chamada novamente.
    title: str | None = None
    description: str | None = None
    hashtags: list[str] | None = None


class RenderCutsRequest(CallbackMixin):
    source_file: MinioObject
    transcript_json: dict[str, Any] = Field(default_factory=dict)
    cuts: list[RenderCutRequest]
    video: dict[str, Any] | None = None
    # Gera título/descrição/hashtags por corte (via LLM) e devolve no callback.
    generate_metadata: bool = True
    llm: str | None = None
