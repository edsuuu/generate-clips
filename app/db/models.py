"""Models SQLAlchemy do auto-post."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.session import Base


class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    url: Mapped[str] = mapped_column(String(2048))
    status: Mapped[str] = mapped_column(String(20), default=JobStatus.PENDING, index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    stage: Mapped[str] = mapped_column(String(40), default="pending")
    message: Mapped[str] = mapped_column(String(500), default="")

    # Webhook (opcional)
    webhook_url: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    webhook_token: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    webhook_header: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    webhook_status: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    webhook_attempt_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    # Opções do pipeline
    llm_provider: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    options: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Resultado
    video_youtube_id: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    video_title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    video_duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    transcript_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    cuts: Mapped[list["Cut"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="Cut.index",
    )


class Cut(Base):
    __tablename__ = "cuts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(36), ForeignKey("jobs.id", ondelete="CASCADE"))

    index: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(16))
    start_seconds: Mapped[float] = mapped_column(Float)
    end_seconds: Mapped[float] = mapped_column(Float)
    duration_seconds: Mapped[float] = mapped_column(Float)
    score: Mapped[float] = mapped_column(Float, default=0)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    video_path: Mapped[str] = mapped_column(String(1024))
    title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    hashtags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    job: Mapped[Job] = relationship(back_populates="cuts")
