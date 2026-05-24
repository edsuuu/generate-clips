from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.support.config import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=3600,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine, autoflush=False, autocommit=False, future=True
)


def get_session() -> Generator[Session, None, None]:
    """Dependency injection helper para FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Cria as tabelas se ainda não existirem (sem migrations)."""
    # Importa models para garantir que estão registrados em Base.metadata
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
