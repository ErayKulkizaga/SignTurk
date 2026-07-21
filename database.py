"""Database configuration for local development and hosted deployments."""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


BASE_DIR = Path(__file__).resolve().parent
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
if not DATABASE_URL:
    DATABASE_URL = f"sqlite:///{(BASE_DIR / 'local_dev.db').as_posix()}"


def _engine_kwargs(url: str) -> dict:
    if url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    return {"pool_pre_ping": True}


engine = create_engine(DATABASE_URL, **_engine_kwargs(DATABASE_URL))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def database_name() -> str:
    """Return a public, credential-free backend label."""
    return "SQLite (local)" if DATABASE_URL.startswith("sqlite") else "PostgreSQL"


def get_db():
    """Yield a SQLAlchemy session and always close it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
