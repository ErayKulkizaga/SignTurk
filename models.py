from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.sql import func
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    full_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(Text, nullable=False)
    role = Column(String(50), default="User")
    status = Column(String(50), default="Active")
    sessions = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class History(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(20), nullable=False)
    mode = Column(String(50), default="Live")
    result = Column(String(255), nullable=False)
    confidence = Column(String(20), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Log(Base):
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    level = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    user = Column(String(255), default="system")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Setting(Base):
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, default=1)
    camera = Column(String(100), default="Default Camera")
    voice = Column(String(100), default="Female Voice")
    speech_rate = Column(Float, default=1.0)
    avatar_speed = Column(Float, default=1.0)
    tts_enabled = Column(Boolean, default=True)
    notifications_enabled = Column(Boolean, default=True)
    avatar_enabled = Column(Boolean, default=True)
    websocket_enabled = Column(Boolean, default=True)
