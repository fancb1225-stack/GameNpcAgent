"""
CoffeeNpcAgent database configuration and ORM models.

Install dependencies:
    pip install sqlalchemy psycopg2-binary python-dotenv

Environment variables:
    DATABASE_URL=postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/game_npc_frame
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from pgvector.sqlalchemy import Vector

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:root123456@127.0.0.1:5432/game_npc_frame",
)

engine = create_engine(
    DATABASE_URL,
    echo=os.getenv("DB_ECHO", "false").lower() == "true",
    pool_pre_ping=True,
    pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
    max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
)

DbSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class NpcType(StrEnum):
    STAFF = "staff"
    FIXED_CUSTOMER = "fixed_customer"
    RANDOM_CUSTOMER = "random_customer"


class Emotion(StrEnum):
    CALM = "calm"
    PLEASED = "pleased"
    CURIOUS = "curious"
    TIRED = "tired"
    STRESSED = "stressed"
    NEUTRAL = "neutral"


class SourceType(StrEnum):
    PLAYER = "player"
    NPC = "npc"


class TimePeriod(StrEnum):
    MORNING = "morning"
    NOON = "noon"
    AFTERNOON = "afternoon"


class CustomerFlow(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class EventType(StrEnum):
    DAILY = "daily"
    CUSTOMER = "customer"
    STAFF = "staff"
    WEATHER = "weather"
    SPECIAL = "special"


class SessionStatus(StrEnum):
    ACTIVE = "active"
    ENDED = "ended"


class MessageRole(StrEnum):
    PLAYER = "player"
    NPC = "npc"
    SYSTEM = "system"


class NpcAction(StrEnum):
    CHAT = "chat"
    ASK_PLAYER = "ask_player"
    COMMENT_ON_NPC = "comment_on_npc"
    MOVE_LOCATION = "move_location"
    END_DIALOGUE = "end_dialogue"


class MemoryType(StrEnum):
    PREFERENCE = "preference"
    RELATIONSHIP = "relationship"
    EVENT = "event"
    HABIT = "habit"
    WARNING = "warning"


class CoffeeKnowledgeCategory(StrEnum):
    BEAN = "bean"
    BREW = "brew"
    MILK = "milk"
    FLAVOR = "flavor"
    MENU = "menu"


class GameDocumentType(StrEnum):
    GAME_SETTING = "game_setting"
    NPC_SETTING = "npc_setting"


class UUIDPrimaryKeyMixin:
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class Player(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "players"

    player_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    nickname: Mapped[str | None] = mapped_column(String(64))
    taste_preferences: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    common_visit_periods: Mapped[list[Any]] = mapped_column(JSONB, default=list, nullable=False)
    frequent_npc_ids: Mapped[list[Any]] = mapped_column(JSONB, default=list, nullable=False)
    consumption_habits: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    personality_tendency: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    known_events: Mapped[list[Any]] = mapped_column(JSONB, default=list, nullable=False)
    npc_subjective_notes: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)


class Npc(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "npcs"

    npc_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    npc_type: Mapped[NpcType] = mapped_column(Enum(NpcType, name="npc_type"), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(64), nullable=False)
    age: Mapped[int | None] = mapped_column(Integer)
    gender: Mapped[str | None] = mapped_column(String(32))
    mbti: Mapped[str | None] = mapped_column(String(16))
    job: Mapped[str | None] = mapped_column(String(64))
    hobbies: Mapped[list[Any]] = mapped_column(JSONB, default=list, nullable=False)
    coffee_preferences: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    personality: Mapped[str | None] = mapped_column(Text)
    speaking_style: Mapped[str | None] = mapped_column(Text)
    background: Mapped[str | None] = mapped_column(Text)
    current_emotion: Mapped[Emotion] = mapped_column(
        Enum(Emotion, name="emotion"), default=Emotion.NEUTRAL, nullable=False
    )
    current_location: Mapped[str] = mapped_column(String(64), default="counter", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class Relationship(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "relationships"
    __table_args__ = (
        UniqueConstraint("source_type", "source_id", "target_type", "target_id", name="uq_relationship_pair"),
        Index("ix_relationship_source_id", "source_id"),
        Index("ix_relationship_target_id", "target_id"),
        Index("ix_relationship_last_interacted_at", "last_interacted_at"),
    )

    source_type: Mapped[SourceType] = mapped_column(Enum(SourceType, name="source_type"), nullable=False)
    source_id: Mapped[str] = mapped_column(String(64), nullable=False)
    target_type: Mapped[SourceType] = mapped_column(Enum(SourceType, name="target_type"), nullable=False)
    target_id: Mapped[str] = mapped_column(String(64), nullable=False)
    trust: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    familiarity: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    fondness: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    dislike: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    stress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_interacted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class WorldState(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "world_states"

    session_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    time_period: Mapped[TimePeriod] = mapped_column(Enum(TimePeriod, name="time_period"), index=True, nullable=False)
    customer_flow: Mapped[CustomerFlow] = mapped_column(Enum(CustomerFlow, name="customer_flow"), nullable=False)
    inventory: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    today_menu: Mapped[list[Any]] = mapped_column(JSONB, default=list, nullable=False)
    weather: Mapped[str] = mapped_column(String(64), nullable=False)
    background_music: Mapped[str] = mapped_column(String(128), nullable=False)
    seat_occupancy: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    current_event_ids: Mapped[list[Any]] = mapped_column(JSONB, default=list, nullable=False)


class Event(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "events"
    __table_args__ = (Index("ix_event_active_period", "is_active", "time_period"),)

    event_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(128), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    event_type: Mapped[EventType] = mapped_column(Enum(EventType, name="event_type"), nullable=False)
    time_period: Mapped[TimePeriod | None] = mapped_column(Enum(TimePeriod, name="event_time_period"))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict, nullable=False)


class EncounterRule(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "encounter_rules"
    __table_args__ = (Index("ix_encounter_rule_period_active", "time_period", "is_active"),)

    time_period: Mapped[TimePeriod] = mapped_column(Enum(TimePeriod, name="encounter_time_period"), nullable=False)
    fixed_npc_ids: Mapped[list[Any]] = mapped_column(JSONB, default=list, nullable=False)
    random_npc_pool: Mapped[list[Any]] = mapped_column(JSONB, default=list, nullable=False)
    random_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class DialogueSession(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "dialogue_sessions"
    __table_args__ = (Index("ix_dialogue_session_player_status", "player_id", "status"),)

    session_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    player_id: Mapped[str] = mapped_column(String(64), nullable=False)
    selected_npc_id: Mapped[str | None] = mapped_column(String(64))
    scene: Mapped[str] = mapped_column(Text, default="morning", nullable=False)
    time_period: Mapped[TimePeriod] = mapped_column(
        Enum(TimePeriod, name="dialogue_time_period"), default=TimePeriod.MORNING, nullable=False
    )
    status: Mapped[SessionStatus] = mapped_column(
        Enum(SessionStatus, name="session_status"), default=SessionStatus.ACTIVE, nullable=False
    )


class DialogueMessage(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "dialogue_messages"
    __table_args__ = (
        Index("ix_dialogue_message_session_created", "session_id", "created_at"),
        Index("ix_dialogue_message_player_npc_created", "player_id", "npc_id", "created_at"),
    )

    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    player_id: Mapped[str] = mapped_column(String(64), nullable=False)
    npc_id: Mapped[str | None] = mapped_column(String(64))
    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole, name="message_role"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str | None] = mapped_column(String(128))
    action: Mapped[NpcAction | None] = mapped_column(Enum(NpcAction, name="npc_action"))
    emotion: Mapped[Emotion | None] = mapped_column(Enum(Emotion, name="message_emotion"))
    relationship_delta: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    state_delta: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict, nullable=False)
    reply_latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class ShortTermMemory(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "short_term_memories"
    __table_args__ = (UniqueConstraint("player_id", "npc_id", name="uq_short_memory_player_npc"),)

    player_id: Mapped[str] = mapped_column(String(64), nullable=False)
    npc_id: Mapped[str] = mapped_column(String(64), nullable=False)
    messages: Mapped[list[Any]] = mapped_column(JSONB, default=list, nullable=False)
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_message_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class LongTermMemory(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "long_term_memories"
    __table_args__ = (
        Index("ix_long_memory_player_npc_type", "player_id", "npc_id", "memory_type"),
        Index("ix_long_memory_importance", "importance"),
        Index(
            "ix_long_memory_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    EMBEDDING_DIM = 384  # Must be consistent with the embedding provider and database configuration.

    memory_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    player_id: Mapped[str] = mapped_column(String(64), nullable=False)
    npc_id: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(128), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    memory_type: Mapped[MemoryType] = mapped_column(Enum(MemoryType, name="memory_type"), nullable=False)
    importance: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    source_message_ids: Mapped[list[Any]] = mapped_column(JSONB, default=list, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict, nullable=False)

    # pgvector 长期记忆向量字段
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)


class GameDocument(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "game_documents"
    __table_args__ = (
        Index("ix_game_document_type_created", "document_type", "created_at"),
    )

    document_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    document_type: Mapped[GameDocumentType] = mapped_column(
        Enum(GameDocumentType, name="game_document_type"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )


class GameSettingChunk(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "game_setting_chunks"
    __table_args__ = (
        Index("ix_game_setting_chunk_document", "document_id", "chunk_index"),
        Index(
            "ix_game_setting_chunk_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    EMBEDDING_DIM = 384

    chunk_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    document_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )


MODEL_REGISTRY: dict[str, type[Base]] = {
    "players": Player,
    "npcs": Npc,
    "relationships": Relationship,
    "world_states": WorldState,
    "events": Event,
    "encounter_rules": EncounterRule,
    "dialogue_sessions": DialogueSession,
    "dialogue_messages": DialogueMessage,
    "short_term_memories": ShortTermMemory,
    "long_term_memories": LongTermMemory,
    "game_documents": GameDocument,
    "game_setting_chunks": GameSettingChunk,
}


def get_db() -> Session:
    return DbSessionLocal()


def check_database_connection() -> bool:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return True
