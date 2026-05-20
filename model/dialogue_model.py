from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from db.db_config import Emotion, MessageRole, NpcAction, SessionStatus, TimePeriod
from model.common_model import JsonDict, OrmSchema
from model.relationship_model import RelationshipDelta


class DialogueSessionCreate(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    player_id: str = Field(min_length=1, max_length=64)
    selected_npc_id: str | None = None
    scene: str = "morning_cafe"
    time_period: TimePeriod = TimePeriod.MORNING
    status: SessionStatus = SessionStatus.ACTIVE


class DialogueSessionRead(OrmSchema):
    id: UUID
    session_id: str
    player_id: str
    selected_npc_id: str | None
    scene: str
    time_period: TimePeriod
    status: SessionStatus
    created_at: datetime
    updated_at: datetime


class DialogueMessageCreate(BaseModel):
    session_id: str
    player_id: str
    role: MessageRole
    content: str = Field(min_length=1)
    npc_id: str | None = None
    intent: str | None = None
    action: NpcAction | None = None
    emotion: Emotion | None = None
    relationship_delta: JsonDict = Field(default_factory=dict)
    state_delta: JsonDict = Field(default_factory=dict)
    metadata_json: JsonDict = Field(default_factory=dict)
    reply_latency_ms: int | None = None


class PlayerMessageCreate(BaseModel):
    session_id: str
    player_id: str
    npc_id: str
    content: str = Field(min_length=1)


class NpcReplyCreate(BaseModel):
    session_id: str
    player_id: str
    npc_id: str
    content: str = Field(min_length=1)
    intent: str | None = None
    action: NpcAction = NpcAction.CHAT
    emotion: Emotion = Emotion.NEUTRAL
    relationship_delta: RelationshipDelta = Field(default_factory=RelationshipDelta)
    state_delta: JsonDict = Field(default_factory=dict)
    metadata_json: JsonDict = Field(default_factory=dict)
    reply_started_at: float | None = None


class DialogueMessageRead(OrmSchema):
    id: UUID
    session_id: str
    player_id: str
    npc_id: str | None
    role: MessageRole
    content: str
    intent: str | None
    action: NpcAction | None
    emotion: Emotion | None
    relationship_delta: JsonDict
    state_delta: JsonDict
    metadata_json: JsonDict
    reply_latency_ms: int | None
    created_at: datetime
