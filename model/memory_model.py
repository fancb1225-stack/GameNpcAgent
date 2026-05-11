from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from db.db_config import MemoryType
from model.common_model import JsonDict, JsonList, OrmSchema


class ShortMemoryMessage(BaseModel):
    role: str
    content: str
    timestamp: str | None = None
    emotion: str | None = None
    affinity_delta: int | None = None
    metadata: JsonDict = Field(default_factory=dict)


class ShortTermMemoryRead(OrmSchema):
    id: UUID
    player_id: str
    npc_id: str
    messages: JsonList
    message_count: int
    last_message_at: datetime | None
    created_at: datetime
    updated_at: datetime


class LongTermMemoryCreate(BaseModel):
    memory_id: str
    player_id: str
    npc_id: str
    title: str
    content: str
    memory_type: MemoryType
    importance: float = Field(default=0.5, ge=0, le=1)
    source_message_ids: JsonList = Field(default_factory=list)
    metadata_json: JsonDict = Field(default_factory=dict)


class LongTermMemoryRead(OrmSchema):
    id: UUID
    memory_id: str
    player_id: str
    npc_id: str
    title: str
    content: str
    memory_type: MemoryType
    importance: float
    source_message_ids: JsonList
    metadata_json: JsonDict
    created_at: datetime
    updated_at: datetime
