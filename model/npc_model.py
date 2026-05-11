from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from db.db_config import Emotion, NpcType
from model.common_model import JsonDict, JsonList, OrmSchema


class NpcCreate(BaseModel):
    npc_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=64)
    npc_type: NpcType
    role: str = Field(min_length=1, max_length=64)
    age: int | None = None
    gender: str | None = None
    mbti: str | None = None
    job: str | None = None
    hobbies: JsonList = Field(default_factory=list)
    coffee_preferences: JsonDict = Field(default_factory=dict)
    personality: str | None = None
    speaking_style: str | None = None
    background: str | None = None
    current_emotion: Emotion = Emotion.NEUTRAL
    current_location: str = "counter"
    is_active: bool = True


class NpcUpdate(BaseModel):
    name: str | None = None
    npc_type: NpcType | None = None
    role: str | None = None
    age: int | None = None
    gender: str | None = None
    mbti: str | None = None
    job: str | None = None
    hobbies: JsonList | None = None
    coffee_preferences: JsonDict | None = None
    personality: str | None = None
    speaking_style: str | None = None
    background: str | None = None
    current_emotion: Emotion | None = None
    current_location: str | None = None
    is_active: bool | None = None

    def to_update_dict(self) -> dict:
        return self.model_dump(exclude_none=True)


class NpcRead(OrmSchema):
    id: UUID
    npc_id: str
    name: str
    npc_type: NpcType
    role: str
    age: int | None
    gender: str | None
    mbti: str | None
    job: str | None
    hobbies: JsonList
    coffee_preferences: JsonDict
    personality: str | None
    speaking_style: str | None
    background: str | None
    current_emotion: Emotion
    current_location: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
