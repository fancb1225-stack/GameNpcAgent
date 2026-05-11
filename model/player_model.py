from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from model.common_model import JsonDict, JsonList, OrmSchema


class PlayerCreate(BaseModel):
    player_id: str = Field(min_length=1, max_length=64)
    nickname: str | None = Field(default=None, max_length=64)
    taste_preferences: JsonDict = Field(default_factory=dict)
    common_visit_periods: JsonList = Field(default_factory=list)
    frequent_npc_ids: JsonList = Field(default_factory=list)
    consumption_habits: JsonDict = Field(default_factory=dict)
    personality_tendency: JsonDict = Field(default_factory=dict)
    known_events: JsonList = Field(default_factory=list)
    npc_subjective_notes: JsonDict = Field(default_factory=dict)


class PlayerUpdate(BaseModel):
    nickname: str | None = None
    taste_preferences: JsonDict | None = None
    common_visit_periods: JsonList | None = None
    frequent_npc_ids: JsonList | None = None
    consumption_habits: JsonDict | None = None
    personality_tendency: JsonDict | None = None
    known_events: JsonList | None = None
    npc_subjective_notes: JsonDict | None = None

    def to_update_dict(self) -> dict:
        return self.model_dump(exclude_none=True)


class PlayerRead(OrmSchema):
    id: UUID
    player_id: str
    nickname: str | None
    taste_preferences: JsonDict
    common_visit_periods: JsonList
    frequent_npc_ids: JsonList
    consumption_habits: JsonDict
    personality_tendency: JsonDict
    known_events: JsonList
    npc_subjective_notes: JsonDict
    created_at: datetime
    updated_at: datetime
