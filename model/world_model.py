from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from db.db_config import CustomerFlow, EventType, TimePeriod
from model.common_model import JsonDict, JsonList, OrmSchema


class CafeWorldStateCreate(BaseModel):
    session_id: str
    time_period: TimePeriod = TimePeriod.MORNING
    customer_flow: CustomerFlow = CustomerFlow.NORMAL
    inventory: JsonDict = Field(default_factory=dict)
    today_menu: JsonList = Field(default_factory=list)
    weather: str = "晴"
    background_music: str = "轻音乐"
    seat_occupancy: JsonDict = Field(default_factory=dict)
    current_event_ids: JsonList = Field(default_factory=list)


class CafeWorldStateUpdate(BaseModel):
    time_period: TimePeriod | None = None
    customer_flow: CustomerFlow | None = None
    inventory: JsonDict | None = None
    today_menu: JsonList | None = None
    weather: str | None = None
    background_music: str | None = None
    seat_occupancy: JsonDict | None = None
    current_event_ids: JsonList | None = None

    def to_update_dict(self) -> dict:
        return self.model_dump(exclude_none=True)


class CafeWorldStateRead(OrmSchema):
    id: UUID
    session_id: str
    time_period: TimePeriod
    customer_flow: CustomerFlow
    inventory: JsonDict
    today_menu: JsonList
    weather: str
    background_music: str
    seat_occupancy: JsonDict
    current_event_ids: JsonList
    created_at: datetime
    updated_at: datetime


class CafeEventRead(OrmSchema):
    id: UUID
    event_id: str
    title: str
    content: str
    event_type: EventType
    time_period: TimePeriod | None
    is_active: bool
    metadata_json: JsonDict
    created_at: datetime
    updated_at: datetime


class EncounterRuleRead(OrmSchema):
    id: UUID
    time_period: TimePeriod
    fixed_npc_ids: JsonList
    random_npc_pool: JsonList
    random_count: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
