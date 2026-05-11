from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from db.db_config import SourceType
from model.common_model import OrmSchema


class RelationshipDelta(BaseModel):
    trust: int = 0
    familiarity: int = 0
    fondness: int = 0
    dislike: int = 0
    stress: int = 0

    def as_dict(self) -> dict[str, int]:
        return self.model_dump()


class RelationshipCreate(BaseModel):
    source_type: SourceType
    source_id: str = Field(min_length=1, max_length=64)
    target_type: SourceType
    target_id: str = Field(min_length=1, max_length=64)


class RelationshipRead(OrmSchema):
    id: UUID
    source_type: SourceType
    source_id: str
    target_type: SourceType
    target_id: str
    trust: int
    familiarity: int
    fondness: int
    dislike: int
    stress: int
    last_interacted_at: datetime | None
    created_at: datetime
    updated_at: datetime
