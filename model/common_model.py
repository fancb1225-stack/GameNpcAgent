from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class OrmSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, use_enum_values=True, arbitrary_types_allowed=True)


class IdSchema(OrmSchema):
    id: UUID


class TimestampSchema(OrmSchema):
    created_at: datetime
    updated_at: datetime


class PageQuery(BaseModel):
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class DeleteResult(BaseModel):
    deleted: int


class ErrorResult(BaseModel):
    detail: str


JsonDict = dict[str, Any]
JsonList = list[Any]
