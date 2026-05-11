from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from db.db_config import CoffeeKnowledgeCategory
from model.common_model import JsonList, OrmSchema


class CoffeeKnowledgeCreate(BaseModel):
    knowledge_id: str
    title: str
    content: str
    category: CoffeeKnowledgeCategory
    tags: JsonList = Field(default_factory=list)
    is_active: bool = True


class CoffeeKnowledgeRead(OrmSchema):
    id: UUID
    knowledge_id: str
    title: str
    content: str
    category: CoffeeKnowledgeCategory
    tags: JsonList
    is_active: bool
    created_at: datetime
    updated_at: datetime
