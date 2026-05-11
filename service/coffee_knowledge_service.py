from __future__ import annotations

from db.db_config import CoffeeKnowledge, CoffeeKnowledgeCategory
from db.db_service import DbService
from model.coffee_model import CoffeeKnowledgeCreate, CoffeeKnowledgeRead


class CoffeeKnowledgeService:
    def __init__(self, db: DbService | None = None):
        self.db = db or DbService()
        self._owns_db = db is None

    def close(self) -> None:
        if self._owns_db:
            self.db.close()

    def create_knowledge(self, data: CoffeeKnowledgeCreate) -> CoffeeKnowledgeRead:
        row = self.db.create(CoffeeKnowledge, data.model_dump())
        return CoffeeKnowledgeRead.model_validate(row)

    def list_knowledge(self, category: CoffeeKnowledgeCategory | str | None = None, limit: int = 50) -> list[CoffeeKnowledgeRead]:
        rows = self.db.list_coffee_knowledge(category=category, active_only=True, limit=limit)
        return [CoffeeKnowledgeRead.model_validate(row) for row in rows]

    def get_knowledge(self, knowledge_id: str) -> CoffeeKnowledgeRead | None:
        row = self.db.get_one_by_field(CoffeeKnowledge, "knowledge_id", knowledge_id)
        return CoffeeKnowledgeRead.model_validate(row) if row else None

    def search_by_tag(self, tag: str, limit: int = 50) -> list[CoffeeKnowledgeRead]:
        rows = self.db.list_coffee_knowledge(active_only=True, limit=limit)
        matched = [row for row in rows if tag in (row.tags or [])]
        return [CoffeeKnowledgeRead.model_validate(row) for row in matched]
