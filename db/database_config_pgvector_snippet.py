"""
Copy these changes into db/database_config.py.
Only the LongTermMemory model and imports are shown here.
"""

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, String, Float, JSON, DateTime, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

# Keep this value consistent across database_config.py, db_init.py and EmbeddingService.
EMBEDDING_DIM = 384

class LongTermMemory(Base):
    __tablename__ = "long_term_memories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    memory_id = Column(String(64), unique=True, nullable=False, index=True)
    player_id = Column(String(64), nullable=False, index=True)
    npc_id = Column(String(64), nullable=False, index=True)

    title = Column(String(200), nullable=False)
    content = Column(String, nullable=False)
    memory_type = Column(String(32), nullable=False, default="event")
    importance = Column(Float, nullable=False, default=0.5)

    # pgvector column. Dimension must match the embedding provider.
    embedding = Column(Vector(EMBEDDING_DIM), nullable=True)
    embedding_model = Column(String(128), nullable=True)

    source_message_ids = Column(JSON, nullable=False, default=list)
    meta_data = Column("metadata", JSON, nullable=False, default=dict)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_long_term_memories_player_npc_type", "player_id", "npc_id", "memory_type"),
    )
