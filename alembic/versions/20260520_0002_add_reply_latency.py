"""add reply latency to dialogue messages

Revision ID: 20260520_0002
Revises: 20260507_0001
Create Date: 2026-05-20
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260520_0002"
down_revision = "20260507_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """为 NPC 回复消息保存实际生成耗时。"""
    op.add_column(
        "dialogue_messages",
        sa.Column("reply_latency_ms", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    """移除 NPC 回复消息实际生成耗时字段。"""
    op.drop_column("dialogue_messages", "reply_latency_ms")
