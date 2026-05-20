from __future__ import annotations

from typing import Any
from uuid import uuid4

from db.hybrid_retrieval import split_text_by_chapter
from db.db_service import DbService
from service.embedding_service import EmbeddingProvider, HashEmbeddingProvider


class GameDocumentService:
    """游戏设定文档导入和 RAG 检索服务。"""

    def __init__(
        self,
        db: Any | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        chunk_size: int = 2000,
        chunk_overlap: int = 120,
    ) -> None:
        # 保存依赖，便于测试替换数据库和 embedding 实现。
        self.db = db or DbService()
        self.embedding_provider = embedding_provider or HashEmbeddingProvider()
        self.chunk_size = max(1, int(chunk_size))
        self.chunk_overlap = max(0, min(int(chunk_overlap), self.chunk_size - 1))

    def ingest_game_setting_text(
        self,
        filename: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """导入游戏设定文本，按章节切分后写入向量知识库。"""

        # 清理输入，避免空文档进入知识库。
        clean_text = (text or "").strip()
        if not clean_text:
            return {"ok": False, "error": "游戏设定文档内容为空"}

        document_id = f"doc_{uuid4().hex[:16]}"
        source_metadata = {"filename": filename, **(metadata or {})}
        self.db.create_game_document(
            document_id=document_id,
            document_type="game_setting",
            filename=filename,
            metadata_json=source_metadata,
        )

        # 按章节切分知识片段，章节过长时再做带重叠窗口切分。
        chunks = []
        for index, content in enumerate(self.split_text(clean_text)):
            chapter_title = self._extract_chapter_title(content)
            chunks.append(
                {
                    "chunk_id": f"chunk_{uuid4().hex[:16]}",
                    "document_id": document_id,
                    "chunk_index": index,
                    "content": content,
                    "embedding": self.embedding_provider.embed_text(content),
                    "embedding_model": self.embedding_provider.model_name,
                    "metadata_json": {
                        "chunk_index": index,
                        "chapter_title": chapter_title,
                        "chunk_strategy": "chapter",
                        **source_metadata,
                    },
                }
            )

        saved_count = self.db.replace_game_chunks(document_id, chunks)
        return {
            "ok": True,
            "document_id": document_id,
            "filename": filename,
            "chunk_count": saved_count,
            "chunk_strategy": "chapter",
        }

    def split_text(self, text: str) -> list[str]:
        """按章节切分文本，兼容超长章节的二次窗口切分。"""

        return split_text_by_chapter(
            text,
            max_chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )

    def search_game_settings(self, query: str, limit: int = 5) -> dict[str, Any]:
        """使用 BM25 + 余弦相似度混合检索游戏设定片段。"""

        # 生成查询向量，并委托仓储层使用 ORM/pgvector 检索候选集。
        query_text = (query or "").strip()
        if not query_text:
            return {"ok": False, "error": "检索问题不能为空"}

        query_embedding = self.embedding_provider.embed_text(query_text)
        chunks = self.db.search_game_chunks(
            query_embedding=query_embedding,
            query=query_text,
            limit=max(1, int(limit)),
        )
        context = self.format_context(chunks)
        return {"ok": True, "query": query_text, "chunks": chunks, "context": context}

    def format_context(self, chunks: list[dict[str, Any]]) -> str:
        """把检索结果整理为 NPC 对话可读的上下文。"""

        # 使用稳定编号，方便模型引用来源但不暴露数据库细节。
        lines = []
        for index, chunk in enumerate(chunks, start=1):
            content = str(chunk.get("content") or "").strip()
            if content:
                lines.append(f"[{index}] {content}")
        return "\n".join(lines)

    def _extract_chapter_title(self, content: str) -> str | None:
        """从章节块第一行提取章节标题，供前端和排查链路使用。"""

        for line in (content or "").splitlines():
            clean_line = line.strip()
            if clean_line:
                return clean_line[:120]
        return None
