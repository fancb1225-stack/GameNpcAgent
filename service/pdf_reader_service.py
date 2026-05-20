from __future__ import annotations

from io import BytesIO


class PdfReaderService:
    """PDF 文档文本抽取服务。"""

    def extract_text(self, content: bytes) -> str:
        """从 PDF 二进制内容中抽取文本。"""

        # 使用 pypdf 按页抽取文本，保留页间换行，便于后续切分。
        if not content:
            return ""
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(page.strip() for page in pages if page.strip())
