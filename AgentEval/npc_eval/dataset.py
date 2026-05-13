from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    # 描述单条 NPC 评测用例的输入和可验证期望。
    case_id: str
    user_input: str
    expected_action: str | None = None
    required_keywords: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)


def load_cases(path: str | Path) -> list[EvalCase]:
    # 从 JSON 文件读取评测用例，便于策划或研发直接增删场景。
    raw_text = Path(path).read_text(encoding="utf-8")
    rows = json.loads(raw_text)

    # 将松散 JSON 转为稳定的数据结构，避免后续评分逻辑直接依赖字典字段。
    return [
        EvalCase(
            case_id=str(row["case_id"]),
            user_input=str(row["user_input"]),
            expected_action=row.get("expected_action"),
            required_keywords=list(row.get("required_keywords", [])),
            context=dict(row.get("context", {})),
        )
        for row in rows
    ]
