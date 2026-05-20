from __future__ import annotations

import json
import re
from typing import Any

from db.db_config import Emotion, NpcType
from db.db_service import DbService
from model.npc_model import NpcRead


class NpcSettingImportService:
    """NPC 设定抽取和入库服务。"""

    def __init__(self, db: Any | None = None, llm_service: Any | None = None) -> None:
        # 保存依赖，LLM 为空或不可用时走规则兜底解析。
        self.db = db or DbService()
        self.llm_service = llm_service

    def import_npc_setting_text(self, text: str) -> dict[str, Any]:
        """从 NPC 设定文本中抽取结构化 NPC，并通过 ORM 写入数据库。"""

        # 先校验 PDF 已抽取出的文本，避免空文档写入脏数据。
        source_text = (text or "").strip()
        if not source_text:
            return {"ok": False, "error": "NPC 设定文档内容为空"}

        payload = self.extract_npc_payload(source_text)
        npc_rows = payload.get("npcs") if isinstance(payload, dict) else None
        if not isinstance(npc_rows, list) or not npc_rows:
            return {"ok": False, "error": "未识别到 NPC 设定"}

        # 数据库层仍使用 ORM upsert，API 响应层只返回可 JSON 序列化的字典。
        imported = []
        for raw_row in npc_rows:
            clean_row = self.normalize_npc_row(raw_row)
            imported.append(self.serialize_imported_npc(self.db.upsert_npc_from_setting(clean_row)))

        return {"ok": True, "imported_count": len(imported), "npcs": imported}

    def serialize_imported_npc(self, row: Any) -> dict[str, Any]:
        """将导入后的 NPC 结果转换为 API 可序列化字典。"""

        # 仓储测试替身可能直接返回字典，真实数据库层通常返回 SQLAlchemy ORM 对象。
        if isinstance(row, dict):
            return row
        return NpcRead.model_validate(row).model_dump(mode="json")

    def extract_npc_payload(self, text: str) -> dict[str, Any]:
        """使用 Agent 抽取 NPC 写入意图，失败时走规则兜底。"""

        # LLM 只负责生成结构化意图，实际数据库写入仍由 ORM 服务执行。
        if self.llm_service is not None:
            prompt = json.dumps(
                {
                    "task": "从游戏 NPC 设定中抽取可写入数据库的结构化 NPC 数据。",
                    "input": text,
                    "output_schema": {
                        "npcs": [
                            {
                                "npc_id": "稳定英文或拼音 ID",
                                "name": "NPC 名称",
                                "npc_type": "staff|fixed_customer|random_customer",
                                "role": "身份或定位",
                                "age": None,
                                "gender": None,
                                "mbti": None,
                                "job": None,
                                "hobbies": [],
                                "personality": "性格",
                                "speaking_style": "说话风格",
                                "background": "背景",
                                "current_location": "当前位置",
                            }
                        ]
                    },
                },
                ensure_ascii=False,
            )
            try:
                response, _ = self.llm_service.chat(
                    prompt=prompt,
                    history=[],
                    system_prompt="你是游戏 NPC 设定入库 Agent，只输出合法 JSON。",
                )
            except Exception:
                response = ""
            parsed = self.extract_json_object(response)
            if parsed:
                return parsed

        # 没有 LLM 或模型输出异常时，使用规则解析单个 NPC。
        return {"npcs": [self.rule_based_extract_one(text)]}

    def extract_json_object(self, text: str) -> dict[str, Any]:
        """从模型输出中提取 JSON 对象。"""

        # 依次支持纯 JSON、Markdown 代码块和正文包裹 JSON。
        raw_text = (text or "").strip()
        candidates = [raw_text]
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.S)
        if fenced:
            candidates.append(fenced.group(1))
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start >= 0 and end > start:
            candidates.append(raw_text[start:end + 1])

        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                return data
        return {}

    def rule_based_extract_one(self, text: str) -> dict[str, Any]:
        """规则兜底解析一个 NPC。"""

        # 兼容 PDF 表格抽取后的“字段 值”和普通“字段：值”两种形式。
        name = self.extract_field(text, ["姓名", "名字", "NPC 设定"]) or "未命名 NPC"
        npc_id = self.extract_field(text, ["npc_id", "NPC ID", "id"]) or self.make_safe_id(name)
        npc_type = self.extract_field(text, ["npc_type", "NPC 类型", "类型"]) or NpcType.FIXED_CUSTOMER.value
        role = self.extract_field(text, ["身份", "角色定位", "定位"])
        job = self.extract_field(text, ["职业", "工作"])
        personality = self.extract_field(text, ["性格", "人格"]) or ""
        speaking_style = self.extract_field(text, ["说话风格", "语言风格"])
        background = self.extract_field(text, ["背景", "人物背景"])
        current_location = self.extract_field(text, ["当前位置", "位置"]) or "unknown"

        return {
            "npc_id": npc_id,
            "name": name,
            "npc_type": npc_type,
            "role": role or job or "未分类 NPC",
            "age": self.extract_int_field(text, ["年龄"]),
            "gender": self.extract_field(text, ["性别"]),
            "mbti": self.extract_field(text, ["MBTI", "mbti"]),
            "job": job,
            "personality": personality,
            "speaking_style": speaking_style,
            "background": background,
            "current_location": current_location,
        }

    def extract_int_field(self, text: str, labels: list[str]) -> int | None:
        """按字段名提取整数。"""

        # 年龄等数值字段允许为空，避免非数字内容导致导入失败。
        value = self.extract_field(text, labels)
        if value is None:
            return None
        match = re.search(r"\d+", value)
        return int(match.group(0)) if match else None

    def extract_field(self, text: str, labels: list[str]) -> str | None:
        """按中文或英文标签提取单行字段。"""

        # 将 PDF 抽取文本拆成干净行，支持表格行和标题行。
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
        for index, line in enumerate(lines):
            for label in labels:
                value = self.extract_inline_field(line, label)
                if value:
                    return value
                if self.same_label(line, label) and index + 1 < len(lines):
                    return self.collect_following_value(lines, index + 1)
        return None

    def collect_following_value(self, lines: list[str], start_index: int) -> str | None:
        """收集标题字段后方的多行内容。"""

        # PDF 会把长段落按视觉宽度断行，因此需要一直读到下一个字段标题。
        values = []
        for line in lines[start_index:]:
            if self.is_known_label_line(line):
                break
            values.append(self.clean_value(line))
        text = "\n".join(value for value in values if value)
        return text or None

    def is_known_label_line(self, line: str) -> bool:
        """判断当前行是否是新的字段标题。"""

        # 用固定标签集合判断段落边界，避免把下一个字段并进当前字段。
        labels = {
            "姓名",
            "名字",
            "NPC 设定",
            "npc_id",
            "NPC ID",
            "id",
            "npc_type",
            "NPC 类型",
            "类型",
            "身份",
            "角色定位",
            "定位",
            "职业",
            "工作",
            "性格",
            "人格",
            "说话风格",
            "语言风格",
            "背景",
            "人物背景",
            "当前位置",
            "位置",
            "年龄",
            "性别",
            "MBTI",
            "mbti",
            "目标",
            "秘密",
            "对玩家态度",
            "一、结构化字段",
            "二、人物设定",
            "三、导入提示",
        }
        normalized = line.rstrip(":：").strip()
        return normalized in labels

    def extract_inline_field(self, line: str, label: str) -> str | None:
        """从单行中提取字段值。"""

        # 支持“姓名：沈照微”“姓名 沈照微”“npc_id master_xxx”。
        pattern = rf"^{re.escape(label)}\s*[:：]?\s+(.+)$"
        match = re.search(pattern, line, flags=re.I)
        if match:
            return self.clean_value(match.group(1))

        # 支持“NPC 设定：沈照微”这种标题式字段。
        pattern = rf"^{re.escape(label)}\s*[:：]\s*(.+)$"
        match = re.search(pattern, line, flags=re.I)
        if match:
            return self.clean_value(match.group(1))
        return None

    def same_label(self, line: str, label: str) -> bool:
        """判断一整行是否只是字段名。"""

        # 去掉常见冒号，兼容 Word/PDF 抽取后的标题行。
        return line.rstrip(":：").lower() == label.lower()

    def clean_value(self, value: str) -> str:
        """清理字段值。"""

        # 去除项目符号和句尾空白，保留中文标点内容本身。
        return value.strip().lstrip("•-").strip()

    def make_safe_id(self, name: str) -> str:
        """为缺少 npc_id 的设定生成稳定兜底 ID。"""

        # 使用 hash 兜底，避免同名空 ID 写库失败。
        return f"npc_{abs(hash(name)) % 100000:05d}"

    def normalize_npc_row(self, row: Any) -> dict[str, Any]:
        """清洗 Agent 输出，限制为 ORM 可写入字段。"""

        # 过滤未知字段并补齐默认值，避免模型输出污染数据库结构。
        raw = row if isinstance(row, dict) else {}
        npc_type = str(raw.get("npc_type") or NpcType.RANDOM_CUSTOMER.value)
        if npc_type not in {item.value for item in NpcType}:
            npc_type = NpcType.RANDOM_CUSTOMER.value

        emotion = str(raw.get("current_emotion") or Emotion.NEUTRAL.value)
        if emotion not in {item.value for item in Emotion}:
            emotion = Emotion.NEUTRAL.value

        name = str(raw.get("name") or "未命名 NPC").strip()
        npc_id = str(raw.get("npc_id") or self.make_safe_id(name)).strip()
        return {
            "npc_id": npc_id,
            "name": name,
            "npc_type": NpcType(npc_type),
            "role": str(raw.get("role") or raw.get("job") or "未分类 NPC"),
            "age": raw.get("age"),
            "gender": raw.get("gender"),
            "mbti": raw.get("mbti"),
            "job": raw.get("job"),
            "hobbies": raw.get("hobbies") if isinstance(raw.get("hobbies"), list) else [],
            "coffee_preferences": (
                raw.get("coffee_preferences")
                if isinstance(raw.get("coffee_preferences"), dict)
                else {}
            ),
            "personality": raw.get("personality"),
            "speaking_style": raw.get("speaking_style"),
            "background": raw.get("background"),
            "current_emotion": Emotion(emotion),
            "current_location": str(raw.get("current_location") or "unknown"),
            "is_active": bool(raw.get("is_active", True)),
        }
