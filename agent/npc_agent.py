"""
Single NPC Agent for CoffeeNpcAgent.

职责：
1. 收集 NPC 回复所需上下文。
2. 调用 LLM 生成结构化决策 JSON。
3. 校验 action schema。
4. 生成或修正 NPC 台词。
5. 写入对话、关系变化、状态变化和记忆。

该文件只依赖已有工具层，不直接访问数据库，不写原始 SQL。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Tuple

from patch_ng import debugmode

from service.long_memory_service import LongMemoryService
from service.short_memory_service import ShortMemoryService

try:
    from agent.coffee_npc_agent_tools import CoffeeNpcAgentTools
except Exception:  # 允许在不同项目结构下由调用方注入 tools
    CoffeeNpcAgentTools = Any  # type: ignore


class ChatModel(Protocol):
    """兼容你已有的 LlmService.chat(prompt, history, system_prompt)。"""

    def chat(
        self,
        prompt: str,
        history: List[Dict[str, str]] | None = None,
        system_prompt: str = "",
    ) -> Tuple[str, List[Dict[str, str]]]:
        ...


ALLOWED_ACTIONS = {
    "chat",
    "recommend_coffee",
    "ask_player",
    "comment_on_npc",
    "move_location",
    "serve_customer",
    "end_dialogue",
}

ALLOWED_EMOTIONS = {
    "neutral",
    "calm",
    "pleased",
    "happy",
    "curious",
    "worried",
    "annoyed",
    "sad",
    "angry",
    "tired",
}

RELATIONSHIP_FIELDS = {"trust", "familiarity", "fondness", "dislike", "stress"}

NPC_AGENT_SYSTEM_PROMPT = """
你是 CoffeeNpcAgent 中的单个咖啡厅 NPC 控制器。
你不能自由规划复杂任务，只能完成“理解-决策-行动-台词”闭环。

必须只输出一个 JSON object，不要输出 Markdown，不要解释。
JSON 字段固定如下：
{
  "intent": "玩家意图，例如 greet / ask_recommendation / ask_event / small_talk / offend / farewell",
  "action": "chat | recommend_coffee | ask_player | comment_on_npc | move_location | serve_customer | end_dialogue",
  "target_npc_id": "当前 NPC ID",
  "emotion": "neutral | calm | pleased | happy | curious | worried | annoyed | sad | angry | tired",
  "line_goal": "本轮台词目标",
  "line": "NPC 最终对玩家说的话，必须符合角色设定，1到3句话",
  "relationship_delta": {
    "trust": 0,
    "familiarity": 0,
    "fondness": 0,
    "dislike": 0,
    "stress": 0
  },
  "state_delta": {
    "npc_location": null,
    "npc_emotion": null
  },
  "memory_tags": []
}

硬性约束：
1. action 只能来自固定枚举，不能自创 action。
2. relationship_delta 每个字段范围建议在 -3 到 3。
3. 推荐咖啡必须基于上下文中的今日菜单和库存，不允许推荐库存不足的咖啡。
4. 事实性内容必须基于上下文，不知道就用角色口吻承认不知道。
5. move_location 必须给出合法 npc_location；否则使用 chat。
6. end_dialogue 只输出结束台词。
""".strip()


@dataclass
class NpcDecision:
    intent: str = "small_talk"
    action: str = "chat"
    target_npc_id: Optional[str] = None
    emotion: str = "neutral"
    line_goal: str = "自然回应玩家"
    line: str = "嗯，我听着。"
    relationship_delta: Dict[str, int] = field(default_factory=lambda: {
        "trust": 0,
        "familiarity": 0,
        "fondness": 0,
        "dislike": 0,
        "stress": 0,
    })
    state_delta: Dict[str, Any] = field(default_factory=dict)
    memory_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "action": self.action,
            "target_npc_id": self.target_npc_id,
            "emotion": self.emotion,
            "line_goal": self.line_goal,
            "line": self.line,
            "relationship_delta": self.relationship_delta,
            "state_delta": self.state_delta,
            "memory_tags": self.memory_tags,
        }


class NpcAgent:
    """单个 NPC Agent。"""

    def __init__(
        self,
        npc_id: str,
        tools: CoffeeNpcAgentTools,
        llm: Optional[ChatModel] = None,
        max_history: int = 20,
    ) -> None:
        if not npc_id:
            raise ValueError("npc_id 不能为空")
        self.npc_id = npc_id
        self.tools = tools
        self.llm = llm
        self.max_history = max(1, min(int(max_history), 100))
        self.llm_history: List[Dict[str, str]] = []
        self.short_memory_service = ShortMemoryService()
        self.long_memory_service = LongMemoryService()

    def handle_player_message(
        self,
        player_id: str,
        session_id: str,
        content: str,
    ) -> Dict[str, Any]:
        """
        处理一轮玩家输入，并返回 NPC 回复结果。

        返回结构：
        {
            "ok": bool,
            "player_message": {...},
            "decision": {...},
            "npc_reply": {...},
            "tool_results": {...}
        }
        """
        if not player_id:
            return self._fail("缺少 player_id")
        if not session_id:
            return self._fail("缺少 session_id")
        if not content or not content.strip():
            return self._fail("玩家输入不能为空")

        tool_results: Dict[str, Any] = {}

        player_msg = self.tools.invoke_tool(
            "write_player_message",
            {
                "session_id": session_id,
                "player_id": player_id,
                "npc_id": self.npc_id,
                "message": content or content.strip(),
            },
        )
        tool_results["write_player_message"] = player_msg
        if not player_msg.get("ok"):
            return self._fail("写入玩家消息失败", tool_results=tool_results)

        context = self.collect_context(player_id=player_id, session_id=session_id, player_input=content)
        tool_results.update(context.get("tool_results", {}))

        decision = self.decide(player_id=player_id, session_id=session_id, player_input=content, context=context)
        decision = self.validate_decision(decision)

        action_check = self.tools.invoke_tool(
            "check_action_allowed",
            {
                "action": decision.action,
                "npc_id": self.npc_id,
                "target_location": decision.state_delta.get("npc_location"),
            },
        )
        tool_results["check_action_allowed"] = action_check
        if not action_check.get("allowed", False):
            decision.action = action_check.get("fallback_action") or "chat"
            decision.state_delta.pop("npc_location", None)

        # 如果 LLM 选择了推荐咖啡，再补一次专用上下文，避免推荐脱离库存和今日菜单。
        if decision.action == "recommend_coffee":
            rec_context = self.tools.invoke_tool(
                "get_recommendation_context",
                {
                    "player_id": player_id,
                    "npc_id": self.npc_id,
                    "session_id": session_id,
                },
            )
            tool_results["get_recommendation_context"] = rec_context
            decision.line = self.ensure_recommendation_line(decision.line, rec_context)

        reply_result = self.tools.invoke_tool(
            "write_npc_reply",
            {
                "session_id": session_id,
                "player_id": player_id,
                "npc_id": self.npc_id,
                "content": decision.line,
                "intent": decision.intent,
                "action": decision.action,
                "emotion": decision.emotion,
                "relationship_delta": decision.relationship_delta,
                "state_delta": decision.state_delta,
                "metadata_json": {
                    "line_goal": decision.line_goal,
                    "memory_tags": decision.memory_tags,
                    "raw_decision": decision.to_dict(),
                },
            },
        )
        tool_results["write_npc_reply"] = reply_result
        if not reply_result.get("ok"):
            return self._fail("写入 NPC 回复失败", tool_results=tool_results, decision=decision.to_dict())

        rel_result = self.tools.invoke_tool(
            "apply_relationship_delta",
            {
                "player_id": player_id,
                "npc_id": self.npc_id,
                "delta": decision.relationship_delta,
            },
        )
        tool_results["apply_relationship_delta"] = rel_result

        profile_result = self.maybe_update_player_profile(player_id=player_id, player_input=content)
        if profile_result is not None:
            tool_results["update_player_profile"] = profile_result

        self.short_memory_service.append_short_term_message(
            player_id=player_id,
            npc_id=self.npc_id,
            message={
                "role": "player",
                "content": player_msg,
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        archive_result = self.short_memory_service.append_short_term_message(
            player_id=player_id,
            npc_id=self.npc_id,
            message={
                "role": "npc",
                "content": reply_result,
                "session_id": session_id,
                "intent": decision.intent,
                "action": decision.action,
                "emotion": decision.emotion,
                "relationship_delta": decision.relationship_delta,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        if archive_result.get("needs_archive"):
            long_result = self.long_memory_service.create_long_term_memory_from_messages(
                player_id=player_id,
                npc_id=self.npc_id,
                messages=archive_result.get("archive_messages", []),
                llm_service=self.llm,
            )
            if long_result.get("ok"):
                self.short_memory_service.trim_short_term_memory(
                    player_id=player_id,
                    npc_id=self.npc_id,
                    remove_count=archive_result.get("archive_message_count", 0),
                )



        return {
            "ok": True,
            "player_id": player_id,
            "npc_id": self.npc_id,
            "session_id": session_id,
            "player_message": player_msg,
            "decision": decision.to_dict(),
            "npc_reply": reply_result,
            "tool_results": tool_results,
        }

    def collect_context(self, player_id: str, session_id: str, player_input: str) -> Dict[str, Any]:
        """收集决策上下文。"""
        results: Dict[str, Any] = {}
        calls = {
            "npc_state": ("get_npc_state", {"npc_id": self.npc_id}),
            "player_profile": ("get_player_profile", {"player_id": player_id}),
            "relationship": ("get_relationship", {"player_id": player_id, "npc_id": self.npc_id}),
            "history": (
                "get_history_message",
                {
                    "player_id": player_id,
                    "npc_id": self.npc_id,
                    "session_id": session_id,
                    "limit": self.max_history,
                },
            ),
            "short_term_memory": ("get_short_term_memory", {"player_id": player_id, "npc_id": self.npc_id}),
            "long_term_memories": (
                "get_long_term_memories",
                {"player_id": player_id, "npc_id": self.npc_id, "limit": 10},
            ),
            "world_state": ("get_world_state", {"session_id": session_id}),
            "cafe_events": ("get_cafe_events", {"time_period": "morning"}),
        }

        for key, (tool_name, args) in calls.items():
            results[key] = self.tools.invoke_tool(tool_name, args)

        if self._looks_like_coffee_question(player_input):
            results["coffee_knowledge"] = self.tools.invoke_tool(
                "get_coffee_knowledge",
                {"limit": 20},
            )

        return {
            "player_id": player_id,
            "npc_id": self.npc_id,
            "session_id": session_id,
            "player_input": player_input,
            "tool_results": results,
        }

    def decide(self, player_id: str, session_id: str, player_input: str, context: Dict[str, Any]) -> NpcDecision:
        """调用 LLM 决策。没有 LLM 时使用规则降级。"""
        if self.llm is None:
            return self.rule_based_decision(player_input=player_input, context=context)

        prompt = self.build_decision_prompt(player_id, session_id, player_input, context)
        try:
            response, updated_history = self.llm.chat(
                prompt=prompt,
                history=self.llm_history,
                system_prompt=NPC_AGENT_SYSTEM_PROMPT,
            )
            self.llm_history = updated_history[-20:]
            data = self.extract_json_object(response)
            return self.decision_from_dict(data)
        except Exception:
            return self.rule_based_decision(player_input=player_input, context=context)

    def build_decision_prompt(
        self,
        player_id: str,
        session_id: str,
        player_input: str,
        context: Dict[str, Any],
    ) -> str:
        compact_context = self.compact_context(context)
        return json.dumps(
            {
                "task": "根据上下文为当前 NPC 生成一轮结构化决策和台词。",
                "player_id": player_id,
                "npc_id": self.npc_id,
                "session_id": session_id,
                "player_input": player_input,
                "context": compact_context,
                "output_requirement": "只输出合法 JSON object。",
            },
            ensure_ascii=False,
            default=str,
        )

    def compact_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """压缩上下文，避免把过大的工具结果直接塞给 LLM。"""
        tool_results = context.get("tool_results", {})
        compact: Dict[str, Any] = {}
        for key, value in tool_results.items():
            if not isinstance(value, dict):
                compact[key] = value
                continue
            clean = {k: v for k, v in value.items() if k not in {"tool_name"}}
            compact[key] = clean
        return compact

    def rule_based_decision(self, player_input: str, context: Dict[str, Any]) -> NpcDecision:
        """无 LLM 或 LLM 失败时的稳定降级逻辑。"""
        text = player_input.strip()
        npc_name = self._get_npc_name(context) or self.npc_id

        if any(word in text for word in ["推荐", "咖啡", "喝什么", "菜单", "拿铁", "美式", "手冲"]):
            return NpcDecision(
                intent="ask_recommendation",
                action="recommend_coffee",
                target_npc_id=self.npc_id,
                emotion="pleased",
                line_goal="基于今日菜单推荐咖啡",
                line="今天可以试试店里的拿铁，口感比较稳。如果你想要更清爽一点，也可以看看今日手冲。",
                relationship_delta={"trust": 0, "familiarity": 1, "fondness": 1, "dislike": 0, "stress": 0},
                state_delta={"npc_emotion": "pleased"},
                memory_tags=["coffee_preference"],
            )

        if any(word in text for word in ["再见", "拜拜", "下次见", "走了"]):
            return NpcDecision(
                intent="farewell",
                action="end_dialogue",
                target_npc_id=self.npc_id,
                emotion="calm",
                line_goal="自然结束对话",
                line=f"{npc_name}点了点头：下次来店里，我还在这儿。",
                relationship_delta={"trust": 0, "familiarity": 1, "fondness": 0, "dislike": 0, "stress": 0},
                state_delta={"npc_emotion": "calm"},
                memory_tags=["farewell"],
            )

        if any(word in text for word in ["讨厌", "难喝", "烦", "闭嘴", "差劲"]):
            return NpcDecision(
                intent="offend",
                action="chat",
                target_npc_id=self.npc_id,
                emotion="annoyed",
                line_goal="克制回应冒犯",
                line="我听到了。要是哪里不合口味，可以直接说具体一点，我会按店里的规矩处理。",
                relationship_delta={"trust": -1, "familiarity": 0, "fondness": -1, "dislike": 1, "stress": 2},
                state_delta={"npc_emotion": "annoyed"},
                memory_tags=["negative_interaction"],
            )

        return NpcDecision(
            intent="small_talk",
            action="chat",
            target_npc_id=self.npc_id,
            emotion="calm",
            line_goal="自然闲聊并引导玩家继续表达",
            line="嗯，我在听。今天店里还算安静，你想聊点什么？",
            relationship_delta={"trust": 0, "familiarity": 1, "fondness": 0, "dislike": 0, "stress": 0},
            state_delta={"npc_emotion": "calm"},
            memory_tags=["small_talk"],
        )

    def validate_decision(self, decision: NpcDecision) -> NpcDecision:
        """本地 schema 校验，保证不会把非法 action 写入数据库。"""
        if decision.action not in ALLOWED_ACTIONS:
            decision.action = "chat"
        if decision.emotion not in ALLOWED_EMOTIONS:
            decision.emotion = "neutral"
        if not decision.target_npc_id:
            decision.target_npc_id = self.npc_id
        if not decision.line or not str(decision.line).strip():
            decision.line = "嗯，我听着。"

        clean_delta: Dict[str, int] = {}
        for field_name in RELATIONSHIP_FIELDS:
            raw = decision.relationship_delta.get(field_name, 0)
            try:
                value = int(raw)
            except Exception:
                value = 0
            clean_delta[field_name] = max(-5, min(5, value))
        decision.relationship_delta = clean_delta

        if not isinstance(decision.state_delta, dict):
            decision.state_delta = {}
        if not isinstance(decision.memory_tags, list):
            decision.memory_tags = []
        return decision

    def decision_from_dict(self, data: Dict[str, Any]) -> NpcDecision:
        if not isinstance(data, dict):
            return NpcDecision(target_npc_id=self.npc_id)
        return NpcDecision(
            intent=str(data.get("intent") or "small_talk"),
            action=str(data.get("action") or "chat"),
            target_npc_id=str(data.get("target_npc_id") or self.npc_id),
            emotion=str(data.get("emotion") or "neutral"),
            line_goal=str(data.get("line_goal") or "自然回应玩家"),
            line=str(data.get("line") or "嗯，我听着。"),
            relationship_delta=data.get("relationship_delta") or {},
            state_delta=data.get("state_delta") or {},
            memory_tags=data.get("memory_tags") or [],
        )

    def extract_json_object(self, text: str) -> Dict[str, Any]:
        """从 LLM 输出中提取第一个 JSON object。"""
        if not text:
            return {}
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
        if fenced:
            return json.loads(fenced.group(1))

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        return {}

    def ensure_recommendation_line(self, line: str, recommendation_context: Dict[str, Any]) -> str:
        """推荐咖啡时做最低限度修正，避免空回复。"""
        if line and line.strip():
            return line.strip()
        if not recommendation_context.get("ok"):
            return "今天菜单我还需要再确认一下。你想要偏甜、偏奶香，还是清爽一点的？"
        available_menu = recommendation_context.get("available_menu") or recommendation_context.get("result", {}).get("available_menu")
        if isinstance(available_menu, list) and available_menu:
            first = available_menu[0]
            name = first.get("name") if isinstance(first, dict) else str(first)
            return f"今天可以先试试{name}。如果你告诉我更喜欢甜一点还是清爽一点，我可以再帮你缩小选择。"
        return "今天可以先看库存充足的菜单。你偏好奶咖、手冲，还是美式？"

    def maybe_update_player_profile(self, player_id: str, player_input: str) -> Optional[Dict[str, Any]]:
        """从明确表达中提取少量画像更新。避免过度推断。"""
        text = player_input.strip()
        taste_patch: Dict[str, Any] = {}
        if "拿铁" in text:
            taste_patch["likes_latte"] = True
        if "美式" in text:
            taste_patch["likes_americano"] = True
        if "手冲" in text:
            taste_patch["likes_pour_over"] = True
        if "甜" in text:
            taste_patch["sweetness_hint"] = "likes_sweet" if any(w in text for w in ["喜欢甜", "甜一点", "偏甜"]) else "mentioned_sweetness"
        if "酸" in text:
            taste_patch["acidity_hint"] = "mentioned_acidity"
        if not taste_patch:
            return None
        return self.tools.invoke_tool(
            "update_player_profile",
            {
                "player_id": player_id,
                "profile_patch": {"taste_preferences": taste_patch},
            },
        )

    def _get_npc_name(self, context: Dict[str, Any]) -> Optional[str]:
        npc_state = context.get("tool_results", {}).get("npc_state", {})
        npc = npc_state.get("npc") if isinstance(npc_state, dict) else None
        if isinstance(npc, dict):
            return npc.get("name")
        return None

    def _looks_like_coffee_question(self, text: str) -> bool:
        return any(word in text for word in ["咖啡", "拿铁", "美式", "手冲", "菜单", "豆", "烘焙", "奶泡", "风味"])

    def _fail(self, error: str, **payload: Any) -> Dict[str, Any]:
        return {
            "ok": False,
            "npc_id": self.npc_id,
            "error": error,
            **payload,
        }
