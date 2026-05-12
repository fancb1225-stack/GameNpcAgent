"""
Director Agent for CoffeeNpcAgent.

职责：
1. 负责咖啡厅场景入口和会话调度。
2. 管理多个 NpcGraphAgent 实例。
3. 将玩家消息路由给当前选择的 NPC。
4. 提供稳定的上层接口，供 FastAPI、CLI 或游戏服务调用。

该文件不直接访问数据库，不写原始 SQL。所有状态读取与写入都通过 CoffeeNpcAgentTools 完成。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple
from uuid import uuid4

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.llm import LlmService


DEFAULT_LLM_ARGUMENT = object()

try:
    from agent.coffee_npc_agent_tools import CoffeeNpcAgentTools
except Exception:  # 允许调用方注入 tools
    CoffeeNpcAgentTools = Any  # type: ignore

try:
    from agent.npc_graph_agent import NpcGraphAgent
except Exception:
    from npc_graph_agent import NpcGraphAgent  # type: ignore


@dataclass
class DirectorSession:
    player_id: str
    session_id: str
    selected_npc_id: Optional[str] = None
    status: str = "active"


class DirectorAgent:
    """咖啡厅 NPC Agent 的导演层。"""

    def __init__(
        self,
        tools: Optional[CoffeeNpcAgentTools] = None,
        llm: Any = DEFAULT_LLM_ARGUMENT,
        enable_web_search: bool = False,
    ) -> None:
        self.tools = tools or CoffeeNpcAgentTools(enable_web_search=enable_web_search)
        # 未传入 llm 时默认创建；显式传入 None 时使用图 Agent 的规则降级。
        self.llm = LlmService.getLLM() if llm is DEFAULT_LLM_ARGUMENT else llm
        self.sessions: Dict[str, DirectorSession] = {}
        self.npc_agents: Dict[str, NpcGraphAgent] = {}

    def close(self) -> None:
        if hasattr(self.tools, "close"):
            self.tools.close()

    def __enter__(self) -> "DirectorAgent":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def enter_morning_cafe(
        self,
        player_id: str,
        nickname: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        创建或进入上午咖啡厅场景。

        返回工具层给出的玩家画像、会话、世界状态、可遇见 NPC 等信息。
        """
        if not player_id:
            return self._fail("缺少 player_id")
        session_id = session_id or f"session_{uuid4().hex[:12]}"
        result = self.tools.invoke_tool(
            "enter_morning_cafe",
            {
                "player_id": player_id,
                "nickname": nickname,
                "session_id": session_id,
            },
        )
        if result.get("ok"):
            self.sessions[session_id] = DirectorSession(
                player_id=player_id,
                session_id=session_id,
                selected_npc_id=None,
            )
        return result

    def select_npc(self, session_id: str, npc_id: str) -> Dict[str, Any]:
        """选择当前对话 NPC。"""
        if not session_id or not npc_id:
            return self._fail("缺少 session_id 或 npc_id")
        result = self.tools.invoke_tool(
            "select_dialogue_npc",
            {"session_id": session_id, "npc_id": npc_id},
        )
        if result.get("ok"):
            session = self.sessions.get(session_id)
            if session is None:
                session = DirectorSession(player_id="", session_id=session_id)
                self.sessions[session_id] = session
            session.selected_npc_id = npc_id
        return result

    def talk(
        self,
        player_id: str,
        session_id: str,
        message: str,
        npc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        玩家发送消息。

        如果 npc_id 为空，则使用当前会话 selected_npc_id。
        如果当前会话还没有选择 NPC，返回明确错误。
        """
        if not player_id:
            return self._fail("缺少 player_id")
        if not session_id:
            return self._fail("缺少 session_id")
        if not message or not message.strip():
            return self._fail("玩家消息不能为空")

        target_npc_id = npc_id or self.get_selected_npc_id(session_id)
        if not target_npc_id:
            return self._fail("当前会话还没有选择 NPC，请先调用 select_npc")

        # 如果调用时显式传入 npc_id，则同步更新会话选择。
        if npc_id:
            self.select_npc(session_id=session_id, npc_id=npc_id)

        npc_agent = self.get_npc_agent(target_npc_id)
        return npc_agent.handle_message(
            session_id=session_id,
            player_id=player_id,
            npc_id=target_npc_id,
            player_message=message,
        )

    def get_scene_snapshot(self, session_id: str) -> Dict[str, Any]:
        """查询当前会话可供前端或调试使用的场景快照。"""
        if not session_id:
            return self._fail("缺少 session_id")
        world_state = self.tools.invoke_tool("get_world_state", {"session_id": session_id})
        events = self.tools.invoke_tool("get_cafe_events", {"time_period": "morning"})
        selected_npc_id = self.get_selected_npc_id(session_id)
        selected_npc = None
        if selected_npc_id:
            selected_npc = self.tools.invoke_tool("get_npc_state", {"npc_id": selected_npc_id})
        return {
            "ok": True,
            "session_id": session_id,
            "selected_npc_id": selected_npc_id,
            "world_state": world_state,
            "events": events,
            "selected_npc": selected_npc,
        }

    def get_player_context(self, player_id: str, npc_id: Optional[str] = None) -> Dict[str, Any]:
        """查询玩家画像和可选 NPC 关系，用于 UI 或调试。"""
        if not player_id:
            return self._fail("缺少 player_id")
        profile = self.tools.invoke_tool("get_player_profile", {"player_id": player_id})
        payload: Dict[str, Any] = {
            "ok": True,
            "player_id": player_id,
            "profile": profile,
        }
        if npc_id:
            payload["relationship"] = self.tools.invoke_tool(
                "get_relationship",
                {"player_id": player_id, "npc_id": npc_id},
            )
            payload["short_term_memory"] = self.tools.invoke_tool(
                "get_short_term_memory",
                {"player_id": player_id, "npc_id": npc_id},
            )
            payload["long_term_memories"] = self.tools.invoke_tool(
                "get_long_term_memories",
                {"player_id": player_id, "npc_id": npc_id, "limit": 10},
            )
        return payload

    def get_npc_agent(self, npc_id: str) -> NpcGraphAgent:
        """获取或创建单个图 Agent。"""
        if npc_id not in self.npc_agents:
            # 图 Agent 内部按每轮消息传入 npc_id，这里只复用编译后的图实例。
            self.npc_agents[npc_id] = NpcGraphAgent(
                tools=self.tools,
                llm_service=self.llm,
            )
        return self.npc_agents[npc_id]

    def get_selected_npc_id(self, session_id: str) -> Optional[str]:
        session = self.sessions.get(session_id)
        if session and session.selected_npc_id:
            return session.selected_npc_id
        return None

    def list_available_tools(self) -> List[str]:
        if hasattr(self.tools, "get_available_tools"):
            return self.tools.get_available_tools()
        return []

    def invoke_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """暴露一个受控工具调用入口，方便调试。"""
        if not hasattr(self.tools, "invoke_tool"):
            return self._fail("当前 tools 不支持 invoke_tool")
        return self.tools.invoke_tool(tool_name, arguments or {})

    def _fail(self, error: str, **payload: Any) -> Dict[str, Any]:
        return {"ok": False, "error": error, **payload}


if __name__ == "__main__":

    director = DirectorAgent()

    scene = director.enter_morning_cafe(player_id="p001", nickname="玩家")
    print("SCENE:", scene)

    session_id = (
        scene.get("result", {}).get("session", {}).get("session_id")
        or "default_morning_session"
    )
    director.select_npc(session_id=session_id, npc_id="barista_001")
    reply = director.talk(
        player_id="p001",
        session_id=session_id,
        message="门口打架的终于走了",
    )
    print("REPLY:", reply.get("reply") or reply.get("npc_reply"))
