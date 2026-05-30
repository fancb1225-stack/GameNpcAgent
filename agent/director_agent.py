"""

职责：
1. 负责场景入口和会话调度。
2. 管理多个 NpcGraphAgent 实例。
3. 将玩家消息路由给当前选择的 NPC。
4. 提供稳定的上层接口，供 FastAPI、CLI 或游戏服务调用。

该文件不直接访问数据库，不写原始 SQL。所有状态读取与写入都通过 NpcAgentTools 完成。
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.llm_service import LlmService

_log = logging.getLogger(__name__)


DEFAULT_LLM_ARGUMENT = object()
DEFAULT_DIRECTOR_BACKGROUND_EXECUTOR = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="director_background",
)

NEXT_ACTION_SYSTEM_PROMPT = """
你是 GameNpcFrame 的 Director Agent。

你的任务是根据玩家消息和 NPC 刚刚回复的两条近期消息，判断 NPC 是否需要追加一次下一步动作。

规则：
1. 只输出一个 JSON object，不要 Markdown，不要解释。
2. need_next_action 为 true 时，必须给出 line，表示 NPC 接着回复玩家的一句补充话。
3. 下一步动作只能触发一次，不能要求继续递归判断。
4. 如果 NPC 回复已经完整、玩家没有等待补充、没有强动作动机，则 need_next_action=false。
5. 不要编造游戏设定、NPC 设定、任务、地点和人物关系；没有依据时不要追加。

输出格式：
{"need_next_action": false, "reason": "无需补充"}
或
{"need_next_action": true, "line": "NPC 补充的一句话", "reason": "需要补充的原因"}
""".strip()

try:
    from agent.agent_tools import NpcAgentTools
except Exception:  # 允许调用方注入 tools
    NpcAgentTools = Any  # type: ignore

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
    """ NPC Agent 的导演层。"""

    def __init__(
        self,
        tools: Optional[NpcAgentTools] = None,
        llm: Any = DEFAULT_LLM_ARGUMENT,
        enable_web_search: bool = False,
        background_executor: Any = None,
    ) -> None:
        self.tools = tools or NpcAgentTools(enable_web_search=enable_web_search)
        if llm is DEFAULT_LLM_ARGUMENT:
            try:
                self.llm = LlmService.getLLM() or None
            except Exception:
                _log.exception("[DirectorAgent] LLMService.getLLM() 失败")
                self.llm = None
        else:
            self.llm = llm

        _log.info("[DirectorAgent] llm=%s", type(self.llm).__name__ if self.llm else "None")

        self.sessions: Dict[str, DirectorSession] = {}
        self.npc_agents: Dict[str, NpcGraphAgent] = {}
        self.background_executor = background_executor or DEFAULT_DIRECTOR_BACKGROUND_EXECUTOR
        self._next_action_locks = defaultdict(Lock)

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
        创建或进入场景。

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

    def enter_game_world(
        self,
        player_id: str,
        nickname: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        创建或进入游戏世界场景。

        返回工具层给出的玩家画像、会话、世界状态、可遇见 NPC 等信息。
        """
        if not player_id:
            return self._fail("缺少 player_id")
        session_id = session_id or f"session_{uuid4().hex[:12]}"
        result = self.tools.invoke_tool(
            "enter_game_world",
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
        result = npc_agent.handle_message(
            session_id=session_id,
            player_id=player_id,
            npc_id=target_npc_id,
            player_message=message,
        )
        return self._apply_next_action_once(
            result=result,
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
            self.npc_agents[npc_id] = NpcGraphAgent(
                tools=self.tools,
                llm_service=self.llm,
                background_executor=self.background_executor,
                defer_memory_write=True,
            )
        return self.npc_agents[npc_id]

    def _apply_next_action_once(
        self,
        result: Dict[str, Any],
        session_id: str,
        player_id: str,
        npc_id: str,
        player_message: str,
    ) -> Dict[str, Any]:
        """用 Director LLM 判断是否追加一次 NPC 下一步动作。"""
        reply = str(result.get("reply") or result.get("npc_reply") or "")
        lock_key = (session_id, npc_id)
        lock = self._next_action_locks[lock_key]

        # 同一会话同一 NPC 的下一步动作判断不并发重入，避免重复追加。
        if not lock.acquire(blocking=False):
            result["next_action"] = {
                "triggered": False,
                "skipped_reason": "next_action_locked",
            }
            return result

        try:
            decision = self._judge_next_action(
                session_id=session_id,
                player_id=player_id,
                npc_id=npc_id,
                player_message=player_message,
                npc_reply=reply,
            )
        finally:
            lock.release()

        if not decision.get("need_next_action"):
            result["next_action"] = {
                "triggered": False,
                "reason": decision.get("reason"),
            }
            return result

        next_line = str(decision.get("line") or "").strip()
        if not next_line:
            result["next_action"] = {
                "triggered": False,
                "reason": "下一步动作缺少 line",
            }
            return result

        # 将追加回复并入 reply，保持现有 API/前端读取 reply 字段即可展示。
        result["reply"] = f"{reply}\n{next_line}" if reply else next_line
        result["next_action"] = {
            "triggered": True,
            "reply": next_line,
            "reason": decision.get("reason"),
        }
        return result

    def _judge_next_action(
        self,
        session_id: str,
        player_id: str,
        npc_id: str,
        player_message: str,
        npc_reply: str,
    ) -> Dict[str, Any]:
        """根据玩家和 NPC 的两条近期消息判断是否需要追加回复。"""
        if self.llm is None:
            return {"need_next_action": False, "reason": "director_llm_disabled"}

        prompt = json.dumps({
            "task": "director_next_action_judge",
            "session_id": session_id,
            "player_id": player_id,
            "npc_id": npc_id,
            "recent_messages": [
                {"role": "player", "content": player_message},
                {"role": "npc", "content": npc_reply},
            ],
            "output_schema": {
                "need_next_action": "bool，是否需要 NPC 追加一次回复",
                "line": "need_next_action=true 时必填，NPC 追加给玩家的一句话",
                "reason": "简短原因",
            },
        }, ensure_ascii=False, default=str)

        try:
            response, _ = self.llm.chat(
                prompt=prompt,
                history=[],
                system_prompt=NEXT_ACTION_SYSTEM_PROMPT,
            )
        except Exception:
            _log.exception("[DirectorAgent] 下一步动作判断失败")
            return {"need_next_action": False, "reason": "llm_call_failed"}

        decision = self._extract_json_object(response)
        return decision if isinstance(decision, dict) else {"need_next_action": False}

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        """解析 LLM 返回的单个 JSON object，失败时返回空字典。"""
        if not text:
            return {}
        raw_text = str(text).strip()
        try:
            data = json.loads(raw_text)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            pass

        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw_text[start:end + 1])
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

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
    director.select_npc(session_id=session_id, npc_id="brother_lu_qingya")
    reply = director.talk(
        player_id="p001",
        session_id=session_id,
        message="师兄在吗",
    )
    print("REPLY:", reply.get("reply") or reply.get("npc_reply"))
