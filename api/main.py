"""
FastAPI backend for CoffeeNpcAgent.

放置建议：CoffeeNpcAgent/api/main.py
启动命令：uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

依赖：
- agent/director_agent.py
- agent/npc_agent.py
- agent/coffee_npc_agent_tools.py
- db/model/service 相关模块

接口目标：
1. 提供最小可用的 HTTP API。
2. 不直接访问数据库。
3. 所有业务能力通过 DirectorAgent / CoffeeNpcAgentTools 调用。
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from agent.director_agent import DirectorAgent
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "无法导入 agent.director_agent.DirectorAgent。"
        "请确认本文件位于项目根目录下的 api/main.py，且 agent 目录存在。"
    ) from exc


class MorningCafeRequest(BaseModel):
    player_id: str = Field(..., min_length=1, description="玩家业务 ID")
    nickname: Optional[str] = Field(default=None, description="玩家昵称")
    session_id: Optional[str] = Field(default=None, description="可选会话 ID，不传则自动生成")


class SelectNpcRequest(BaseModel):
    npc_id: str = Field(..., min_length=1, description="要选择的 NPC ID")


class DialogueRequest(BaseModel):
    player_id: str = Field(..., min_length=1, description="玩家业务 ID")
    session_id: str = Field(..., min_length=1, description="会话 ID")
    message: str = Field(..., min_length=1, description="玩家输入内容")
    npc_id: Optional[str] = Field(default=None, description="可选 NPC ID；不传则使用当前会话选中的 NPC")


class DirectNpcDialogueRequest(BaseModel):
    player_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class ToolInvokeRequest(BaseModel):
    tool_name: str = Field(..., min_length=1)
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ApiResponse(BaseModel):
    ok: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def _create_llm_if_enabled() -> Optional[Any]:
    """
    按环境变量决定是否启用 LLM。

    默认不启用，方便先跑通后端、数据库和规则降级逻辑。

    启用方式：
    ENABLE_LLM=true
    并保证 llm.py / agent_config.py 中的 LlmService 与 llm_api_key 可用。
    """
    enabled = os.getenv("ENABLE_LLM", "false").lower() in {"1", "true", "yes", "y"}
    if not enabled:
        return None

    try:
        from agent.llm import LlmService
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ENABLE_LLM=true，但无法导入 llm.py ") from exc

    return LlmService.getLLM()


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm = _create_llm_if_enabled()
    enable_web_search = os.getenv("ENABLE_WEB_SEARCH", "false").lower() in {"1", "true", "yes", "y"}
    app.state.director = DirectorAgent(llm=llm, enable_web_search=enable_web_search)
    try:
        yield
    finally:
        director = getattr(app.state, "director", None)
        if director is not None and hasattr(director, "close"):
            director.close()


app = FastAPI(
    title="CoffeeNpcAgent API",
    description="咖啡厅 NPC Agent 的最小 FastAPI 后端接口。",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_director() -> DirectorAgent:
    director = getattr(app.state, "director", None)
    if director is None:
        raise HTTPException(status_code=500, detail="DirectorAgent 未初始化")
    return director


def _return_or_raise(result: Dict[str, Any], status_code: int = 400) -> Dict[str, Any]:
    if not isinstance(result, dict):
        raise HTTPException(status_code=500, detail="Agent 返回了非法结果")
    if result.get("ok") is False:
        raise HTTPException(status_code=status_code, detail=result)
    return result


@app.get("/health", response_model=ApiResponse)
def health() -> ApiResponse:
    """服务健康检查，同时尝试通过工具层检查数据库连接。"""
    director = get_director()
    db_result: Dict[str, Any]
    try:
        db_result = director.invoke_tool("health_check", {})
    except Exception as exc:
        db_result = {"ok": False, "error": str(exc)}

    # 如果 tools 中没有 health_check，也不阻断服务启动。
    if db_result.get("ok") is False and "未知工具" in str(db_result.get("error", "")):
        db_result = {"ok": True, "message": "health_check 工具未定义，仅完成 API 层健康检查"}

    return ApiResponse(
        ok=True,
        data={
            "service": "CoffeeNpcAgent API",
            "api": "running",
            "database_or_tools": db_result,
            "available_tools": director.list_available_tools(),
        },
    )


@app.post("/scene/morning", response_model=ApiResponse)
def enter_morning_cafe(request: MorningCafeRequest) -> ApiResponse:
    """进入固定上午咖啡厅场景，创建或恢复一个会话。"""
    result = get_director().enter_morning_cafe(
        player_id=request.player_id,
        nickname=request.nickname,
        session_id=request.session_id,
    )
    return ApiResponse(ok=True, data=_return_or_raise(result))


@app.post("/sessions/{session_id}/select-npc", response_model=ApiResponse)
def select_npc(session_id: str, request: SelectNpcRequest) -> ApiResponse:
    """选择当前会话要对话的 NPC。"""
    result = get_director().select_npc(session_id=session_id, npc_id=request.npc_id)
    return ApiResponse(ok=True, data=_return_or_raise(result))


@app.post("/dialogue/messages", response_model=ApiResponse)
def send_dialogue_message(request: DialogueRequest) -> ApiResponse:
    """向当前 NPC 发送一句玩家消息，并返回 NPC 回复。"""
    result = get_director().talk(
        player_id=request.player_id,
        session_id=request.session_id,
        message=request.message,
        npc_id=request.npc_id,
    )
    return ApiResponse(ok=True, data=_return_or_raise(result))


@app.post("/npcs/{npc_id}/messages", response_model=ApiResponse)
def send_message_to_npc(npc_id: str, request: DirectNpcDialogueRequest) -> ApiResponse:
    """绕过会话已选 NPC，直接向指定 NPC 发送消息。"""
    result = get_director().talk(
        player_id=request.player_id,
        session_id=request.session_id,
        message=request.message,
        npc_id=npc_id,
    )
    return ApiResponse(ok=True, data=_return_or_raise(result))


@app.get("/sessions/{session_id}/snapshot", response_model=ApiResponse)
def get_scene_snapshot(session_id: str) -> ApiResponse:
    """查询当前场景快照：世界状态、事件、当前选择 NPC。"""
    result = get_director().get_scene_snapshot(session_id=session_id)
    return ApiResponse(ok=True, data=_return_or_raise(result))


@app.get("/players/{player_id}/context", response_model=ApiResponse)
def get_player_context(
    player_id: str,
    npc_id: Optional[str] = Query(default=None, description="可选 NPC ID；传入后返回关系和记忆"),
) -> ApiResponse:
    """查询玩家画像，以及玩家与某个 NPC 的关系、记忆。"""
    result = get_director().get_player_context(player_id=player_id, npc_id=npc_id)
    return ApiResponse(ok=True, data=_return_or_raise(result))


@app.get("/players/{player_id}/relationships/{npc_id}", response_model=ApiResponse)
def get_player_npc_relationship(player_id: str, npc_id: str) -> ApiResponse:
    """查询玩家与指定 NPC 的五维关系。"""
    result = get_director().invoke_tool(
        "get_relationship",
        {"player_id": player_id, "npc_id": npc_id},
    )
    return ApiResponse(ok=True, data=_return_or_raise(result))


@app.get("/npcs/{npc_id}", response_model=ApiResponse)
def get_npc_state(npc_id: str) -> ApiResponse:
    """查询 NPC 当前状态。"""
    result = get_director().invoke_tool("get_npc_state", {"npc_id": npc_id})
    return ApiResponse(ok=True, data=_return_or_raise(result))


@app.get("/world-state/{session_id}", response_model=ApiResponse)
def get_world_state(session_id: str) -> ApiResponse:
    """查询指定会话的咖啡厅世界状态。"""
    result = get_director().invoke_tool("get_world_state", {"session_id": session_id})
    return ApiResponse(ok=True, data=_return_or_raise(result))


@app.get("/coffee-knowledge", response_model=ApiResponse)
def get_coffee_knowledge(
    query: Optional[str] = Query(default=None, description="可选查询文本"),
    category: Optional[str] = Query(default=None, description="可选分类：bean / brew / milk / flavor / menu"),
    limit: int = Query(default=10, ge=1, le=50),
) -> ApiResponse:
    """查询咖啡知识库。"""
    args: Dict[str, Any] = {"limit": limit}
    if query is not None:
        args["query"] = query
    if category is not None:
        args["category"] = category
    result = get_director().invoke_tool("get_coffee_knowledge", args)
    return ApiResponse(ok=True, data=_return_or_raise(result))


@app.get("/tools", response_model=ApiResponse)
def list_tools() -> ApiResponse:
    """列出当前 Agent 可用工具。"""
    director = get_director()
    return ApiResponse(
        ok=True,
        data={"tools": director.list_available_tools()},
    )


@app.post("/tools/invoke", response_model=ApiResponse)
def invoke_tool(request: ToolInvokeRequest) -> ApiResponse:
    """
    调试用工具调用入口。

    生产环境可以删除或加鉴权，避免前端任意调用内部工具。
    """
    if os.getenv("ENABLE_TOOL_DEBUG_API", "true").lower() not in {"1", "true", "yes", "y"}:
        raise HTTPException(status_code=403, detail="工具调试接口未开启")

    result = get_director().invoke_tool(request.tool_name, request.arguments)
    return ApiResponse(ok=True, data=_return_or_raise(result))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)