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

import logging
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

_log_dir = Path(__file__).resolve().parent.parent / "output"
_log_dir.mkdir(exist_ok=True)
_log_file = _log_dir / "logs.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(_log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
    force=True,
)
from typing import Any, Dict, Iterator, List, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from service.game_document_service import GameDocumentService
from service.npc_setting_import_service import NpcSettingImportService
from service.npc_service import NpcService
from service.pdf_reader_service import PdfReaderService

try:
    from agent.director_agent import DirectorAgent
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "无法导入 agent.director_agent.DirectorAgent。"
        "请确认本文件位于项目根目录下的 api/main.py，且 agent 目录存在。"
    ) from exc


class SessionCreateRequest(BaseModel):
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


class RagSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="要检索的游戏设定问题")
    limit: int = Field(default=5, ge=1, le=20, description="返回片段数量")


class ApiResponse(BaseModel):
    ok: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


COFFEE_TOOL_NAMES = {
    "enter_morning_cafe",
    "get_cafe_events",
    "get_coffee_knowledge",
    "get_world_state",
    "update_world_state",
}


def _public_tools(director: DirectorAgent) -> list[str]:
    """返回对前端公开的非咖啡场景工具列表。"""

    return [tool for tool in director.list_available_tools() if tool not in COFFEE_TOOL_NAMES]


def _create_llm_if_enabled() -> Optional[Any]:
    """
    按环境变量决定是否启用 LLM。

    默认启用，避免图 Agent 在已配置 LLM 时意外走规则降级逻辑。

    禁用方式：
    ENABLE_LLM=false
    """
    import logging
    _log = logging.getLogger(__name__)

    disabled = os.getenv("ENABLE_LLM", "true").lower() in {"0", "false", "no", "n", "off"}
    if disabled:
        _log.warning("[_create_llm_if_enabled] ENABLE_LLM=false，LLM 已禁用")
        return None

    try:
        from agent.llm_service import LlmService
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ENABLE_LLM=true，但无法导入 llm_service.py ") from exc

    try:
        llm = LlmService.getLLM()
        _log.info("[_create_llm_if_enabled] LLM 创建成功: %s", type(llm).__name__)
        return llm
    except Exception:
        _log.exception("[_create_llm_if_enabled] LLM 创建失败，将走规则降级")
        return None


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
    title="GameNpcFrame API",
    description="面向游戏 NPC 框架的 FastAPI 后端接口。",
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


def _ensure_pdf(file: UploadFile) -> None:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持上传 PDF 文件")


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
            "service": "GameNpcFrame API",
            "api": "running",
            "database_or_tools": db_result,
            "available_tools": _public_tools(director),
        },
    )


@app.post("/documents/game-settings/upload", response_model=ApiResponse)
async def upload_game_setting_pdf(file: UploadFile = File(...)) -> ApiResponse:
    """上传游戏设定 PDF，抽取文本后切分并写入 RAG 知识库。"""

    _ensure_pdf(file)
    content = await file.read()
    text = PdfReaderService().extract_text(content)
    result = GameDocumentService().ingest_game_setting_text(
        filename=file.filename or "game_setting.pdf",
        text=text,
        metadata={"content_type": file.content_type},
    )
    if result.get("ok") is False:
        return ApiResponse(ok=False, error=str(result.get("error")))
    return ApiResponse(ok=True, data=result)


@app.post("/documents/npc-settings/upload", response_model=ApiResponse)
async def upload_npc_setting_pdf(file: UploadFile = File(...)) -> ApiResponse:
    """上传 NPC 设定 PDF，由 Agent 识别结构并通过 ORM 写入 NPC 表。"""

    _ensure_pdf(file)
    content = await file.read()
    text = PdfReaderService().extract_text(content)
    director = get_director()
    result = NpcSettingImportService(
        llm_service=getattr(director, "llm", None),
    ).import_npc_setting_text(text)
    if result.get("ok") is False:
        return ApiResponse(ok=False, error=str(result.get("error")))
    return ApiResponse(ok=True, data=result)


@app.post("/rag/game-settings/search", response_model=ApiResponse)
def search_game_setting_context(request: RagSearchRequest) -> ApiResponse:
    """检索游戏设定 RAG 上下文，供调试或对话注入使用。"""

    result = GameDocumentService().search_game_settings(
        query=request.query,
        limit=request.limit,
    )
    if result.get("ok") is False:
        return ApiResponse(ok=False, error=str(result.get("error")))
    return ApiResponse(ok=True, data=result)


@app.post("/sessions", response_model=ApiResponse)
def create_session(request: SessionCreateRequest) -> ApiResponse:
    """创建或恢复一个通用 NPC 对话会话。"""
    result = get_director().enter_morning_cafe(
        player_id=request.player_id,
        nickname=request.nickname,
        session_id=request.session_id,
    )
    data = _return_or_raise(result)
    raw_result = data.get("result", data)
    session = dict(raw_result.get("session") or {})
    session.pop("scene", None)
    session.pop("time_period", None)
    return ApiResponse(
        ok=True,
        data={
            "result": {
                "player": raw_result.get("player"),
                "session": session,
            }
        },
    )


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


def _sse_event(event_name: str, data: Dict[str, Any]) -> str:
    # 将结构化数据编码为 SSE 事件，供前端逐条解析。
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event_name}\ndata: {payload}\n\n"


def _split_reply_chunks(reply: str, chunk_size: int = 8) -> Iterator[str]:
    # 第一版使用固定长度分片，后续可替换为模型原生 token streaming。
    text = reply or ""
    for index in range(0, len(text), chunk_size):
        yield text[index:index + chunk_size]


def _stream_dialogue_result(request: DialogueRequest) -> Iterator[str]:
    # 复用现有对话链路，确保数据库写入和普通接口行为一致。
    try:
        result = get_director().talk(
            player_id=request.player_id,
            session_id=request.session_id,
            message=request.message,
            npc_id=request.npc_id,
        )
        data = _return_or_raise(result)
        reply = data.get("reply") or data.get("decision", {}).get("line") or ""
        for chunk in _split_reply_chunks(str(reply)):
            yield _sse_event("chunk", {"content": chunk})
        yield _sse_event("done", data)
    except Exception as exc:
        yield _sse_event("error", {"error": str(exc)})


@app.post("/dialogue/messages/stream")
def stream_dialogue_message(request: DialogueRequest) -> StreamingResponse:
    """以 SSE 流式返回 NPC 回复，同时复用完整消息保存链路。"""
    return StreamingResponse(
        _stream_dialogue_result(request),
        media_type="text/event-stream",
    )


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


@app.get("/npcs", response_model=ApiResponse)
def list_npcs() -> ApiResponse:
    """列出所有活跃 NPC。"""
    result = get_director().invoke_tool("get_npc_state", {"npc_id": "__list_all__"})
    return ApiResponse(ok=True, data=_return_or_raise(result))


@app.get("/npcs/{npc_id}", response_model=ApiResponse)
def get_npc_state(npc_id: str) -> ApiResponse:
    """查询 NPC 当前状态。"""
    result = get_director().invoke_tool("get_npc_state", {"npc_id": npc_id})
    return ApiResponse(ok=True, data=_return_or_raise(result))


@app.delete("/npcs/{npc_id}", response_model=ApiResponse)
def delete_npc(npc_id: str) -> ApiResponse:
    """删除指定 NPC。"""

    service = NpcService()
    try:
        deleted_count = service.delete_npc(npc_id)
    finally:
        service.close()
    if deleted_count <= 0:
        raise HTTPException(status_code=404, detail=f"未找到 NPC: {npc_id}")
    return ApiResponse(ok=True, data={"npc_id": npc_id, "deleted_count": deleted_count})


@app.get("/dialogue/history", response_model=ApiResponse)
def get_dialogue_history(
    player_id: str = Query(..., min_length=1, description="玩家 ID"),
    npc_id: str = Query(..., min_length=1, description="NPC ID"),
    limit: int = Query(default=50, ge=1, le=200),
) -> ApiResponse:
    """查询玩家与指定 NPC 的所有历史对话，按时间先后排序。"""
    from service.dialogue_service import DialogueService
    from db.db_service import DbService as _DbService
    db = _DbService()
    try:
        dialogues = DialogueService(db)
        rows = dialogues.list_messages(
            session_id=None,
            player_id=player_id,
            npc_id=npc_id,
            limit=limit,
        )
        messages = [r.model_dump(mode="json") for r in rows]
        return ApiResponse(ok=True, data={"messages": messages, "count": len(messages)})
    except Exception as exc:
        return ApiResponse(ok=False, error=str(exc))
    finally:
        db.close()


@app.get("/tools", response_model=ApiResponse)
def list_tools() -> ApiResponse:
    """列出当前 Agent 可用工具。"""
    director = get_director()
    return ApiResponse(
        ok=True,
        data={"tools": _public_tools(director)},
    )


@app.post("/tools/invoke", response_model=ApiResponse)
def invoke_tool(request: ToolInvokeRequest) -> ApiResponse:
    """
    调试用工具调用入口。

    生产环境可以删除或加鉴权，避免前端任意调用内部工具。
    """
    if os.getenv("ENABLE_TOOL_DEBUG_API", "true").lower() not in {"1", "true", "yes", "y"}:
        raise HTTPException(status_code=403, detail="工具调试接口未开启")

    if request.tool_name in COFFEE_TOOL_NAMES:
        raise HTTPException(status_code=404, detail=f"工具已不再公开: {request.tool_name}")

    result = get_director().invoke_tool(request.tool_name, request.arguments)
    return ApiResponse(ok=True, data=_return_or_raise(result))

# @app.get("/", response_model=ApiResponse)
# def index():
#     return template.TemplateResponse("./frontend/index.html", {"request": None, "message": "Welcome to GameNpcFrame API!"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
