import importlib
import sys
import types


class FakeGraphAgent:
    instances = []

    def __init__(self, tools, llm_service=None):
        # 记录构造参数，确认 Director 把依赖传给图 Agent。
        self.tools = tools
        self.llm_service = llm_service
        self.calls = []
        FakeGraphAgent.instances.append(self)

    def handle_message(self, session_id, player_id, npc_id, player_message):
        # 返回固定结果，确认 Director.talk 直接走图 Agent 入口。
        self.calls.append({
            "session_id": session_id,
            "player_id": player_id,
            "npc_id": npc_id,
            "player_message": player_message,
        })
        return {
            "ok": True,
            "reply": "图 Agent 回复",
            "decision": {"action": "chat"},
        }


class FakeTools:
    def __init__(self):
        # 记录工具调用，避免测试访问真实数据库。
        self.calls = []

    def invoke_tool(self, tool_name, arguments=None):
        # Director 显式传入 npc_id 时会同步选择 NPC。
        self.calls.append((tool_name, arguments or {}))
        return {"ok": True, "tool_name": tool_name}


class FakeLlm:
    pass


def import_director_with_fake_graph_agent(monkeypatch):
    # 注入假的图 Agent 模块，避免测试依赖真实 LangGraph 和数据库。
    fake_graph_module = types.ModuleType("agent.npc_graph_agent")
    fake_graph_module.NpcGraphAgent = FakeGraphAgent
    monkeypatch.setitem(sys.modules, "agent.npc_graph_agent", fake_graph_module)
    sys.modules.pop("agent.director_agent", None)
    FakeGraphAgent.instances = []

    return importlib.import_module("agent.director_agent")


def test_director_routes_messages_to_npc_graph_agent(monkeypatch):
    director_agent = import_director_with_fake_graph_agent(monkeypatch)
    tools = FakeTools()
    llm = FakeLlm()
    director = director_agent.DirectorAgent(tools=tools, llm=llm)

    result = director.talk(
        player_id="p001",
        session_id="s001",
        npc_id="passerby_001",
        message="你好",
    )

    assert result["reply"] == "图 Agent 回复"
    assert len(FakeGraphAgent.instances) == 1
    assert FakeGraphAgent.instances[0].tools is tools
    assert FakeGraphAgent.instances[0].llm_service is llm
    assert FakeGraphAgent.instances[0].calls == [{
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "passerby_001",
        "player_message": "你好",
    }]


def test_director_keeps_none_llm_for_graph_agent(monkeypatch):
    director_agent = import_director_with_fake_graph_agent(monkeypatch)
    tools = FakeTools()

    def fail_get_llm():
        # 未显式启用 LLM 时，Director 不应自行创建模型。
        raise AssertionError("不应该创建默认 LLM")

    monkeypatch.setattr(director_agent.LlmService, "getLLM", fail_get_llm)
    director = director_agent.DirectorAgent(tools=tools, llm=None)
    graph_agent = director.get_npc_agent("passerby_001")

    assert graph_agent.llm_service is None


def test_director_creates_default_llm_when_llm_argument_is_omitted(monkeypatch):
    director_agent = import_director_with_fake_graph_agent(monkeypatch)
    tools = FakeTools()
    default_llm = FakeLlm()

    monkeypatch.setattr(director_agent.LlmService, "getLLM", lambda: default_llm)
    director = director_agent.DirectorAgent(tools=tools)
    graph_agent = director.get_npc_agent("passerby_001")

    assert graph_agent.llm_service is default_llm
