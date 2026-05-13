import importlib
import sys
import types

from agent.llm_service import LlmService


class FakeLlm:
    pass


def import_api_main(monkeypatch):
    # 使用最小 LangGraph 替身，避免测试依赖真实图运行时。
    graph_module = types.ModuleType("langgraph.graph")
    graph_module.START = "__start__"
    graph_module.END = "__end__"
    graph_module.StateGraph = object
    monkeypatch.setitem(sys.modules, "langgraph", types.ModuleType("langgraph"))
    monkeypatch.setitem(sys.modules, "langgraph.graph", graph_module)
    sys.modules.pop("api.main", None)
    sys.modules.pop("agent.director_agent", None)
    sys.modules.pop("agent.npc_graph_agent", None)

    return importlib.import_module("api.main")


def test_api_creates_llm_by_default_when_not_explicitly_disabled(monkeypatch):
    api_main = import_api_main(monkeypatch)
    default_llm = FakeLlm()

    monkeypatch.delenv("ENABLE_LLM", raising=False)
    monkeypatch.setattr(LlmService, "getLLM", lambda: default_llm)

    assert api_main._create_llm_if_enabled() is default_llm


def test_api_keeps_llm_disabled_when_env_is_false(monkeypatch):
    api_main = import_api_main(monkeypatch)

    monkeypatch.setenv("ENABLE_LLM", "false")
    monkeypatch.setattr(
        LlmService,
        "getLLM",
        lambda: (_ for _ in ()).throw(AssertionError("不应该创建 LLM")),
    )

    assert api_main._create_llm_if_enabled() is None
