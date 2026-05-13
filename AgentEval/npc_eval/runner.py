from __future__ import annotations

from dataclasses import asdict
from typing import Any

from agent.npc_graph_agent import NpcGraphAgent

from .dataset import EvalCase
from .fake_tools import EvalFakeTools
from .local_metrics import CaseReport, score_case


class RuleLlmService:
    # 为默认评测提供稳定回复，避免本地评测依赖外部模型服务和网络。
    def __init__(self, user_input: str) -> None:
        # 保存原始玩家输入，让替身回复只受用例本身影响。
        self.user_input = user_input

    def chat(
        self,
        prompt: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str = "",
    ) -> tuple[str, list[dict[str, str]]]:
        # 评测替身直接返回结构化 JSON，让真实 graph 继续走标准解析和节点流转。
        if "拿铁" in self.user_input:
            reply = "好的，一杯拿铁，我马上帮你准备。"
        elif "推荐" in self.user_input or "喝什么" in self.user_input:
            reply = "今天适合来一杯手冲，口感清爽，也能配合上午的节奏。"
        else:
            reply = "我听到了，我们慢慢聊。"

        # 返回 NPC 决策 JSON，覆盖动作、情绪、台词和关系变化。
        content = (
            '{"intent": "chat", "action": "chat", "emotion": "calm", '
            f'"line": "{reply}", "relationship_delta": {{"familiarity": 1}}}}'
        )
        return content, history or []


def evaluate_case(case: EvalCase, use_deepeval_trace: bool = False) -> CaseReport:
    # 每条用例创建独立工具替身，避免工具调用记录互相污染。
    tools = EvalFakeTools()
    llm_service = RuleLlmService(case.user_input)
    agent = NpcGraphAgent(tools=tools, llm_service=llm_service)

    # DeepEval trace 是可选增强；未安装时保持本地评测可用。
    callbacks = _build_deepeval_callbacks() if use_deepeval_trace else []
    result = agent.graph.invoke(
        {
            "session_id": case.context.get("session_id", f"eval_{case.case_id}"),
            "player_id": case.context.get("player_id", "eval_player"),
            "npc_id": case.context.get("npc_id", "barista_001"),
            "player_message": case.user_input,
            "tools": tools,
            "llm_service": llm_service,
            "next_action_delay_seconds": 0.01,
            "errors": [],
        },
        config={"callbacks": callbacks} if callbacks else None,
    )

    # 将 graph 结果收敛为和线上 handle_message 类似的响应结构。
    output = {
        "ok": True,
        "reply": result.get("npc_reply", ""),
        "action": result.get("action"),
        "intent": result.get("intent"),
        "decision": result.get("decision"),
    }
    return score_case(case, output)


def evaluate_cases(cases: list[EvalCase], use_deepeval_trace: bool = False) -> list[CaseReport]:
    # 顺序执行评测用例，保证输出顺序与数据集顺序一致。
    return [evaluate_case(case, use_deepeval_trace=use_deepeval_trace) for case in cases]


def reports_to_dicts(reports: list[CaseReport]) -> list[dict[str, Any]]:
    # 将 dataclass 报告转为 JSON 友好的字典，供 CLI 和文件输出复用。
    return [asdict(report) for report in reports]


def _build_deepeval_callbacks() -> list[Any]:
    # 延迟导入 DeepEval，避免没有安装依赖时影响本地规则评测。
    try:
        from deepeval.integrations.langchain import CallbackHandler
        from deepeval.metrics import TaskCompletionMetric
    except ImportError as exc:
        raise RuntimeError(
            "未安装 DeepEval，请先执行：pip install -r AgentEval/requirements.txt"
        ) from exc

    # 按 DeepEval 的 LangGraph/LangChain 回调方式接入任务完成度评测。
    return [CallbackHandler(metrics=[TaskCompletionMetric(threshold=0.7)])]
