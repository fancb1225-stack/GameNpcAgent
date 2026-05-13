from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .dataset import EvalCase


ALLOWED_ACTIONS = {
    # 与项目 NPC graph 的动作集合保持一致，用于本地评分时避免导入 LangGraph。
    "chat",
    "recommend_coffee",
    "ask_player",
    "comment_on_npc",
    "move_location",
    "serve_customer",
    "end_dialogue",
}


@dataclass(frozen=True)
class CaseReport:
    # 保存单条用例的评分结果，方便 CLI 输出和后续接入报表系统。
    case_id: str
    score: float
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    reply: str = ""
    action: str | None = None


def score_case(case: EvalCase, result: dict[str, Any]) -> CaseReport:
    # 提取 agent 输出中的核心字段，评分逻辑只关心玩家可见回复和动作。
    reply = str(result.get("reply") or "").strip()
    action = result.get("action")

    # 定义可解释的本地检查项，作为没有 DeepEval judge 时的快速质量门禁。
    checks = {
        "reply_not_empty": bool(reply),
        "allowed_action": action in ALLOWED_ACTIONS,
        "expected_action": True,
        "required_keywords": True,
    }
    reasons: list[str] = []

    # 如果用例声明了期望动作，则要求 agent 的动作与场景目标一致。
    if case.expected_action:
        checks["expected_action"] = action == case.expected_action
        if not checks["expected_action"]:
            reasons.append(f"动作应为 {case.expected_action}，实际为 {action}")

    # 如果用例声明了关键词，则要求回复里至少包含所有关键内容。
    missing_keywords = [keyword for keyword in case.required_keywords if keyword not in reply]
    checks["required_keywords"] = not missing_keywords
    if missing_keywords:
        reasons.append(f"回复缺少关键词：{', '.join(missing_keywords)}")

    # 基础结构检查失败时，也给出清晰的失败原因。
    if not checks["reply_not_empty"]:
        reasons.append("回复为空")
    if not checks["allowed_action"]:
        reasons.append(f"动作不在允许列表内：{action}")

    # 用通过比例形成简单分数，便于 CI 阈值判断。
    score = sum(1 for passed in checks.values() if passed) / len(checks)

    return CaseReport(
        case_id=case.case_id,
        score=score,
        passed=all(checks.values()),
        checks=checks,
        reasons=reasons,
        reply=reply,
        action=str(action) if action is not None else None,
    )
