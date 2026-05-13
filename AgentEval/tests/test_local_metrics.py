from AgentEval.npc_eval.dataset import EvalCase
from AgentEval.npc_eval.local_metrics import score_case


def test_score_case_passes_when_reply_matches_expected_shape():
    # 构造一条符合预期的 NPC 回复结果，用来验证本地评分通过路径。
    case = EvalCase(
        case_id="coffee_order",
        user_input="我想要一杯拿铁",
        expected_action="chat",
        required_keywords=["拿铁"],
    )
    result = {
        "ok": True,
        "reply": "好的，一杯拿铁，我马上帮你准备。",
        "action": "chat",
    }

    # 评分应同时覆盖回复非空、动作合法、动作匹配和关键词命中。
    report = score_case(case, result)

    assert report.case_id == "coffee_order"
    assert report.passed is True
    assert report.score == 1.0
    assert report.checks["reply_not_empty"] is True
    assert report.checks["expected_action"] is True
    assert report.checks["required_keywords"] is True


def test_score_case_fails_when_required_keywords_are_missing():
    # 构造一条缺少关键内容的回复，保证评分能暴露 NPC 偏离场景的问题。
    case = EvalCase(
        case_id="coffee_order",
        user_input="我想要一杯拿铁",
        expected_action="chat",
        required_keywords=["拿铁"],
    )
    result = {
        "ok": True,
        "reply": "好的，我马上帮你准备。",
        "action": "chat",
    }

    # 缺少关键词时，用例应失败并给出可读原因。
    report = score_case(case, result)

    assert report.passed is False
    assert report.score < 1.0
    assert report.checks["required_keywords"] is False
    assert any("关键词" in reason for reason in report.reasons)
