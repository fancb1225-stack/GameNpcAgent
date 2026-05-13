from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 将项目根目录加入模块路径，保证从 AgentEval 子目录运行时也能导入主项目。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from npc_eval.dataset import load_cases
from npc_eval.runner import evaluate_cases, reports_to_dicts


def parse_args() -> argparse.Namespace:
    # 解析命令行参数，支持本地规则评测和 DeepEval trace 两种模式。
    parser = argparse.ArgumentParser(description="运行咖啡厅 NPC Agent 评测")
    parser.add_argument(
        "--cases",
        default=str(Path(__file__).parent / "eval_cases.json"),
        help="评测用例 JSON 文件路径",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "deepeval"],
        default="local",
        help="local 使用本地规则评分，deepeval 额外启用 DeepEval 回调",
    )
    parser.add_argument(
        "--output",
        default="",
        help="可选的 JSON 报告输出路径",
    )
    return parser.parse_args()


def main() -> int:
    # 加载用例并运行评测，默认不需要外部模型或数据库。
    args = parse_args()
    cases = load_cases(args.cases)
    reports = evaluate_cases(cases, use_deepeval_trace=args.mode == "deepeval")
    report_rows = reports_to_dicts(reports)

    # 打印人类可读摘要，便于本地快速查看失败用例。
    passed_count = sum(1 for report in reports if report.passed)
    print(f"通过：{passed_count}/{len(reports)}")
    for report in reports:
        status = "PASS" if report.passed else "FAIL"
        reason = "；".join(report.reasons) if report.reasons else "无"
        print(f"[{status}] {report.case_id} score={report.score:.2f} reason={reason}")

    # 如果指定输出路径，则保存完整 JSON 报告。
    if args.output:
        Path(args.output).write_text(
            json.dumps(report_rows, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )

    return 0 if all(report.passed for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
