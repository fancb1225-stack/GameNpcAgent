# NPC Agent 评测工具

这个目录用于评测咖啡厅 NPC Agent 的对话质量。默认模式使用本地规则评分，
不依赖真实数据库、外部模型或 DeepEval 账号；安装依赖后可以切换到 DeepEval trace 模式。

## 安装

```bash
pip install -r AgentEval/requirements.txt
```

如果只跑本地规则评测，可以先不安装 DeepEval。

## 运行本地评测

```bash
python AgentEval/run_eval.py
```

本地模式会运行 `AgentEval/eval_cases.json` 中的用例，并检查：

- NPC 回复不能为空
- action 必须属于项目允许的 NPC 动作
- 如果用例设置了 `expected_action`，实际 action 必须匹配
- 如果用例设置了 `required_keywords`，回复必须包含这些关键词

## 运行 DeepEval 模式

```bash
python AgentEval/run_eval.py --mode deepeval
```

DeepEval 模式会在 LangGraph 执行时挂载 DeepEval 回调，并使用
`TaskCompletionMetric` 作为额外评测信号。首次使用前请根据 DeepEval 官方文档完成
登录或评测模型配置。

## 新增用例

编辑 `AgentEval/eval_cases.json`，增加如下结构：

```json
{
    "case_id": "unique_case_id",
    "user_input": "玩家输入",
    "expected_action": "chat",
    "required_keywords": ["必须出现的词"],
    "context": {
        "session_id": "eval_session",
        "player_id": "eval_player",
        "npc_id": "barista_001"
    }
}
```

## 输出 JSON 报告

```bash
python AgentEval/run_eval.py --output AgentEval/report.json
```

报告会包含每条用例的分数、检查项、失败原因、NPC 回复和 action。
