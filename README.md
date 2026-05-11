# CoffeeNpcAgent

面向游戏的咖啡厅 NPC Agent 后端。首版采用“理解-决策-行动”状态机，
把 NPC 行动限制在固定 action schema 内，优先保证稳定、有趣、低延迟。

## 功能范围

- 固定上午咖啡厅场景：老板、咖啡师、固定顾客、随机路人。
- NPC 动作限制：聊天、推荐咖啡、询问玩家、评价其他 NPC、移动位置、
  服务顾客、结束对话。
- 五维关系：信任、熟悉、喜欢、厌恶、压力。
- 短期记忆：每个玩家与 NPC 保留最近 20 条对话。
- 长期记忆：短期窗口溢出时归档最早 10 条并生成摘要。
- 玩家画像：口味偏好、常互动 NPC、性格倾向等。
- 咖啡厅世界状态：时间段、客流量、库存、今日菜单、天气、音乐、座位、事件。
- 咖啡知识库：用于稳定推荐和事实性回复。

## 安装依赖

```bash
python -m pip install -e .[dev]
```

## 环境变量

复制 `.env.example` 并按本机 PostgreSQL 配置修改：

```bash
DATABASE_URL=postgresql+psycopg://postgres:root123456@localhost:5432/coffee_npc_agent
```

首版可以不配置 LLM，服务会使用规则化流程完成对话。

## 数据库迁移

```bash
alembic upgrade head
```

应用代码通过 SQLAlchemy ORM 访问数据库，不编写业务原始 SQL 查询。

## 启动服务

```bash
python -m coffee_npc_agent.main
```

服务默认监听 `http://localhost:8000`。

## 示例请求

创建会话：

```bash
curl -X POST http://localhost:8000/sessions ^
  -H "Content-Type: application/json" ^
  -d "{\"player_id\":\"p001\",\"session_id\":\"session_001\",\"time_period\":\"morning\"}"
```

查询上午遇见 NPC：

```bash
curl http://localhost:8000/sessions/session_001/encounters
```

与咖啡师对话：

```bash
curl -X POST http://localhost:8000/dialogue/messages ^
  -H "Content-Type: application/json" ^
  -d "{\"player_id\":\"p001\",\"session_id\":\"session_001\",\"npc_id\":\"barista_001\",\"message\":\"今天推荐什么咖啡？\"}"
```

查询关系：

```bash
curl http://localhost:8000/players/p001/relationships
```

查询玩家画像：

```bash
curl http://localhost:8000/players/p001/profile
```

查询咖啡厅世界状态：

```bash
curl http://localhost:8000/world-state/session_001
```

## 运行测试

```bash
python -m pytest -q
```

测试覆盖 ORM 表声明、Repository 基础读写、状态机 action 校验、关系规则、
短期记忆滑动窗口、长期摘要、固定上午场景和 API。
