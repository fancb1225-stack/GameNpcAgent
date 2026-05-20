# GameNpcFrame

GameNpcFrame 是一个面向游戏的通用 NPC Agent 框架。项目支持 NPC 对话、PDF
导入游戏设定、PDF 导入 NPC 设定、游戏设定 RAG 检索、长短期记忆、关系系统、
PostgreSQL + pgvector 持久化，以及 Docker Compose 一键启动。

当前项目仍保留少量早期咖啡厅示例数据和内部类名，但公开 API 与前端页面已经向通用
游戏 NPC 框架迁移。

## 核心能力

- 通用 NPC 对话：支持玩家、会话、NPC、关系、短期记忆、长期记忆和对话历史。
- 游戏设定 RAG：上传 PDF 后按章节切分设定文本，并在 NPC 对话时混合检索注入。
- NPC 设定导入：上传 NPC PDF 后识别结构化字段，并通过 ORM 写入数据库。
- 记忆系统：短期记忆保存最近交互，溢出后摘要为长期记忆并支持混合检索。
- NPC 管理：支持查询 NPC、选择 NPC 对话、删除 NPC。
- Docker 部署：编排后端服务和带 pgvector 扩展的 PostgreSQL 数据库。

## 模块与技术栈

| 模块 | 路径 | 职责 | 主要技术 |
| --- | --- | --- | --- |
| API 服务 | `api/` | 会话、NPC、对话、文档上传和 RAG 检索接口 | FastAPI、Pydantic、Uvicorn |
| Agent 编排 | `agent/` | NPC 对话状态图、工具调用和模型适配 | LangGraph、OpenAI SDK 兼容接口 |
| 业务服务 | `service/` | 玩家、NPC、关系、记忆、PDF 导入和 RAG 服务 | SQLAlchemy ORM、pypdf |
| 数据模型 | `model/` | API 和业务层数据读写模型 | Pydantic v2 |
| 数据库层 | `db/` | ORM 表、CRUD、混合检索、pgvector 查询 | SQLAlchemy 2、PostgreSQL、pgvector |
| 前端页面 | `frontend/` | 本地调试 UI，支持上传、聊天、删除 NPC 和 RAG 检索 | HTML、CSS、JavaScript |
| 数据库迁移 | `alembic/` | 管理数据库结构迁移 | Alembic |
| 容器化 | `Dockerfile`、`docker-compose.yml` | 构建 backend + PostgreSQL(pgvector) | Docker Compose |
| 测试 | `tests/` | 覆盖 API、Agent、RAG、记忆检索和 PDF 导入 | pytest |
| 示例文档 | `output/doc/` | 武侠世界背景和 NPC 设定示例文档 | DOCX、PDF |

## 目录说明

```text
GameNpcFrame/
├── agent/                  # Agent 编排、LLM 适配、NPC 对话图
├── api/                    # FastAPI 接口
├── db/                     # SQLAlchemy ORM、数据库服务、混合检索
├── model/                  # Pydantic 数据模型
├── service/                # 业务服务层
├── frontend/               # 本地调试前端页面
├── tests/                  # pytest 测试
├── docker/postgres/init/   # PostgreSQL 初始化脚本
├── output/doc/             # 示例设定文档
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## 技术实现细节

### 1. API 层

API 入口位于 `api/main.py`，使用 FastAPI 提供统一的 `ApiResponse` 返回结构：

```json
{
  "ok": true,
  "data": {},
  "error": null
}
```

主要接口：

- `GET /health`：健康检查，并返回当前公开工具列表。
- `POST /sessions`：创建或恢复玩家会话。
- `GET /npcs`：列出 NPC。
- `GET /npcs/{npc_id}`：查询单个 NPC。
- `DELETE /npcs/{npc_id}`：删除 NPC。
- `POST /sessions/{session_id}/select-npc`：选择当前会话要对话的 NPC。
- `POST /dialogue/messages`：发送玩家消息并获得 NPC 回复。
- `POST /documents/game-settings/upload`：上传游戏设定 PDF。
- `POST /documents/npc-settings/upload`：上传 NPC 设定 PDF。
- `POST /rag/game-settings/search`：调试游戏设定 RAG 检索。

实现要点：

- 文件上传使用 `python-multipart` 和 FastAPI 的 `UploadFile`。
- PDF 文件只在服务端被解析为文本，模型不直接接收 PDF 二进制。
- 删除 NPC 通过 `NpcService.delete_npc` 调用 ORM 层，不手写原始 SQL。
- 旧 coffee/cafe/world-state 公开接口已移除；调试工具列表也过滤旧工具名。

### 2. Agent 对话图

NPC 对话由 `agent/npc_graph_agent.py` 中的 LangGraph 状态图驱动。每次玩家发送消息时，
`NpcGraphAgent.handle_message` 会构造 `NpcGraphState`，再交给图执行。

图节点顺序：

1. `load_context`：读取 NPC 状态、玩家画像、关系和世界状态。
2. `retrieve_memories`：读取短期记忆、长期记忆，并检索游戏设定 RAG 上下文。
3. `decide_action`：调用 LLM 生成结构化决策；未启用 LLM 时走规则回复。
4. `normalize_decision`：清洗 LLM 输出，限制 action、emotion、关系变化字段。
5. `validate_action`：检查动作是否合法，不合法时降级为普通聊天。
6. `build_recommendation_context`：仅在特定动作下读取推荐上下文。
7. `generate_line`：生成最终 NPC 台词。
8. `decide_next_action`：判断是否需要延迟触发后续动作。
9. `apply_state_change`：写入关系变化、NPC 情绪和位置变化。
10. `update_player_profile`：从玩家输入中提取少量画像更新。
11. `write_memory`：写入对话历史和短期记忆。

结构化决策约束：

- `action` 必须在固定枚举内，例如 `chat`、`ask_player`、`move_location`。
- `emotion` 必须在固定枚举内，例如 `neutral`、`calm`、`curious`。
- `relationship_delta` 只允许 `trust`、`familiarity`、`fondness`、`dislike`、`stress`。
- 每个关系变化值会被限制在安全范围内，避免模型输出污染数据库。

### 3. 短期记忆实现

短期记忆由 `service/short_memory_service.py` 负责，数据库表为 `short_term_memories`。

设计模型：

- 每个玩家和每个 NPC 维护一个独立短期记忆窗口。
- 唯一逻辑键是 `player_id + npc_id`。
- 记忆内容保存在 JSONB 字段 `messages` 中。
- 每条消息包含 `role`、`content`、`emotion`、`timestamp`、`metadata`。

窗口策略：

- `SHORT_TERM_LIMIT = 20`
- `ARCHIVE_BATCH_SIZE = 10`

写入流程：

1. `append_short_term_message` 接收一条玩家或 NPC 消息。
2. `_normalize_message` 标准化消息结构，并补充时间戳。
3. 查询当前 `player_id + npc_id` 的短期记忆记录。
4. 如果不存在，则创建一条新记录。
5. 将新消息追加到 `messages` 数组。
6. 更新 `message_count` 和 `last_message_at`。
7. 如果消息数超过 20，返回 `needs_archive = true` 和最早 10 条消息。

归档触发：

- `npc_graph_agent.write_memory` 会在后台线程写入短期记忆。
- 写入后调用 `_archive_short_memory_if_needed` 检查是否溢出。
- 如果需要归档，则调用长期记忆服务生成长期记忆。
- 长期记忆写入成功后，调用 `trim_short_term_memory` 移除已归档消息。

这样短期记忆始终保留最近互动，同时较早互动会被压缩进入长期记忆。

### 4. 长期记忆实现

长期记忆由 `service/long_memory_service.py` 负责，数据库表为 `long_term_memories`。

长期记忆字段：

- `memory_id`：业务 ID。
- `player_id`：玩家 ID。
- `npc_id`：NPC ID。
- `title`：记忆标题。
- `content`：记忆摘要内容。
- `memory_type`：记忆类型。
- `importance`：重要性，范围 0 到 1。
- `embedding`：向量字段，用于 pgvector 检索。
- `embedding_model`：向量模型名称。
- `source_message_ids`：来源消息 ID 列表。
- `metadata_json`：额外元数据。

从短期记忆生成长期记忆：

1. 短期记忆窗口超过 20 条后，取最早 10 条消息。
2. `create_long_term_memory_from_messages` 调用 `summarize_messages`。
3. 如果传入了 LLM，则要求模型输出 JSON 摘要。
4. 如果 LLM 不可用或输出异常，则使用 `_rule_based_summary` 规则摘要。
5. 规则摘要会根据关键词判断记忆类型，例如偏好、警告、关系、习惯。
6. `create_long_term_memory` 写入长期记忆。
7. 写入前会调用 `_embed_memory_safely` 尝试生成 embedding。
8. embedding 失败不会阻断长期记忆写入，只是向量字段为空。

长期记忆检索：

1. `search_long_memories` 接收 `player_id`、`npc_id`、`query`、`limit`。
2. 优先调用 `EmbeddingService.embed_query` 生成查询向量。
3. `_load_memory_candidates` 使用 ORM + pgvector 取向量候选，并计算 cosine distance。
4. 没有 embedding 的历史记忆也会作为 BM25 候选进入混合排序。
5. `rank_hybrid_results` 计算 BM25 分数和向量相似度。
6. 最终按 `hybrid_score = 0.5 * bm25_score + 0.5 * vector_score` 排序。
7. 返回结果包含 `hybrid_score`、`bm25_score`、`vector_score`、`distance`。

### 5. Embedding 实现

Embedding 入口位于 `service/embedding_service.py`。

当前默认实现是 `HashEmbeddingProvider`：

- 向量维度：384。
- 模型名：`local-hash-embedding-384`。
- 不依赖外部模型 API。
- 对英文做简单分词，对中文做字符级 fallback token。
- 使用 MD5 hash 将 token 映射到固定维度。
- 使用正负号累加后做 L2 归一化。

注意：

当前 hash embedding 适合本地开发和端到端联调，但不是语义向量模型。生产环境建议替换为
真实 embedding 模型，同时保持向量维度与数据库 `Vector(384)` 字段一致，或同步修改表结构。

### 6. 混合检索实现

BM25 + 余弦相似度的公共逻辑位于 `db/hybrid_retrieval.py`。

核心函数：

- `split_text_by_chapter`：按章节标题切分游戏设定文档。
- `rank_hybrid_results`：对候选结果做 BM25 + 向量相似度融合排序。

分词策略：

- 英文、数字、下划线按词切分。
- 中文按字符切分，作为轻量 BM25 fallback。
- 不额外引入搜索引擎依赖，保持 Docker 环境简单。

向量分数：

- pgvector 的 `cosine_distance` 越小越相似。
- 系统将其转换为 `vector_score = 1 - distance`。
- 分数会被裁剪到 0 到 1。

BM25 分数：

- 使用轻量 BM25 实现，参数为 `k1 = 1.5`、`b = 0.75`。
- BM25 原始分会在当前候选集内归一化到 0 到 1。

融合公式：

```text
hybrid_score = 0.5 * bm25_score + 0.5 * vector_score
```

这样可以同时兼顾关键词命中和语义相似度。对游戏设定中的专有名词、门派名、地名、
法术名等，BM25 能补足纯向量检索的稳定性；对自然语言追问，向量相似度能补足同义表达。

### 7. 游戏设定 RAG 实现

RAG 由 `service/game_document_service.py`、`service/pdf_reader_service.py`、
`db/hybrid_retrieval.py` 和数据库表 `game_documents`、`game_setting_chunks` 共同实现。

#### 7.1 PDF 文本抽取

`PdfReaderService.extract_text` 使用 `pypdf.PdfReader`：

1. 接收 PDF 二进制内容。
2. 按页调用 `page.extract_text()`。
3. 过滤空页。
4. 用空行拼接页面文本。

#### 7.2 文档记录

上传游戏设定 PDF 后，`GameDocumentService.ingest_game_setting_text` 会：

1. 清理文本，空文档直接返回错误。
2. 生成 `document_id`。
3. 写入 `game_documents` 表。
4. metadata 中保存文件名和 content type。

#### 7.3 章节切分

游戏设定不再按固定长度优先切分，而是先识别章节标题。

支持的章节标题形式包括：

- `# 第一章 北境雪线`
- `第一章 北境雪线`
- `第二节 门派规矩`
- `1. 世界背景`
- `一、势力格局`
- `Chapter 1 World`

切分策略：

1. 扫描每一行文本。
2. 遇到章节标题时开启新的章节块。
3. 标题和后续正文会保留在同一个 chunk 中。
4. 如果单个章节超过 `chunk_size`，再使用带 overlap 的窗口二次切分。
5. 每个 chunk 的 metadata 会写入 `chapter_title` 和 `chunk_strategy = chapter`。

默认参数：

- `chunk_size = 2000`
- `chunk_overlap = 120`

章节切分比固定长度切分更适合游戏设定，因为世界观、势力、法术、NPC 背景通常天然按章节组织。
RAG 命中时可以返回完整语义单元，减少片段只包含半段规则或缺少前提的问题。

#### 7.4 向量写入

每个 chunk 会生成：

- `chunk_id`
- `document_id`
- `chunk_index`
- `content`
- `embedding`
- `embedding_model`
- `metadata_json`

这些数据写入 `game_setting_chunks` 表。该表的 `embedding` 字段使用 `Vector(384)`，
并创建 HNSW 索引：

```text
ix_game_setting_chunk_embedding_hnsw
postgresql_using = hnsw
postgresql_ops = vector_cosine_ops
```

#### 7.5 RAG 混合检索

检索由 `search_game_settings` 和 `DbService.search_game_chunks` 实现：

1. 清理查询文本。
2. 使用当前 embedding provider 生成查询向量。
3. ORM 层使用 `GameSettingChunk.embedding.cosine_distance(query_embedding)` 取候选集。
4. 候选集大小为 `max(20, min(limit * 10, 200))`。
5. `rank_hybrid_results` 对候选集计算 BM25 分数。
6. 将 `cosine_distance` 转换为 `vector_score`。
7. 按 `hybrid_score` 排序后返回前 `limit` 条。
8. `format_context` 将片段格式化为 Prompt 可读上下文。

格式化结果示例：

```text
[1] 第一章 北境雪线
北境禁止火焰法术，违反者会被巡夜司追捕。

[2] 第二章 南境炼金
南境允许炼金术研究，但炼金师必须登记。
```

#### 7.6 对话注入

NPC 对话图中的 `retrieve_memories` 节点会调用工具：

```text
get_game_setting_context
```

调用参数：

- `query`：当前玩家消息。
- `limit`：默认取 5 个片段。

返回的 `game_setting_context` 会进入 `decide_action` 的结构化 Prompt，供 NPC 决策和台词生成使用。

### 8. NPC 设定导入实现

NPC 设定导入由 `service/npc_setting_import_service.py` 实现。

链路：

1. 前端上传 NPC 设定 PDF。
2. `PdfReaderService` 抽取文本。
3. `NpcSettingImportService.import_npc_setting_text` 检查文本是否为空。
4. `extract_npc_payload` 优先调用 LLM，让模型输出结构化 JSON。
5. 如果 LLM 未启用、调用失败或 JSON 解析失败，则使用规则解析。
6. `normalize_npc_row` 清洗字段并补齐默认值。
7. 调用 `DbService.upsert_npc_from_setting` 通过 ORM 写入 `npcs` 表。
8. 返回结果前使用 `NpcRead.model_validate(...).model_dump(mode="json")` 转成字典。

LLM 输出目标结构：

```json
{
  "npcs": [
    {
      "npc_id": "master_shen_zhaowei",
      "name": "沈照微",
      "npc_type": "fixed_customer",
      "role": "玩家的师父",
      "age": 52,
      "gender": "女",
      "mbti": "INFJ",
      "job": "天阙山掌门",
      "personality": "沉静、克制",
      "speaking_style": "语速慢，措辞冷静",
      "background": "人物背景",
      "current_location": "天阙山"
    }
  ]
}
```

规则解析支持：

- `字段：值`
- `字段 值`
- PDF 表格抽取后的逐行字段
- 独立标题后的多行内容

当前支持的常见字段包括：

- `npc_id`
- `姓名`
- `npc_type`
- `身份`
- `年龄`
- `性别`
- `MBTI`
- `职业`
- `当前位置`
- `性格`
- `说话风格`
- `背景`

### 9. 数据库设计

数据库使用 PostgreSQL，ORM 使用 SQLAlchemy 2。Docker 环境使用
`pgvector/pgvector:pg16` 镜像，并通过 `docker/postgres/init/001_create_vector_extension.sql`
启用 vector 扩展。

核心表：

- `players`：玩家画像。
- `npcs`：NPC 基础设定、情绪、位置和状态。
- `relationships`：玩家与 NPC 的五维关系。
- `dialogue_sessions`：会话状态。
- `dialogue_messages`：对话历史。
- `short_term_memories`：短期记忆窗口。
- `long_term_memories`：长期记忆和向量。
- `game_documents`：导入的游戏设定文档。
- `game_setting_chunks`：游戏设定切片和向量。

关系系统使用五个维度：

- `trust`
- `familiarity`
- `fondness`
- `dislike`
- `stress`

NPC 每轮对话可以通过 `relationship_delta` 修改这些维度。

## 环境要求

- Python 3.13
- Docker Desktop
- PostgreSQL 16 + pgvector，Docker Compose 会自动启动

## Docker 启动

```bash
docker compose up --build
```

服务默认地址：

- 后端 API：`http://localhost:8000`
- 前端页面：打开 `frontend/index.html`
- PostgreSQL：`localhost:5432`

## 常用接口

健康检查：

```bash
curl http://localhost:8000/health
```

创建或恢复对话会话：

```bash
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d "{\"player_id\":\"p001\",\"nickname\":\"测试玩家\"}"
```

列出 NPC：

```bash
curl http://localhost:8000/npcs
```

删除 NPC：

```bash
curl -X DELETE http://localhost:8000/npcs/master_shen_zhaowei
```

上传游戏设定 PDF：

```bash
curl -X POST http://localhost:8000/documents/game-settings/upload \
  -F "file=@game-setting.pdf"
```

上传 NPC 设定 PDF：

```bash
curl -X POST http://localhost:8000/documents/npc-settings/upload \
  -F "file=@npc-setting.pdf"
```

检索游戏设定 RAG 上下文：

```bash
curl -X POST http://localhost:8000/rag/game-settings/search \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"北境魔法规则\",\"limit\":3}"
```

## 本地开发

```bash
python -m pip install -e .[dev]
python -m db.db_init
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

## 配置项

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DATABASE_URL` | Docker 中指向 `postgres` 服务 | 数据库连接地址 |
| `ENABLE_LLM` | `false` | 是否启用 LLM；关闭时使用规则兜底 |
| `ENABLE_WEB_SEARCH` | `false` | 是否启用联网搜索工具 |
| `ENABLE_TOOL_DEBUG_API` | `true` | 是否开放 `/tools/invoke` 调试接口 |
| `CORS_ALLOW_ORIGINS` | `*` | CORS 允许来源 |
| `LLM_API_KEY` | 空 | 模型服务 API Key |
| `LLM_BASE_URL` | 空 | 模型服务 Base URL |
| `LLM_MODEL` | 空 | 模型名称 |

## 测试

```bash
python -m pytest -q
```

当前相关测试覆盖：

- `tests/test_hybrid_retrieval.py`：章节切分和混合排序。
- `tests/test_game_document_service.py`：游戏设定章节导入和 RAG 检索。
- `tests/test_long_memory_hybrid_search.py`：长期记忆混合检索。
- `tests/test_npc_graph_agent.py`：NPC 对话图和记忆注入链路。

## 开发约定

- 后端使用 SQLAlchemy ORM 访问数据库，业务代码避免手写原始 SQL 查询。
- README 和项目说明使用中文。
- 新增后端能力时，应补充对应测试。
- 前端当前为轻量调试页面，使用原生 HTML/CSS/JavaScript 实现。
