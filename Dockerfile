# 使用 Python 3.13 精简镜像作为运行环境
FROM python:3.13-slim

# 设置 Python 运行参数，减少缓存并让日志实时输出
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 设置容器内工作目录
WORKDIR /app

# 先复制依赖声明文件，利用 Docker 构建缓存加速后续构建
COPY pyproject.toml README.md ./

# 从 pyproject.toml 读取项目依赖，避免当前包结构不匹配导致构建失败
RUN python - <<'PY' > /tmp/requirements.txt
import tomllib

with open("pyproject.toml", "rb") as file:
    config = tomllib.load(file)

for dependency in config["project"]["dependencies"]:
    print(dependency)
PY
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements.txt

# 复制应用源码和数据库迁移配置
COPY agent ./agent
COPY api ./api
COPY db ./db
COPY model ./model
COPY service ./service
COPY alembic ./alembic
COPY alembic.ini ./
COPY frontend ./frontend

# 声明 FastAPI 默认服务端口
EXPOSE 8000

# 启动 FastAPI 服务，对外监听容器网络
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
