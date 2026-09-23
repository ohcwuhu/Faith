FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.lock .
# 本镜像只装核心运行时（requirements.lock）。AI 实验室的重型依赖
# （torch / funasr / deepface / opensmile / pandas 等，见 requirements-ai.txt）
# 体积过大，需要按平台单独安装；未安装时 ASR / 语调情感 / VLM 等能力会降级，
# 但应用本身必须能正常启动——因此 socketio、requests、jieba 已列入核心依赖。
RUN pip install --no-cache-dir -r requirements.lock

COPY alembic.ini .
COPY alembic ./alembic
COPY app ./app
COPY scripts ./scripts
# 知识库检索索引：宿主机用 scripts/build_kb_index.py 生成后随镜像发布。
# 缺失时检索会静默降级为"不带参考资料"（见 app/services/ai_lab/kb_service.py）。
COPY data ./data

# 运行数据目录（挂载持久化盘）
RUN mkdir -p /app/uploads /app/exports

EXPOSE 8000

# 生产：单 worker（AI 模型常驻），启动前自动迁移
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:socket_app --host 0.0.0.0 --port 8000 --workers 1"]
