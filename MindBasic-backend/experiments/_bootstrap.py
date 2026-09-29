"""实验脚本的公共引导：让 `python experiments/xxx.py` 可直接运行。

实验脚本导入的是**线上同一份**规则与融合代码（app.services.*），
因此这里需要提供最小可用的应用配置占位值。

占位值只在"环境变量与项目根目录的 .env 都没有提供该项"时写入：
环境变量的优先级高于 .env，如果先写入占位值，会覆盖 .env 里的真实配置
（例如 ``DATABASE_URL``），导致需要数据库的脚本连不上库。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: 后端项目根目录（experiments/ 的上一级）
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "experiments" / "data"
RESULT_DIR: Path = PROJECT_ROOT / "experiments" / "results"


def ensure_app_importable() -> None:
    """把项目根目录加入 sys.path，并补齐最小配置项。"""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from_env_file = _env_file_values()
    for key, placeholder in (
        (
            "DATABASE_URL",
            "mysql+pymysql://experiment:experiment@127.0.0.1:3306/mindbasic?charset=utf8mb4",
        ),
        ("JWT_SECRET_KEY", "experiments-only-placeholder-secret-key-0123456789abcdef"),
        ("DEBUG", "true"),
    ):
        if key not in os.environ and key not in from_env_file:
            os.environ[key] = placeholder


def _env_file_values() -> dict[str, str]:
    """读取项目根目录 .env 的键（值交由 pydantic-settings 解析）。"""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return {}
    try:
        from dotenv import dotenv_values
    except ImportError:  # pragma: no cover - python-dotenv 属于运行依赖
        return {}
    return {key: value for key, value in dotenv_values(env_path).items() if value is not None}


__all__ = ["PROJECT_ROOT", "DATA_DIR", "RESULT_DIR", "ensure_app_importable"]
