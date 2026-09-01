"""
结构化日志模块。
替换全局 print，支持控制台 + 文件双输出，格式统一。
"""
import logging
import os
from pathlib import Path

from config import LOG_LEVEL, LOG_FILE

# 确保日志目录存在
_log_path = Path(LOG_FILE)
_log_path.parent.mkdir(parents=True, exist_ok=True)

# 构建 logger
logger = logging.getLogger("multi_agent_vision_decision")
logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

# 防止重复添加 handler
if not logger.handlers:
    # 格式：时间 | 级别 | 模块 | 消息
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # 文件 handler
    file_handler = logging.FileHandler(_log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)


def get_logger(name: str | None = None) -> logging.Logger:
    """获取带命名空间的 logger。"""
    if name:
        return logging.getLogger(f"multi_agent_vision_decision.{name}")
    return logger
