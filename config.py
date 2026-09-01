"""
统一配置管理模块。
所有配置项从 .env 文件加载，提供默认值兜底。
"""
import os
import sys
from pathlib import Path

# 将本地 .venv_lib 目录加入 sys.path（用于项目级依赖安装）
_VENV_LIB = Path(__file__).parent / ".venv_lib"
_USING_VIRTUALENV = sys.prefix != sys.base_prefix
if not _USING_VIRTUALENV and _VENV_LIB.exists() and str(_VENV_LIB) not in sys.path:
    sys.path.insert(0, str(_VENV_LIB))

from dotenv import load_dotenv

# 加载 .env 文件（优先当前目录，其次项目根目录）
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path, override=False)


# ===== 应用标识 =====
APP_NAME = os.getenv("APP_NAME", "多agent视觉感知与可解释决策")
DECISION_LABELS = {
    "analysis_failed": "分析未完成",
    "manual_review_required": "建议人工复核",
    "automatic_analysis_available": "可用于自动分析",
}


# ===== DeepSeek API =====
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ===== YOLO 模型 =====
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8n.pt")
YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.25"))

# ===== 输出路径 =====
REPORTS_DIR = os.getenv("REPORTS_DIR", "reports/enhanced")

# ===== 日志 =====
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/agent.log")

# ===== 缓存 =====
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_DIR = os.getenv("CACHE_DIR", "cache/detections")


def validate_config() -> list[str]:
    """校验必要配置项，返回缺失/警告列表。"""
    warnings = []
    if not DEEPSEEK_API_KEY:
        warnings.append("DEEPSEEK_API_KEY 未设置，LLM 调用将失败")
    if not YOLO_WEIGHTS:
        warnings.append("YOLO_WEIGHTS 未设置，目标检测将失败")
    return warnings