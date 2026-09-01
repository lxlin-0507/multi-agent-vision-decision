"""
自定义异常类，用于精细化的错误分类与处理。
每个异常类携带 error_code 和可选的 detail 字典，便于审计追踪。
"""


class AgentBaseError(Exception):
    """所有 Agent 异常的基类。"""
    error_code: str = "AGENT_BASE_ERROR"

    def __init__(self, message: str, detail: dict | None = None):
        super().__init__(message)
        self.detail = detail or {}


class ConfigError(AgentBaseError):
    """配置相关错误：缺少必要配置项、配置格式错误等。"""
    error_code = "CONFIG_ERROR"


class ImageError(AgentBaseError):
    """图像相关错误：文件不存在、格式不支持、无法读取等。"""
    error_code = "IMAGE_ERROR"


class DetectionError(AgentBaseError):
    """目标检测错误：模型加载失败、推理异常、显存不足等。"""
    error_code = "DETECTION_ERROR"


class SemanticError(AgentBaseError):
    """语义分析错误：场景推断失败、数据不足无法判定等。"""
    error_code = "SEMANTIC_ERROR"


class ReportError(AgentBaseError):
    """报告生成错误：模板渲染失败、LLM 调用异常等。"""
    error_code = "REPORT_ERROR"


class PersistenceError(AgentBaseError):
    """持久化错误：磁盘空间不足、权限不足、写入失败等。"""
    error_code = "PERSISTENCE_ERROR"