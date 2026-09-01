"""
对话式交互模块。
在流水线分析完成后，支持用户对分析结果进行多轮追问。
"""
import json
from collections.abc import Iterator
from typing import Any, Dict, List

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore

import config
from logger import get_logger

_log = get_logger("chat")


class AnalysisChat:
    """分析结果对话管理器。保存分析上下文，支持多轮追问。"""

    def __init__(self, pipeline_result: Dict[str, Any]):
        """
        Args:
            pipeline_result: run_pipeline() 返回的完整状态
        """
        self.run_id = pipeline_result.get("run_id", "unknown")
        self.context = pipeline_result

        # 构建对话上下文
        det = pipeline_result.get("tool_outputs", {}).get("detect_objects", {})
        semantic = pipeline_result.get("agent_outputs", {}).get("semantic_agent", {})
        compliance = pipeline_result.get("agent_outputs", {}).get("compliance_agent", {})
        geofence = pipeline_result.get("agent_outputs", {}).get("geofence_agent", {})
        decision = pipeline_result.get("decision_result", {})
        review = pipeline_result.get("review_result", {})

        self.analysis_summary = json.dumps({
            "检测统计": {
                "总数": det.get("total", 0),
                "类别": det.get("class_counts", {}),
                "置信度": det.get("avg_confidence", {}),
            },
            "场景": {
                "类型": semantic.get("scene_type", "未知"),
                "分类": semantic.get("scene_category", "未知"),
                "密度": semantic.get("density_level", "未知"),
            },
            "新版决策": {
                "分数": decision.get("total_score", 0),
                "等级": decision.get("score_level", "未知"),
                "结论": decision.get("decision", "未知"),
                "需要人工复核": decision.get("review_required", False),
                "复核原因": decision.get("review_reasons", []),
                "评分分解": decision.get("factors", []),
            },
            "质量复核": {
                "质量等级": review.get("quality_level", "未知"),
                "允许自动决策": review.get("allow_automatic_decision", False),
                "原因": review.get("reasons", []),
            },
            "风险": {
                "分数": compliance.get("risk_score", 0),
                "等级": compliance.get("risk_level", "未知"),
                "建议": compliance.get("audit_advice", ""),
            },
            "围栏": geofence.get("candidates", []),
        }, ensure_ascii=False, indent=2)

        self.messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    f"你是{config.APP_NAME}的专业视觉分析助手。用户已经完成了一次图像分析，"
                    "现在会基于分析结果向你提问。请根据分析上下文，用中文专业地回答用户的问题。\n\n"
                    "分析上下文：\n" + self.analysis_summary
                ),
            }
        ]
        _log.info("[%s] 对话会话已创建", self.run_id)

    def ask_stream(self, question: str) -> Iterator[str]:
        """流式处理追问，逐步产出回答并在完成后保存完整历史。"""
        self.messages.append({"role": "user", "content": question})

        if not _OPENAI_AVAILABLE:
            fallback = "⚠️ openai 库未安装，无法使用对话功能。请运行 pip install openai 安装。"
            self.messages.append({"role": "assistant", "content": fallback})
            yield fallback
            return

        answer_parts: list[str] = []
        try:
            client = OpenAI(
                api_key=config.DEEPSEEK_API_KEY,
                base_url=config.DEEPSEEK_BASE_URL,
            )
            stream = client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=self.messages,
                temperature=0.5,
                max_tokens=1500,
                stream=True,
            )

            for chunk in stream:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if not content:
                    continue
                answer_parts.append(content)
                yield "".join(answer_parts)

            answer = "".join(answer_parts).strip()
            if not answer:
                answer = "抱歉，本次没有生成可显示的回答。"
                yield answer
            self.messages.append({"role": "assistant", "content": answer})
            _log.info("[%s] 追问已回复: %s...", self.run_id, question[:50])
        except Exception as e:
            _log.error("[%s] 追问失败: %s", self.run_id, e)
            prefix = "".join(answer_parts)
            fallback = f"抱歉，回答生成失败：{e}"
            answer = f"{prefix}\n\n{fallback}" if prefix else fallback
            self.messages.append({"role": "assistant", "content": answer})
            yield answer

    def ask(self, question: str) -> str:
        """兼容非流式调用方，返回完整回答。"""
        answer = ""
        for answer in self.ask_stream(question):
            pass
        return answer

    def get_history(self) -> List[Dict[str, str]]:
        """返回对话历史（不含 system prompt）。"""
        return self.messages[1:]  # 跳过 system prompt


# 全局会话缓存（按 run_id 存储）
_chat_sessions: Dict[str, AnalysisChat] = {}


def get_or_create_chat(run_id: str, pipeline_result: Dict[str, Any] | None = None) -> AnalysisChat | None:
    """获取或创建对话会话。"""
    if run_id in _chat_sessions:
        return _chat_sessions[run_id]

    if pipeline_result is None:
        return None

    chat = AnalysisChat(pipeline_result)
    _chat_sessions[run_id] = chat
    return chat


def clear_chat(run_id: str) -> None:
    """清除对话会话。"""
    _chat_sessions.pop(run_id, None)
