"""
多agent视觉感知与可解释决策系统。

默认运行完全离线的视觉、规则、审计与报告流程。只有 run_pipeline()
显式传入 enable_llm=True 时才会调用配置的 DeepSeek API。
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

try:
    from openai import OpenAI

    _OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None  # type: ignore
    _OPENAI_AVAILABLE = False

try:
    from ultralytics import YOLO

    _YOLO_AVAILABLE = True
except ImportError:
    YOLO = None  # type: ignore
    _YOLO_AVAILABLE = False

import config
from agents.common import load_rules, now_text
from agents.decision_scoring import DecisionScoringAgent
from agents.quality_review import QualityReviewAgent
from agents.scene_analysis import SceneAnalysisAgent
from agents.vision_perception import VisionPerceptionAgent
from exceptions import ConfigError, DetectionError, PersistenceError
from logger import get_logger
from schemas import (
    AgentStatus,
    AuditEntry,
    DecisionResult,
    QualityReviewResult,
    SceneAnalysisResult,
    VisionPerceptionResult,
)
from visualizer import draw_detections

_log = get_logger(__name__)
_ROOT_DIR = Path(__file__).parent
_RULES_PATH = _ROOT_DIR / "scoring_rules.yaml"
_YOLO_MODEL: Any = None
_LLM_CLIENT: Any = None


class WorkflowState(TypedDict, total=False):
    image_path: str
    run_id: str
    started_at: str
    finished_at: str
    enable_llm: bool
    audit_log: list[dict[str, Any]]
    vision_result: dict[str, Any]
    scene_result: dict[str, Any]
    review_result: dict[str, Any]
    decision_result: dict[str, Any]
    tool_outputs: dict[str, Any]
    agent_outputs: dict[str, Any]
    final_report: str
    output_files: dict[str, str]
    error: str
    error_code: str


@dataclass
class AgentServices:
    vision: VisionPerceptionAgent
    scene: SceneAnalysisAgent
    review: QualityReviewAgent
    decision: DecisionScoringAgent


def get_yolo_model() -> Any:
    """Load the configured YOLO model once per process."""
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        if not _YOLO_AVAILABLE:
            raise DetectionError("ultralytics 未安装，无法使用 YOLO 检测")
        try:
            _YOLO_MODEL = YOLO(config.YOLO_WEIGHTS)
        except Exception as error:
            raise DetectionError(
                f"YOLO 模型加载失败: {error}", {"weights": config.YOLO_WEIGHTS}
            ) from error
    return _YOLO_MODEL


def get_llm_client() -> Any:
    """Create the external LLM client only for explicitly enabled runs."""
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        if not _OPENAI_AVAILABLE:
            raise ConfigError("openai 库未安装，无法使用 LLM 报告生成")
        if not config.DEEPSEEK_API_KEY:
            raise ConfigError("DEEPSEEK_API_KEY 未配置，无法使用 LLM 报告生成")
        _LLM_CLIENT = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )
    return _LLM_CLIENT


def create_services() -> AgentServices:
    rules = load_rules(str(_RULES_PATH))
    return AgentServices(
        vision=VisionPerceptionAgent(
            model_loader=get_yolo_model,
            model_name=config.YOLO_WEIGHTS,
            thresholds=rules["thresholds"],
        ),
        scene=SceneAnalysisAgent(rules["thresholds"]),
        review=QualityReviewAgent(rules["thresholds"]),
        decision=DecisionScoringAgent(rules),
    )


def add_audit(
    state: WorkflowState, node: str, action: str, detail: dict[str, Any]
) -> None:
    entry = AuditEntry(time=now_text(), node=node, action=action, detail=detail)
    state.setdefault("audit_log", []).append(entry.model_dump(mode="json"))


def intake_agent(state: WorkflowState) -> WorkflowState:
    image_path = state.get("image_path", "").strip()
    state["run_id"] = state.get("run_id") or f"run_{uuid.uuid4().hex[:12]}"
    state["started_at"] = now_text()
    state.setdefault("audit_log", [])
    state.setdefault("tool_outputs", {})
    state.setdefault("agent_outputs", {})

    if not image_path:
        state["error"] = "missing_image_path"
        state["error_code"] = "IMAGE_ERROR"
        add_audit(state, "intake_agent", "validation_failed", {"reason": "empty_path"})
    elif not os.path.exists(image_path):
        state["error"] = "image_not_found"
        state["error_code"] = "IMAGE_ERROR"
        add_audit(
            state,
            "intake_agent",
            "validation_failed",
            {"reason": "image_not_found", "image_path": image_path},
        )
    else:
        state["image_path"] = image_path
        add_audit(state, "intake_agent", "accepted", {"image_path": image_path})
    return state


def vision_perception_node(services: AgentServices):
    def run(state: WorkflowState) -> WorkflowState:
        result = services.vision.analyze(
            state["image_path"], config.YOLO_CONF_THRESHOLD
        )
        if result.detections:
            annotations = [
                {
                    "class": item.class_name,
                    "conf": item.confidence,
                    "bbox": (
                        [item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2]
                        if item.bbox
                        else None
                    ),
                }
                for item in result.detections
            ]
            annotated_path = draw_detections(
                state["image_path"],
                annotations,
                output_dir=str(_run_dir(state)),
                run_id=state["run_id"],
            )
            result = result.model_copy(update={"annotated_image_path": annotated_path})

        serialized = result.model_dump(mode="json")
        state["vision_result"] = serialized
        state["tool_outputs"]["detect_objects"] = {
            "ok": result.meta.status != AgentStatus.FAILED,
            "class_counts": result.class_counts,
            "avg_confidence": result.average_confidence,
            "total": result.total_detections,
            "raw_items": [
                {
                    "class": item.class_name,
                    "conf": item.confidence,
                    "bbox": (
                        [item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2]
                        if item.bbox
                        else None
                    ),
                }
                for item in result.detections
            ],
        }
        add_audit(
            state,
            "vision_perception_agent",
            "analysis_completed",
            {
                "status": result.meta.status.value,
                "total_detections": result.total_detections,
                "cache_hit": result.cache_hit,
                "quality_level": result.quality_level,
                "warnings": result.meta.warnings,
                "errors": result.meta.errors,
            },
        )
        return state

    return run


def route_after_vision(state: WorkflowState) -> str:
    vision = VisionPerceptionResult.model_validate(state["vision_result"])
    if vision.meta.status == AgentStatus.FAILED:
        route, reason = "quality_review_agent", "视觉感知执行失败"
    elif vision.quality_level != "acceptable":
        route, reason = "quality_review_agent", f"质量等级为 {vision.quality_level}"
    else:
        route, reason = "scene_analysis_agent", "视觉结果满足自动场景分析条件"
    add_audit(
        state,
        "supervisor_agent",
        "route_selected",
        {"from": "vision_perception_agent", "to": route, "reason": reason},
    )
    return route


def scene_analysis_node(services: AgentServices):
    def run(state: WorkflowState) -> WorkflowState:
        vision = VisionPerceptionResult.model_validate(state["vision_result"])
        result = services.scene.analyze(vision)
        state["scene_result"] = result.model_dump(mode="json")
        state["agent_outputs"]["semantic_agent"] = {
            "scene_type": result.scene_type,
            "scene_category": result.scene_category,
            "density_level": result.density_level,
            "person_count": result.person_count,
            "vehicle_count": result.vehicle_count,
            "animal_count": result.animal_count,
            "primary_targets": result.primary_targets,
            "evidence": result.evidence,
        }
        add_audit(
            state,
            "scene_analysis_agent",
            "scene_inferred",
            {"scene_type": result.scene_type, "evidence": result.evidence},
        )
        return state

    return run


def quality_review_node(services: AgentServices):
    def run(state: WorkflowState) -> WorkflowState:
        vision = VisionPerceptionResult.model_validate(state["vision_result"])
        result = services.review.review(vision)
        state["review_result"] = result.model_dump(mode="json")
        add_audit(
            state,
            "quality_review_agent",
            "review_completed",
            {
                "review_required": result.review_required,
                "reasons": result.reasons,
            },
        )
        return state

    return run


def ensure_review_node(services: AgentServices):
    def run(state: WorkflowState) -> WorkflowState:
        if "review_result" not in state:
            return quality_review_node(services)(state)
        return state

    return run


def decision_scoring_node(services: AgentServices):
    def run(state: WorkflowState) -> WorkflowState:
        vision = VisionPerceptionResult.model_validate(state["vision_result"])
        review = QualityReviewResult.model_validate(state["review_result"])
        scene = (
            SceneAnalysisResult.model_validate(state["scene_result"])
            if "scene_result" in state
            else None
        )
        result = services.decision.score(vision, review, scene)
        state["decision_result"] = result.model_dump(mode="json")
        state["agent_outputs"]["decision_scoring_agent"] = state["decision_result"]
        add_audit(
            state,
            "decision_scoring_agent",
            "score_calculated",
            {
                "total_score": result.total_score,
                "decision": result.decision,
                "review_required": result.review_required,
            },
        )
        return state

    return run


def _template_report(state: WorkflowState) -> str:
    vision = VisionPerceptionResult.model_validate(state["vision_result"])
    decision = DecisionResult.model_validate(state["decision_result"])
    scene = state.get("scene_result")
    decision_label = config.DECISION_LABELS.get(decision.decision, decision.decision)
    lines = [
        f"# {config.APP_NAME}报告",
        "",
        f"> run_id: {state['run_id']} | 时间: {state['started_at']}",
        "",
        "## 决策结论",
        f"- 总分: {decision.total_score}",
        f"- 评分等级: {decision.score_level}",
        f"- 决策: {decision_label}",
        f"- 需要人工复核: {'是' if decision.review_required else '否'}",
        f"- 摘要: {decision.summary}",
        "",
        "## 视觉感知",
        f"- 检测总数: {vision.total_detections}",
        f"- 类别统计: {json.dumps(vision.class_counts, ensure_ascii=False)}",
        f"- 平均置信度: {json.dumps(vision.average_confidence, ensure_ascii=False)}",
        f"- 质量等级: {vision.quality_level}",
        "",
        "## 评分分解",
    ]
    for factor in decision.factors:
        lines.append(
            f"- {factor.factor_name}: {factor.score}/{factor.max_score}；"
            f"{'；'.join(factor.evidence)}"
        )
    if scene:
        scene_result = SceneAnalysisResult.model_validate(scene)
        lines.extend(
            [
                "",
                "## 场景分析",
                f"- 场景类型: {scene_result.scene_type}",
                f"- 场景分类: {scene_result.scene_category}",
                f"- 推断依据: {'；'.join(scene_result.evidence)}",
            ]
        )
    if decision.review_reasons:
        lines.extend(["", "## 人工复核原因"])
        lines.extend(f"- {reason}" for reason in decision.review_reasons)
    return "\n".join(lines)


def _generate_llm_report(state: WorkflowState) -> str:
    """Use the LLM only after a caller explicitly enables it."""
    if not state.get("enable_llm", False):
        return ""
    client = get_llm_client()
    payload = {
        "vision": state["vision_result"],
        "scene": state.get("scene_result"),
        "review": state["review_result"],
        "decision": state["decision_result"],
    }
    response = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    f"你是{config.APP_NAME}的报告助手。"
                    "只依据给出的结构化数据输出中文 Markdown，"
                    "不要扩展数据中未提供的业务语境或现实结论。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, indent=2),
            },
        ],
        temperature=0.3,
        max_tokens=1600,
    )
    return response.choices[0].message.content


def report_agent(state: WorkflowState) -> WorkflowState:
    report = _template_report(state)
    mode = "template"
    if state.get("enable_llm", False):
        try:
            generated = _generate_llm_report(state)
            if generated:
                report = report.split("## 决策结论", maxsplit=1)[0] + generated
                mode = "llm"
        except Exception as error:
            add_audit(
                state,
                "report_agent",
                "llm_fallback",
                {"reason": str(error)},
            )
    state["final_report"] = report
    add_audit(state, "report_agent", "report_generated", {"mode": mode})
    return state


def _run_dir(state: WorkflowState) -> Path:
    return Path(config.REPORTS_DIR) / state["run_id"]


def persistence_agent(state: WorkflowState) -> WorkflowState:
    run_dir = _run_dir(state)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": run_dir / "report.md",
        "audit": run_dir / "audit.json",
        "state": run_dir / "state.json",
        "decision": run_dir / "decision.json",
    }
    try:
        state["finished_at"] = now_text()
        state["output_files"] = {name: str(path) for name, path in paths.items()}
        vision = VisionPerceptionResult.model_validate(state["vision_result"])
        if vision.annotated_image_path:
            state["output_files"]["annotated"] = vision.annotated_image_path
        add_audit(
            state,
            "persistence_agent",
            "artifacts_saved",
            {"run_dir": str(run_dir), "files": state["output_files"]},
        )
        paths["report"].write_text(state["final_report"], encoding="utf-8")
        paths["audit"].write_text(
            json.dumps(state["audit_log"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths["decision"].write_text(
            json.dumps(state["decision_result"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths["state"].write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as error:
        raise PersistenceError(
            f"工件保存失败: {error}", {"run_id": state.get("run_id")}
        ) from error
    return state


def build_graph(services: AgentServices | None = None):
    services = services or create_services()
    builder = StateGraph(WorkflowState)
    builder.add_node("intake_agent", intake_agent)
    builder.add_node("vision_perception_agent", vision_perception_node(services))
    builder.add_node("scene_analysis_agent", scene_analysis_node(services))
    builder.add_node("quality_review_agent", quality_review_node(services))
    builder.add_node("ensure_review_agent", ensure_review_node(services))
    builder.add_node("decision_scoring_agent", decision_scoring_node(services))
    builder.add_node("report_agent", report_agent)
    builder.add_node("persistence_agent", persistence_agent)
    builder.set_entry_point("intake_agent")
    builder.add_edge("intake_agent", "vision_perception_agent")
    builder.add_conditional_edges(
        "vision_perception_agent",
        route_after_vision,
        {
            "scene_analysis_agent": "scene_analysis_agent",
            "quality_review_agent": "quality_review_agent",
        },
    )
    builder.add_edge("scene_analysis_agent", "ensure_review_agent")
    builder.add_edge("quality_review_agent", "decision_scoring_agent")
    builder.add_edge("ensure_review_agent", "decision_scoring_agent")
    builder.add_edge("decision_scoring_agent", "report_agent")
    builder.add_edge("report_agent", "persistence_agent")
    builder.add_edge("persistence_agent", END)
    return builder.compile()


def run_pipeline(
    image_path: str,
    *,
    enable_llm: bool = False,
    services: AgentServices | None = None,
) -> WorkflowState:
    """Run one image analysis. LLM usage is disabled unless explicitly enabled."""
    if enable_llm:
        _log.warning("已显式启用外部 LLM API 调用。")
    warnings = config.validate_config()
    if warnings and enable_llm:
        for warning in warnings:
            _log.warning("配置警告: %s", warning)
    graph = build_graph(services)
    return graph.invoke({"image_path": image_path, "enable_llm": enable_llm})


if __name__ == "__main__":
    image_path = input("请输入图像路径: ").strip().strip('"')
    result = run_pipeline(image_path)
    print(result.get("final_report", ""))
