"""Offline image and detection quality review agent."""
from __future__ import annotations

from time import perf_counter

from agents.common import elapsed_ms, now_text
from schemas import (
    AgentMeta,
    AgentStatus,
    QualityReviewResult,
    VisionPerceptionResult,
)


class QualityReviewAgent:
    name = "quality_review_agent"

    def __init__(self, thresholds: dict[str, float | int]):
        self.thresholds = thresholds

    def review(self, vision: VisionPerceptionResult) -> QualityReviewResult:
        started_at = now_text()
        start = perf_counter()
        reasons: list[str] = []

        if vision.meta.status == AgentStatus.FAILED:
            reasons.append("视觉感知子 Agent 执行失败。")
        if vision.total_detections == 0:
            reasons.append("未检测到模型支持的目标，当前图像证据不足。")
        if (
            vision.image_width is not None
            and vision.image_height is not None
            and (
                vision.image_width < self.thresholds["minimum_recommended_width"]
                or vision.image_height < self.thresholds["minimum_recommended_height"]
            )
        ):
            reasons.append(
                f"图像分辨率 {vision.image_width}x{vision.image_height} 低于推荐值。"
            )

        low_confidence = [
            name
            for name, confidence in vision.average_confidence.items()
            if confidence < self.thresholds["low_confidence"]
        ]
        if low_confidence:
            reasons.append(f"低置信度类别：{', '.join(low_confidence)}。")

        if vision.meta.warnings:
            reasons.extend(vision.meta.warnings)

        review_required = bool(reasons)
        if vision.meta.status == AgentStatus.FAILED:
            quality_level = "failed"
            allow_automatic_decision = False
        elif review_required:
            quality_level = "review_required"
            allow_automatic_decision = False
        else:
            quality_level = "acceptable"
            allow_automatic_decision = True

        return QualityReviewResult(
            meta=AgentMeta(
                agent_name=self.name,
                status=AgentStatus.DEGRADED if review_required else AgentStatus.SUCCESS,
                started_at=started_at,
                finished_at=now_text(),
                latency_ms=elapsed_ms(start),
                warnings=reasons if review_required else [],
            ),
            review_required=review_required,
            allow_automatic_decision=allow_automatic_decision,
            quality_level=quality_level,
            reasons=reasons,
        )
