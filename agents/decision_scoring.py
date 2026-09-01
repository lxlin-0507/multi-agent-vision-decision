"""Explainable rule-based decision scoring agent."""
from __future__ import annotations

from time import perf_counter

from agents.common import elapsed_ms, now_text
from schemas import (
    AgentMeta,
    AgentStatus,
    DecisionResult,
    QualityReviewResult,
    SceneAnalysisResult,
    ScoreFactor,
    VisionPerceptionResult,
)


class DecisionScoringAgent:
    name = "decision_scoring_agent"

    def __init__(self, rules: dict):
        self.rules = rules
        self.weights = rules["weights"]
        self.thresholds = rules["thresholds"]

    def score(
        self,
        vision: VisionPerceptionResult,
        review: QualityReviewResult,
        scene: SceneAnalysisResult | None,
    ) -> DecisionResult:
        started_at = now_text()
        start = perf_counter()
        factors = [
            self._evidence_factor(vision),
            self._confidence_factor(vision),
            self._complexity_factor(scene),
            self._readiness_factor(review),
        ]
        total = round(sum(item.score for item in factors), 2)
        review_reasons = review.reasons

        if review.quality_level == "failed":
            decision, score_level = "analysis_failed", "不可用"
        elif review.review_required:
            decision, score_level = "manual_review_required", self._score_level(total)
        else:
            decision, score_level = "automatic_analysis_available", self._score_level(total)

        return DecisionResult(
            meta=AgentMeta(
                agent_name=self.name,
                status=AgentStatus.DEGRADED if review.review_required else AgentStatus.SUCCESS,
                started_at=started_at,
                finished_at=now_text(),
                latency_ms=elapsed_ms(start),
                warnings=review_reasons,
            ),
            total_score=total,
            score_level=score_level,
            decision=decision,
            review_required=review.review_required,
            review_reasons=review_reasons,
            factors=factors,
            summary=self._summary(total, decision, review_reasons),
        )

    def _evidence_factor(self, vision: VisionPerceptionResult) -> ScoreFactor:
        maximum = self.weights["evidence_completeness"]
        if vision.meta.status == AgentStatus.FAILED:
            score, evidence = 0.0, ["视觉子 Agent 未产生可用结果。"]
        elif vision.total_detections == 0:
            score, evidence = maximum * 0.4, ["图像可读取，但未检测到模型支持的目标。"]
        else:
            score = maximum
            evidence = [f"检测到 {vision.total_detections} 个目标。"]
        return ScoreFactor(
            factor_id="evidence_completeness",
            factor_name="视觉证据完整度",
            score=score,
            max_score=maximum,
            evidence=evidence,
            source_agent="vision_perception_agent",
        )

    def _confidence_factor(self, vision: VisionPerceptionResult) -> ScoreFactor:
        maximum = self.weights["confidence_quality"]
        averages = list(vision.average_confidence.values())
        if not averages:
            score, evidence = 0.0, ["没有可用于置信度评估的检测目标。"]
        else:
            average = sum(averages) / len(averages)
            score = round(maximum * average, 2)
            evidence = [f"类别平均置信度均值为 {average:.2f}。"]
        return ScoreFactor(
            factor_id="confidence_quality",
            factor_name="检测可信度",
            score=score,
            max_score=maximum,
            evidence=evidence,
            source_agent="vision_perception_agent",
        )

    def _complexity_factor(self, scene: SceneAnalysisResult | None) -> ScoreFactor:
        maximum = self.weights["scene_complexity"]
        if scene is None:
            score, evidence = 0.0, ["场景分析未执行。"]
        elif scene.density_level == "高":
            score, evidence = maximum * 0.35, ["高密度场景增加自动判断复杂度。"]
        elif scene.density_level == "中":
            score, evidence = maximum * 0.65, ["中密度场景存在一定判断复杂度。"]
        else:
            score, evidence = maximum, ["低密度或无目标场景的规则复杂度较低。"]
        return ScoreFactor(
            factor_id="scene_complexity",
            factor_name="场景可分析性",
            score=score,
            max_score=maximum,
            evidence=evidence,
            source_agent="scene_analysis_agent",
        )

    def _readiness_factor(self, review: QualityReviewResult) -> ScoreFactor:
        maximum = self.weights["decision_readiness"]
        if review.allow_automatic_decision:
            score, evidence = maximum, ["质量复核允许自动分析。"]
        elif review.quality_level == "failed":
            score, evidence = 0.0, ["图像或视觉推理失败，不能自动分析。"]
        else:
            score, evidence = maximum * 0.25, ["质量复核要求人工确认。"]
        return ScoreFactor(
            factor_id="decision_readiness",
            factor_name="自动决策可用性",
            score=score,
            max_score=maximum,
            evidence=evidence,
            source_agent="quality_review_agent",
        )

    @staticmethod
    def _score_level(score: float) -> str:
        if score >= 75:
            return "高"
        if score >= 50:
            return "中"
        return "低"

    @staticmethod
    def _summary(score: float, decision: str, reasons: list[str]) -> str:
        if reasons:
            return f"评分 {score}，结论为 {decision}；原因：{'；'.join(reasons)}"
        return f"评分 {score}，结论为 {decision}。"
