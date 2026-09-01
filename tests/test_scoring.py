from agents.decision_scoring import DecisionScoringAgent
from agents.quality_review import QualityReviewAgent
from agents.scene_analysis import SceneAnalysisAgent
from schemas import (
    AgentMeta,
    AgentStatus,
    VisionPerceptionResult,
)


RULES = {
    "thresholds": {
        "low_confidence": 0.45,
        "high_density_vehicle_count": 10,
        "medium_density_vehicle_count": 5,
        "high_density_person_count": 6,
        "medium_density_person_count": 3,
        "minimum_recommended_width": 640,
        "minimum_recommended_height": 480,
    },
    "weights": {
        "evidence_completeness": 25,
        "confidence_quality": 25,
        "scene_complexity": 25,
        "decision_readiness": 25,
    },
}


def make_vision(**overrides):
    payload = {
        "meta": AgentMeta(
            agent_name="vision_perception_agent",
            status=AgentStatus.SUCCESS,
            started_at="2026-01-01 00:00:00",
            finished_at="2026-01-01 00:00:00",
            latency_ms=1,
        ),
        "image_path": "test.jpg",
        "model_name": "mock",
        "confidence_threshold": 0.25,
        "cache_hit": False,
        "image_width": 1280,
        "image_height": 720,
        "quality_level": "acceptable",
        "total_detections": 3,
        "class_counts": {"car": 3},
        "average_confidence": {"car": 0.9},
    }
    payload.update(overrides)
    return VisionPerceptionResult(**payload)


def test_low_confidence_requires_review_and_reduces_score():
    vision = make_vision(
        average_confidence={"car": 0.3},
        quality_level="review_required",
    )
    review = QualityReviewAgent(RULES["thresholds"]).review(vision)
    scene = SceneAnalysisAgent(RULES["thresholds"]).analyze(vision)
    decision = DecisionScoringAgent(RULES).score(vision, review, scene)

    assert review.review_required is True
    assert decision.decision == "manual_review_required"
    assert any(factor.factor_id == "confidence_quality" for factor in decision.factors)
