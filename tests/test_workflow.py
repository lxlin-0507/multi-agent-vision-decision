from dataclasses import dataclass

import config
from agents.decision_scoring import DecisionScoringAgent
from agents.quality_review import QualityReviewAgent
from agents.scene_analysis import SceneAnalysisAgent
from enhanced_auditable_agent import AgentServices, build_graph
from schemas import AgentMeta, AgentStatus, VisionPerceptionResult


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


@dataclass
class FakeVisionAgent:
    result: VisionPerceptionResult

    def analyze(self, image_path: str, confidence_threshold: float) -> VisionPerceptionResult:
        return self.result.model_copy(update={"image_path": image_path})


def test_low_confidence_route_reaches_review_and_persistence(tmp_path, monkeypatch):
    monkeypatch.setattr("enhanced_auditable_agent.config.REPORTS_DIR", str(tmp_path))
    vision = VisionPerceptionResult(
        meta=AgentMeta(
            agent_name="vision_perception_agent",
            status=AgentStatus.DEGRADED,
            started_at="2026-01-01 00:00:00",
            finished_at="2026-01-01 00:00:00",
            latency_ms=1,
            warnings=["低置信度类别：car。"],
        ),
        image_path="placeholder.jpg",
        model_name="mock",
        confidence_threshold=0.25,
        cache_hit=False,
        image_width=1280,
        image_height=720,
        quality_level="review_required",
        total_detections=1,
        class_counts={"car": 1},
        average_confidence={"car": 0.3},
    )
    services = AgentServices(
        vision=FakeVisionAgent(vision),
        scene=SceneAnalysisAgent(RULES["thresholds"]),
        review=QualityReviewAgent(RULES["thresholds"]),
        decision=DecisionScoringAgent(RULES),
    )
    image_path = tmp_path / "placeholder.jpg"
    image_path.write_bytes(b"placeholder")
    result = build_graph(services).invoke({"image_path": str(image_path)})

    assert result["decision_result"]["decision"] == "manual_review_required"
    assert result["review_result"]["review_required"] is True
    assert any(
        entry["node"] == "supervisor_agent"
        and entry["detail"]["to"] == "quality_review_agent"
        for entry in result["audit_log"]
    )
    assert result["final_report"].startswith(f"# {config.APP_NAME}报告")
    assert "遥感" not in result["final_report"]
    assert (tmp_path / result["run_id"] / "decision.json").exists()
