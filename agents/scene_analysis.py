"""Rule-based scene analysis agent."""
from __future__ import annotations

from time import perf_counter

from agents.common import elapsed_ms, now_text
from schemas import AgentMeta, AgentStatus, SceneAnalysisResult, VisionPerceptionResult


class SceneAnalysisAgent:
    name = "scene_analysis_agent"

    def __init__(self, thresholds: dict[str, float | int]):
        self.thresholds = thresholds

    def analyze(self, vision: VisionPerceptionResult) -> SceneAnalysisResult:
        started_at = now_text()
        start = perf_counter()
        counts = vision.class_counts
        person_count = counts.get("person", 0)
        vehicle_count = sum(
            counts.get(name, 0)
            for name in ("car", "bus", "truck", "motorcycle", "bicycle")
        )
        animal_count = sum(
            counts.get(name, 0)
            for name in ("bird", "cat", "dog", "horse", "sheep", "cow")
        )
        evidence: list[str] = []

        if (
            vehicle_count >= self.thresholds["high_density_vehicle_count"]
            or person_count >= self.thresholds["high_density_person_count"]
        ):
            scene_type, scene_category, density = "高密度交通场景", "城市交通", "高"
            evidence.append(
                f"车辆={vehicle_count}、人员={person_count}，达到高密度阈值。"
            )
        elif (
            vehicle_count >= self.thresholds["medium_density_vehicle_count"]
            or person_count >= self.thresholds["medium_density_person_count"]
        ):
            scene_type, scene_category, density = "中密度交通场景", "城镇/郊区", "中"
            evidence.append(
                f"车辆={vehicle_count}、人员={person_count}，达到中密度阈值。"
            )
        elif vision.total_detections == 0:
            scene_type, scene_category, density = "无目标场景", "未知/空旷区域", "低"
            evidence.append("未检测到当前模型支持的目标类别。")
        elif animal_count >= 3:
            scene_type, scene_category, density = "自然环境场景", "自然区域", "低"
            evidence.append(f"检测到动物目标={animal_count}。")
        else:
            scene_type, scene_category, density = "低密度场景", "稀疏目标区域", "低"
            evidence.append(f"仅检测到 {vision.total_detections} 个稀疏目标。")

        return SceneAnalysisResult(
            meta=AgentMeta(
                agent_name=self.name,
                status=AgentStatus.SUCCESS,
                started_at=started_at,
                finished_at=now_text(),
                latency_ms=elapsed_ms(start),
            ),
            scene_type=scene_type,
            scene_category=scene_category,
            density_level=density,
            person_count=person_count,
            vehicle_count=vehicle_count,
            animal_count=animal_count,
            primary_targets=sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5],
            evidence=evidence,
        )
