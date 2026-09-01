"""YOLO-backed vision perception agent."""
from __future__ import annotations

import os
from time import perf_counter
from typing import Any, Callable

try:
    import cv2
except ImportError:
    cv2 = None

from agents.common import elapsed_ms, now_text
from cache import get_cached, set_cache
from schemas import (
    AgentMeta,
    AgentStatus,
    BoundingBox,
    DetectionItem,
    VisionPerceptionResult,
)


class VisionPerceptionAgent:
    name = "vision_perception_agent"

    def __init__(
        self,
        model_loader: Callable[[], Any],
        model_name: str,
        thresholds: dict[str, float | int],
    ):
        self.model_loader = model_loader
        self.model_name = model_name
        self.thresholds = thresholds

    def analyze(self, image_path: str, confidence_threshold: float) -> VisionPerceptionResult:
        started_at = now_text()
        start = perf_counter()
        dimensions = self._image_dimensions(image_path)
        if not os.path.exists(image_path):
            return self._failed(
                image_path,
                confidence_threshold,
                started_at,
                start,
                ["图片文件不存在。"],
                dimensions,
            )
        if dimensions is None:
            return self._failed(
                image_path,
                confidence_threshold,
                started_at,
                start,
                ["图片无法被 OpenCV 读取。"],
                dimensions,
            )

        cached = get_cached(image_path, confidence_threshold)
        if cached is not None:
            return self._from_cached(
                cached, image_path, confidence_threshold, started_at, start, dimensions, True
            )

        try:
            model = self.model_loader()
            results = model(image_path, verbose=False)
        except Exception as error:
            return self._failed(
                image_path,
                confidence_threshold,
                started_at,
                start,
                [f"YOLO 推理失败：{error}"],
                dimensions,
            )

        raw_items: list[dict[str, Any]] = []
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < confidence_threshold:
                    continue
                raw_items.append(
                    {
                        "class": model.names[int(box.cls[0])],
                        "conf": confidence,
                        "bbox": box.xyxy[0].tolist() if box.xyxy is not None else None,
                    }
                )

        output = self._build_output(raw_items, image_path, confidence_threshold, False, dimensions)
        set_cache(image_path, confidence_threshold, output)
        return self._from_cached(
            output, image_path, confidence_threshold, started_at, start, dimensions, False
        )

    def _from_cached(
        self,
        cached: dict[str, Any],
        image_path: str,
        confidence_threshold: float,
        started_at: str,
        start: float,
        dimensions: tuple[int, int],
        cache_hit: bool,
    ) -> VisionPerceptionResult:
        detections = [
            DetectionItem(
                class_name=item["class"],
                confidence=item["conf"],
                bbox=BoundingBox(
                    x1=item["bbox"][0],
                    y1=item["bbox"][1],
                    x2=item["bbox"][2],
                    y2=item["bbox"][3],
                )
                if item.get("bbox")
                else None,
            )
            for item in cached.get("raw_items", [])
        ]
        warnings = self._quality_warnings(
            dimensions, cached.get("total", 0), cached.get("avg_confidence", {})
        )
        quality_level = "acceptable" if not warnings else "review_required"
        return VisionPerceptionResult(
            meta=AgentMeta(
                agent_name=self.name,
                status=AgentStatus.DEGRADED if warnings else AgentStatus.SUCCESS,
                started_at=started_at,
                finished_at=now_text(),
                latency_ms=elapsed_ms(start),
                warnings=warnings,
            ),
            image_path=image_path,
            model_name=self.model_name,
            confidence_threshold=confidence_threshold,
            cache_hit=cache_hit,
            image_width=dimensions[0],
            image_height=dimensions[1],
            quality_level=quality_level,
            total_detections=cached.get("total", 0),
            class_counts=cached.get("class_counts", {}),
            average_confidence=cached.get("avg_confidence", {}),
            detections=detections,
        )

    def _build_output(
        self,
        raw_items: list[dict[str, Any]],
        image_path: str,
        confidence_threshold: float,
        cache_hit: bool,
        dimensions: tuple[int, int],
    ) -> dict[str, Any]:
        class_counts: dict[str, int] = {}
        confidence_pools: dict[str, list[float]] = {}
        for item in raw_items:
            class_name = item["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidence_pools.setdefault(class_name, []).append(item["conf"])
        return {
            "ok": True,
            "class_counts": class_counts,
            "avg_confidence": {
                name: round(sum(values) / len(values), 4)
                for name, values in confidence_pools.items()
            },
            "total": len(raw_items),
            "raw_items": raw_items,
        }

    def _failed(
        self,
        image_path: str,
        confidence_threshold: float,
        started_at: str,
        start: float,
        errors: list[str],
        dimensions: tuple[int, int] | None,
    ) -> VisionPerceptionResult:
        return VisionPerceptionResult(
            meta=AgentMeta(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                started_at=started_at,
                finished_at=now_text(),
                latency_ms=elapsed_ms(start),
                errors=errors,
            ),
            image_path=image_path,
            model_name=self.model_name,
            confidence_threshold=confidence_threshold,
            cache_hit=False,
            image_width=dimensions[0] if dimensions else None,
            image_height=dimensions[1] if dimensions else None,
            quality_level="failed",
            total_detections=0,
        )

    def _quality_warnings(
        self,
        dimensions: tuple[int, int],
        total: int,
        average_confidence: dict[str, float],
    ) -> list[str]:
        warnings: list[str] = []
        if (
            dimensions[0] < self.thresholds["minimum_recommended_width"]
            or dimensions[1] < self.thresholds["minimum_recommended_height"]
        ):
            warnings.append(f"图像分辨率偏低：{dimensions[0]}x{dimensions[1]}。")
        if total == 0:
            warnings.append("未检测到模型支持的目标。")
        low_classes = [
            name
            for name, value in average_confidence.items()
            if value < self.thresholds["low_confidence"]
        ]
        if low_classes:
            warnings.append(f"低置信度类别：{', '.join(low_classes)}。")
        return warnings

    @staticmethod
    def _image_dimensions(image_path: str) -> tuple[int, int] | None:
        if cv2 is None or not os.path.exists(image_path):
            return None
        image = cv2.imread(image_path)
        if image is None:
            return None
        height, width = image.shape[:2]
        return width, height
