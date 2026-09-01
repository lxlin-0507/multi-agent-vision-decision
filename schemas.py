"""Versioned contracts exchanged by workflow agents."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = "1.0"


class AgentStatus(str, Enum):
    SUCCESS = "success"
    DEGRADED = "degraded"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    agent_name: str
    status: AgentStatus
    started_at: str
    finished_at: str
    latency_ms: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x1: float
    y1: float
    x2: float
    y2: float


class DetectionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_name: str
    confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox | None = None


class VisionPerceptionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: AgentMeta
    image_path: str
    model_name: str
    confidence_threshold: float = Field(ge=0, le=1)
    cache_hit: bool
    image_width: int | None = Field(default=None, ge=1)
    image_height: int | None = Field(default=None, ge=1)
    quality_level: str
    total_detections: int = Field(ge=0)
    class_counts: dict[str, int] = Field(default_factory=dict)
    average_confidence: dict[str, float] = Field(default_factory=dict)
    detections: list[DetectionItem] = Field(default_factory=list)
    annotated_image_path: str | None = None


class SceneAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: AgentMeta
    scene_type: str
    scene_category: str
    density_level: str
    person_count: int = Field(ge=0)
    vehicle_count: int = Field(ge=0)
    animal_count: int = Field(ge=0)
    primary_targets: list[tuple[str, int]] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class QualityReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: AgentMeta
    review_required: bool
    allow_automatic_decision: bool
    quality_level: str
    reasons: list[str] = Field(default_factory=list)


class ScoreFactor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factor_id: str
    factor_name: str
    score: float = Field(ge=0)
    max_score: float = Field(gt=0)
    evidence: list[str] = Field(default_factory=list)
    source_agent: str


class DecisionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: AgentMeta
    total_score: float = Field(ge=0, le=100)
    score_level: str
    decision: str
    review_required: bool
    review_reasons: list[str] = Field(default_factory=list)
    factors: list[ScoreFactor] = Field(default_factory=list)
    summary: str


class AuditEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time: str
    node: str
    action: str
    detail: dict[str, Any]
