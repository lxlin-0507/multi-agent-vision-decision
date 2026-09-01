from pydantic import ValidationError
import pytest

from schemas import BoundingBox, DetectionItem


def test_detection_confidence_is_validated():
    with pytest.raises(ValidationError):
        DetectionItem(
            class_name="car",
            confidence=1.1,
            bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
        )
