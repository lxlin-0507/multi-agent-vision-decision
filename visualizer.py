"""
检测结果可视化模块。
在图片上绘制 YOLO 检测框、类别标签和置信度。
"""
import os
from pathlib import Path
from typing import Any, Dict, List

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

from config import REPORTS_DIR
from logger import get_logger

_log = get_logger("visualizer")

# 类别颜色映射（常见 COCO 类别）
_COLOR_MAP: Dict[str, tuple] = {
    "person": (0, 255, 0),
    "car": (255, 0, 0),
    "bus": (0, 0, 255),
    "truck": (255, 255, 0),
    "motorcycle": (255, 0, 255),
    "bicycle": (0, 255, 255),
    "bird": (128, 255, 128),
    "cat": (255, 128, 128),
    "dog": (128, 128, 255),
    "horse": (255, 128, 0),
    "sheep": (128, 255, 0),
    "cow": (0, 128, 255),
    "boat": (0, 128, 128),
    "traffic light": (255, 255, 128),
    "stop sign": (128, 255, 255),
}


def _get_color(cls_name: str) -> tuple:
    """获取类别对应的颜色，未知类别随机生成。"""
    if cls_name in _COLOR_MAP:
        return _COLOR_MAP[cls_name]
    # 基于类名 hash 生成稳定颜色
    h = hash(cls_name) % 256
    return (h, (h * 3) % 256, (h * 7) % 256)


def draw_detections(
    image_path: str,
    detections: List[Dict[str, Any]],
    output_dir: str | None = None,
    run_id: str | None = None,
) -> str:
    """
    在图片上绘制检测框并保存。

    Args:
        image_path: 原始图片路径
        detections: raw_items 列表，每项含 class/conf/bbox
        output_dir: 输出目录，默认使用 REPORTS_DIR
        run_id: 运行 ID，用于文件命名

    Returns:
        标注图片的保存路径
    """
    _log.info("开始绘制检测框: %s (%d 个目标)", image_path, len(detections))

    if not _CV2_AVAILABLE:
        _log.warning("opencv-python 未安装，跳过可视化")
        return ""

    img = cv2.imread(image_path)
    if img is None:
        _log.error("无法读取图片: %s", image_path)
        return ""

    h, w = img.shape[:2]

    for det in detections:
        cls_name = det.get("class", "unknown")
        conf = det.get("conf", 0.0)
        bbox = det.get("bbox")

        if bbox is None:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = _get_color(cls_name)

        # 绘制检测框
        thickness = max(2, int(min(w, h) / 300))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # 绘制标签
        label = f"{cls_name} {conf:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        cv2.rectangle(
            img,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w + 4, y1),
            color,
            -1,
        )
        cv2.putText(
            img,
            label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    # 保存
    save_dir = output_dir or REPORTS_DIR
    os.makedirs(save_dir, exist_ok=True)
    prefix = run_id or "annotated"
    save_path = os.path.join(save_dir, f"{prefix}_annotated.jpg")
    cv2.imwrite(save_path, img)

    _log.info("标注图已保存: %s", save_path)
    return save_path