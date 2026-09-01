"""
检测结果缓存模块。
基于图片路径的哈希值缓存 YOLO 检测结果，避免重复推理。
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from config import CACHE_ENABLED, CACHE_DIR
from logger import get_logger

_log = get_logger("cache")


def _cache_dir() -> Path:
    p = Path(CACHE_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _image_hash(image_path: str) -> str:
    """计算图片文件的 SHA256 哈希，作为缓存键。"""
    h = hashlib.sha256()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_key(image_path: str, conf_threshold: float) -> str:
    """生成缓存键：图片哈希 + 阈值。"""
    img_hash = _image_hash(image_path)
    return f"{img_hash}_thr{conf_threshold}"


def get_cached(image_path: str, conf_threshold: float) -> dict[str, Any] | None:
    """读取缓存，命中返回检测结果，未命中返回 None。"""
    if not CACHE_ENABLED:
        return None

    key = _cache_key(image_path, conf_threshold)
    cache_path = _cache_dir() / f"{key}.json"

    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _log.info("缓存命中: %s", image_path)
            return data
        except Exception:
            _log.warning("缓存读取失败，将重新检测: %s", image_path)
            return None

    _log.debug("缓存未命中: %s", image_path)
    return None


def set_cache(image_path: str, conf_threshold: float, result: dict[str, Any]) -> None:
    """写入缓存。"""
    if not CACHE_ENABLED:
        return

    key = _cache_key(image_path, conf_threshold)
    cache_path = _cache_dir() / f"{key}.json"

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        _log.info("缓存已写入: %s", image_path)
    except Exception as e:
        _log.warning("缓存写入失败: %s, 原因: %s", image_path, e)


def clear_cache() -> int:
    """清空所有缓存，返回删除条目数。"""
    d = _cache_dir()
    count = 0
    for f in d.glob("*.json"):
        f.unlink()
        count += 1
    _log.info("缓存已清空，共删除 %d 条", count)
    return count