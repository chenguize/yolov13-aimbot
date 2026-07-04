# 可配置「只打某几类」— 与 YOLO 导出的 class 列一致，逗号分隔，空=不限制
from __future__ import annotations

from typing import Optional, Set

import numpy as np

from config import config


def get_aim_target_class_set() -> Optional[Set[int]]:
    s = config.getstr("Inference", "aim_target_classes", "").strip()
    if not s:
        return None
    try:
        return {int(x.strip()) for x in s.replace(";", ",").split(",") if x.strip()}
    except ValueError:
        return None


def filter_yolo_boxes(detections: np.ndarray) -> np.ndarray:
    """
    输入 ultralytics 风格 (N,6): x1,y1,x2,y2,conf,cls。
    未配置白名单时原样返回；全被滤掉时返回 (0,6)。
    """
    if detections is None:
        return np.empty((0, 6), dtype=np.float32)
    if len(detections) == 0:
        return detections
    allow = get_aim_target_class_set()
    if allow is None:
        return detections
    cls_col = np.asarray(detections[:, 5], dtype=np.int64)
    keep = np.fromiter((int(c) in allow for c in cls_col), dtype=bool, count=len(cls_col))
    if not np.any(keep):
        return np.empty((0, 6), dtype=detections.dtype)
    out = detections[keep]
    if len(out) > 1:
        order = np.argsort(-out[:, 4])
        out = out[order]
    return out
