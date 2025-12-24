# utils/types.py
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np


@dataclass
class FrameInfo:
    """单帧信息 - 包含采集时刻鼠标位置（时间一致性核心）"""
    frame: np.ndarray
    frame_id: int
    timestamp: float
    mouse_pos_at_capture: Tuple[int, int]  # 采集瞬间的屏幕绝对坐标


@dataclass
class Detection:
    """单目标检测结果（后处理后格式）"""
    bbox: Tuple[float, float, float, float]  # x1,y1,x2,y2
    conf: float
    cls: int
    screen_x: float              # 屏幕绝对坐标中心
    screen_y: float
    center_rel: Tuple[float, float]  # 256x256 相对中心


@dataclass
class MouseHistoryEntry:
    """鼠标位置历史条目"""
    timestamp: float
    x: int
    y: int


class DetectionList(List[Detection]):
    """类型别名 - 检测列表"""
    pass