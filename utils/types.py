# utils/types.py - 类型定义模块
#
# 核心职责：
# 1. 定义项目中使用的数据结构
# 2. 提供类型提示支持
# 3. 标准化数据格式
#

from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np


@dataclass
class FrameInfo:
    """单帧信息 - 包含采集时刻鼠标位置（时间一致性核心）"""
    frame: np.ndarray  # 图像帧数据
    frame_id: int  # 帧唯一标识符
    timestamp: float  # 捕获时间戳
    mouse_pos_at_capture: Tuple[int, int]  # 采集瞬间的屏幕绝对坐标


@dataclass
class Detection:
    """单目标检测结果（后处理后格式）"""
    bbox: Tuple[float, float, float, float]  # 边界框 [x1,y1,x2,y2]
    conf: float  # 置信度
    cls: int  # 类别ID
    screen_x: float  # 屏幕绝对坐标中心X
    screen_y: float  # 屏幕绝对坐标中心Y
    center_rel: Tuple[float, float]  # 256x256 相对中心


@dataclass
class MouseHistoryEntry:
    """鼠标位置历史条目"""
    timestamp: float  # 时间戳
    x: int  # X坐标
    y: int  # Y坐标


class DetectionList(List[Detection]):
    """类型别名 - 检测列表"""
    pass