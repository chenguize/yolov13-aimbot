# perception/bus.py - 帧状态总线模块
#
# 核心职责：
# 1. 帧状态管理（存储最新帧信息）
# 2. 时间一致性保证（根据时间戳查找对应鼠标位置）
# 3. 消息传递（连接采集和推理模块）
#
# 架构特点：轻量消息总线（状态驱动核心）
#

import threading
from typing import Optional, Tuple, List
from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass
class FrameInfo:
    """帧信息数据类 - 包含单帧的完整信息"""
    frame: np.ndarray  # 图像帧数据
    frame_id: int  # 帧唯一标识符
    timestamp: float  # 捕获时间戳
    mouse_pos_at_capture: Tuple[int, int]  # 捕获瞬间的鼠标屏幕绝对坐标


class FrameBus:
    """
    帧状态总线 + 鼠标位置环形缓冲
    支持查询任意时间戳对应的鼠标位置（用于时间一致性）
    实现 controller 不等帧的异步架构
    """

    def __init__(self, history_duration: float = 0.2):  # 默认保存 200ms 历史
        """初始化帧总线"""
        self._lock = threading.Lock()  # 线程安全锁
        self._latest: Optional[FrameInfo] = None  # 存储最新帧信息
        # 时间戳 -> mouse_pos 环形缓冲（按时间排序）
        # 用于根据时间戳查找对应的鼠标位置
        self._mouse_history = deque(maxlen=100)  # (timestamp, x, y)
        self.history_duration = history_duration  # 历史保留时长

    def publish_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp: float,
        mouse_pos_at_capture: Tuple[int, int]
    ):
        """
        发布新帧，同时记录采集时刻鼠标位置
        这是实现时间一致性的关键步骤
        """
        info = FrameInfo(frame, frame_id, timestamp, mouse_pos_at_capture)

        with self._lock:
            self._latest = info  # 更新最新帧
            self._mouse_history.append((timestamp, *mouse_pos_at_capture))  # 记录鼠标位置

            # 清理过期数据，保持历史记录在指定时长内
            while self._mouse_history and self._mouse_history[0][0] < timestamp - self.history_duration:
                self._mouse_history.popleft()

    def get_latest(self) -> Optional[FrameInfo]:
        """获取最新帧信息"""
        with self._lock:
            return self._latest

    def get_mouse_pos_at_timestamp(self, target_ts: float) -> Optional[Tuple[int, int]]:
        """
        根据时间戳查找最接近的鼠标位置（线性插值）
        用于解决推理延迟导致的坐标不准问题
        这是时间一致性的核心功能
        """
        with self._lock:
            if not self._mouse_history:
                return None

            # 找到最近的两个点进行线性插值
            prev = None
            for ts, x, y in self._mouse_history:
                if ts >= target_ts:
                    if prev is None:
                        # 如果目标时间正好在第一个点或之前，直接返回该点
                        return x, y
                    # 执行线性插值
                    prev_ts, prev_x, prev_y = prev
                    t = (target_ts - prev_ts) / (ts - prev_ts)  # 插值参数
                    ix = int(prev_x + (x - prev_x) * t)  # 插值后的 x 坐标
                    iy = int(prev_y + (y - prev_y) * t)  # 插值后的 y 坐标
                    return ix, iy
                prev = (ts, x, y)

            # 如果目标时间比最早记录的还早，返回最早位置
            if prev:
                return prev[1], prev[2]
            return None