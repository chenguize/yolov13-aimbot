# perception/bus.py
import threading
from typing import Optional, Tuple, List
from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass
class FrameInfo:
    frame: np.ndarray
    frame_id: int
    timestamp: float
    mouse_pos_at_capture: Tuple[int, int]  # 采集瞬间的鼠标位置


class FrameBus:
    """
    帧状态总线 + 鼠标位置环形缓冲
    支持查询任意时间戳对应的鼠标位置（用于时间一致性）
    """

    def __init__(self, history_duration: float = 0.2):  # 默认保存 200ms 历史
        self._lock = threading.Lock()
        self._latest: Optional[FrameInfo] = None
        # 时间戳 -> mouse_pos 环形缓冲（按时间排序）
        self._mouse_history = deque(maxlen=100)  # (timestamp, x, y)
        self.history_duration = history_duration

    def publish_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp: float,
        mouse_pos_at_capture: Tuple[int, int]
    ):
        """发布新帧，同时记录采集时刻鼠标位置"""
        info = FrameInfo(frame, frame_id, timestamp, mouse_pos_at_capture)

        with self._lock:
            self._latest = info
            self._mouse_history.append((timestamp, *mouse_pos_at_capture))

            # 清理过期数据
            while self._mouse_history and self._mouse_history[0][0] < timestamp - self.history_duration:
                self._mouse_history.popleft()

    def get_latest(self) -> Optional[FrameInfo]:
        with self._lock:
            return self._latest

    def get_mouse_pos_at_timestamp(self, target_ts: float) -> Optional[Tuple[int, int]]:
        """
        根据时间戳查找最接近的鼠标位置（线性插值）
        用于解决推理延迟导致的坐标不准问题
        """
        with self._lock:
            if not self._mouse_history:
                return None

            # 找到最近的两个点进行插值
            prev = None
            for ts, x, y in self._mouse_history:
                if ts >= target_ts:
                    if prev is None:
                        return x, y
                    prev_ts, prev_x, prev_y = prev
                    t = (target_ts - prev_ts) / (ts - prev_ts)
                    ix = int(prev_x + (x - prev_x) * t)
                    iy = int(prev_y + (y - prev_y) * t)
                    return ix, iy
                prev = (ts, x, y)

            # 如果目标时间比最早还早，返回最早位置
            if prev:
                return prev[1], prev[2]
            return None