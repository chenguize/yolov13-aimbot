import threading
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class FrameInfo:
    """
    不可拆分的帧原子单位
    Phase 3 强调：Frame 必须携带 t_cap，否则没有任何意义。
    """
    frame: np.ndarray       # 图像数据 (BGR)
    frame_id: int          # 序列号
    t_cap: float           # 关键：截图完成时的时间戳 (Time Anchor)
    center_pos: Tuple[int, int] # 截图时的屏幕中心坐标

class FrameBus:
    """
    轻量级帧总线
    只保留最新的一帧，丢弃旧帧 (Drop-Oldest)。
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._latest: Optional[FrameInfo] = None

    def publish(self, info: FrameInfo):
        """写入最新帧"""
        with self._lock:
            self._latest = info

    def get_latest(self) -> Optional[FrameInfo]:
        """获取最新帧 (非阻塞)"""
        with self._lock:
            return self._latest