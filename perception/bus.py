# perception/bus.py
import threading
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class FrameInfo:
    """包含单帧及其捕获时刻的状态"""
    frame: np.ndarray
    frame_id: int
    timestamp: float
    mouse_pos_at_capture: Tuple[int, int]

class FrameBus:
    """轻量化帧总线：去除了 FPS 模式下的冗余插值逻辑"""
    def __init__(self):
        self._lock = threading.Lock()
        self._latest: Optional[FrameInfo] = None

    def publish_frame(self, frame, frame_id, timestamp, mouse_pos_at_capture):
        """更新最新帧状态"""
        info = FrameInfo(frame, frame_id, timestamp, mouse_pos_at_capture)
        with self._lock:
            self._latest = info

    def get_latest(self) -> Optional[FrameInfo]:
        """获取最新检测目标"""
        with self._lock:
            return self._latest

    def get_mouse_pos_at_timestamp(self, target_ts: float) -> Tuple[int, int]:
        """
        在 FPS 模式下，直接返回屏幕中心
        保留此接口是为了兼容后续 world_model 的坐标转换逻辑
        """
        # 这里的返回值应从 config 读取或由初始化传入
        from config import config
        return (config.getint("General", "screen_width") // 2,
                config.getint("General", "screen_height") // 2)