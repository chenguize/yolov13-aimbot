# perception/bus.py
from typing import Optional, List
from dataclasses import dataclass
import threading
import numpy as np

@dataclass
class FrameInfo:
    frame: np.ndarray
    id: int
    timestamp: float

class FrameBus:
    """轻量级最新帧总线 + 有限历史（防抖/丢帧用）"""

    def __init__(self, max_history: int = 8):
        self._lock = threading.Lock()
        self._latest: Optional[FrameInfo] = None
        self._history: List[FrameInfo] = []
        self.max_history = max_history

    def publish_frame(self, frame: np.ndarray, frame_id: int, timestamp: float):
        info = FrameInfo(frame, frame_id, timestamp)
        with self._lock:
            self._latest = info
            self._history.append(info)
            if len(self._history) > self.max_history:
                self._history.pop(0)

    def get_latest(self) -> Optional[FrameInfo]:
        with self._lock:
            return self._latest

    def get_history(self, count: int = 1) -> List[FrameInfo]:
        with self._lock:
            return self._history[-count:] if count > 0 else []