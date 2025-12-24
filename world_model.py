# world_model.py
import threading
from typing import List, Any, Optional, Dict

class WorldModel:
    """最简世界模型：当前只保留最新有效检测"""

    def __init__(self):
        self._lock = threading.Lock()
        self.current_detections: List[Dict] = []
        self.last_update_time = 0.0
        self.last_frame_id = -1

    def update_detections(self, detections: List[Any], frame_id: int, timestamp: float):
        """最简实现：直接覆盖"""
        with self._lock:
            # 简单过滤：只保留 conf > 0.4 的检测（可调）
            valid = [d for d in detections if d[4] > 0.4]

            self.current_detections = [
                {
                    "bbox": d[:4],
                    "conf": d[4],
                    "cls": d[5],
                    "center": ((d[0]+d[2])/2, (d[1]+d[3])/2)
                }
                for d in valid
            ]

            self.last_update_time = timestamp
            self.last_frame_id = frame_id

    def get_best_target(self) -> Optional[Dict]:
        with self._lock:
            if not self.current_detections:
                return None
            # 最简单策略：离屏幕中心最近的（后续可改 IOU/卡尔曼/多目标等）
            return self.current_detections[0]