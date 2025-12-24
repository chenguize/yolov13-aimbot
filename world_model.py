# world_model.py
import threading
import time
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from filterpy.kalman import KalmanFilter
from config import config
from perception.bus import FrameBus


class WorldModel:
    """
    状态中枢 - 坐标转换 / 跟踪 / 预测 / 自运动补偿（Kalman + u_k 反馈）
    """

    def __init__(self, frame_bus: FrameBus):
        self._lock = threading.Lock()
        self.bus = frame_bus
        self.current_detections: List[Dict] = []
        self.last_update_time = 0.0
        self.last_frame_id = -1

        # Kalman 滤波器（简单 2D 位置 + 速度）
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([0., 0., 0., 0.])  # [x, y, vx, vy]
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        self.kf.P *= 1000.
        self.kf.R = np.eye(2) * 5
        self.kf.Q = np.eye(4) * 0.1

        self.last_control_u = np.array([0., 0.])  # 上一帧实际控制量 u_k

    def update_detections(self, detections: List[Any], frame_id: int, timestamp: float):
        with self._lock:
            valid = [d for d in detections if d[4] > config.getfloat("Inference", "conf_threshold", 0.38)]

            new_dets = []
            for d in valid:
                # 关键：使用采集时刻的鼠标位置进行坐标转换
                mouse_pos = self.bus.get_mouse_pos_at_timestamp(timestamp)
                if mouse_pos is None:
                    mouse_x, mouse_y = 960, 540  # 回退到屏幕中心
                else:
                    mouse_x, mouse_y = mouse_pos

                center_x = (d[0] + d[2]) / 2
                center_y = (d[1] + d[3]) / 2

                # 相对坐标 -> 屏幕绝对坐标
                screen_x = mouse_x - 128 + center_x
                screen_y = mouse_y - 128 + center_y

                new_dets.append({
                    "bbox": d[:4],
                    "conf": d[4],
                    "cls": d[5],
                    "screen_x": screen_x,
                    "screen_y": screen_y,
                    "center": (center_x, center_y)
                })

            self.current_detections = new_dets
            self.last_update_time = timestamp
            self.last_frame_id = frame_id

            # 如果有目标，更新 Kalman
            if new_dets:
                best = new_dets[0]
                z = np.array([[best["screen_x"]], [best["screen_y"]]])
                self.kf.predict(u=self.last_control_u.reshape(2, 1))
                self.kf.update(z)

    def receive_control_feedback(self, dx: float, dy: float):
        """接收 output_ghub 实际发送的控制量 u_k"""
        with self._lock:
            self.last_control_u = np.array([dx, dy])

    def get_best_target(self) -> Optional[Dict]:
        with self._lock:
            if not self.current_detections:
                return None

            # 使用 Kalman 滤波后的位置（更稳定）
            kf_pos = self.kf.x[:2].flatten()
            best = self.current_detections[0]
            best["screen_x"] = kf_pos[0]
            best["screen_y"] = kf_pos[1]

            return best