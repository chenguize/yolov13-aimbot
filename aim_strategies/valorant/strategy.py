# aim_strategies/valorant/strategy.py
from typing import Tuple, List, Optional, Dict, Any
from config import config
import time
import math


class ValorantStrategy:
    """
    Valorant 专用瞄准策略（2025年底实战版）
    - 映射：高速比例映射 + 死区
    - 预测：最近8帧平均速度 + 自适应衰减
    """

    def __init__(self):
        self.mapping_factor = config.getfloat("Controller", "sensitivity", 5.0)  # 推荐 4.5~6.0
        self.deadzone = config.getfloat("Output", "deadzone_pixels", 1.2)
        self.history_max_len = config.getint("AimStrategy", "prediction_history_frames", 8)
        self.history: List[Tuple[float, float, float]] = []  # (x, y, timestamp)

    def calculate_mouse_move(
        self,
        target_x: float,
        target_y: float,
        center_x: float,
        center_y: float,
        dt: float
    ) -> Tuple[float, float]:
        dx = target_x - center_x
        dy = target_y - center_y

        if abs(dx) < self.deadzone and abs(dy) < self.deadzone:
            return 0.0, 0.0

        move_x = dx * self.mapping_factor
        move_y = dy * self.mapping_factor

        return move_x, move_y

    def calculate_prediction(
        self,
        current_x: float,
        current_y: float,
        dt: float,
        hardware_latency: float = 0.012
    ) -> Tuple[float, float]:
        self.history.append((current_x, current_y, time.perf_counter()))
        if len(self.history) > self.history_max_len:
            self.history.pop(0)

        if len(self.history) < 3:
            return current_x, current_y

        vx_total = vy_total = count = 0.0
        for i in range(1, len(self.history)):
            x2, y2, t2 = self.history[i]
            x1, y1, t1 = self.history[i - 1]
            gap = t2 - t1
            if gap > 1e-6:
                vx_total += (x2 - x1) / gap
                vy_total += (y2 - y1) / gap
                count += 1

        if count == 0:
            return current_x, current_y

        avg_vx = vx_total / count
        avg_vy = vy_total / count

        velocity_mag = math.hypot(avg_vx, avg_vy)
        decay = math.exp(-2.5 * velocity_mag / 1920)  # 屏幕宽度归一化

        pred_time = dt + hardware_latency
        pred_x = current_x + avg_vx * pred_time * decay
        pred_y = current_y + avg_vy * pred_time * decay

        return pred_x, pred_y