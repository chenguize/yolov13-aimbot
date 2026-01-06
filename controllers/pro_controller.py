import time
import threading
import numpy as np
from typing import Tuple
from .base_controller import BaseController


class PROController(BaseController):
    """
    PROController: 纯粹的轨迹规划器 (Motion Planner)
    职责：
    1. 接收目标位移 (intent)
    2. 计算符合生物力学的速度曲线 (Minimum Jerk)
    3. 处理动量融合 (Momentum Blending)

    [变更] 不再包含任何 Trigger/开火判断逻辑
    """

    def __init__(self):
        super().__init__()
        self.snap_duration = 0.08
        self._lock = threading.Lock()

        # 运动状态
        self.start_time = 0.0
        self.is_moving = False

        # 轨迹控制 (float64 高精度)
        self.start_counts = np.array([0.0, 0.0])
        self.target_counts = np.array([0.0, 0.0])
        self.current_counts = np.array([0.0, 0.0])

    def compute(self, intent_dx: float, intent_dy: float, dt: float) -> Tuple[float, float]:
        """
        [主线程调用] 路径规划
        注意：这里的距离判断是为了'动量融合'，属于运动学范畴，而非目标选择范畴。
        """
        if abs(intent_dx) < 1.0 and abs(intent_dy) < 1.0:
            return 0.0, 0.0

        with self._lock:
            now = time.perf_counter()
            # 计算新目标绝对位置
            new_target = self.current_counts + np.array([intent_dx, intent_dy])
            dist_change = np.linalg.norm(new_target - self.target_counts)

            # 物理判定：如果目标位置突变（>5px），说明是新的甩枪，需要重置起点
            # 如果变化很小，说明是跟踪微调，保留当前速度，只更新终点
            if dist_change > 5.0 or not self.is_moving:
                self.start_time = now
                self.start_counts = self.current_counts.copy()
                self.target_counts = new_target
                self.is_moving = True
            else:
                self.target_counts = new_target

        return 0.0, 0.0

    def tick_mouse(self) -> Tuple[int, int]:
        """
        [MouseWorker线程调用] 执行 1ms 微步插值
        """
        if not self.is_moving: return 0, 0

        t = time.perf_counter()

        with self._lock:
            if np.isnan(self.target_counts).any() or np.isnan(self.current_counts).any():
                self.is_moving = False
                return 0, 0

            elapsed = t - self.start_time

            if elapsed >= self.snap_duration:
                self.is_moving = False
                delta = self.target_counts - self.current_counts
                self.current_counts = self.target_counts.copy()
                return int(delta[0]), int(delta[1])

            tau = elapsed / self.snap_duration
            s_val = 10 * (tau ** 3) - 15 * (tau ** 4) + 6 * (tau ** 5)

            ideal_pos = self.start_counts + (self.target_counts - self.start_counts) * s_val
            delta_float = ideal_pos - self.current_counts
            move_x = int(delta_float[0])
            move_y = int(delta_float[1])

            if move_x != 0 or move_y != 0:
                self.current_counts[0] += move_x
                self.current_counts[1] += move_y
                return move_x, move_y

            return 0, 0