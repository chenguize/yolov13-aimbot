# aim_strategies/valorant/strategy.py
import math
from typing import Tuple
from config import config  # [新增] 导入配置


class CalibrationState:
    def __init__(self, init_k_x: float = 1.9, init_k_y: float = 1.2):
        self.k_x = init_k_x
        self.k_y = init_k_y
        self.learning_rate = 0.05
        self.min_k = 0.5
        self.max_k = 5.0

        # [新增] 读取开关：是否允许修改参数？
        self.enabled = config.getbool("AimStrategy", "enable_auto_calibration", False)

    def update(self, ai_counts: float, real_pixels: float, axis: str):
        # [新增] 安全熔断：如果开关没开，直接拒绝学习
        # 这能防止 PID 控制器的非线性行为污染 K 值
        if not self.enabled:
            return

        if abs(real_pixels) < 1.0 or abs(ai_counts) < 5.0:
            return

        observed_k = ai_counts / real_pixels

        # 异常值过滤
        if not (self.min_k < observed_k < self.max_k):
            return

        # 更新 K 值并打印日志 (方便你看到校准结果)
        if axis == 'x':
            self.k_x = (1 - self.learning_rate) * self.k_x + self.learning_rate * observed_k
            print(f"[AutoTune] New K_x: {self.k_x:.4f} (Obs: {observed_k:.2f})")
        else:
            self.k_y = (1 - self.learning_rate) * self.k_y + self.learning_rate * observed_k
            print(f"[AutoTune] New K_y: {self.k_y:.4f} (Obs: {observed_k:.2f})")


class ValorantStrategy:
    def __init__(self):
        # 从配置读取初始值
        kx = config.getfloat("AimStrategy", "k_factor_x", 1.9)
        ky = config.getfloat("AimStrategy", "k_factor_y", 1.2)
        self.calib = CalibrationState(init_k_x=kx, init_k_y=ky)
        self.last_intent_x = 0
        self.last_intent_y = 0

    def calculate_mouse_move(self, dx: float, dy: float) -> Tuple[int, int]:
        raw_x = dx * self.calib.k_x
        raw_y = dy * self.calib.k_y
        out_x = int(raw_x)
        out_y = int(raw_y)
        self.last_intent_x = out_x
        self.last_intent_y = out_y
        return out_x, out_y

    def reverse_map(self, counts_x: float, counts_y: float) -> Tuple[float, float]:
        kx = self.calib.k_x if abs(self.calib.k_x) > 0.01 else 1.0
        ky = self.calib.k_y if abs(self.calib.k_y) > 0.01 else 1.0
        return counts_x / kx, counts_y / ky

    def feedback_update(self,
                        history_ai_counts_x: int, actual_pixel_dx: float,
                        history_ai_counts_y: int, actual_pixel_dy: float):
        """
        Main Loop 调用的入口
        """
        self.calib.update(history_ai_counts_x, actual_pixel_dx, 'x')
        self.calib.update(history_ai_counts_y, actual_pixel_dy, 'y')