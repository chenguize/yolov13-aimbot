# aim_strategies/valorant/strategy.py
import math
from typing import Tuple
from config import config

class CalibrationState:
    def __init__(self, init_k_x: float = 1.9, init_k_y: float = 1.2):
        self.k_x = init_k_x
        self.k_y = init_k_y
        self.learning_rate = 0.02  # 保守学习率，过滤手动操作毛刺
        self.min_k = 0.5
        self.max_k = 20.0
        self.enabled = config.getbool("AimStrategy", "enable_auto_calibration", False)

    def update_with_raw_observation(self, total_counts: float, raw_pixel_diff: float, axis: str):
        """
        纯物理闭环：不关心对冲，只关心账本支出与画面产出
        """
        if not self.enabled: return

        # 样本筛选：位移太小（噪声大）或指令太小则不学习
        if abs(raw_pixel_diff) < 2.0 or abs(total_counts) < 5.0:
            return

        # 核心物理公式：K = 账本总指令 / 画面原始位移
        observed_k = abs(total_counts / raw_pixel_diff)

        if not (self.min_k < observed_k < self.max_k):
            return

        # EMA 平滑更新
        if axis == 'x':
            old = self.k_x
            self.k_x = (1 - self.learning_rate) * self.k_x + self.learning_rate * observed_k
            if abs(old - self.k_x) > 0.001:
                print(f"[CausalTune] 🎯 X轴物理结算: 指令({total_counts:.0f}) / 像素({raw_pixel_diff:.1f}) = K:{self.k_x:.4f}")
        else:
            old = self.k_y
            self.k_y = (1 - self.learning_rate) * self.k_y + self.learning_rate * observed_k
            if abs(old - self.k_y) > 0.001:
                print(f"[CausalTune] 🎯 Y轴物理结算: 指令({total_counts:.0f}) / 像素({raw_pixel_diff:.1f}) = K:{self.k_y:.4f}")

class ValorantStrategy:
    def __init__(self):
        kx = config.getfloat("AimStrategy", "k_factor_x", 1.9)
        ky = config.getfloat("AimStrategy", "k_factor_y", 1.2)
        self.calib = CalibrationState(init_k_x=kx, init_k_y=ky)

    def calculate_mouse_move(self, dx: float, dy: float) -> Tuple[int, int]:
        return int(dx * self.calib.k_x), int(dy * self.calib.k_y)

    def reverse_map(self, counts_x: float, counts_y: float) -> Tuple[float, float]:
        return counts_x / self.calib.k_x, counts_y / self.calib.k_y

    def feedback_update(self, total_x, raw_pix_dx, total_y, raw_pix_dy):
        """
        由 WorldModel 传入原始物理观测值进行调参
        """
        self.calib.update_with_raw_observation(total_x, raw_pix_dx, 'x')
        self.calib.update_with_raw_observation(total_y, raw_pix_dy, 'y')