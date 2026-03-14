import math
import logging
from typing import Tuple
import numpy as np
from config import config

logger = logging.getLogger("AimStrategy")


class CalibrationState:
    def __init__(self, init_k_x: float = 1.9, init_k_y: float = 1.2):
        self.k_x = init_k_x
        self.k_y = init_k_y
        self.learning_rate = 0.05
        self.min_k = 0.5
        self.max_k = 10.0
        self.enabled = config.getbool("AimStrategy", "enable_auto_calibration", False)
        self.change_threshold = 0.05

    def update(self, ai_counts: float, real_pixels: float, axis: str):
        if not self.enabled: return
        if abs(real_pixels) < 1.0 or abs(ai_counts) < 5.0: return
        observed_k = ai_counts / real_pixels
        if observed_k <= 0 or not (self.min_k < observed_k < self.max_k): return
        current_k = self.k_x if axis == 'x' else self.k_y
        if abs(observed_k - current_k) / current_k < self.change_threshold: return

        new_k = (1 - self.learning_rate) * current_k + self.learning_rate * observed_k
        if axis == 'x':
            self.k_x = new_k
        else:
            self.k_y = new_k


class ValorantStrategy:
    """
    Tier S+++ │ 纯物理映射引擎 (UE5 Compatible)
    已剔除所有动态缩放私货，确保正逆变换绝对对称，保证 Kalman Ego-motion 精准。
    """

    def __init__(self):
        kx = config.getfloat("AimStrategy", "k_factor_x", 1.9)
        ky = config.getfloat("AimStrategy", "k_factor_y", 1.2)
        self.calib = CalibrationState(init_k_x=kx, init_k_y=ky)
        self.game_fov = config.getfloat("General", "game_fov", 103.0)
        self.screen_width = config.getfloat("General", "screen_width", 1920)
        self.focal_length = (self.screen_width / 2) / math.tan(math.radians(self.game_fov / 2))

    def apply_fov_distortion(self, dx: float, dy: float) -> Tuple[float, float]:
        angle_x = math.atan(dx / self.focal_length)
        angle_y = math.atan(dy / self.focal_length)
        return angle_x * self.focal_length, angle_y * self.focal_length

    def calculate_mouse_move(self, dx: float, dy: float, bbox_w: float | None = None) -> Tuple[float, float]:
        """位置正变换：像素偏移 → 物理 Count (含 FOV 非线性)"""
        c_dx, c_dy = self.apply_fov_distortion(dx, dy)
        return c_dx * self.calib.k_x, c_dy * self.calib.k_y

    def calculate_velocity_move(self, vx: float, vy: float, bbox_w: float | None = None) -> Tuple[float, float]:
        """速度正变换：px/s → counts/s (线性近似)"""
        return vx * self.calib.k_x, vy * self.calib.k_y

    def reverse_map(self, counts_x: float, counts_y: float, bbox_w: float | None = None) -> Tuple[float, float]:
        """位置逆变换：物理 Count → 屏幕像素"""
        kx = self.calib.k_x if abs(self.calib.k_x) > 0.01 else 1.0
        ky = self.calib.k_y if abs(self.calib.k_y) > 0.01 else 1.0
        corrected_dx = counts_x / kx
        corrected_dy = counts_y / ky
        dx = math.tan(corrected_dx / self.focal_length) * self.focal_length
        dy = math.tan(corrected_dy / self.focal_length) * self.focal_length
        return dx, dy

    def reverse_map_velocity(self, counts_x: float, counts_y: float, bbox_w: float | None = None) -> Tuple[
        float, float]:
        """速度逆变换：counts/s → px/s"""
        kx = self.calib.k_x if abs(self.calib.k_x) > 0.01 else 1.0
        ky = self.calib.k_y if abs(self.calib.k_y) > 0.01 else 1.0
        return counts_x / kx, counts_y / ky

    def feedback_update(self, h_ai_x, pix_dx, h_ai_y, pix_dy):
        self.calib.update(h_ai_x, pix_dx, 'x')
        self.calib.update(h_ai_y, pix_dy, 'y')