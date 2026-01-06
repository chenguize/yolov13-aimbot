import math
import logging
from typing import Tuple
from config import config

# 配置 logger
logger = logging.getLogger("AimStrategy")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class CalibrationState:
    def __init__(self, init_k_x: float = 1.9, init_k_y: float = 1.2):
        self.k_x = init_k_x
        self.k_y = init_k_y
        self.learning_rate = 0.05
        self.min_k = 0.5
        self.max_k = 10.0  # 缩紧上限
        self.enabled = config.getbool("AimStrategy", "enable_auto_calibration", False)

        # [Fix] 噪声抑制阈值 (百分比)
        self.change_threshold = 0.05

    def update(self, ai_counts: float, real_pixels: float, axis: str):
        if not self.enabled: return
        if abs(real_pixels) < 1.0 or abs(ai_counts) < 5.0: return

        observed_k = ai_counts / real_pixels

        # [Fix] 符号检查 & 极值过滤
        if observed_k <= 0:
            return  # 忽略反向移动

        if not (self.min_k < observed_k < self.max_k):
            return

        current_k = self.k_x if axis == 'x' else self.k_y

        # [Fix] 死区控制：只有变化超过阈值才更新，避免在最优值附近震荡
        if abs(observed_k - current_k) / current_k < self.change_threshold:
            return

        new_k = (1 - self.learning_rate) * current_k + self.learning_rate * observed_k

        if axis == 'x':
            self.k_x = new_k
            logger.info(f"K_x Updated: {current_k:.3f} -> {self.k_x:.3f} (Obs: {observed_k:.3f})")
        else:
            self.k_y = new_k
            logger.info(f"K_y Updated: {current_k:.3f} -> {self.k_y:.3f} (Obs: {observed_k:.3f})")


class ValorantStrategy:
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
        corrected_dx = angle_x * self.focal_length
        corrected_dy = angle_y * self.focal_length
        return corrected_dx, corrected_dy

    def calculate_mouse_move(self, dx: float, dy: float) -> Tuple[int, int]:
        c_dx, c_dy = self.apply_fov_distortion(dx, dy)
        raw_x = c_dx * self.calib.k_x
        raw_y = c_dy * self.calib.k_y
        return int(raw_x), int(raw_y)

    def reverse_map(self, counts_x: float, counts_y: float) -> Tuple[float, float]:
        # [Fix] 防止除以零或极小值
        kx = self.calib.k_x if abs(self.calib.k_x) > 0.01 else 1.0
        ky = self.calib.k_y if abs(self.calib.k_y) > 0.01 else 1.0
        return counts_x / kx, counts_y / ky

    def feedback_update(self, h_ai_x, pix_dx, h_ai_y, pix_dy):
        self.calib.update(h_ai_x, pix_dx, 'x')
        self.calib.update(h_ai_y, pix_dy, 'y')