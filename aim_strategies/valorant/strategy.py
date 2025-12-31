# aim_strategies/valorant/strategy.py
import math
from typing import Tuple
from config import config

class CalibrationState:
    def __init__(self, init_k_x: float = 1.9, init_k_y: float = 1.2):
        self.k_x = init_k_x
        self.k_y = init_k_y
        self.learning_rate = 0.05
        self.min_k = 0.5
        self.max_k = 20.0 # 稍微放宽上限以适应高灵敏度
        self.enabled = config.getbool("AimStrategy", "enable_auto_calibration", False)

    def update(self, ai_counts: float, real_pixels: float, axis: str):
        if not self.enabled: return

        # [Debug] 强制打印原始输入，看看是不是 RingBuffer 传了 0 过来
        # print(f"[AutoTune Raw] Axis={axis} AI={ai_counts:.1f} Pix={real_pixels:.1f}")

        if abs(real_pixels) < 1.0 or abs(ai_counts) < 5.0:
            return

        observed_k = ai_counts / real_pixels

        # [Debug] 核心诊断：如果这里打印负数，说明 main.py 的正负号需要反转
        print(f"[AutoTune Probe] {axis}: AI({ai_counts:.0f}) / Pix({real_pixels:.1f}) = K_obs({observed_k:.2f})")

        if not (self.min_k < observed_k < self.max_k):
            print(f"[AutoTune] ❌ 过滤异常 K 值: {observed_k:.2f} (不在 {self.min_k}-{self.max_k} 之间)")
            return

        # 更新参数
        if axis == 'x':
            old = self.k_x
            self.k_x = (1 - self.learning_rate) * self.k_x + self.learning_rate * observed_k
            print(f"[AutoTune] ✅ K_x 更新: {old:.2f} -> {self.k_x:.4f}")
        else:
            old = self.k_y
            self.k_y = (1 - self.learning_rate) * self.k_y + self.learning_rate * observed_k
            print(f"[AutoTune] ✅ K_y 更新: {old:.2f} -> {self.k_y:.4f}")

class ValorantStrategy:
    def __init__(self):
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

    def feedback_update(self, h_ai_x, pix_dx, h_ai_y, pix_dy):
        self.calib.update(h_ai_x, pix_dx, 'x')
        self.calib.update(h_ai_y, pix_dy, 'y')