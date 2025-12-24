# humanize/noise.py
import random
from typing import Tuple

from config import config


class Noise:
    """添加高斯噪声 + 周期性抖动，模拟人类手抖"""

    def __init__(self):
        self.enabled = config.getbool("Humanize", "enable_noise", False)
        self.level = config.getfloat("Humanize", "noise_level", 0.12)

    def apply(self, dx: float, dy: float) -> Tuple[float, float]:
        if not self.enabled:
            return dx, dy

        noise_x = random.gauss(0, self.level * abs(dx))
        noise_y = random.gauss(0, self.level * abs(dy))

        return dx + noise_x, dy + noise_y