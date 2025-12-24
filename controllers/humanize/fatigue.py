# humanize/fatigue.py
import time
from typing import Tuple

from config import config


class Fatigue:
    """疲劳累积 - 连续瞄准时间越长，精度逐渐下降"""

    def __init__(self):
        self.enabled = config.getbool("Humanize", "enable_fatigue", False)
        self.start_time = time.perf_counter()
        self.max_degrade = 0.25  # 最大精度损失 25%

    def apply(self, dx: float, dy: float) -> Tuple[float, float]:
        if not self.enabled:
            return dx, dy

        elapsed = time.perf_counter() - self.start_time
        degrade = min(self.max_degrade, elapsed / 60.0 * self.max_degrade)  # 60秒满疲劳

        return dx * (1 - degrade), dy * (1 - degrade)