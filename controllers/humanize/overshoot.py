# humanize/overshoot.py
from typing import Tuple

from config import config


class Overshoot:
    """过冲回正 - 快速移动后轻微过冲再回拉"""

    def __init__(self):
        self.enabled = config.getbool("Humanize", "enable_overshoot", False)
        self.factor = 0.15  # 过冲比例

    def apply(self, dx: float, dy: float, prev_dx: float, prev_dy: float) -> Tuple[float, float]:
        if not self.enabled:
            return dx, dy

        # 只在快速变化时触发
        if abs(dx) > 10 and abs(dx - prev_dx) > 5:
            return dx * (1 + self.factor), dy * (1 + self.factor)
        return dx, dy