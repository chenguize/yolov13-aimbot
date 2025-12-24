# humanize/curve.py
from config import config
from utils.helpers import smoothstep


class Curve:
    """鼠标移动曲线整形 - 使用 smoothstep 做 ease-in-out"""

    def __init__(self):
        self.enabled = config.getbool("Humanize", "enable_curve", False)

    def apply(self, t: float) -> float:
        if not self.enabled:
            return t
        return smoothstep(0.0, 1.0, t)