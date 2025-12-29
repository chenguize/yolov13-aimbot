# controllers/pid_controller.py
from .base_controller import BaseController
from typing import Tuple


class PIDController(BaseController):
    """
    与逐帧 strategy 语义适配的执行器
    不做 PID，不放大，只限幅
    """

    def __init__(self):
        super().__init__()
        self.max_step = 140.0
        self.min_step = 1.0

    def compute(self, intent_dx: float, intent_dy: float, dt: float) -> Tuple[float, float]:
        def clamp(v: float) -> float:
            if abs(v) < self.min_step:
                return 0.0
            if v > self.max_step:
                return self.max_step
            if v < -self.max_step:
                return -self.max_step
            return v

        return clamp(intent_dx), clamp(intent_dy)
