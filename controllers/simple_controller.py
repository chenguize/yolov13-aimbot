# controllers/simple_controller.py

from .base_controller import BaseController
from typing import Tuple, Dict, Optional, Any


class SimpleController(BaseController):
    def __init__(self):
        super().__init__()
        self.max_step = 500.0
        self.min_step = 0.0

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