# controllers/simple_controller.py
from .base_controller import BaseController


class SimpleController(BaseController):
    """最基础的比例控制器"""

    def __init__(self):
        from config import config
        self.sensitivity = config.getfloat("Controller", "sensitivity", 0.42)
        self.center = (
            config.getint("General", "screen_width", 1920) / 2,
            config.getint("General", "screen_height", 1080) / 2
        )

    def compute(self, target: Dict, current_pos: Tuple[float, float], dt: float) -> Tuple[float, float]:
        if not target:
            return 0.0, 0.0
        cx, cy = target["center"]
        error_x = cx - self.center[0]
        error_y = cy - self.center[1]
        return error_x * self.sensitivity, error_y * self.sensitivity