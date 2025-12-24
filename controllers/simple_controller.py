# controllers/simple_controller.py
from .base_controller import BaseController
from typing import Tuple, Dict, Optional, Any


class SimpleController(BaseController):
    """
    简单比例控制器 - 直接线性映射误差
    最基础、最可预测的实现，适合调试和低灵敏度场景
    """

    def __init__(self):
        super().__init__()
        self.sensitivity = self.config.getfloat("Controller", "sensitivity", 0.42)

    def compute(
        self,
        target: Optional[Dict[str, Any]],
        current_mouse_pos: Tuple[float, float],
        dt: float
    ) -> Tuple[float, float]:
        if not target:
            return 0.0, 0.0

        tx = target.get("screen_x", self.screen_center[0])
        ty = target.get("screen_y", self.screen_center[1])

        error_x = tx - current_mouse_pos[0]
        error_y = ty - current_mouse_pos[1]

        # 应用死区
        if abs(error_x) <= self.deadzone and abs(error_y) <= self.deadzone:
            return 0.0, 0.0

        # 比例映射
        move_x = error_x * self.sensitivity
        move_y = error_y * self.sensitivity

        return move_x, move_y