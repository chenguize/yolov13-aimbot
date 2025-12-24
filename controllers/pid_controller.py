# controllers/pid_controller.py
from .base_controller import BaseController
from typing import Tuple, Dict, Optional, Any


class PIDController(BaseController):
    """
    完整PID控制器实现（主力控制器）
    包含比例、积分、微分项 + 积分限幅
    """

    def __init__(self):
        super().__init__()
        self.kp = self.config.getfloat("Controller", "pid_kp", 2.2)
        self.ki = self.config.getfloat("Controller", "pid_ki", 0.08)
        self.kd = self.config.getfloat("Controller", "pid_kd", 0.35)

        # 状态
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0

        self.integral_limit = self.config.getfloat("Controller", "pid_integral_limit", 60.0)

    def compute(
        self,
        target: Optional[Dict[str, Any]],
        current_mouse_pos: Tuple[float, float],
        dt: float
    ) -> Tuple[float, float]:
        if not target or dt <= 1e-6:
            # 目标丢失时清空积分，防止持续漂移
            self.integral_x = self.integral_y = 0.0
            return 0.0, 0.0

        tx = target.get("screen_x", self.screen_center[0])
        ty = target.get("screen_y", self.screen_center[1])

        error_x = tx - current_mouse_pos[0]
        error_y = ty - current_mouse_pos[1]

        # 积分（带抗饱和）
        self.integral_x += error_x * dt
        self.integral_y += error_y * dt
        self.integral_x = max(-self.integral_limit, min(self.integral_limit, self.integral_x))
        self.integral_y = max(-self.integral_limit, min(self.integral_limit, self.integral_y))

        # 微分项
        derivative_x = (error_x - self.prev_error_x) / dt
        derivative_y = (error_y - self.prev_error_y) / dt

        # PID 输出
        output_x = self.kp * error_x + self.ki * self.integral_x + self.kd * derivative_x
        output_y = self.kp * error_y + self.ki * self.integral_y + self.kd * derivative_y

        # 更新上一帧误差
        self.prev_error_x = error_x
        self.prev_error_y = error_y

        return output_x, output_y