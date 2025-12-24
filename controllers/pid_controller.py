# controllers/pid_controller.py （预留高级 PID，当前实现简单版）
from .base_controller import BaseController
from typing import Tuple


class PIDController(BaseController):
    """PID + 前馈（当前只实现比例部分，高级功能靠开关控制）"""

    def __init__(self):
        from config import config
        self.kp = config.getfloat("Controller", "sensitivity", 0.42) * 2.2  # 粗调
        self.ki = 0.0
        self.kd = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.use_desired_vel = config.getbool("Controller", "use_desired_velocity", False)
        # ... 更多高级参数可后续加

    def compute(self, target: Dict, current_pos: Tuple[float, float], dt: float) -> Tuple[float, float]:
        if not target or dt <= 0:
            return 0.0, 0.0

        cx, cy = target["center"]
        error_x = cx - current_pos[0]
        error_y = cy - current_pos[1]

        # 简单比例（后续可扩展 I/D/前馈/非线性区）
        output_x = error_x * self.kp
        output_y = error_y * self.kp

        # 预留高级模式（目前不生效）
        if self.use_desired_vel:
            # 未来实现期望速度模式
            pass

        return output_x, output_y