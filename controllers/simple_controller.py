# controllers/simple_controller.py

from .base_controller import BaseController
from typing import Tuple


class SimpleController(BaseController):
    """
    Phase 3: 调试专用控制器 (Pass-through)

    不进行任何 PID 平滑，直接执行 Strategy 计算出的几何全量。
    用于验证 Strategy 的映射参数 (K-Factor) 是否准确。
    如果 SimpleController 能瞬间锁准但 AdvancedController 锁不准，
    说明问题出在 PID 参数上，而不是 Strategy 映射上。
    """

    def __init__(self):
        super().__init__()
        # 调试模式允许较大的单帧爆发，但仍需防疯转
        self.max_step = 800.0
        self.min_step = 0.5  # 稍微放宽死区，过滤极小抖动

    def compute(self, intent_dx: float, intent_dy: float, dt: float) -> Tuple[float, float]:
        """
        Args:
            intent_dx/dy: Strategy 算出的'理论应动量'
            dt: 时间差 (Simple模式下忽略此参数，因为没有时间积分)
        """

        # 内部辅助函数：仅做死区和限幅
        def clamp(v: float) -> float:
            # 1. 死区过滤 (Deadzone)
            if abs(v) < self.min_step:
                return 0.0

            # 2. 安全限幅 (Safety Limit)
            if v > self.max_step:
                return self.max_step
            if v < -self.max_step:
                return -self.max_step

            # 3. 直通输出 (Pass-through)
            return v

        return clamp(intent_dx), clamp(intent_dy)