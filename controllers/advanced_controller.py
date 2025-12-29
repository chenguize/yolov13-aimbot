# controllers/advanced_controller.py
from .base_controller import BaseController
from typing import Tuple


class ADVANCEDController(BaseController):
    """
    针对 10 帧快速收敛优化的物理控制器。
    """

    def __init__(self):
        super().__init__()

        # --- PID 核心参数 [调试区] ---
        # [Kp] 比例增益。1.05 表示略微超前驱动。
        # 如果 Simple 模式准准，但 PID 模式总是差一点，可微增此值。
        self.kp = 1.05

        # [Ki] 积分增益。用于消除微小残余误差。
        self.ki = 0.02

        # [Kd] 微分增益（物理刹车）。[10帧收敛的关键]
        # 值越大，接近目标时的“阻尼感”越强，防止过冲震荡。
        self.kd = 1.65

        # --- 状态与限制 ---
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0

        # [Output Limit] 解锁单帧位移上限。
        # 要实现 10 帧拉大枪，必须允许单帧移动较大的物理计数。
        self.output_limit = 80.0
        self.integral_limit = 40.0
        self.leak_rate = 0.85

    def compute(self, intent_dx: float, intent_dy: float, dt: float) -> Tuple[float, float]:
        # intent_dx 已经是 strategy 算出的物理 Counts
        error_x = intent_dx
        error_y = intent_dy

        # 1. 积分累加（带泄放，防止老旧误差积压）
        self.integral_x = self.integral_x * self.leak_rate + error_x * dt
        self.integral_y = self.integral_y * self.leak_rate + error_y * dt

        self.integral_x = max(-self.integral_limit, min(self.integral_limit, self.integral_x))
        self.integral_y = max(-self.integral_limit, min(self.integral_limit, self.integral_y))

        # 2. 微分计算（计算误差变化率）
        if dt > 0:
            derivative_x = (error_x - self.prev_error_x) / dt
            derivative_y = (error_y - self.prev_error_y) / dt
        else:
            derivative_x = derivative_y = 0.0

        # 3. PID 物理输出合成
        out_x = (self.kp * error_x) + (self.ki * self.integral_x) + (self.kd * derivative_x)
        out_y = (self.kp * error_y) + (self.ki * self.integral_y) + (self.kd * derivative_y)

        # 4. 更新历史数据
        self.prev_error_x = error_x
        self.prev_error_y = error_y

        # 5. 限幅输出
        final_x = max(-self.output_limit, min(self.output_limit, out_x))
        final_y = max(-self.output_limit, min(self.output_limit, out_y))

        return final_x, final_y