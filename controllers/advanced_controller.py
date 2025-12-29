from .base_controller import BaseController
from typing import Tuple


class ADVANCEDController(BaseController):
    """
    Phase 3: PID 动力学控制器
    通过积分消除误差，通过微分抑制过冲。
    """

    def __init__(self):
        super().__init__()

        # ==========================================================
        # PID 参数调优区 (也可以放入 config.ini)
        # ==========================================================

        # [Kp] 比例项: 基础响应速度
        # Phase 3 建议: 0.5 - 0.8 (配合 WorldModel 的精准预测)
        self.kp = self.config.getfloat("Controller", "pid_kp", 0.65)

        # [Ki] 积分项: 消除微小距离的静差
        # 建议极小，否则会导致准星在目标身上转圈
        self.ki = self.config.getfloat("Controller", "pid_ki", 0.02)

        # [Kd] 微分项: "刹车"力度
        # 当准星快速接近目标时，反向用力防止过冲
        self.kd = self.config.getfloat("Controller", "pid_kd", 0.25)

        # 状态记忆 (用于 I 和 D 计算)
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0

        # 安全限制
        self.max_output = 100.0  # 单帧最大输出 (防止发疯)
        self.max_integral = 20.0  # 积分限幅 (防止积分饱和)

        # 积分泄放系数 (防止历史误差长期积累)
        self.integral_decay = 0.9

    def compute(self, intent_dx: float, intent_dy: float, dt: float) -> Tuple[float, float]:
        """
        PID 计算流程
        intent_dx/dy: 这一帧 Strategy 认为"还需要走多少" (即 Error)
        """
        # 1. 误差即输入
        error_x = intent_dx
        error_y = intent_dy

        # 2. 积分计算 (累加误差，带衰减)
        self.integral_x = (self.integral_x * self.integral_decay) + (error_x * dt)
        self.integral_y = (self.integral_y * self.integral_decay) + (error_y * dt)

        # 积分限幅
        self.integral_x = max(-self.max_integral, min(self.max_integral, self.integral_x))
        self.integral_y = max(-self.max_integral, min(self.max_integral, self.integral_y))

        # 3. 微分计算 (误差变化率 / 速度)
        # 如果 dt 异常(如卡顿)，则跳过微分项
        if dt > 0.001:
            derivative_x = (error_x - self.prev_error_x) / dt
            derivative_y = (error_y - self.prev_error_y) / dt
        else:
            derivative_x = 0.0
            derivative_y = 0.0

        # 4. PID 输出合成
        # u = Kp*e + Ki*∫e + Kd*de/dt
        out_x = (self.kp * error_x) + (self.ki * self.integral_x) + (self.kd * derivative_x)
        out_y = (self.kp * error_y) + (self.ki * self.integral_y) + (self.kd * derivative_y)

        # 5. 更新历史状态
        self.prev_error_x = error_x
        self.prev_error_y = error_y

        # 6. 最终安全限幅
        final_x = max(-self.max_output, min(self.max_output, out_x))
        final_y = max(-self.max_output, min(self.max_output, out_y))

        return final_x, final_y