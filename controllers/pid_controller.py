# controllers/pid_controller.py - PID控制器
#
# 核心职责：
# 1. 实现完整PID控制算法
# 2. 提供比例、积分、微分项控制
# 3. 包含积分限幅和抗饱和处理
#
# 架构特点：主力控制器，包含积分限幅和抗饱和
#

from .base_controller import BaseController
from typing import Tuple, Dict, Optional, Any


class PIDController(BaseController):
    """
    完整PID控制器实现（主力控制器）
    包含比例、积分、微分项 + 积分限幅
    """

    def __init__(self):
        """初始化PID控制器参数和状态"""
        super().__init__()
        # PID 参数，从配置中读取
        self.kp = self.config.getfloat("Controller", "pid_kp", 2.2)  # 比例系数
        self.ki = self.config.getfloat("Controller", "pid_ki", 0.08)  # 积分系数
        self.kd = self.config.getfloat("Controller", "pid_kd", 0.35)  # 微分系数

        # 积分项状态 - 用于累积误差
        self.integral_x = 0.0
        self.integral_y = 0.0
        # 上一帧误差 - 用于计算微分项
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0

        # 积分限幅 - 防止积分饱和
        self.integral_limit = self.config.getfloat("Controller", "pid_integral_limit", 60.0)

    def compute(
        self,
        target: Optional[Dict[str, Any]],
        current_mouse_pos: Tuple[float, float],
        dt: float
    ) -> Tuple[float, float]:
        """
        计算鼠标移动量 - 使用PID控制算法
        """
        # 如果无目标或时间间隔无效，返回零移动并清空积分
        if not target or dt <= 1e-6:
            # 目标丢失时清空积分，防止持续漂移
            self.integral_x = self.integral_y = 0.0
            return 0.0, 0.0

        # 获取目标屏幕坐标
        tx = target.get("screen_x", self.screen_center[0])
        ty = target.get("screen_y", self.screen_center[1])

        # 计算位置误差
        error_x = tx - current_mouse_pos[0]
        error_y = ty - current_mouse_pos[1]

        # 积分项计算（带抗饱和限幅）
        self.integral_x += error_x * dt  # 累积误差
        self.integral_y += error_y * dt
        # 应用积分限幅，防止积分饱和
        self.integral_x = max(-self.integral_limit, min(self.integral_limit, self.integral_x))
        self.integral_y = max(-self.integral_limit, min(self.integral_limit, self.integral_y))

        # 微分项计算 - 使用差分近似导数
        derivative_x = (error_x - self.prev_error_x) / dt
        derivative_y = (error_y - self.prev_error_y) / dt

        # PID 输出计算 - 比例 + 积分 + 微分
        output_x = self.kp * error_x + self.ki * self.integral_x + self.kd * derivative_x
        output_y = self.kp * error_y + self.ki * self.integral_y + self.kd * derivative_y

        # 更新上一帧误差，用于下一次微分计算
        self.prev_error_x = error_x
        self.prev_error_y = error_y

        return output_x, output_y