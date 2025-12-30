# controllers/pro_controller.py
from .base_controller import BaseController
from typing import Tuple
import math


class PROController(BaseController):
    """
    Phase 3 Pro: 变增益 PID 控制器 (Gain Scheduling)

    比普通 PID 更强的地方在于：它像人类一样，
    知道"离得远要快拉，离得近要慢调"。
    """

    def __init__(self):
        super().__init__()

        # --- 基础配置 (配置文件中只需保留一组基准值) ---
        # 这里的基准值建议对应"近距离微调"的参数
        self.base_kp = self.config.getfloat("Controller", "pid_kp", 0.65)
        self.base_ki = self.config.getfloat("Controller", "pid_ki", 0.02)
        self.base_kd = self.config.getfloat("Controller", "pid_kd", 0.25)

        # --- 增益调度曲线配置 ---
        # 当误差超过这个像素距离时，开始增加 Kp
        self.flick_threshold = 50.0
        # 最大激进倍率 (甩枪时 Kp 最多翻几倍)
        self.max_boost = 2.5

        # 状态记忆
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0

        # 积分防饱和
        self.max_integral = 15.0
        self.integral_decay = 0.92

    def _get_scheduled_params(self, error_magnitude: float):
        """
        根据当前误差大小，动态计算 PID 参数
        """
        # 默认使用基准参数 (适合微调)
        kp, ki, kd = self.base_kp, self.base_ki, self.base_kd

        # 如果误差很大，进入"甩枪模式"
        if error_magnitude > 10.0:
            # 计算激进系数：误差越大，系数越高，最高不超过 max_boost
            # 使用对数曲线让过渡更平滑
            boost = 1.0 + math.log10(error_magnitude / 5.0)
            boost = min(boost, self.max_boost)

            # 动态调整：
            # 1. 增加 P (跑得更快)
            kp *= boost
            # 2. 减少 I (大范围移动不需要积分，防止积分饱和过冲)
            ki *= (1.0 / boost)
            # 3. 略微减少 D (减少阻力，让它滑行)
            kd *= 0.8

        return kp, ki, kd

    def compute(self, intent_dx: float, intent_dy: float, dt: float) -> Tuple[float, float]:
        # 1. 计算总误差距离 (欧几里得距离)
        error_dist = math.hypot(intent_dx, intent_dy)

        # 2. 获取当前时刻的最佳 PID 参数
        kp, ki, kd = self._get_scheduled_params(error_dist)

        # --- 标准 PID 计算流程 ---

        # 积分项 (累加)
        self.integral_x = (self.integral_x * self.integral_decay) + (intent_dx * dt)
        self.integral_y = (self.integral_y * self.integral_decay) + (intent_dy * dt)

        # 积分限幅 (防疯转)
        self.integral_x = max(-self.max_integral, min(self.max_integral, self.integral_x))
        self.integral_y = max(-self.max_integral, min(self.max_integral, self.integral_y))

        # 微分项 (变化率)
        if dt > 0.0001:
            derivative_x = (intent_dx - self.prev_error_x) / dt
            derivative_y = (intent_dy - self.prev_error_y) / dt
        else:
            derivative_x = 0.0
            derivative_y = 0.0

        # 合成输出
        out_x = (kp * intent_dx) + (ki * self.integral_x) + (kd * derivative_x)
        out_y = (kp * intent_dy) + (ki * self.integral_y) + (kd * derivative_y)

        # 更新历史
        self.prev_error_x = intent_dx
        self.prev_error_y = intent_dy

        # 硬限幅 (最后一道防线)
        max_out = 150.0  # 稍微放宽一点上限，允许甩枪
        out_x = max(-max_out, min(max_out, out_x))
        out_y = max(-max_out, min(max_out, out_y))

        return out_x, out_y