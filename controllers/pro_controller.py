# controllers/pro_controller.py
# Phase 4 仿生弹道规划器 (Bio-Trajectory Planner)

import math
import time
import random
import numpy as np
from typing import Tuple, Optional
from .base_controller import BaseController
from config import config


class PROController(BaseController):
    """
    PROController: 基于生物力学的路径规划执行器
    核心：Fitts's Law + Minimum Jerk Curve + Biometric Noise
    """

    def __init__(self):
        super().__init__()

        # --- Fitts 定律参数 (MT = a + b * log2(2D/W + 1)) ---
        # a: 固有反应时间, b: 运动信息速率
        self.fitts_a = config.getfloat("Controller", "fitts_a", 0.12)  # 锁定在人类巅峰约 120ms
        self.fitts_b = config.getfloat("Controller", "fitts_b", 0.05)

        # --- 拟人化噪声参数 (8-12Hz 肌肉颤抖) ---
        self.enable_noise = config.getbool("Controller", "enable_biometric_noise", True)
        self.noise_intensity = 0.05  # 叠加幅度

        # --- 轨迹状态机 ---
        self.is_planning = False
        self.start_time = 0.0
        self.total_duration = 0.0
        self.start_pos = (0.0, 0.0)
        self.target_pos = (0.0, 0.0)

        # 记录已执行的位移累计 (Mickeys)
        self.executed_x = 0.0
        self.executed_y = 0.0

    def compute(self, intent_dx: float, intent_dy: float, dt: float) -> Tuple[float, float]:
        """
        主计算入口：不再追逐 intent，而是根据规划的时间线步进
        """
        current_time = time.perf_counter()

        # 1. 判定是否需要重新规划 (当目标移动偏差过大或新目标出现时)
        # intent_dx/dy 代表当前准星距离目标的实时总差距
        error_dist = math.hypot(intent_dx, intent_dy)

        if not self.is_planning or error_dist > 50.0:
            self._plan_trajectory(intent_dx, intent_dy)
            self.start_time = current_time
            self.executed_x = 0.0
            self.executed_y = 0.0

        # 2. 计算当前轨迹进度 tau (0.0 -> 1.0)
        elapsed = current_time - self.start_time
        tau = elapsed / self.total_duration if self.total_duration > 0 else 1.0

        if tau >= 1.0:
            # 轨迹结束，执行微调或进入空闲
            self.is_planning = False
            return self._apply_safety_limits(intent_dx * 0.2, intent_dy * 0.2)

        # 3. Minimum Jerk 插值计算
        # 公式: s(t) = 10t^3 - 15t^4 + 6t^5
        s_tau = 10 * (tau ** 3) - 15 * (tau ** 4) + 6 * (tau ** 5)

        # 计算当前时刻理论上应达到的累计位移
        ideal_x = self.target_pos[0] * s_tau
        ideal_y = self.target_pos[1] * s_tau

        # 4. 计算当前帧步进量 (当前理想位置 - 已执行位置)
        step_x = ideal_x - self.executed_x
        step_y = ideal_y - self.executed_y

        # 5. 叠加生物震颤噪声 (8-12Hz)
        if self.enable_noise:
            noise_x, noise_y = self._generate_biometric_noise(tau)
            step_x += noise_x
            step_y += noise_y

        # 更新已执行记录
        self.executed_x += step_x
        self.executed_y += step_y

        return self._apply_safety_limits(step_x, step_y)

    def _plan_trajectory(self, dx: float, dy: float):
        """
        根据 Fitts 定律规划运动时长与目标
        """
        distance = math.hypot(dx, dy)
        if distance < 1.0:
            self.total_duration = 0.01
            return

        # 假设目标平均宽度 W=20 像素 (可由 WorldModel 动态下发)
        W = 20.0
        # Fitts 公式: MT = a + b * log2(2D/W + 1)
        self.total_duration = self.fitts_a + self.fitts_b * math.log2(2 * distance / W + 1)

        # 主动注入意图性过冲 (15%~45%)
        # 距离越远，过冲概率和幅度越大
        overshoot_factor = 1.0
        if distance > 100:
            overshoot_factor = 1.0 + random.uniform(0.15, 0.35)

        self.target_pos = (dx * overshoot_factor, dy * overshoot_factor)
        self.is_planning = True

    def _generate_biometric_noise(self, tau: float) -> Tuple[float, float]:
        """
        生成频率在 8-12Hz 的低频肌肉颤抖
        """
        # 颤抖幅度随速度（曲线导数）变化，加速阶段颤抖更明显
        velocity_weight = 30 * (tau ** 2) - 60 * (tau ** 3) + 30 * (tau ** 4)
        freq = random.uniform(8, 12)
        amplitude = self.noise_intensity * velocity_weight

        noise_x = math.sin(time.perf_counter() * freq * 2 * math.pi) * amplitude
        noise_y = math.cos(time.perf_counter() * freq * 2 * math.pi) * amplitude
        return noise_x, noise_y

    def _apply_safety_limits(self, ux: float, uy: float) -> Tuple[float, float]:
        """
        执行输出层的铁律限制
        """
        # Deadzone 死区限制
        if math.hypot(ux, uy) < 0.5:
            return 0.0, 0.0

        # Max Step 单帧硬限幅
        max_step = config.getfloat("Controller", "max_step", 300.0)
        mag = math.hypot(ux, uy)
        if mag > max_step:
            ux = (ux / mag) * max_step
            uy = (uy / mag) * max_step

        return ux, uy