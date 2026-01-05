# controllers/pro_controller.py
# Phase 4 仿生弹道规划器 (Turbo Edition / 极速解限版)
# 专为 256x256 小范围截图优化，追求极致的锁敌速度

import math
import time
import random
import numpy as np
from typing import Tuple
from .base_controller import BaseController
from config import config


class PROController(BaseController):
    """
    PROController Turbo: 牺牲部分拟人化特征，换取极致的响应速度
    """

    def __init__(self):
        super().__init__()

        # --- 极速版默认参数 ---
        # 正常人: a=0.12, b=0.05
        # 职业哥: a=0.06, b=0.03
        # 挂:     a=0.02, b=0.01 (几乎瞬移)
        self.fitts_a = config.getfloat("Controller", "fitts_a", 0.02)
        self.fitts_b = config.getfloat("Controller", "fitts_b", 0.01)

        # 震颤幅度减小，因为速度太快了不需要太多抖动掩盖
        self.enable_noise = config.getbool("Controller", "enable_biometric_noise", True)
        self.noise_intensity = 0.02

        self.is_planning = False
        self.start_time = 0.0
        self.duration = 0.0
        self.executed_pos = np.array([0.0, 0.0])
        self.current_overshoot_factor = 1.0

    def compute(self, intent_dx: float, intent_dy: float, dt: float) -> Tuple[float, float]:
        current_time = time.perf_counter()
        dist_remaining = math.hypot(intent_dx, intent_dy)

        # [状态机重置]
        # 增加灵敏度：如果目标剧烈移动（距离变化超过 30px），立即重置规划，重新爆发
        # 原来的 512px 太大了，导致长距离跟枪时一直在这个慢速周期里
        if not self.is_planning or dist_remaining > 512.0 or abs(dist_remaining - self.last_dist) > 30.0:
            self._plan_trajectory(intent_dx, intent_dy)
            self.start_time = current_time
            self.executed_pos = np.array([0.0, 0.0])

        self.last_dist = dist_remaining  # 记录上一帧距离用于检测突变

        elapsed = current_time - self.start_time

        # [核心优化 1: 热启动]
        # 欺骗算法，让它以为已经过了 15% 的时间。
        # 这样起步就是最大速度，跳过了 Min-Jerk 的"墨迹"阶段
        hot_start_offset = self.duration * 0.15
        effective_elapsed = elapsed + hot_start_offset

        if self.duration <= 0.0001:
            tau = 1.0
        else:
            tau = min(effective_elapsed / self.duration, 1.0)

        # [核心优化 2: 爆发曲线]
        # 抛弃 S 曲线 (Slow-Fast-Slow)，改用 Ease-Out (Fast-Slow)
        # s_tau = 1 - (1 - tau)^4
        # 这种曲线起步极快，后段平滑吸附，像磁铁一样
        s_tau = 1.0 - (1.0 - tau) ** 4

        # 动态终点更新
        current_total_goal_x = (self.executed_pos[0] + intent_dx) * self.current_overshoot_factor
        current_total_goal_y = (self.executed_pos[1] + intent_dy) * self.current_overshoot_factor

        ideal_x = current_total_goal_x * s_tau
        ideal_y = current_total_goal_y * s_tau

        step_x = ideal_x - self.executed_pos[0]
        step_y = ideal_y - self.executed_pos[1]

        # 极速模式下只保留极微小的微调抖动
        if self.enable_noise and 0.5 < tau < 0.9:
            nx, ny = self._generate_biometric_noise(tau)
            # 降低抖动权重，避免影响吸附
            step_x += nx * 0.3
            step_y += ny * 0.3

        self.executed_pos[0] += step_x
        self.executed_pos[1] += step_y

        if tau >= 1.0:
            self.is_planning = False
            # 结束时给予 100% 的吸附力，不要 0.8，直接锁死
            return intent_dx, intent_dy

        return self._apply_safety_limits(step_x, step_y)
    def _plan_trajectory(self, dx: float, dy: float):
        distance = math.hypot(dx, dy)

        # [针对 256 截图的优化]
        # 如果距离小于 30px (约屏幕的 1/8)，直接瞬移，不走 Fitts
        if distance < 30.0:
            self.duration = 0.005  # 5ms 极速
            self.current_overshoot_factor = 1.0
            self.is_planning = True
            return

        # Fitts Law 计算
        W = 20.0
        index_of_difficulty = math.log2(2 * distance / W + 1)
        mt = self.fitts_a + self.fitts_b * index_of_difficulty

        # [加速倍率] 强制将所有规划时间缩短 20%
        self.duration = mt * 0.8

        # 远距离依然保留一点点过冲，为了更自然的急停感
        self.current_overshoot_factor = 1.0
        if distance > 100:
            self.current_overshoot_factor = random.uniform(1.01, 1.05)

        self.is_planning = True

    def _generate_biometric_noise(self, tau: float) -> Tuple[float, float]:
        velocity_profile = 30 * (tau ** 2) - 60 * (tau ** 3) + 30 * (tau ** 4)
        freq = random.uniform(10, 15)  # 提高频率
        amp = self.noise_intensity * velocity_profile * 3.0
        t = time.perf_counter()
        nx = math.sin(t * freq * 6.28) * amp
        ny = math.cos(t * freq * 6.28) * amp
        return nx, ny

    def _apply_safety_limits(self, ux: float, uy: float) -> Tuple[float, float]:
        if math.hypot(ux, uy) < 0.5: return 0.0, 0.0

        # [放开限速] 允许单帧移动更多像素
        max_step = config.getfloat("Controller", "max_step", 500.0)
        mag = math.hypot(ux, uy)
        if mag > max_step:
            scale = max_step / mag
            ux *= scale
            uy *= scale
        return ux, uy