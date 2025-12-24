# humanize/fatigue.py - 疲劳累积模块
#
# 核心职责：
# 1. 模拟长时间瞄准精度下降
# 2. 根据连续瞄准时间降低精度
# 3. 增加行为的自然性
#
# 架构特点：时间依赖的精度衰减
#

import time
from typing import Tuple

from config import config


class Fatigue:
    """疲劳累积 - 连续瞄准时间越长，精度逐渐下降"""

    def __init__(self):
        """初始化疲劳模块"""
        self.enabled = config.getbool("Humanize", "enable_fatigue", False)  # 是否启用
        self.start_time = time.perf_counter()  # 开始时间
        self.max_degrade = 0.25  # 最大精度损失 25%

    def apply(self, dx: float, dy: float) -> Tuple[float, float]:
        """
        应用疲劳效应到移动量
        根据连续瞄准时间降低移动精度
        """
        if not self.enabled:
            return dx, dy  # 如果未启用，直接返回原值

        # 计算连续瞄准时间
        elapsed = time.perf_counter() - self.start_time
        # 计算精度衰减比例，随时间线性增长，但不超过最大值
        degrade = min(self.max_degrade, elapsed / 60.0 * self.max_degrade)  # 60秒满疲劳

        # 应用精度衰减
        return dx * (1 - degrade), dy * (1 - degrade)