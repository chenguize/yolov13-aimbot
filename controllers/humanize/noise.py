# humanize/noise.py - 随机噪声模块
#
# 核心职责：
# 1. 添加高斯噪声模拟手抖
# 2. 增加移动的随机性
# 3. 使行为更像人类
#
# 架构特点：基于移动幅度的噪声强度
#

import random
from typing import Tuple

from config import config


class Noise:
    """添加高斯噪声 + 周期性抖动，模拟人类手抖"""

    def __init__(self):
        """初始化噪声模块"""
        self.enabled = config.getbool("Humanize", "enable_noise", False)  # 是否启用
        self.level = config.getfloat("Humanize", "noise_level", 0.12)  # 噪声强度

    def apply(self, dx: float, dy: float) -> Tuple[float, float]:
        """
        应用噪声到移动量
        噪声强度与移动幅度成正比
        """
        if not self.enabled:
            return dx, dy  # 如果未启用，直接返回原值

        # 生成高斯噪声，噪声幅度与移动幅度成正比
        noise_x = random.gauss(0, self.level * abs(dx))  # X轴噪声
        noise_y = random.gauss(0, self.level * abs(dy))  # Y轴噪声

        # 将噪声添加到原始移动量
        return dx + noise_x, dy + noise_y