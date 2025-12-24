# humanize/overshoot.py - 过冲回正模块
#
# 核心职责：
# 1. 模拟快速移动后的过冲现象
# 2. 增加瞄准行为的自然性
# 3. 模拟人类瞄准的微调过程
#
# 架构特点：基于移动变化率的过冲触发
#

from typing import Tuple

from config import config


class Overshoot:
    """过冲回正 - 快速移动后轻微过冲再回拉"""

    def __init__(self):
        """初始化过冲模块"""
        self.enabled = config.getbool("Humanize", "enable_overshoot", False)  # 是否启用
        self.factor = 0.15  # 过冲比例

    def apply(self, dx: float, dy: float, prev_dx: float, prev_dy: float) -> Tuple[float, float]:
        """
        应用过冲效应到移动量
        只在快速变化时触发过冲
        """
        if not self.enabled:
            return dx, dy  # 如果未启用，直接返回原值

        # 只在快速变化时触发过冲
        # 条件：当前移动幅度大于10像素，且与上一帧的差值大于5
        if abs(dx) > 10 and abs(dx - prev_dx) > 5:
            return dx * (1 + self.factor), dy * (1 + self.factor)
        return dx, dy