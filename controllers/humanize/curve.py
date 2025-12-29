# humanize/curve.py - 鼠标移动曲线整形模块
#
# 核心职责：
# 1. 对鼠标移动轨迹进行曲线整形
# 2. 实现 ease-in-out 效果
# 3. 使移动更自然
#
# 架构特点：使用 smoothstep 函数实现平滑曲线
#

from config import config
from utils.helpers import smoothstep


class Curve:
    """鼠标移动曲线整形 - 使用 smoothstep 做 ease-in-out"""

    def __init__(self):
        """初始化曲线模块"""
        self.enabled = config.getbool("Humanize", "enable_curve", False)  # 是否启用

    def apply(self, t: float) -> float:
        """
        应用曲线整形
        t: 归一化参数 [0, 1]，表示移动完成的比例
        """
        if not self.enabled:
            return t  # 如果未启用，直接返回原值（线性移动）
        # 使用 smoothstep 函数实现平滑曲线
        return smoothstep(0.0, 1.0, t)