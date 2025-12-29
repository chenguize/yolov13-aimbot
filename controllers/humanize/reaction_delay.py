# humanize/reaction_delay.py - 反应延迟模块
#
# 核心职责：
# 1. 模拟人类反应时间
# 2. 在瞄准前随机延迟一段时间
# 3. 增加行为的自然性
#
# 架构特点：状态化延迟，避免连续触发
#

import random
import time
from config import config


class ReactionDelay:
    """
    反应延迟模拟器
    在每次计算移动前随机延迟一段时间（模拟人类反应）
    """

    def __init__(self):
        """初始化反应延迟模块"""
        self.enabled = config.getbool("Humanize", "enable_reaction_delay", False)  # 是否启用
        self.min_ms = config.getint("Humanize", "reaction_delay_min_ms", 40)  # 最小延迟毫秒
        self.max_ms = config.getint("Humanize", "reaction_delay_max_ms", 120)  # 最大延迟毫秒
        self.last_trigger_time = 0.0  # 上次触发延迟的时间

    def apply_delay(self) -> None:
        """应用反应延迟"""
        if not self.enabled:
            return  # 如果未启用，直接返回

        now = time.perf_counter()
        # 每段连续瞄准只触发一次延迟
        # 防止在连续瞄准过程中不断添加延迟
        if now - self.last_trigger_time > 0.8:  # 超过800ms重新触发
            # 随机选择延迟时间
            delay_sec = random.uniform(self.min_ms / 1000, self.max_ms / 1000)
            time.sleep(delay_sec)  # 执行延迟
            self.last_trigger_time = time.perf_counter()  # 更新触发时间