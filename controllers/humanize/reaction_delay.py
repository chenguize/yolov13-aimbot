# humanize/reaction_delay.py
import random
import time
from config import config


class ReactionDelay:
    """
    反应延迟模拟器
    在每次计算移动前随机延迟一段时间（模拟人类反应）
    """

    def __init__(self):
        self.enabled = config.getbool("Humanize", "enable_reaction_delay", False)
        self.min_ms = config.getint("Humanize", "reaction_delay_min_ms", 40)
        self.max_ms = config.getint("Humanize", "reaction_delay_max_ms", 120)
        self.last_trigger_time = 0.0

    def apply_delay(self) -> None:
        if not self.enabled:
            return

        now = time.perf_counter()
        # 每段连续瞄准只触发一次延迟
        if now - self.last_trigger_time > 0.8:  # 超过800ms重新触发
            delay_sec = random.uniform(self.min_ms / 1000, self.max_ms / 1000)
            time.sleep(delay_sec)
            self.last_trigger_time = time.perf_counter()