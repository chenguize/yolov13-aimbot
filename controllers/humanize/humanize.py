# controllers/humanize/humanize.py
import random
import math
import time
from typing import Tuple
from config import config


class Humanizer:
    """
    Phase 3: 拟人化后处理管道
    职责：对 Controller 输出的"完美轨迹"进行污染，使其更像人类操作。
    """

    def __init__(self):
        # 配置读取
        self.enable_humanize = config.getbool("Humanize", "enable_humanize", False)

        # 1. 静态噪声 (高斯分布)
        self.enable_noise = config.getbool("Humanize", "enable_noise", False)
        self.noise_level = config.getfloat("Humanize", "noise_level", 0.2)

        # 2. 动态抖动 (模拟紧张手抖 - 正弦波叠加)
        self.enable_jitter = config.getbool("Humanize", "enable_jitter", False)
        self.jitter_freq = 0.5  # 抖动频率
        self.jitter_amp = 1.5  # 抖动幅度

        # 状态
        self.start_time = time.perf_counter()

    def apply(self, dx: float, dy: float) -> Tuple[float, float]:
        """
        对控制量应用拟人化滤镜
        """
        # 如果总开关没开，直接返回
        if not self.enable_humanize:
            return dx, dy

        out_x, out_y = dx, dy

        # 只有在有显著移动意图时才添加噪声 (防止静止时准星乱跳)
        is_moving = (abs(dx) > 0.1 or abs(dy) > 0.1)

        if is_moving:
            # --- 1. 静态噪声 (随机误差) ---
            if self.enable_noise:
                # noise_level 越大，随机偏差越大
                noise_x = random.gauss(0, self.noise_level)
                noise_y = random.gauss(0, self.noise_level)
                out_x += noise_x
                out_y += noise_y

            # --- 2. 动态抖动 (周期性手抖) ---
            if self.enable_jitter:
                t = time.perf_counter() - self.start_time
                # 抖动幅度与当前移动速度成正比 (动得越快抖得越厉害)
                magnitude = (abs(dx) + abs(dy)) * 0.05 * self.jitter_amp

                jitter_x = math.sin(t * self.jitter_freq * 10) * magnitude
                jitter_y = math.cos(t * self.jitter_freq * 13) * magnitude  # 用不同频率避免画圈

                out_x += jitter_x
                out_y += jitter_y

        return out_x, out_y