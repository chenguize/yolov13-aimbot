
# controllers/humanize/humanize.py
import random
import time
import math
from typing import Tuple
from config import config


class Humanizer:
    """
    Phase 4: 生物力学拟人化模型 (Bio-Mechanical Simulation)

    核心思想：
    1. 摒弃 math.sin 和 random.gauss，使用平滑随机游走 (Smoothed Random Walk)。
    2. 引入物理动量 (Momentum)，模拟手腕/鼠标的质量惯性，自然产生过冲(Overshoot)。
    3. 基于速度的动态噪声增益 (Velocity-based Gain)。
    """

    def __init__(self):
        self.enable = config.getbool("Humanize", "enable_humanize", False)

        # --- 生物噪声配置 ---
        # 噪声目标游走范围 (像素)
        self.max_noise_amp = config.getfloat("Humanize", "noise_level", 1.5)
        # 噪声平滑因子 (0.0~1.0)，越大变化越慢(越像手腕)，越小越抖(越像帕金森)
        self.noise_smoothness = 0.92

        # --- 物理动量配置 (模拟过冲的核心) ---
        # 惯性系数 (0.0~1.0)。0.0=无惯性，0.9=冰面滑行。
        # 建议 0.4~0.7，这会让准星无法瞬间停下，产生自然的"滑步过冲"
        self.inertia_factor = 0.65

        # --- 内部状态 ---
        self.noise_x = 0.0
        self.noise_y = 0.0
        self.last_out_x = 0.0
        self.last_out_y = 0.0
        self.last_time = time.perf_counter()

    def _update_bio_noise(self):
        """
        生成平滑的生物噪声 (类 Perlin 效果)
        代替原本的高频高斯噪声和机械正弦波
        """
        # 1. 随机选择一个新的“漂移目标”
        target_x = random.uniform(-self.max_noise_amp, self.max_noise_amp)
        target_y = random.uniform(-self.max_noise_amp, self.max_noise_amp)

        # 2. 使用低通滤波 (Lerp) 让当前噪声慢慢向目标移动
        # 这产生了连续的、不规则的波形，符合肌肉微颤特征
        self.noise_x = self.noise_x * self.noise_smoothness + target_x * (1 - self.noise_smoothness)
        self.noise_y = self.noise_y * self.noise_smoothness + target_y * (1 - self.noise_smoothness)

    def apply(self, dx: float, dy: float) -> Tuple[float, float]:
        """
        Input: Controller 计算出的完美修正量 (dx, dy)
        Output: 经过物理模拟后的实际鼠标移动量
        """
        if not self.enable:
            return dx, dy

        current_time = time.perf_counter()
        dt = current_time - self.last_time
        self.last_time = current_time

        # 防止 dt 过大导致积分爆炸 (如卡顿或切屏)
        if dt > 0.1: dt = 0.01

        # 1. 计算输入速度强度 (用于动态调整噪声)
        speed = math.hypot(dx, dy)

        # 2. 只有在有移动意图或保留惯性时才计算
        if speed > 0.1 or (abs(self.last_out_x) > 0.1 or abs(self.last_out_y) > 0.1):

            # --- 步骤 A: 更新生物噪声 ---
            self._update_bio_noise()

            # 动态噪声增益：动得越快，手越抖；慢下来时，抖动减小
            # 限制最大抖动倍率，防止甩枪时飞出天际
            noise_gain = min(1.0, speed / 10.0) + 0.2
            current_noise_x = self.noise_x * noise_gain
            current_noise_y = self.noise_y * noise_gain

            # --- 步骤 B: 物理动量叠加 (最关键的一步) ---
            # 目标输出 = 完美修正量 + 生物噪声
            target_out_x = dx + current_noise_x
            target_out_y = dy + current_noise_y

            # 实际输出 = 上一帧输出 * 惯性 + 目标输出 * (1-惯性)
            # 这就是低通滤波器 (Low Pass Filter)，它会造成两个效果：
            # 1. 延迟 (Delay): 你的准星会比 Controller 的指令慢半拍
            # 2. 过冲 (Overshoot): 当 Controller 停下(dx=0)时，惯性会让你继续滑行一小段，然后被下一帧的 PID 拉回来

            final_x = self.last_out_x * self.inertia_factor + target_out_x * (1 - self.inertia_factor)
            final_y = self.last_out_y * self.inertia_factor + target_out_y * (1 - self.inertia_factor)

            # 更新历史状态
            self.last_out_x = final_x
            self.last_out_y = final_y

            return final_x, final_y

        else:
            # 静止状态，缓慢归零
            self.last_out_x *= 0.5
            self.last_out_y *= 0.5
            return 0.0, 0.0