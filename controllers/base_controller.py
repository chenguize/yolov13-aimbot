from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional
import time
from config import config


class BaseController(ABC):
    def __init__(self):
        """
        初始化基础控制器
        Phase 3: 统一管理屏幕中心与扳机状态
        """
        self.config = config

        # 屏幕参数
        w = self.config.getint("General", "screen_width", 1920)
        h = self.config.getint("General", "screen_height", 1080)
        self.screen_center = (w / 2, h / 2)

        # 死区设置 (Strategy 可能会处理，但 Controller 做最后一道防线)
        self.deadzone = self.config.getfloat("Output", "deadzone_pixels", 1.0)

        # ==========================================================
        # Trigger / Burst Logic (点射控制)
        # ==========================================================
        self.enable_burst = config.getbool("Controller", "enable_burst_cooldown", True)
        self.burst_threshold = config.getint("Controller", "burst_count_threshold", 3)
        self.burst_cooldown = config.getfloat("Controller", "burst_cooldown_seconds", 0.18)
        self.burst_reset_time = 5.0  # 5秒不开枪重置计数

        # 运行时状态
        self.burst_count = 0  # 当前连续开火次数
        self.last_shot_time = 0.0  # 上次开火时间
        self.cooldown_until = 0.0  # 冷却结束时间戳

    @abstractmethod
    def compute(
            self,
            intent_dx: float,
            intent_dy: float,
            dt: float
    ) -> Tuple[float, float]:
        """
        核心计算接口
        Args:
            intent_dx/dy: Strategy 传来的几何意图 (Counts)
            dt: 距离上一帧的时间间隔 (用于积分/微分)
        Returns:
            (out_x, out_y): 最终硬件执行量
        """
        pass

    def should_trigger(self, target: Dict, screen_center: Tuple[int, int]) -> bool:
        """
        扳机判定逻辑：FOV 检查 + 连发冷却限制
        """
        if not target:
            return False

        # 1. 解析目标数据 (兼容 Phase 3 的字典结构)
        tx = target.get("screen_x")
        ty = target.get("screen_y")

        if tx is None or ty is None:
            return False

        # 2. Trigger FOV 检查 (比 Aim FOV 小得多)
        tfov_x = self.config.getfloat("Triggerbot", "trigger_fov_x", 3.0)
        tfov_y = self.config.getfloat("Triggerbot", "trigger_fov_y", 3.0)

        dx = abs(tx - screen_center[0])
        dy = abs(ty - screen_center[1])

        in_fov = (dx <= tfov_x) and (dy <= tfov_y)

        if not in_fov:
            return False

        # 3. Burst 冷却逻辑
        if not self.enable_burst:
            return True  # 如果没开启点射模式，只要在 FOV 内就开枪

        now = time.perf_counter()

        # A. 长期闲置重置
        if now - self.last_shot_time > self.burst_reset_time:
            if self.burst_count > 0:
                # print(f"[Trigger] 闲置重置 Burst: {self.burst_count} -> 0")
                self.burst_count = 0

        # B. 检查是否处于强制冷却期
        if now < self.cooldown_until:
            return False

        # C. 允许开火，更新状态
        self.burst_count += 1
        self.last_shot_time = now

        # D. 达到阈值，触发冷却
        if self.burst_count >= self.burst_threshold:
            self.cooldown_until = now + self.burst_cooldown
            self.burst_count = 0  # 冷却后重置，准备下一轮
            # print(f"[Trigger] 触发冷却 ({self.burst_cooldown}s)")

        return True