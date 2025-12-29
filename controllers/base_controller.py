from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, Any
from config import config
import time  # 导入 time


class BaseController(ABC):
    def __init__(self):
        """初始化基础控制器参数"""
        self.config = config
        self.screen_center = (
            self.config.getint("General", "screen_width", 1920) / 2,
            self.config.getint("General", "screen_height", 1080) / 2
        )
        self.deadzone = self.config.getfloat("Output", "deadzone_pixels", 1.2)

        # 新增：三枪后冷却 + 长时间5s不开枪重置
        self.enable_burst_cooldown = config.getbool("Controller", "enable_burst_cooldown", False)
        self.burst_threshold = config.getint("Controller", "burst_count_threshold", 3)
        self.burst_cooldown_seconds = config.getfloat("Controller", "burst_cooldown_seconds", 0.15)
        self.burst_reset_idle_seconds = 5.0

        self.burst_count = 0                   # 当前连续开枪次数
        self.last_shot_time = 0.0              # 上次开枪时间戳
        self.cooldown_end_time = 0.0           # 当前冷却结束时间

    @abstractmethod
    def compute(
        self,
        intent_dx: float,
        intent_dy: float,
        dt: float
    ) -> Tuple[float, float]:
        pass

    def should_trigger(self, target: Dict, screen_center: Tuple[int, int]) -> bool:
        """
        扳机判定 + 三枪后冷却 + 长时间不开枪重置计数
        """
        if not target:
            return False

        tx = target.get("screen_x")
        ty = target.get("screen_y")
        conf = target.get("conf", 0.0)

        if tx is None or ty is None:
            return False

        fov_x = config.getfloat("Triggerbot", "trigger_fov_x", 3.8)
        fov_y = config.getfloat("Triggerbot", "trigger_fov_y", 3.8)

        dx = abs(tx - screen_center[0])
        dy = abs(ty - screen_center[1])

        is_inside_box = (dx <= fov_x) and (dy <= fov_y)

        print(
            f"[Trigger DEBUG] | dx={dx:.1f}px | dy={dy:.1f}px | box_ok={is_inside_box} | trigger={is_inside_box}")

        if not is_inside_box:
            return False

        current_time = time.perf_counter()

        # 长时间不开枪 → 重置计数器
        if current_time - self.last_shot_time > self.burst_reset_idle_seconds:
            if self.burst_count > 0:
                print(f"[Trigger] 长时间未开枪 ({self.burst_reset_idle_seconds}s)，burst 计数器归零")
            self.burst_count = 0

        # 检查是否在冷却中
        if current_time < self.cooldown_end_time:
            remaining = self.cooldown_end_time - current_time
            print(f"[Trigger] 三枪冷却中，还剩 {remaining:.2f}秒")
            return False

        # 可以开枪 → 更新计数
        self.burst_count += 1
        self.last_shot_time = current_time

        print(f"[Trigger] 开枪计数: {self.burst_count}/{self.burst_threshold}")

        # 达到阈值 → 进入冷却
        if self.burst_count >= self.burst_threshold:
            print(f"[Trigger] 连续开 {self.burst_threshold} 枪，进入冷却 {self.burst_cooldown_seconds}秒")
            self.cooldown_end_time = current_time + self.burst_cooldown_seconds
            self.burst_count = 0  # 冷却后重置计数

        return True