# utils/workers.py
"""
基础设施线程：MovementTracker、MouseWorker、TriggerWorker。

从 agent.py 抽离，与主驱动逻辑解耦，便于独立测试和维护。
"""

import math
import random
import threading
import time

import numpy as np
import win32api
from ctypes import windll

from utils.logger import get_logger

logger = get_logger("Workers")


# ══════════════════════════════════════════════════════════════════════════════
# MovementTracker —— 人物急停状态
# ══════════════════════════════════════════════════════════════════════════════
class MovementTracker:
    """监控 WASD 按键，判断 '已停止位移 + 滑行衰减' 窗口。"""

    def __init__(self):
        self.W, self.A, self.S, self.D = 0x57, 0x41, 0x53, 0x44
        self.stop_cooldown_duration = 0.10
        self.last_moving_time = 0.0

    def is_accurate_to_shoot(self) -> bool:
        keys = (self.W, self.A, self.S, self.D)
        currently_moving = any(win32api.GetAsyncKeyState(k) & 0x8000 for k in keys)
        current_time = time.perf_counter()
        if currently_moving:
            self.last_moving_time = current_time
            return False
        return (current_time - self.last_moving_time) >= self.stop_cooldown_duration


# ══════════════════════════════════════════════════════════════════════════════
# MouseWorker —— 1000Hz 输出循环
# ══════════════════════════════════════════════════════════════════════════════
class MouseWorker(threading.Thread):
    """1kHz 主 tick + 高斯抖动（±100μs 打破机器特征）。"""

    def __init__(self, controller, output_dev, shutdown_evt: threading.Event,
                 ring_buffer=None):
        super().__init__(name="MouseWorker", daemon=True)
        self.controller = controller
        self.output = output_dev
        self.shutdown_evt = shutdown_evt
        self.ring_buffer = ring_buffer

    def run(self):
        logger.info("MouseWorker (1000Hz) starting")
        try:
            windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass

        base_period = 0.001
        while not self.shutdown_evt.is_set():
            loop_start = time.perf_counter()
            jitter = random.gauss(0, 0.0001)
            target_period = max(0.0008, min(0.0012, base_period + jitter))

            try:
                dx, dy = self.controller.tick_mouse()
                if dx or dy:
                    self.output.mouse_xy(dx, dy)
            except Exception as e:
                logger.debug("MouseWorker tick error: %s", e)

            elapsed = time.perf_counter() - loop_start
            remaining = target_period - elapsed
            if remaining > 0.0005:
                time.sleep(remaining - 0.0002)
            while (time.perf_counter() - loop_start) < target_period:
                pass

        try:
            windll.winmm.timeEndPeriod(1)
        except Exception:
            pass
        logger.info("MouseWorker stopped")


# ══════════════════════════════════════════════════════════════════════════════
# TriggerWorker —— 异步射击
# ══════════════════════════════════════════════════════════════════════════════
class TriggerWorker(threading.Thread):
    """
    把 mouse_down/sleep/mouse_up 异步化，避免 sleep 阻塞主 tick。
    调用方只需 fire(click_duration_s)；类内保证 15~60ms 点击时长。
    """

    def __init__(self, output_dev, shutdown_evt: threading.Event):
        super().__init__(name="TriggerWorker", daemon=True)
        self.output = output_dev
        self.shutdown_evt = shutdown_evt
        self._fire_evt = threading.Event()
        self._click_duration = 0.03

    def fire(self, click_duration: float):
        self._click_duration = float(np.clip(click_duration, 0.015, 0.06))
        self._fire_evt.set()

    def run(self):
        logger.info("TriggerWorker starting")
        while not self.shutdown_evt.is_set():
            if not self._fire_evt.wait(timeout=0.1):
                continue
            self._fire_evt.clear()
            try:
                self.output.mouse_down(1)
                time.sleep(self._click_duration)
                self.output.mouse_up(1)
            except Exception as e:
                logger.debug("TriggerWorker fire error: %s", e)
        logger.info("TriggerWorker stopped")
