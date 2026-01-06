import time
import math
import threading
import random
import logging
from ctypes import windll

from config import config
from utils.types import InferenceContext
from perception.bus import FrameBus
from perception.capture import CaptureThread
from perception.ring_buffer import RingBuffer
from inference import InferenceThread
from world_model import WorldModel
from output import gHub as output_device
from utils.recorder import TraceRecorder

logger = logging.getLogger("Agent")


# MouseWorker 类保持不变 (略去以节省篇幅，请保留原代码)
class MouseWorker(threading.Thread):
    # ... (代码与上一版相同) ...
    def __init__(self, controller, output_dev, shutdown_evt):
        super().__init__(name="MouseWorker", daemon=True)
        self.controller = controller
        self.output = output_dev
        self.shutdown_evt = shutdown_evt

    def run(self):
        # ... (代码与上一版相同) ...
        print("[Agent] 🚀 MouseWorker (1000Hz) Started")
        try:
            windll.winmm.timeBeginPeriod(1)
        except:
            pass
        target_period = 0.001
        while not self.shutdown_evt.is_set():
            loop_start = time.perf_counter()
            try:
                dx, dy = self.controller.tick_mouse()
                if dx != 0 or dy != 0:
                    self.output.mouse_xy(dx, dy)
            except Exception:
                pass
            elapsed = time.perf_counter() - loop_start
            remaining = target_period - elapsed
            if remaining > 0.0005: time.sleep(remaining - 0.0002)
            while (time.perf_counter() - loop_start) < target_period: pass
        try:
            windll.winmm.timeEndPeriod(1)
        except:
            pass


class AIAgent:
    def __init__(self):
        self.shutdown_event = threading.Event()
        self.paused = False
        self.enable_aimbot = True

        # 基础设施
        self.ring_buffer = RingBuffer(max_duration=2.0)
        self.frame_bus = FrameBus()
        output_device.set_ring_buffer(self.ring_buffer)

        # 调试
        self.recorder = TraceRecorder(save_path="debug_trace.pkl")
        if config.getbool("Debug", "enable_trace", False): self.recorder.enable()

        # 核心模块
        self.world_model = WorldModel()
        self.controller = self.world_model.controller

        # 线程
        self.capture_thread = CaptureThread(self.frame_bus, self.shutdown_event)
        self.inference_thread = InferenceThread(self.frame_bus, self.world_model, self.shutdown_event)
        self.mouse_worker = MouseWorker(self.controller, output_device, self.shutdown_event)

        # 上下文
        self.ctx = InferenceContext()
        self.loop_counter = 0
        self.crop_center = self.world_model.crop_center

        # [配置缓存] Trigger 相关参数
        self.enable_trigger = config.getbool("General", "enable_triggerbot", True)
        self.trigger_fov = config.getfloat("Triggerbot", "trigger_fov_x", 3.0)
        self.trigger_conf = config.getfloat("Triggerbot", "trigger_confidence", 0.6)

        # Burst 状态 (从 Controller 移出)
        self.last_shot_time = 0.0

        print("[Agent] Initialized. Ready to start.")

    def start(self):
        self.capture_thread.start()
        self.inference_thread.start()
        self.mouse_worker.start()
        print("[Agent] All threads started.")

    def stop(self):
        self.shutdown_event.set()
        self.recorder.save_to_disk()
        print("[Agent] Shutdown sequence completed.")

    def tick(self):
        if self.paused:
            time.sleep(0.1)
            return

        # Event 驱动等待
        if not self.world_model.wait_for_frame(timeout=0.1):
            return

        loop_start = time.perf_counter()
        self.loop_counter += 1

        try:
            # 1. 感知与决策 (WorldModel)
            self.world_model.step(self.ctx, self.ring_buffer)

            # 2. 行为执行 (Trigger)
            # 只有 Agent 有权决定是否开火
            if self._check_trigger_condition():
                self._perform_shoot()

            # 日志与录制
            if self.loop_counter % 100 == 0:
                self._print_stats(loop_start)
            self.recorder.record_frame(self.ctx)

        except Exception as e:
            logger.error(f"Agent Tick Error: {e}")
            time.sleep(0.01)

    def _check_trigger_condition(self) -> bool:
        """
        [决策逻辑] 判断是否满足开火条件
        完全接管原 Controller 的判定逻辑
        """
        if not (self.enable_aimbot and self.enable_trigger and self.ctx.is_valid):
            return False

        # 1. 置信度检查
        if self.ctx.conf < self.trigger_conf:
            return False

        # 2. FOV 检查 (是否在准星中心)
        tx, ty = self.ctx.p_predict
        dx = abs(tx - self.crop_center)
        dy = abs(ty - self.crop_center)

        # 这是一个很小的矩形区域，通常是 3x3 像素
        if dx > self.trigger_fov or dy > self.trigger_fov:
            return False

        return True

    def _perform_shoot(self):
        """执行射击动作 (含随机延迟)"""
        now = time.perf_counter()

        # 简单的连发限制 (例如至少间隔 150ms)
        if now - self.last_shot_time < 0.15:
            return

        output_device.mouse_down(1)
        time.sleep(random.uniform(0.02, 0.04))  # 模拟按下耗时
        output_device.mouse_up(1)

        self.last_shot_time = now

    def _print_stats(self, loop_start):
        total_ms = (time.perf_counter() - loop_start) * 1000
        lag = getattr(self.ctx, 'dynamic_lag_ms', 0.0)
        print(f"[Agent] FPS: {1000 / total_ms:.1f} | Lat: {lag:.1f}ms | K: {self.world_model.strategy.calib.k_x:.2f}")

    def toggle_pause(self):
        self.paused = not self.paused
        print(f"[System] PAUSED: {self.paused}")

    def toggle_aimbot(self):
        self.enable_aimbot = not self.enable_aimbot
        print(f"[System] AIMBOT: {self.enable_aimbot}")