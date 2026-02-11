# agent.py
import time
import threading
import random
import logging
import math
import numpy as np
import win32api
from ctypes import windll

import noise  # [必须安装] pip install noise

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


# ==============================================================================
class HumanMouseListener(threading.Thread):
    def __init__(self, ring_buffer, shutdown_evt):
        super().__init__(name="HumanMouseListener", daemon=True)
        self.ring_buffer = ring_buffer
        self.shutdown_evt = shutdown_evt

    def run(self):
        print("[Agent] 🖐️ Human Mouse Listener Started")
        try:
            last_pos = win32api.GetCursorPos()
        except Exception:
            last_pos = (0, 0)

        while not self.shutdown_evt.is_set():
            try:
                current_pos = win32api.GetCursorPos()
                dx = current_pos[0] - last_pos[0]
                dy = current_pos[1] - last_pos[1]

                if dx != 0 or dy != 0:
                    self.ring_buffer.add_event(dx, dy, is_ai=False)

                last_pos = current_pos
            except Exception:
                pass

            time.sleep(0.001)

        print("[Agent] 🖐️ Human Mouse Listener Stopped")


# ==============================================================================
class MouseWorker(threading.Thread):
    def __init__(self, controller, output_dev, shutdown_evt):
        super().__init__(name="MouseWorker", daemon=True)
        self.controller = controller
        self.output = output_dev
        self.shutdown_evt = shutdown_evt

    def run(self):
        print("[Agent] 🚀 MouseWorker (1000Hz) Started")
        try:
            windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass

        base_period = 0.001

        while not self.shutdown_evt.is_set():
            loop_start = time.perf_counter()

            # [核心反作弊优化：添加 ±100μs 的时间抖动，打破完美的 1000Hz 机器特征]
            jitter = random.gauss(0, 0.0001)
            current_target_period = max(0.0008, min(0.0012, base_period + jitter))

            try:
                dx, dy = self.controller.tick_mouse()
                if dx != 0 or dy != 0:
                    self.output.mouse_xy(dx, dy)
            except Exception:
                pass

            elapsed = time.perf_counter() - loop_start
            remaining = current_target_period - elapsed
            if remaining > 0.0005:
                time.sleep(remaining - 0.0002)

            while (time.perf_counter() - loop_start) < current_target_period:
                pass

        try:
            windll.winmm.timeEndPeriod(1)
        except Exception:
            pass
        print("[Agent] 🚀 MouseWorker Stopped")


# ==============================================================================
class AIAgent:
    def __init__(self):
        self.shutdown_event = threading.Event()
        self.paused = False
        self.enable_aimbot = True

        self.ring_buffer = RingBuffer(max_duration=2.0)
        self.frame_bus = FrameBus()
        output_device.set_ring_buffer(self.ring_buffer)

        self.recorder = TraceRecorder(save_path="debug_trace.pkl")
        if config.getbool("Debug", "enable_trace", False):
            self.recorder.enable()

        self.world_model = WorldModel()
        self.controller = self.world_model.controller
        self.aim_strategy = self.world_model.strategy

        self.cap_to_inf_event = threading.Event()

        self.capture_thread = CaptureThread(self.frame_bus, self.shutdown_event, self.cap_to_inf_event)
        self.inference_thread = InferenceThread(self.frame_bus, self.world_model, self.shutdown_event,
                                                self.cap_to_inf_event)
        self.mouse_worker = MouseWorker(self.controller, output_device, self.shutdown_event)
        self.human_mouse_listener = HumanMouseListener(self.ring_buffer, self.shutdown_event)

        self.ctx = InferenceContext()
        self.crop_center = self.world_model.crop_center

        self.loop_counter = 0
        self.last_stat_time = time.perf_counter()
        self.last_tick_time = time.perf_counter()
        self.frames_in_cycle = 0

        self.enable_trigger = config.getbool("General", "enable_triggerbot", True)
        self.trigger_fov = config.getfloat("Triggerbot", "trigger_fov_x", 3.0)
        self.trigger_conf = config.getfloat("Triggerbot", "trigger_confidence", 0.6)
        self.last_shot_time = 0.0

        self.target_first_seen_time = 0.0
        self.target_in_crosshair_time = 0.0
        self.is_target_in_crosshair = False

        # [核心反作弊优化：Perlin 噪声的随机初始偏移量]
        self.noise_offset_x = random.uniform(0, 1000.0)
        self.noise_offset_y = random.uniform(0, 1000.0)

        print("[Agent] Initialized. Ready to start.")

    def start(self):
        self.capture_thread.start()
        self.inference_thread.start()
        self.mouse_worker.start()
        self.human_mouse_listener.start()
        print("[Agent] All threads started.")

    def stop(self):
        self.shutdown_event.set()
        self.cap_to_inf_event.set()
        self.world_model.frame_ready_event.set()
        time.sleep(0.5)
        self.recorder.save_to_disk()
        print("[Agent] Shutdown sequence completed.")

    def tick(self):
        self.world_model.frame_ready_event.wait()
        self.world_model.frame_ready_event.clear()

        now = time.perf_counter()
        if self.paused:
            self.last_tick_time = now
            return

        self.world_model.step(self.ctx, self.ring_buffer)

        dt = now - self.last_tick_time
        self.last_tick_time = now

        if dt <= 0 or dt > 0.1:
            return

        if not self.ctx.is_valid or not self.ctx.p_predict:
            self.target_first_seen_time = 0.0
            self.is_target_in_crosshair = False
            return

        if self.target_first_seen_time == 0.0:
            self.target_first_seen_time = now

        dx_h, dy_h = self.ring_buffer.get_human_delta_sum(now - dt, now)
        vx = dx_h / dt if dt > 0 else 0.0
        vy = dy_h / dt if dt > 0 else 0.0

        bbox_w = self.ctx.targets[0].w if self.ctx.targets else None

        # ==============================================================================
        # [核心反作弊优化：Perlin 噪声替代固定正弦波] 绝对无规律的连续漂移
        # ==============================================================================
        noise_scale = 0.5
        amplitude = 3.0
        drift_x = noise.pnoise1(now * noise_scale + self.noise_offset_x) * amplitude
        drift_y = noise.pnoise1(now * noise_scale + self.noise_offset_y) * amplitude

        drifted_p_x = self.ctx.p_predict[0] + drift_x
        drifted_p_y = self.ctx.p_predict[1] + drift_y

        intent_x, intent_y = self.aim_strategy.calculate_mouse_move(
            drifted_p_x, drifted_p_y, bbox_w=bbox_w
        )

        v_real_pixels = self.ctx.v_real
        intent_vx, intent_vy = self.aim_strategy.calculate_mouse_move(
            v_real_pixels[0], v_real_pixels[1], bbox_w=bbox_w
        )

        pixel_error_dist = np.linalg.norm([self.ctx.p_predict[0], self.ctx.p_predict[1]])
        spatial_factor = np.clip(1.0 - (pixel_error_dist / 128.0) ** 2, 0.1, 1.0)
        time_since_seen = now - self.target_first_seen_time
        reaction_factor = np.clip(time_since_seen / 0.15, 0.0, 1.0)

        power_factor = spatial_factor * reaction_factor

        self.controller.compute(
            target_x=intent_x, target_y=intent_y, dt=dt,
            human_v=np.array([vx, vy]), v_real=np.array([intent_vx, intent_vy]),
            power_factor=power_factor
        )

        if self.enable_aimbot:
            self._check_and_trigger()

        self.frames_in_cycle += 1
        self._print_stats()

    def _check_and_trigger(self):
        if self._check_trigger_condition():
            self._perform_shoot()

    def _check_trigger_condition(self) -> bool:
        if not (self.enable_aimbot and self.enable_trigger and self.ctx.is_valid):
            self.is_target_in_crosshair = False
            return False

        if self.ctx.conf < self.trigger_conf:
            self.is_target_in_crosshair = False
            return False

        tx, ty = self.ctx.p_predict
        dx = abs(tx)
        dy = abs(ty)

        if dx <= self.trigger_fov and dy <= self.trigger_fov:
            if not self.is_target_in_crosshair:
                self.is_target_in_crosshair = True
                self.target_in_crosshair_time = time.perf_counter()
            return True
        else:
            self.is_target_in_crosshair = False
            return False

    def _perform_shoot(self):
        now = time.perf_counter()

        dx_sum, dy_sum = self.ring_buffer.get_human_delta_sum(now - 0.2, now)
        human_movement_dist = math.hypot(dx_sum, dy_sum)

        # ==============================================================================
        # [核心反作弊优化：Gamma分布生理学延迟] 拥有真实的“拖沓长尾”
        # ==============================================================================
        # Gamma分布：shape=8.0, scale=0.0125 -> 均值约 100ms，最高允许到350ms大分心
        raw_reaction = np.random.gamma(shape=8.0, scale=0.0125)
        base_delay = np.clip(raw_reaction, 0.05, 0.35)

        activity_factor = min(human_movement_dist / 80.0, 1.0)
        dynamic_delay = base_delay * (1.0 - activity_factor)

        if now - self.target_in_crosshair_time < dynamic_delay:
            return

        if now - self.last_shot_time < 0.15:
            return

        output_device.mouse_down(1)

        raw_click_duration = random.gauss(0.03, 0.005)
        click_duration = max(0.015, min(0.06, raw_click_duration))

        time.sleep(click_duration)
        output_device.mouse_up(1)

        self.last_shot_time = now

    def _print_stats(self):
        now = time.perf_counter()
        dt = now - self.last_stat_time
        if dt <= 0 or dt < 1.0:
            return
        real_fps = self.frames_in_cycle / dt
        lag = getattr(self.ctx, 'dynamic_lag_ms', 0.0)
        conf = self.ctx.conf
        print(f"[Agent] FPS: {real_fps:.1f} | Lat: {lag:.1f}ms | Conf: {conf:.2f}")
        self.last_stat_time = now
        self.frames_in_cycle = 0

    def toggle_pause(self):
        self.paused = not self.paused
        print(f"[System] PAUSED: {self.paused}")

    def toggle_aimbot(self):
        self.enable_aimbot = not self.enable_aimbot
        if not self.enable_aimbot:
            self.controller.crosshair_velocity[:] = 0
            self.controller._subpixel[:] = 0
        print(f"[System] AIMBOT: {self.enable_aimbot}")