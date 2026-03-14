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
# 🌟 人物急停与运动状态监控器
# ==============================================================================
class MovementTracker:
    def __init__(self):
        # 虚拟键码 (WASD)
        self.W = 0x57
        self.A = 0x41
        self.S = 0x53
        self.D = 0x44

        # 急停物理恢复时间 (Valorant 建议 0.08~0.12 秒，CS2 建议 0.15 秒)
        self.stop_cooldown_duration = 0.10
        self.last_moving_time = 0.0

    def is_accurate_to_shoot(self) -> bool:
        # 检测 WASD 是否有任意一个被按下 (最高位为 1 表示按下)
        is_w = win32api.GetAsyncKeyState(self.W) & 0x8000
        is_a = win32api.GetAsyncKeyState(self.A) & 0x8000
        is_s = win32api.GetAsyncKeyState(self.S) & 0x8000
        is_d = win32api.GetAsyncKeyState(self.D) & 0x8000

        currently_moving = is_w or is_a or is_s or is_d
        current_time = time.perf_counter()

        if currently_moving:
            self.last_moving_time = current_time
            return False

        # 检查松开按键后，是否度过了物理滑行期
        if current_time - self.last_moving_time < self.stop_cooldown_duration:
            return False

        return True


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

        # 人机 Flick 状态检测
        self._prev_human_flicking = False

        # Perlin 噪声的随机初始偏移量
        self.noise_offset_x = random.uniform(0, 1000.0)
        self.noise_offset_y = random.uniform(0, 1000.0)
        self.movement_tracker = MovementTracker()

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

        # 🌟 对齐点 1：目标无效时重置生命周期
        if not self.ctx.is_valid or not self.ctx.p_predict:
            self.target_first_seen_time = 0.0
            self.is_target_in_crosshair = False
            if hasattr(self.controller, 'reset_target_state'):
                self.controller.reset_target_state()
            return

        if self.target_first_seen_time == 0.0:
            self.target_first_seen_time = now
            print(f"[{time.strftime('%H:%M:%S')}] 🎯 发现目标！")
        dx_h_inst, dy_h_inst = self.ring_buffer.get_pure_human_delta_sum(now - dt, now)
        human_vx_inst = dx_h_inst / dt if dt > 0 else 0.0
        human_vy_inst = dy_h_inst / dt if dt > 0 else 0.0

        bbox_w = self.ctx.targets[0].w if self.ctx.targets else 60.0

        # ==============================================================================
        # 视觉坐标映射与漂移
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
        intent_vx, intent_vy = self.aim_strategy.calculate_velocity_move(
            v_real_pixels[0], v_real_pixels[1], bbox_w=bbox_w
        )

        a_real_pixels = getattr(self.ctx, 'a_real', (0.0, 0.0))
        intent_ax, intent_ay = self.aim_strategy.calculate_velocity_move(
            a_real_pixels[0], a_real_pixels[1], bbox_w=bbox_w
        )

        pixel_error_dist = np.linalg.norm([drifted_p_x, drifted_p_y])

        # ==============================================================================
        # 🌟 对齐点 4：将发力空间系数对齐到 800px 二次曲面衰减
        # ==============================================================================
        spatial_factor = np.clip(1.0 - (pixel_error_dist / 800.0) ** 2, 0.1, 1.0)

        time_since_seen = now - self.target_first_seen_time
        reaction_factor = np.clip(time_since_seen / 0.15, 0.0, 1.0)

        # ==============================================================================
        # 🌟 对齐点 3：人机动态离合器 (基于距离自适应的人类优先阈值)
        # ==============================================================================
        dx_h_recent, dy_h_recent = self.ring_buffer.get_pure_human_delta_sum(now - 0.1, now)
        human_speed = math.hypot(dx_h_recent, dy_h_recent) / 0.1

        if pixel_error_dist < 40.0:
            speed_thresh_min = 1200.0
            speed_thresh_max = 2500.0
        elif pixel_error_dist > 150.0:
            speed_thresh_min = 150.0
            speed_thresh_max = 800.0
        else:
            progress = (150.0 - pixel_error_dist) / 110.0
            speed_thresh_min = 150.0 + progress * 1050.0
            speed_thresh_max = 800.0 + progress * 1700.0

        if human_speed > speed_thresh_max:
            human_override_factor = 0.0
        elif human_speed < speed_thresh_min:
            human_override_factor = 1.0
        else:
            human_override_factor = 1.0 - ((human_speed - speed_thresh_min) / (speed_thresh_max - speed_thresh_min))

        power_factor = spatial_factor * reaction_factor * human_override_factor

        # ==============================================================================
        # 🌟 对齐点 2：人类 Flick 结束边缘检测 (通知控制器清空积分)
        # ==============================================================================
        cur_human_flicking = human_speed > (speed_thresh_max * 0.7)
        if self._prev_human_flicking and not cur_human_flicking:
            if hasattr(self.controller, 'notify_flick_end'):
                self.controller.notify_flick_end()
        self._prev_human_flicking = cur_human_flicking

        # 核心 LQR 计算
        self.controller.compute(
            target_x=intent_x, target_y=intent_y, dt=dt,
            human_v=np.array([human_vx_inst, human_vy_inst]),
            v_real=np.array([intent_vx, intent_vy]),
            a_real=np.array([intent_ax, intent_ay]),
            power_factor=power_factor,
            bbox_w=bbox_w
        )

        if self.enable_aimbot:
            self._check_and_trigger()

        self.frames_in_cycle += 1
        self._print_stats()

    def _check_and_trigger(self):
        if self._check_trigger_condition():
            if self.movement_tracker.is_accurate_to_shoot():
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