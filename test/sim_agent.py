# sim_agent.py
"""
仿真引擎 —— 管道延迟模拟 + WorldModel/Controller 驱动 + 场景委托。

架构：
  SimAIAgent (引擎)       MoonlightLatencyModel (网络)
       │                        │
       ├─ scenario.spawn()      ├─ sample()
       ├─ scenario.tick_physics()
       ├─ scenario.tick_human_input()
       └─ WorldModel / RingBuffer / Controller

场景 (test/scenarios/) 只定义「目标怎么动、人类怎么甩、怎么判击杀」，
引擎负责「延迟怎么变、AI 怎么算、鼠标怎么发」。

仿真管道延迟（串流 / 推理 / CV 噪声）默认由 config.ini [Test] 控制；
实战 agent 不读 [Test]。
"""

import time
import threading
import numpy as np
import random
import math
from collections import deque
from config import config
from utils import runtime_defaults as _rtd
from utils.human_intent import HumanIntentTracker

try:
    import noise
except ImportError:
    noise = None

from perception.ring_buffer import RingBuffer
from utils.types import Detection, InferenceContext
from test.scenarios.base import BaseScenario
from test.scenarios.ball_tracking import BallTrackingScenario


# ==============================================================================
#  Moonlight 串流延迟模型
# ==============================================================================
class MoonlightLatencyModel:
    """
    模拟真实 Moonlight 串流的延迟特征：
      - 基础帧延迟 (编码 + 网络传输)：10~20ms
      - 随机抖动 (Jitter)：+/-3~8ms，对数正态分布
      - 偶发毛刺 (Spike)：每隔若干帧一次 25~40ms 突增
      - 帧间平滑：EMA 滤波
    """

    def __init__(
        self,
        base_ms: float = 15.0,
        jitter_std_ms: float = 4.0,
        spike_prob: float = 0.04,
        spike_extra_ms: float = 18.0,
        min_ms: float = 8.0,
        max_ms: float = 45.0,
        ema_alpha: float = 0.25,
    ):
        self.base    = base_ms    / 1000.0
        self.jitter  = jitter_std_ms / 1000.0
        self.spike_p = spike_prob
        self.spike_e = spike_extra_ms / 1000.0
        self.min_lat = min_ms / 1000.0
        self.max_lat = max_ms / 1000.0
        self.alpha   = ema_alpha

        self._ema = self.base
        _var_ratio = (self.jitter / max(self.base, 1e-6)) ** 2
        self._ln_sigma = math.sqrt(math.log(1.0 + _var_ratio)) if _var_ratio > 0 else 0.001
        self._ln_mu    = math.log(max(self.base, 1e-6)) - 0.5 * self._ln_sigma ** 2

    def sample(self) -> float:
        """采样本帧的实际串流延迟 (秒)"""
        raw = np.random.lognormal(self._ln_mu, self._ln_sigma)
        if np.random.random() < self.spike_p:
            raw += self.spike_e * np.random.uniform(0.5, 1.5)
        self._ema = self.alpha * raw + (1.0 - self.alpha) * self._ema
        return float(np.clip(self._ema, self.min_lat, self.max_lat))

    def reset(self):
        self._ema = self.base


# ==============================================================================
#  仿真 AI Agent (引擎层)
# ==============================================================================
class SimAIAgent:
    """
    串流延迟模拟版仿真代理。

    用法:
        scenario = BallTrackingScenario(max_kills=30)
        agent   = SimAIAgent(scenario)
        while not agent.is_done:
            agent.step()
    """

    def __init__(
        self,
        scenario: BaseScenario = None,
        noise_level: float = None,
        dropout_rate: float = None,
        stream_base_ms: float = None,
        stream_jitter_ms: float = None,
        stream_spike_prob: float = None,
        stream_spike_ms: float = None,
        inference_base_ms: float = None,
        inference_std_ms: float = None,
    ):
        self.scenario = scenario or BallTrackingScenario()

        self._test_sim_stream = config.getbool("Test", "simulate_stream_latency", True)
        self._test_sim_infer = config.getbool("Test", "simulate_inference_latency", True)
        self._test_print_stream_stats = config.getbool(
            "Test", "print_stream_latency_stats", True
        )

        if noise_level is None:
            noise_level = config.getfloat("Test", "cv_noise_level", 5.0)
        if dropout_rate is None:
            dropout_rate = config.getfloat("Test", "cv_dropout_rate", 0.08)
        if stream_jitter_ms is None:
            stream_jitter_ms = config.getfloat("Test", "stream_jitter_ms", 4.0)
        if stream_spike_prob is None:
            stream_spike_prob = config.getfloat("Test", "stream_spike_prob", 0.04)
        if stream_spike_ms is None:
            stream_spike_ms = config.getfloat("Test", "stream_spike_extra_ms", 18.0)
        if inference_base_ms is None:
            inference_base_ms = config.getfloat("Test", "inference_base_ms", 6.0)
        if inference_std_ms is None:
            inference_std_ms = config.getfloat("Test", "inference_std_ms", 1.5)

        # ── 串流延迟（仿真）；关 simulate_stream_latency 时不参与采样 ──
        if stream_base_ms is None:
            if self._test_sim_stream:
                base_ov = config.getfloat("Test", "stream_base_ms", 0.0)
                stream_base_ms = (
                    base_ov
                    if base_ov > 0.0
                    else config.getfloat("WorldModel", "moonlight_latency_ms", 15.0)
                )
            else:
                stream_base_ms = 0.0

        self.sim_time = 0.0
        self._original_perf_counter = time.perf_counter
        time.perf_counter = lambda: self.sim_time

        from world_model import WorldModel
        self.world_model = WorldModel()
        self.ring_buffer  = RingBuffer()
        _hib = (config.getstr("General", "human_input_backend", "inputs") or "inputs").strip().lower()
        self._human_intent_subtract_ai = (
            _hib in ("pyn", "pynput", "hook")
            and config.getbool("General", "human_intent_subtract_ai", True)
        )

        self.sim_steps   = 0
        self.noise_level  = noise_level
        self.dropout_rate = dropout_rate
        self.burst_state = 0

        # ── 双延迟模型 ──
        self.stream_latency_model = MoonlightLatencyModel(
            base_ms=stream_base_ms,
            jitter_std_ms=stream_jitter_ms,
            spike_prob=stream_spike_prob,
            spike_extra_ms=stream_spike_ms,
        )
        self._inf_base = inference_base_ms / 1000.0
        self._inf_std  = inference_std_ms  / 1000.0
        self._lat_history: deque = deque(maxlen=500)
        self._lat_print_timer = 0.0

        # ── 准星与场景 ──
        self.crosshair_pos = np.zeros(2, dtype=np.float64)
        self.center = np.array(
            [self.world_model.crop_center, self.world_model.crop_center],
            dtype=np.float64
        )

        self.ctx = InferenceContext()
        self.ctx.center_pos = self.center

        # ── 3D 摄像机 (场景共享) ──
        self.screen_w     = config.getfloat("General", "screen_width",  1920.0)
        self.screen_height = config.getfloat("General", "screen_height", 1080.0)
        self.focal_length = (self.screen_w / 2.0) / math.tan(math.radians(103.0 / 2.0))

        # ── 场景写入属性 ──
        self.enemy_pos = np.zeros(2, dtype=np.float64)
        self.enemy_vel = np.zeros(2, dtype=np.float64)
        self.target_hitbox_x: float = 5.0
        self.target_hitbox_y: float = 5.0
        self.chase_mode: str = "pure_ai"
        self.target_first_seen_time: float = 0.0
        self.kill_count: int = 0
        self._prev_flick_active: bool = False
        self.takeover_release_time: float = 0.0   # 接管场景: 人类松手时刻 (0=未设置/非接管)

        # ── 噪声 ──
        self.noise_offset_x = random.uniform(0, 1000.0)
        self.noise_offset_y = random.uniform(0, 1000.0)

        # ── 意图追踪 ──
        self._intent_tracker = HumanIntentTracker()

        # ── 评分用 ──
        self.last_ai_factor: float = 0.0

        # 初始化场景
        self.scenario.init(self)

    # ==========================================================================
    #  助手
    # ==========================================================================
    def _intent_delta(self, t_start: float, t_end: float):
        return self.ring_buffer.get_intent_delta_sum(
            t_start, t_end, subtract_injected_ai=self._human_intent_subtract_ai
        )

    def _sample_pipeline_delay(self):
        """返回 (stream_latency, inference_delay, total_delay) 秒。"""
        if self._test_sim_stream:
            stream_lat = self.stream_latency_model.sample()
            self._lat_history.append(stream_lat * 1000.0)
        else:
            stream_lat = 0.0

        if self._test_sim_infer:
            inf_delay = max(0.002, np.random.normal(self._inf_base, self._inf_std))
        else:
            inf_delay = 0.002

        total = stream_lat + inf_delay
        return stream_lat, inf_delay, total

    def _print_latency_stats(self):
        if not self._test_print_stream_stats or not self._test_sim_stream:
            return
        if self.sim_time - self._lat_print_timer < 1.0:
            return
        self._lat_print_timer = self.sim_time
        if len(self._lat_history) > 10:
            arr = np.array(self._lat_history)
            print(
                f"[串流延迟] 均值={arr.mean():.1f}ms  "
                f"P50={np.percentile(arr, 50):.1f}ms  "
                f"P95={np.percentile(arr, 95):.1f}ms  "
                f"最大={arr.max():.1f}ms"
            )

    # ==========================================================================
    #  计算机视觉模拟 (CV)
    # ==========================================================================
    def tick_cv(self):
        """带有 3D 尺寸计算的机器视觉。委托场景 chase_mode 获取目标。"""
        if self.scenario.is_done:
            return

        relative_pos = self.enemy_pos - self.crosshair_pos
        dist_px      = np.linalg.norm(relative_pos)

        if dist_px > 1000.0:
            self.ctx.targets = []
            self.target_first_seen_time = 0.0
            return

        if self.burst_state > 0 or np.random.random() < self.dropout_rate:
            self.burst_state = (
                max(0, self.burst_state - 1)
                if self.burst_state > 0
                else np.random.randint(2, 6)
            )
            self.ctx.targets = []
            return

        if self.target_first_seen_time == 0.0:
            self.target_first_seen_time = self.sim_time

        screen_x = relative_pos[0] + self.center[0]
        screen_y = relative_pos[1] + self.center[1]

        # 场景通过 spawn 设置了 target_z；用 focal_length/任意参考 做 bbox
        scale_factor = self.focal_length / 10.0  # 默认 z=10m 参考
        box_w = 0.45 * scale_factor
        box_h = 1.7 * scale_factor

        noise_off = np.random.randn(2) * self.noise_level

        detection = Detection(
            x=screen_x + noise_off[0], y=screen_y + noise_off[1],
            w=box_w, h=box_h, conf=0.95, class_id=0,
            xyxy=np.array([
                screen_x - box_w / 2, screen_y - box_h / 2,
                screen_x + box_w / 2, screen_y + box_h / 2,
            ])
        )
        self.ctx.targets = [detection]

    # ==========================================================================
    #  鼠标输出
    # ==========================================================================
    def tick_mouse(self):
        mx, my = self.world_model.controller.tick_mouse()
        last_delta = getattr(self.world_model.controller, "_last_delta_float", None)
        if last_delta is not None:
            dcx, dcy = last_delta
            px, py = self.world_model.strategy.reverse_map_velocity(
                float(dcx), float(dcy), bbox_w=60.0
            )
        else:
            px, py = self.world_model.strategy.reverse_map_velocity(
                float(mx), float(my), bbox_w=60.0
            )
        self.crosshair_pos[0] += px
        self.crosshair_pos[1] += py
        self.ring_buffer.add_event(mx, my, is_ai=True)

    # ==========================================================================
    #  主步进
    # ==========================================================================
    def step(self):
        dt = 0.002
        self.sim_time += dt
        self.sim_steps += 1

        # ── 管道延迟采样 ──
        stream_lat, inf_delay, total_delay = self._sample_pipeline_delay()
        t_capture = self.sim_time - total_delay

        # ── 委托场景: 物理 + 人类输入 ──
        self.scenario.tick_physics(self, dt)
        human_dx, human_dy = self.scenario.tick_human_input(self, dt)

        # ── 计算机视觉 ──
        self.tick_cv()

        # ── 构造检测 → WorldModel ──
        dets = np.empty((0, 6), dtype=np.float32)
        if self.ctx.targets:
            det = self.ctx.targets[0]
            if det.xyxy is not None:
                dets = np.array(
                    [[det.xyxy[0], det.xyxy[1], det.xyxy[2], det.xyxy[3], det.conf, 0.0]],
                    dtype=np.float32,
                )

        self.world_model.update_detections(
            detections=dets,
            frame_id=self.sim_steps,
            t_capture=t_capture,
            t_done=self.sim_time,
        )
        self.world_model.step(self.ctx, self.ring_buffer)

        # ── 人类输入注入 RingBuffer ──
        if not hasattr(self, '_human_subpixel'):
            self._human_subpixel = np.zeros(2, dtype=np.float64)

        if human_dx or human_dy:
            delta_x = human_dx + self._human_subpixel[0]
            delta_y = human_dy + self._human_subpixel[1]
            mx = int(np.floor(delta_x))
            my = int(np.floor(delta_y))
            self._human_subpixel[0] = delta_x - mx
            self._human_subpixel[1] = delta_y - my
            self.ring_buffer.add_event(mx, my, is_ai=False)

        # 场景通过 _prev_flick_active 控制 flick end 通知
        cur_flick_active = getattr(self, '_prev_flick_active', False)
        prev_flick_active = getattr(self, '_prev_flick_was', False)
        if prev_flick_active and not cur_flick_active:
            if hasattr(self.world_model.controller, 'notify_flick_end'):
                self.world_model.controller.notify_flick_end()
        self._prev_flick_was = cur_flick_active

        # ── AI 联合发力 ──
        if self.ctx.p_predict is not None and self.ctx.v_real is not None:
            dx_h_inst, dy_h_inst = self._intent_delta(self.sim_time - dt, self.sim_time)
            human_vx_inst = dx_h_inst / dt if dt > 0 else 0.0
            human_vy_inst = dy_h_inst / dt if dt > 0 else 0.0

            bbox_w = self.ctx.targets[0].w if self.ctx.targets else 60.0

            # Perlin 漂移噪声
            try:
                if noise is not None:
                    ns = 0.5
                    amp = 3.0
                    drift_x = noise.pnoise1(self.sim_time * ns + self.noise_offset_x) * amp
                    drift_y = noise.pnoise1(self.sim_time * ns + self.noise_offset_y) * amp
                else:
                    drift_x, drift_y = 0.0, 0.0
            except Exception:
                drift_x, drift_y = 0.0, 0.0

            drifted_p_x = self.ctx.p_predict[0] + drift_x
            drifted_p_y = self.ctx.p_predict[1] + drift_y

            intent_x, intent_y = self.world_model.strategy.calculate_mouse_move(
                drifted_p_x, drifted_p_y, bbox_w=bbox_w
            )
            v_real_pixels = self.ctx.v_real
            intent_vx, intent_vy = self.world_model.strategy.calculate_velocity_move(
                v_real_pixels[0], v_real_pixels[1], bbox_w=bbox_w
            )
            a_real_pixels = getattr(self.ctx, 'a_real', (0.0, 0.0))
            intent_ax, intent_ay = self.world_model.strategy.calculate_velocity_move(
                a_real_pixels[0], a_real_pixels[1], bbox_w=bbox_w
            )

            pixel_error_dist = np.linalg.norm([drifted_p_x, drifted_p_y])
            spatial_factor   = np.clip(1.0 - (pixel_error_dist / 800.0) ** 2, 0.1, 1.0)

            time_since_seen = self.sim_time - self.target_first_seen_time
            # 纯 AI 模式：目标"凭空出现"，模拟人类反应延迟 (0→1 ramp over 0.15s)
            # 人机协同：人类已经做了认知决策，AI 即时辅助无需额外延迟
            if self.scenario.chase_mode == 'pure_ai':
                # 纯 AI 评测：默认 0 延迟（测真实管线性能）。
                # 需要拟人反应时间时，在 [Test] 里设 pure_ai_reaction_ramp_sec=0.15
                _ramp = config.getfloat("Test", "pure_ai_reaction_ramp_sec", 0.0)
                if _ramp > 0.0:
                    reaction_factor = float(np.clip(time_since_seen / _ramp, 0.0, 1.0))
                else:
                    reaction_factor = 1.0
            else:
                reaction_factor = 1.0

            ai_weight = self._intent_tracker.update(
                human_vx_inst, human_vy_inst,
                drifted_p_x, drifted_p_y,
                dt,
            )

            power_factor = spatial_factor * reaction_factor * ai_weight

            ctrl = self.world_model.controller
            if hasattr(ctrl, "set_mouse_emit"):
                ctrl.set_mouse_emit(True)
            ctrl.compute(
                target_x=intent_x, target_y=intent_y, dt=dt,
                human_v=np.array([human_vx_inst, human_vy_inst]),
                v_real=np.array([intent_vx, intent_vy]),
                a_real=np.array([intent_ax, intent_ay]),
                power_factor=power_factor,
                bbox_w=bbox_w,
            )
            self.last_ai_factor = power_factor
        else:
            ctrl = self.world_model.controller
            if hasattr(ctrl, "set_mouse_emit"):
                ctrl.set_mouse_emit(False)
            self.last_ai_factor = 0.0

        self.tick_mouse()
        self._print_latency_stats()

    @property
    def is_done(self) -> bool:
        return self.scenario.is_done
