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
                    else config.getfloat("Latency", "moonlight_latency_ms", 15.0)
                )
            else:
                stream_base_ms = 0.0

        self.sim_time = 0.0
        self._perf_time = 0.0
        self._original_perf_counter = time.perf_counter
        time.perf_counter = lambda: self._perf_time

        from world_model import WorldModel
        self.world_model = WorldModel()
        self.ring_buffer  = RingBuffer()
        # Synthetic human events are already isolated by RingBuffer's cursor
        # channel. Subtracting the separate AI channel again would create a
        # fictitious negative human input and make the controller fight itself.
        self._human_intent_subtract_ai = False

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
        self.screen_w     = config.getfloat("Hardware", "screen_width",  1920.0)
        self.screen_height = config.getfloat("Hardware", "screen_height", 1080.0)
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

        # ════════════════════════════════════════════════════════════════════════════
        # 模块A 重构新增：α 连续融合 + 接管状态机 + warm_start + conf 滞回 + 1000Hz 鼠标
        # ════════════════════════════════════════════════════════════════════════════
        # ── α 连续融合状态（镜像 agent.py）──
        self._current_chase_mode: str = 'pure_ai'
        self._flick_end_time: float = 0.0          # reaction_factor 计时起点
        # ── 预测性 warm_start 状态（人手速度导数 → 预测放手时刻）──
        self._prev_human_speed_raw: float = 0.0
        self._human_accel_ema: float = 0.0
        self._prev_human_flicking: bool = False
        # ── conf 滞回 + 多帧确认（镜像 agent.py 的锁目标逻辑）──
        self._lock_confirm_count: int = 0
        self._invalid_streak: int = 0
        self._min_aim_conf: float = config.getfloat("Aim", "min_aim_conf", 0.32)
        self._min_aim_conf_drop: float = config.getfloat("Aim", "min_aim_conf_drop", 0.20)
        self._aim_lock_confirm: int = max(1, int(config.getint("Aim", "aim_lock_confirm_count", 3)))
        self._aim_drop_invalid_frames: int = max(1, int(config.getint("Aim", "aim_drop_invalid_frames", 2)))
        # ── 1000Hz 鼠标模拟（子步积分，模拟 MouseWorker 独立线程）──
        self._mouse_tick_rate: float = 1000.0       # Hz
        self._mouse_tick_dt: float = 1.0 / self._mouse_tick_rate
        self._mouse_tick_accumulator: float = 0.0
        self._mouse_subpixel: np.ndarray = np.zeros(2, dtype=np.float64)  # 取整残差累积
        # ── 多目标支持（模块C 场景需要）──
        self._multi_target_mode: bool = False       # 场景可设 True 启用多目标检测
        # ── 检测噪声模型升级（模块B 前置：conf 抖动 + bbox 抖动 + 运动方向偏置）──
        self._conf_jitter_enable: bool = config.getbool("Test", "cv_conf_jitter_enable", True)
        self._conf_base: float = config.getfloat("Test", "cv_conf_base", 0.62)   # 近距基线 conf
        self._conf_jitter_alpha: float = config.getfloat("Test", "cv_conf_jitter_alpha", 3.0)  # Beta 分布参数
        self._conf_jitter_beta: float = config.getfloat("Test", "cv_conf_jitter_beta", 2.0)
        self._conf_ema_alpha: float = config.getfloat("Test", "cv_conf_ema_alpha", 0.15)  # conf EMA 平滑（YOLO conf 帧间漂移而非跳变）
        self._conf_bbox_ref_w: float = config.getfloat("Test", "cv_conf_bbox_ref_w", 60.0)  # bbox 宽度参考值（近距典型），用于 conf 衰减
        # per-target conf EMA（多目标场景下每个目标独立，避免串扰）
        self._conf_ema_map: dict = {}  # key=enemy_index, value=EMA conf
        self._bbox_jitter_enable: bool = config.getbool("Test", "cv_bbox_jitter_enable", True)
        self._bbox_jitter_scale: float = config.getfloat("Test", "cv_bbox_jitter_scale", 0.08)  # 8% 尺寸抖动
        self._motion_bias_enable: bool = config.getbool("Test", "cv_motion_bias_enable", True)
        self._motion_bias_scale: float = config.getfloat("Test", "cv_motion_bias_scale", 3.0)   # 运动方向偏置 px

        # 初始化场景
        self.scenario.init(self)

    # ==========================================================================
    #  助手
    # ==========================================================================
    def _intent_delta(self, t_start: float, t_end: float):
        return self.ring_buffer.get_intent_delta_sum(
            t_start, t_end, subtract_injected_ai=self._human_intent_subtract_ai
        )

    def _target_velocity_counts(self) -> np.ndarray:
        vr = self.ctx.v_real
        if vr is None:
            return np.zeros(2, dtype=np.float64)
        bbox_w = self.ctx.targets[0].w if self.ctx.targets else 60.0
        strategy = self.world_model.strategy
        try:
            vx, vy = strategy.calculate_velocity_move(
                float(vr[0]), float(vr[1]), px_x=0.0, px_y=0.0, bbox_w=bbox_w
            )
        except TypeError:
            vx, vy = strategy.calculate_velocity_move(
                float(vr[0]), float(vr[1]), bbox_w=bbox_w
            )
        return np.array([vx, vy], dtype=np.float64)

    def _publish_target_observation(self, visible: bool, confidence: float = 0.0):
        ctrl = self.world_model.controller
        if hasattr(ctrl, 'observe_target'):
            velocity = self._target_velocity_counts() if visible else None
            ctrl.observe_target(visible, velocity, confidence)

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
    #  计算机视觉模拟 (CV)  —— 模块B 升级：conf 抖动 + bbox 抖动 + 运动方向偏置
    # ==========================================================================
    def tick_cv(self):
        """
        带有 3D 尺寸计算的机器视觉模拟。
        模块B 升级：
          - conf Beta 分布抖动（替代固定 0.95）
          - bbox 尺寸抖动（替代固定 w/h）
          - 运动方向偏置（替代各向同性高斯）
          - 多目标支持（场景可提供 enemy_list）
        """
        if self.scenario.is_done:
            return

        # ── 多目标模式：场景提供 enemy_list ──
        enemy_list = getattr(self, 'enemy_list', None)
        if enemy_list is None:
            enemy_list = [(
                self.enemy_pos,
                getattr(self, 'enemy_vel', np.zeros(2)),
                getattr(self, '_target_z', 10.0),
            )]

        detections = []
        for enemy_idx, enemy in enumerate(enemy_list):
            e_pos, e_vel, e_z = enemy[0], enemy[1], enemy[2]
            relative_pos = e_pos - self.crosshair_pos
            dist_px = np.linalg.norm(relative_pos)

            perception_radius = float(getattr(
                self.scenario, 'perception_radius_px', 1000.0
            ))
            visibility_fn = getattr(self.scenario, 'is_target_visible', None)
            target_visible = (
                bool(visibility_fn(float(dist_px)))
                if callable(visibility_fn) else dist_px <= perception_radius
            )
            if not target_visible:
                continue

            # 丢框（burst + 随机）
            if self.burst_state > 0 or np.random.random() < self.dropout_rate:
                if self.burst_state > 0:
                    self.burst_state = max(0, self.burst_state - 1)
                else:
                    self.burst_state = np.random.randint(2, 6)
                continue

            screen_x = relative_pos[0] + self.center[0]
            screen_y = relative_pos[1] + self.center[1]

            # bbox 尺寸（带抖动）
            scale_factor = self.focal_length / max(e_z, 1.0)
            box_w_base = 0.45 * scale_factor
            box_h_base = 1.7 * scale_factor
            if self._bbox_jitter_enable:
                w_jit = 1.0 + np.random.randn() * self._bbox_jitter_scale
                h_jit = 1.0 + np.random.randn() * self._bbox_jitter_scale
                box_w = max(2.0, box_w_base * w_jit)
                box_h = max(2.0, box_h_base * h_jit)
            else:
                box_w = box_w_base
                box_h = box_h_base

            # 位置噪声（高斯 + 运动方向偏置）
            noise_off = np.random.randn(2) * self.noise_level
            if self._motion_bias_enable and np.linalg.norm(e_vel) > 1.0:
                # YOLO 检测偏向运动方向拖尾：沿速度方向加偏置
                v_dir = e_vel / (np.linalg.norm(e_vel) + 1e-9)
                noise_off += v_dir * np.random.uniform(0, self._motion_bias_scale)

            # conf 抖动（Beta 分布 + bbox 尺寸衰减 + per-target EMA 平滑）
            # 实战中 YOLO 不输出距离，但 bbox 越大 = 目标越近 = conf 越高
            # （透视投影：近距目标 bbox 占比大，特征清晰，conf 自然更高）
            if self._conf_jitter_enable:
                # bbox 尺寸衰减：box_w >= ref_w → factor=1.0（近距高 conf）
                #                box_w → 0     → factor=0.3（远距低 conf）
                bbox_factor = float(np.clip(box_w / self._conf_bbox_ref_w, 0.3, 1.0))
                conf_base_eff = self._conf_base * bbox_factor
                conf_raw = float(np.clip(
                    np.random.beta(self._conf_jitter_alpha, self._conf_jitter_beta)
                    * 0.35 + conf_base_eff, 0.1, 0.95
                ))
                # per-target EMA 平滑：YOLO conf 帧间漂移而非跳变
                # 多目标场景下每个目标独立 EMA，避免串扰
                prev_ema = self._conf_ema_map.get(enemy_idx, 0.0)
                if prev_ema < 0.01:
                    new_ema = conf_raw  # 首帧直接用
                else:
                    new_ema = (1.0 - self._conf_ema_alpha) * prev_ema + self._conf_ema_alpha * conf_raw
                self._conf_ema_map[enemy_idx] = new_ema
                conf = new_ema
            else:
                conf = 0.95

            detection = Detection(
                x=screen_x + noise_off[0], y=screen_y + noise_off[1],
                w=box_w, h=box_h, conf=conf, class_id=0,
                xyxy=np.array([
                    screen_x - box_w / 2 + noise_off[0],
                    screen_y - box_h / 2 + noise_off[1],
                    screen_x + box_w / 2 + noise_off[0],
                    screen_y + box_h / 2 + noise_off[1],
                ])
            )
            detections.append(detection)

        self.ctx.targets = detections
        # 注意：target_first_seen_time 由 _tick_alpha_scheduling() 管理（conf 滞回 + 多帧确认）

    # ==========================================================================
    #  鼠标输出  —— 模块A 升级：1000Hz 子步积分（模拟 MouseWorker 独立线程）
    # ==========================================================================
    def tick_mouse(self, dt_main: float = 0.002):
        """
        模拟 MouseWorker 1000Hz 独立线程消费 controller.tick_mouse()。
        主循环 dt=2ms(500Hz) → 子步 2 次 1ms(1000Hz) 取整积分。
        这模拟了实战中 1kHz 取整积分的量化抖动效应。
        """
        self._mouse_tick_accumulator += dt_main
        n_substeps = int(self._mouse_tick_accumulator / self._mouse_tick_dt)
        if n_substeps <= 0:
            return
        self._mouse_tick_accumulator -= n_substeps * self._mouse_tick_dt

        ctrl = self.world_model.controller
        strategy = self.world_model.strategy

        total_mx, total_my = 0, 0
        total_dcx, total_dcy = 0.0, 0.0
        has_float_delta = False
        main_time = self._perf_time
        substep_start = main_time - n_substeps * self._mouse_tick_dt
        try:
            for substep in range(n_substeps):
                self._perf_time = substep_start + (substep + 1) * self._mouse_tick_dt
                mx, my = ctrl.tick_mouse()
                total_mx += mx
                total_my += my

                # ``_last_delta_float`` is per mouse tick. Summing it here is
                # essential when the 500 Hz simulation advances a 1 kHz actuator.
                # Do not reuse it while emission is disabled: in that state the
                # controller intentionally leaves the previous value untouched.
                if getattr(ctrl, "_emit_mouse", True):
                    last_delta = getattr(ctrl, "_last_delta_float", None)
                    if last_delta is not None:
                        total_dcx += float(last_delta[0])
                        total_dcy += float(last_delta[1])
                        has_float_delta = True
        finally:
            self._perf_time = main_time

        # 用浮点 delta 反映射到像素（比整数更精确）
        if has_float_delta:
            # crosshair_pos is a world-space accumulated aim coordinate. It is
            # not a screen offset and therefore must not be used as a FOV
            # Jacobian anchor. Mouse actuation happens around screen center.
            try:
                px, py = strategy.reverse_map_velocity(
                    total_dcx, total_dcy,
                    px_x=0.0, px_y=0.0, bbox_w=60.0
                )
            except TypeError:
                px, py = strategy.reverse_map_velocity(
                    total_dcx, total_dcy, bbox_w=60.0
                )
        else:
            px, py = strategy.reverse_map_velocity(
                float(total_mx), float(total_my), bbox_w=60.0
            )

        self.crosshair_pos[0] += px
        self.crosshair_pos[1] += py
        self.ring_buffer.add_event(total_mx, total_my, is_ai=True)

    # ==========================================================================
    #  模块A 新增：α 连续融合调度（镜像 agent.py tick 逻辑）
    # ==========================================================================
    def _tick_alpha_scheduling(self, dt: float, now: float) -> bool:
        """
        α 连续融合调度：按人手速度 EMA 连续调整 alpha_target。
        替代原 chase_mode 二值切换 + 3 帧滞回。

        返回 True 表示有有效目标可继续 compute；False 表示应 freeze。
        """
        ctrl = self.world_model.controller
        min_aim = self._min_aim_conf
        min_drop = min(self._min_aim_conf - 1e-3, self._min_aim_conf_drop)

        # ── conf 滞回 + 多帧确认（镜像 agent.py）──
        if not self.ctx.targets:
            self._invalid_streak += 1
            # WorldModel deliberately coasts a track through short detector
            # dropouts. Keep executing that prediction; stopping on the first
            # missing raw box defeats the estimator and turns a 2-6 frame CV
            # burst into repeated motor power cuts.
            if (
                self.target_first_seen_time > 0.0
                and self.ctx.p_predict is not None
            ):
                if hasattr(ctrl, 'set_mouse_emit'):
                    ctrl.set_mouse_emit(True)
                return True
            if self._invalid_streak >= self._aim_drop_invalid_frames:
                if self.target_first_seen_time > 0.0:
                    # 拆锁
                    self.target_first_seen_time = 0.0
                    self._lock_confirm_count = 0
                    self._invalid_streak = 0
                    self._conf_ema_map.clear()  # 重置 per-target conf EMA
                    if hasattr(ctrl, 'set_mouse_emit'):
                        ctrl.set_mouse_emit(False)
                physical_error = float(np.linalg.norm(
                    self.enemy_pos - self.crosshair_pos
                ))
                perception_radius = float(getattr(
                    self.scenario, 'perception_radius_px', 1000.0
                ))
                visibility_fn = getattr(self.scenario, 'is_target_visible', None)
                physically_visible = (
                    bool(visibility_fn(physical_error))
                    if callable(visibility_fn)
                    else physical_error <= perception_radius
                )
                if not physically_visible:
                    self._publish_target_observation(False)
            return False

        self._invalid_streak = 0
        cur_conf = self.ctx.targets[0].conf if self.ctx.targets else 0.0
        if cur_conf >= min_aim:
            # Confidence confirmation and covert reaction planning run in
            # parallel; confirmation therefore adds no post-reaction latency.
            self._publish_target_observation(True, cur_conf)

        # 首次锁目标：需连续 N 帧有效
        if self.target_first_seen_time == 0.0:
            if cur_conf < min_aim:
                if hasattr(ctrl, 'set_mouse_emit'):
                    ctrl.set_mouse_emit(False)
                return False
            self._lock_confirm_count += 1
            if (
                getattr(ctrl, '_handoff_reason', '') == 'human_release'
                and getattr(ctrl, 'takeover_state', '') == 'REACTION'
            ):
                self._lock_confirm_count = self._aim_lock_confirm
            if self._lock_confirm_count < self._aim_lock_confirm:
                if hasattr(ctrl, 'set_mouse_emit'):
                    ctrl.set_mouse_emit(False)
                return False
        else:
            # 已锁：conf 跌破 min_drop 才拆锁
            if cur_conf < min_drop:
                handoff_state = getattr(ctrl, 'takeover_state', 'ACTIVE_LOCK')
                if handoff_state in ('HUMAN_LEAD', 'REACTION', 'PRIMING'):
                    # A weak frame must not cut motor power during handoff. The
                    # world model is still carrying a valid predicted track.
                    return True
                self.target_first_seen_time = 0.0
                self._lock_confirm_count = 0
                self._conf_ema = 0.0  # 重置 conf EMA
                if hasattr(ctrl, 'set_mouse_emit'):
                    ctrl.set_mouse_emit(False)
                return False

        # ── 首次见到目标：决定 alpha_target ──
        first_frame = (self.target_first_seen_time == 0.0)
        if first_frame:
            self._intent_tracker.reset()
            self.target_first_seen_time = now
            # 按最近 100ms 人类速度计算 alpha_target
            dx_100, dy_100 = self._intent_delta(now - 0.1, now)
            recent_speed = math.hypot(dx_100, dy_100) / 0.1
            alpha_target = float(np.clip(
                1.0 - (recent_speed - 100.0) / 400.0, 0.0, 1.0
            ))
            handoff_in_progress = (
                getattr(ctrl, '_handoff_reason', '') == 'human_release'
                and bool(getattr(self.scenario, '_human_released', False))
            )
            if handoff_in_progress:
                alpha_target = 1.0
            self._current_chase_mode = 'pure_ai' if alpha_target > 0.5 else 'human_flick'

            if hasattr(ctrl, 'set_blend_alpha'):
                if handoff_in_progress:
                    init_state = getattr(ctrl, 'takeover_state', 'REACTION')
                else:
                    init_state = 'HUMAN_LEAD' if alpha_target < 0.3 else 'ACTIVE_LOCK'
                ctrl.set_blend_alpha(alpha_target, takeover_state=init_state)
            if hasattr(ctrl, 'reset_target_state'):
                try:
                    ctrl.reset_target_state(
                        mode=self._current_chase_mode, hard=False
                    )
                except TypeError:
                    ctrl.reset_target_state()

        # ── alpha 持续调度：速度基础 + 加速度触发的提前让位（v5.0 升级）────
        # 原版（线性）: alpha = 1 - speed/500，斜率固定 → flick 启动也让位 0.6
        # 新版（外科）: 速度部分保持线性（不破坏稳态 natural），
        #               加速度触发额外的提前让位（最多 0.40，flick 启动时 α 急降）
        #   - 慢动 (speed=100, acc=0)  → alpha=0.80 (与原版同)
        #   - flick 启动 (speed=200, acc=800) → alpha=0.20 (从 0.60 提前让位)
        #   - 强 flick (speed=500+)   → alpha=0.0 (与原版同)
        # 物理意义：flick 启动瞬间 acc_proxy 暴涨（30ms 速度 >> 80ms 速度），
        #           触发 alpha 提前下降 → AI 让位更早，避免"拉锯"
        FLICK_END_PX_S = 450.0
        _recent_dx, _recent_dy = self._intent_delta(now - 0.08, now)
        _recent_spd = math.hypot(_recent_dx, _recent_dy) / 0.08
        _30ms_dx, _30ms_dy = self._intent_delta(now - 0.03, now)
        _30ms_spd = math.hypot(_30ms_dx, _30ms_dy) / 0.03
        _inst_dx, _inst_dy = self._intent_delta(now - dt, now)
        _inst_spd = math.hypot(_inst_dx, _inst_dy) / max(dt, 1e-6)
        # 加速度代理：flick 启动时 30ms 速度比 80ms 速度大
        _acc_proxy = max(0.0, _30ms_spd - _recent_spd)
        # 速度基础（保持原线性，稳态 natural 不破坏）
        _speed_alpha = 1.0 - min(1.0, _recent_spd / 500.0)
        # 加速度触发的额外让位（flick 启动时 α 急降）
        # speed gate：仅在中等以上速度（speed>150）才允许 acc_drop 生效，
        #            避免低速小噪声时 acc_drop 误触发导致 natural 下降
        # acc_drop 上限 0.10：只让 flick 最尖锐的启动瞬间生效，影响小
        _speed_gate = 1.0 / (1.0 + math.exp(-(_recent_spd - 150.0) / 40.0))  # spd=150→0.5, 250→0.93
        _acc_drop = min(0.10, _acc_proxy / 2000.0) * _speed_gate
        _new_alpha_target = float(np.clip(_speed_alpha - _acc_drop, 0.0, 1.0))

        # 接管状态机
        _cur_state = getattr(ctrl, 'takeover_state', 'ACTIVE_LOCK')
        _new_state = None
        if (_new_alpha_target < 0.2 and _inst_spd > 100.0
                and not bool(getattr(self.scenario, '_human_released', False))
                and _cur_state != 'HUMAN_LEAD'):
            _new_state = 'HUMAN_LEAD'
        elif _new_alpha_target > 0.8 and _cur_state == 'HUMAN_LEAD':
            _new_state = 'POST_TAKEOVER'
            self._flick_end_time = now
        elif _new_alpha_target > 0.5 and _cur_state == 'POST_TAKEOVER':
            _new_state = 'ACTIVE_LOCK'

        if hasattr(ctrl, 'set_blend_alpha'):
            ctrl.set_blend_alpha(_new_alpha_target, takeover_state=_new_state)
        self._current_chase_mode = 'pure_ai' if _new_alpha_target > 0.5 else 'human_flick'

        return True

    # ==========================================================================
    #  模块A 新增：预测性 + 方向感知 warm_start（镜像 agent.py）
    # ==========================================================================
    def _tick_warm_start(self, dt: float, now: float, p_x: float, p_y: float):
        """
        预测性 warm_start：人手速度导数 → 预测放手时刻 → 提前预热。
        方向感知 warm_start：放手时按方向对齐选权重。
        """
        ctrl = self.world_model.controller

        # ── 人手速度 raw + 导数 ──
        dx_h_inst, dy_h_inst = self._intent_delta(now - dt, now)
        human_speed_raw = math.hypot(dx_h_inst, dy_h_inst) / dt if dt > 0 else 0.0
        human_accel = (human_speed_raw - self._prev_human_speed_raw) / max(dt, 0.001)
        self._human_accel_ema = 0.8 * self._human_accel_ema + 0.2 * human_accel
        self._prev_human_speed_raw = human_speed_raw

        # ── 预测性预热：人手强减速 + 高速 → 即将放手 ──
        FLICK_END_PX_S = 450.0
        cur_human_flicking = human_speed_raw > FLICK_END_PX_S

        if (human_speed_raw > 500.0
                and self._human_accel_ema < -2000.0
                and hasattr(ctrl, 'pre_warm')):
            t_predict = human_speed_raw / max(abs(self._human_accel_ema), 1.0)
            if t_predict < 0.05:
                vr = self.ctx.v_real
                if vr is not None:
                    try:
                        k = getattr(self.world_model.strategy, 'k_factor_x', 1.0)
                        tgt_v_arr = np.array(
                            [float(vr[0]) * k, float(vr[1]) * k],
                            dtype=np.float64,
                        )
                        blend = float(np.clip(1.0 - t_predict / 0.05, 0.0, 0.5))
                        ctrl.pre_warm(tgt_v_arr, blend_factor=blend)
                    except Exception:
                        pass

        # ── 实际放手 → 方向感知 warm_start ──
        if self._prev_human_flicking and not cur_human_flicking:
            self._flick_end_time = now
            if hasattr(ctrl, 'begin_handoff'):
                try:
                    _hdx, _hdy = self._intent_delta(now - 0.02, now)
                    bbox_w = self.ctx.targets[0].w if self.ctx.targets else 60.0
                    strategy = self.world_model.strategy
                    try:
                        hvx, hvy = strategy.calculate_velocity_move(
                            _hdx / 0.02, _hdy / 0.02,
                            px_x=0.0, px_y=0.0, bbox_w=bbox_w,
                        )
                    except TypeError:
                        hvx, hvy = strategy.calculate_velocity_move(
                            _hdx / 0.02, _hdy / 0.02, bbox_w=bbox_w,
                        )
                    ex, ey = strategy.calculate_mouse_move(p_x, p_y, bbox_w=bbox_w)
                    ctrl.begin_handoff(
                        human_velocity=np.array([hvx, hvy], dtype=np.float64),
                        target_velocity=self._target_velocity_counts(),
                        error=np.array([ex, ey], dtype=np.float64),
                        reason='human_release',
                    )
                except Exception:
                    pass
            elif hasattr(ctrl, 'notify_flick_end'):
                ctrl.notify_flick_end()
            elif hasattr(ctrl, 'warm_start_from_velocity_directional'):
                try:
                    _hdx, _hdy = self._intent_delta(now - 0.02, now)
                    ctrl.warm_start_from_velocity_directional(
                        _hdx / 0.02, _hdy / 0.02, 0.0, 0.0, p_x, p_y,
                    )
                except Exception:
                    pass
            elif hasattr(ctrl, 'warm_start_from_velocity'):
                try:
                    _hdx, _hdy = self._intent_delta(now - 0.02, now)
                    human_v_inst = np.array([_hdx / 0.02, _hdy / 0.02], dtype=np.float64)
                    vr = self.ctx.v_real
                    if vr is not None:
                        k = getattr(self.world_model.strategy, 'k_factor_x', 1.0)
                        tgt_v = np.array([float(vr[0]) * k, float(vr[1]) * k], dtype=np.float64)
                        seed = human_v_inst * 0.7 + tgt_v * 0.3
                    else:
                        seed = human_v_inst
                    ctrl.warm_start_from_velocity(float(seed[0]), float(seed[1]))
                except Exception:
                    pass
        self._prev_human_flicking = cur_human_flicking

        return dx_h_inst, dy_h_inst, human_speed_raw

    # ==========================================================================
    #  主步进  —— 模块A 全面重写：镜像 agent.py tick() 接口
    # ==========================================================================
    def step(self):
        dt = 0.002
        self._perf_time += dt
        if not bool(getattr(self.scenario, 'clock_paused', False)):
            self.sim_time += dt
        self.sim_steps += 1
        now = self._perf_time

        # ── 管道延迟采样 ──
        stream_lat, inf_delay, total_delay = self._sample_pipeline_delay()
        t_capture = self._perf_time - total_delay

        # ── 委托场景: 物理 + 人类输入 ──
        self.scenario.tick_physics(self, dt)
        human_dx, human_dy = self.scenario.tick_human_input(self, dt)

        # Human input is physical motion, not merely an intent signal. Apply
        # the exact floating displacement before capture, while recording the
        # quantized device event for latency reconstruction and intent logic.
        if human_dx or human_dy:
            self.crosshair_pos[0] += human_dx
            self.crosshair_pos[1] += human_dy

            if not hasattr(self, '_human_subpixel'):
                self._human_subpixel = np.zeros(2, dtype=np.float64)
            delta_x = human_dx + self._human_subpixel[0]
            delta_y = human_dy + self._human_subpixel[1]
            mx = int(np.floor(delta_x))
            my = int(np.floor(delta_y))
            self._human_subpixel[0] = delta_x - mx
            self._human_subpixel[1] = delta_y - my
            self.ring_buffer.add_event(mx, my, is_ai=False)

        # ── 计算机视觉（模块B 升级版）──
        self.tick_cv()

        # ── 构造检测 → WorldModel（多目标支持）──
        dets = np.empty((0, 6), dtype=np.float32)
        if self.ctx.targets:
            rows = []
            for det in self.ctx.targets:
                if det.xyxy is not None:
                    rows.append([
                        det.xyxy[0], det.xyxy[1], det.xyxy[2], det.xyxy[3],
                        det.conf, float(det.class_id if hasattr(det, 'class_id') else 0)
                    ])
            if rows:
                dets = np.array(rows, dtype=np.float32)

        self.world_model.update_detections(
            detections=dets,
            frame_id=self.sim_steps,
            t_capture=t_capture,
            t_done=self._perf_time,
        )
        self.world_model.step(self.ctx, self.ring_buffer)

        # ── 兼容旧场景：_prev_flick_active → notify_flick_end ──
        cur_flick_active = getattr(self, '_prev_flick_active', False)
        prev_flick_active = getattr(self, '_prev_flick_was', False)
        if prev_flick_active and not cur_flick_active:
            ctrl = self.world_model.controller
            if (hasattr(ctrl, 'notify_flick_end')
                    and getattr(ctrl, '_handoff_reason', '') != 'human_release'):
                self.world_model.controller.notify_flick_end()
        self._prev_flick_was = cur_flick_active

        # ── α 连续融合调度（conf 滞回 + 多帧确认 + alpha_target）──
        has_target = self._tick_alpha_scheduling(dt, now)
        if not has_target:
            self.last_ai_factor = 0.0
            self.tick_mouse(dt)
            self._print_latency_stats()
            return

        # ── AI 联合发力 ──
        if self.ctx.p_predict is not None and self.ctx.v_real is not None:
            p_x = float(self.ctx.p_predict[0])
            p_y = float(self.ctx.p_predict[1])

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

            drifted_p_x = p_x + drift_x
            drifted_p_y = p_y + drift_y

            bbox_w = self.ctx.targets[0].w if self.ctx.targets else 60.0

            intent_x, intent_y = self.world_model.strategy.calculate_mouse_move(
                drifted_p_x, drifted_p_y, bbox_w=bbox_w
            )

            # FOV Jacobian 修正：传 px_x/px_y（镜像 agent.py P0 修复）
            v_real_pixels = self.ctx.v_real
            try:
                intent_vx, intent_vy = self.world_model.strategy.calculate_velocity_move(
                    v_real_pixels[0], v_real_pixels[1],
                    px_x=0.0, px_y=0.0, bbox_w=bbox_w
                )
                a_real_pixels = getattr(self.ctx, 'a_real', (0.0, 0.0))
                intent_ax, intent_ay = self.world_model.strategy.calculate_velocity_move(
                    a_real_pixels[0], a_real_pixels[1],
                    px_x=0.0, px_y=0.0, bbox_w=bbox_w
                )
            except TypeError:
                # 老接口兜底
                intent_vx, intent_vy = self.world_model.strategy.calculate_velocity_move(
                    v_real_pixels[0], v_real_pixels[1], bbox_w=bbox_w
                )
                a_real_pixels = getattr(self.ctx, 'a_real', (0.0, 0.0))
                intent_ax, intent_ay = self.world_model.strategy.calculate_velocity_move(
                    a_real_pixels[0], a_real_pixels[1], bbox_w=bbox_w
                )

            pixel_error_dist = math.hypot(drifted_p_x, drifted_p_y)
            spatial_factor = float(np.clip(1.0 - (pixel_error_dist / 800.0) ** 2, 0.1, 1.0))

            # ── reaction_factor: alpha 加权（镜像 agent.py）──
            _alpha_now = getattr(self.world_model.controller, 'alpha', 1.0)
            if _alpha_now > 0.95:
                reaction_factor = 1.0
            else:
                t_ref = self._flick_end_time if self._flick_end_time > 0.0 else self.target_first_seen_time
                reaction_factor_full = float(np.clip((now - t_ref) / 0.03, 0.30, 1.0))
                reaction_factor = _alpha_now * 1.0 + (1.0 - _alpha_now) * reaction_factor_full

            # ── warm_start + 人手速度计算 ──
            dx_h_inst, dy_h_inst, human_speed_raw = self._tick_warm_start(
                dt, now, drifted_p_x, drifted_p_y
            )

            # 人手速度 EMA 窗（用于意图追踪）
            _vw = float(np.clip(
                max(float(self._intent_tracker.human_vel_window_s), float(dt)),
                0.015, 0.120
            ))
            dx_hs, dy_hs = self._intent_delta(now - _vw, now)
            human_vx = (dx_hs / _vw) if _vw > 0 else 0.0
            human_vy = (dy_hs / _vw) if _vw > 0 else 0.0

            # ── 人手意图 → AI 权重 ──
            ai_weight = self._intent_tracker.update(
                human_vx, human_vy,
                drifted_p_x, drifted_p_y,
                dt, now,
            )

            power_factor = spatial_factor * reaction_factor * ai_weight

            ctrl = self.world_model.controller
            if hasattr(ctrl, "set_mouse_emit"):
                ctrl.set_mouse_emit(True)
            ctrl.compute(
                target_x=intent_x, target_y=intent_y, dt=dt,
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

        # ── 1000Hz 鼠标子步积分 ──
        self.tick_mouse(dt)
        self._print_latency_stats()

    @property
    def is_done(self) -> bool:
        return self.scenario.is_done
