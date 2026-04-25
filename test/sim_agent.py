# sim_agent.py
# 串流延迟模拟版：独立模拟 Moonlight 网络抖动延迟 (10~30ms) + 推理处理延迟
import time
import threading
import numpy as np
import random
import math
from collections import deque
from config import config

try:
    import noise
except ImportError:
    print("⚠️ 警告: 未安装 noise 库，仿真中的 Perlin 噪点将失效，请使用 pip install noise 安装")

from perception.ring_buffer import RingBuffer
from utils.types import Detection, InferenceContext


# ==============================================================================
# 🌐 Moonlight 串流延迟模型
#
# 模拟真实 Moonlight 串流的延迟特征：
#   - 基础帧延迟 (编码 + 网络传输)：10~20ms
#   - 随机抖动 (Jitter)：±3~8ms，采用对数正态分布（符合实际网络包分布）
#   - 偶发毛刺 (Spike)：每隔若干帧出现一次 25~40ms 的突增延迟，模拟无线包重传
#   - 帧间平滑：EMA 滤波避免前后帧延迟剧烈跳变
# ==============================================================================
class MoonlightLatencyModel:
    def __init__(
        self,
        base_ms: float = 15.0,       # 基础串流延迟 (ms)
        jitter_std_ms: float = 4.0,  # 正常抖动标准差 (ms)
        spike_prob: float = 0.04,    # 每帧发生毛刺的概率
        spike_extra_ms: float = 18.0,# 毛刺时额外增加的延迟 (ms)
        min_ms: float = 8.0,         # 下限保护 (ms)
        max_ms: float = 45.0,        # 上限保护 (ms)
        ema_alpha: float = 0.25,     # EMA 平滑系数 (越大越跳跃，越小越平滑)
    ):
        self.base    = base_ms    / 1000.0
        self.jitter  = jitter_std_ms / 1000.0
        self.spike_p = spike_prob
        self.spike_e = spike_extra_ms / 1000.0
        self.min_lat = min_ms / 1000.0
        self.max_lat = max_ms / 1000.0
        self.alpha   = ema_alpha

        # 初始化 EMA 状态为基础延迟
        self._ema = self.base
        # 用对数正态的 sigma 参数来匹配目标 std
        # ln-normal: mean=base, std=jitter → sigma = sqrt(ln(1 + (jitter/base)^2))
        _var_ratio = (self.jitter / max(self.base, 1e-6)) ** 2
        self._ln_sigma = math.sqrt(math.log(1.0 + _var_ratio)) if _var_ratio > 0 else 0.001
        self._ln_mu    = math.log(max(self.base, 1e-6)) - 0.5 * self._ln_sigma ** 2

    def sample(self) -> float:
        """采样本帧的实际串流延迟 (秒)"""
        # 对数正态基础抖动
        raw = np.random.lognormal(self._ln_mu, self._ln_sigma)

        # 偶发毛刺 (模拟无线重传 / 编码器 I-frame 突增)
        if np.random.random() < self.spike_p:
            raw += self.spike_e * np.random.uniform(0.5, 1.5)

        # EMA 平滑：避免前后帧延迟从 10ms 瞬跳到 35ms
        self._ema = self.alpha * raw + (1.0 - self.alpha) * self._ema

        return float(np.clip(self._ema, self.min_lat, self.max_lat))

    def reset(self):
        self._ema = self.base


# ==============================================================================
# 仿真 AI Agent
# ==============================================================================
class SimAIAgent:
    def __init__(
        self,
        noise_level: float = 5.0,
        dropout_rate: float = 0.08,
        # --- 延迟参数 (全部可从外部传入或由 config 读取) ---
        stream_base_ms: float = None,   # 串流基础延迟；None 则读 config
        stream_jitter_ms: float = 4.0,  # 串流抖动 std (ms)
        stream_spike_prob: float = 0.04,
        stream_spike_ms: float = 18.0,
        inference_base_ms: float = 6.0, # YOLO 推理耗时基础值 (ms)
        inference_std_ms: float = 1.5,  # 推理耗时抖动
    ):
        # ── 从 config 读取串流延迟基础值（可被参数覆盖）──────────────────────
        if stream_base_ms is None:
            stream_base_ms = config.getfloat("WorldModel", "moonlight_latency_ms", 15.0)

        self.sim_time = 0.0
        self._original_perf_counter = time.perf_counter
        time.perf_counter = lambda: self.sim_time

        from world_model import WorldModel
        self.world_model = WorldModel()
        self.ring_buffer  = RingBuffer()

        self.sim_steps   = 0
        self.noise_level  = noise_level
        self.dropout_rate = dropout_rate

        # ── 双延迟模型 ────────────────────────────────────────────────────────
        # 1) 串流网络延迟 (Moonlight)
        self.stream_latency_model = MoonlightLatencyModel(
            base_ms       = stream_base_ms,
            jitter_std_ms = stream_jitter_ms,
            spike_prob    = stream_spike_prob,
            spike_extra_ms= stream_spike_ms,
        )
        # 2) 本地推理延迟 (YOLO forward pass)
        self._inf_base = inference_base_ms / 1000.0
        self._inf_std  = inference_std_ms  / 1000.0

        # 延迟统计（供 print 用）
        self._lat_history: deque = deque(maxlen=500)
        self._lat_print_timer = 0.0

        # ── 准星与场景 ────────────────────────────────────────────────────────
        self.crosshair_pos = np.zeros(2, dtype=np.float64)
        self.center = np.array(
            [self.world_model.crop_center, self.world_model.crop_center],
            dtype=np.float64
        )

        self.ctx = InferenceContext()
        self.ctx.center_pos = self.center

        self.burst_state = 0

        # ── 实战靶场状态机 ────────────────────────────────────────────────────
        self.max_kills        = 30
        self.kill_count       = 0
        self.is_done          = False
        self.current_target_id = 0

        # 🌟 3D 摄像机投影常量 (Valorant FOV = 103)
        self.fov          = 103.0
        self.screen_w     = config.getfloat("General", "screen_width",  1920.0)
        self.screen_height= config.getfloat("General", "screen_height", 1080.0)
        self.focal_length = (self.screen_w / 2.0) / math.tan(math.radians(self.fov / 2.0))

        # 🌟 3D 物理空间变量
        self.target_z      = 10.0
        self.enemy_vel_3d  = np.zeros(2, dtype=np.float64)
        self.target_vel_3d = np.zeros(2, dtype=np.float64)
        self.current_action= 'stop'
        self.height_3d     = 0.0

        self.enemy_pos = np.zeros(2, dtype=np.float64)
        self.enemy_vel = np.zeros(2, dtype=np.float64)

        self.tot_threshold         = 0.08
        self.target_first_seen_time = 0.0
        self.noise_offset_x = random.uniform(0, 1000.0)
        self.noise_offset_y = random.uniform(0, 1000.0)

        self._prev_flick_active = False

        self.spawn_new_target()

    # ==========================================================================
    # 内部工具：采样本帧的完整管道延迟
    # ==========================================================================
    def _sample_pipeline_delay(self):
        """
        返回 (stream_latency, inference_delay, total_delay) 单位：秒

        管道时序：
            t_real_capture  ──[stream_latency]──▶  帧到达本机
                            ──[inference_delay]──▶  检测结果可用  (= sim_time)

        因此：
            t_capture = sim_time - stream_latency - inference_delay
        """
        stream_lat = self.stream_latency_model.sample()
        # 推理耗时：高斯分布，下限 2ms
        inf_delay  = max(0.002, np.random.normal(self._inf_base, self._inf_std))
        total      = stream_lat + inf_delay
        self._lat_history.append(stream_lat * 1000.0)
        return stream_lat, inf_delay, total

    # ==========================================================================
    def spawn_new_target(self):
        """生成新目标：初始化 3D 距离与霓虹物理参数"""
        if self.kill_count >= self.max_kills:
            self.is_done = True
            self.ctx.targets = []
            return

        self.current_target_id += 1
        self.world_model.targets.clear()

        # Bug 6 修复：重置 Kalman 的自适应状态（innov_ema, Q_scale），避免上一
        # 目标（高速 adad/slide）的 innov 尾巴污染下一目标前 100ms 的 v_est。
        # innov_ema EMA 下降常数 α=0.1 → 100ms 才衰减 ~30%，期间 Q 偏高会让
        # 新 target 的协方差膨胀更快，ff_vel 被放大噪声 → precision 方差变大。
        if hasattr(self.world_model, 'kalman') and hasattr(self.world_model.kalman, 'reset_adaptive_state'):
            self.world_model.kalman.reset_adaptive_state()

        self.chase_mode = np.random.choice(['pure_ai', 'human_flick'])
        # 把 chase_mode 传给控制器（决定是否激活 entry_ticks 微调窗口）
        if hasattr(self.world_model.controller, 'reset_target_state'):
            try:
                self.world_model.controller.reset_target_state(mode=self.chase_mode)
            except TypeError:
                # 兼容不支持 mode 参数的老控制器
                self.world_model.controller.reset_target_state()

        self.target_z    = np.random.uniform(5.0, 35.0)
        scale_factor     = self.focal_length / self.target_z
        base_radius_px   = 3.0 * scale_factor

        if self.chase_mode == 'pure_ai':
            # 实战语境：AI 的 YOLO 识别半径 ≈ 256px；pure_ai 模拟"目标从侧面
            # 切入 FOV / 已经被人类粗略对过枪 → AI 独立精修最后一段"。
            # 初始距离限制在 60~256 px，避免让 AI 替人类走 1000+px 的甩枪段。
            spawn_radius_px = np.random.uniform(60, 256)
        else:
            spawn_radius_px = np.random.uniform(600, 1400)

        angle    = np.random.uniform(0, 2 * np.pi)
        offset_x = np.cos(angle) * spawn_radius_px
        offset_y = np.sin(angle) * spawn_radius_px

        half_w, half_h = self.screen_w / 2.0, self.screen_height / 2.0
        margin   = 50
        offset_x = np.clip(offset_x, -half_w + margin, half_w - margin)
        offset_y = np.clip(offset_y, -half_h + margin, half_h - margin)

        self.enemy_pos = self.crosshair_pos + np.array([offset_x, offset_y])

        initial_dist = np.linalg.norm(self.enemy_pos - self.crosshair_pos)
        print(
            f"New target spawned | mode: {self.chase_mode} | z: {self.target_z:.1f}m"
            f" | initial_dist: {initial_dist:.0f} px"
        )

        self.enemy_vel_3d  = np.array([np.random.choice([-1, 1]) * 8.5, 0.0])
        self.target_vel_3d = self.enemy_vel_3d.copy()
        self.height_3d     = 0.0
        self.current_action= 'sprint'

        self.tot_timer              = 0.0
        self.target_first_seen_time = 0.0
        self.move_timer             = 0.5

        if self.chase_mode == 'human_flick':
            error_offset      = np.random.randn(2) * 20.0
            self.flick_target = self.enemy_pos + error_offset
            self.flick_duration  = np.random.uniform(0.15, 0.22)
            self.flick_timer     = 0.0
            self.flick_start_pos = self.crosshair_pos.copy()

        self._prev_flick_active = False

        # 新目标出现时重置串流延迟模型的 EMA（避免上一局高延迟污染新局）
        self.stream_latency_model.reset()

    # ==========================================================================
    def tick_enemy(self, dt: float):
        if self.is_done: return

        # 霓虹 (Neon) 3D 物理与身法模型
        if getattr(self, 'move_timer', 0) <= 0:
            self.move_timer = np.random.uniform(0.15, 0.4)
            action = np.random.choice(
                ['sprint', 'slide', 'adad', 'jump', 'stop'],
                p=[0.30, 0.20, 0.30, 0.15, 0.05]
            )
            self.current_action = action

            if action == 'sprint':
                self.target_vel_3d[0] = np.random.choice([-1, 1]) * 8.5
            elif action == 'adad':
                self.target_vel_3d[0] = np.random.choice([-1, 1]) * 5.4
            elif action == 'slide':
                dir_x = np.sign(self.enemy_vel_3d[0]) if self.enemy_vel_3d[0] != 0 else np.random.choice([-1, 1])
                self.enemy_vel_3d[0] = dir_x * 14.0
                self.target_vel_3d[0] = 0.0
                self.move_timer = 0.6
            elif action == 'jump':
                if self.height_3d <= 0.01:
                    self.enemy_vel_3d[1] = -5.8
            elif action == 'stop':
                self.target_vel_3d[0] = 0.0

        self.move_timer -= dt

        # ----- 1. 3D 水平引擎 -----
        if self.current_action == 'slide':
            accel_x = (self.target_vel_3d[0] - self.enemy_vel_3d[0]) * 5.0
        else:
            accel_x = (self.target_vel_3d[0] - self.enemy_vel_3d[0]) * 40.0
        self.enemy_vel_3d[0] += accel_x * dt

        # ----- 2. 3D 垂直抛物线引擎 -----
        if self.height_3d > 0.0 or self.enemy_vel_3d[1] < 0:
            self.enemy_vel_3d[1] += 16.0 * dt
            self.height_3d -= self.enemy_vel_3d[1] * dt
            if self.height_3d <= 0.0:
                self.height_3d     = 0.0
                self.enemy_vel_3d[1] = 0.0

        # ----- 3. 3D → 2D 摄像机投影 -----
        scale_factor   = self.focal_length / self.target_z
        self.enemy_vel[0] = self.enemy_vel_3d[0] * scale_factor
        self.enemy_vel[1] = self.enemy_vel_3d[1] * scale_factor
        self.enemy_pos   += self.enemy_vel * dt

        # 动态判定体积
        self.target_hitbox_x = 0.15 * scale_factor
        self.target_hitbox_y = 0.15 * scale_factor

        err_x = abs(self.enemy_pos[0] - self.crosshair_pos[0])
        err_y = abs(self.enemy_pos[1] - self.crosshair_pos[1])

        if err_x <= self.target_hitbox_x and err_y <= self.target_hitbox_y:
            self.tot_timer += dt
            if self.tot_timer >= self.tot_threshold:
                self.kill_count += 1
                mode_str = "🧑 人机" if self.chase_mode == 'human_flick' else "🤖 纯AI"
                print(
                    f"🎯 击杀 {self.kill_count:2d}/30! [{mode_str}] | "
                    f"距离: {self.target_z:.1f}m | 动作: {self.current_action}"
                )
                self.spawn_new_target()
        else:
            self.tot_timer = max(0.0, self.tot_timer - dt * 2.0)

    # ==========================================================================
    def tick_cv(self):
        """带有真实 3D 尺寸计算的机器视觉"""
        if self.is_done: return

        relative_pos = self.enemy_pos - self.crosshair_pos
        dist_px      = np.linalg.norm(relative_pos)

        if dist_px > 1000.0:
            self.ctx.targets = []
            # Bug 1 修复：目标飞出识别半径时防御性重置 first_seen_time，
            # 避免目标重新进入时 reaction_factor 跳过冷启动斜坡。
            # sim 场景极少触发，但避免未来修改 spawn 逻辑后引入静默 bug。
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

        scale_factor = self.focal_length / self.target_z
        box_w        = 0.45 * scale_factor

        height_3d = 0.8 if self.current_action == 'slide' else 1.7
        box_h     = height_3d * scale_factor

        noise_off = np.random.randn(2) * self.noise_level

        detection = Detection(
            x=screen_x + noise_off[0], y=screen_y + noise_off[1],
            w=box_w, h=box_h, conf=0.95, class_id=self.current_target_id,
            xyxy=np.array([
                screen_x - box_w / 2,
                screen_y - box_h / 2,
                screen_x + box_w / 2,
                screen_y + box_h / 2
            ])
        )
        self.ctx.targets = [detection]

    # ==========================================================================
    def tick_mouse(self):
        mx, my = self.world_model.controller.tick_mouse()
        # Sim 用浮点位移（counts）更新 crosshair_pos，避免整数量化在 Natural
        # 评分里被当成"高频抖动"（量化脉冲每几 ms 跳 1px → hf_penalty 爆）
        # 真实硬件仍然消费整数 mx/my
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
    def _print_latency_stats(self):
        """每秒打印一次当前串流延迟统计"""
        if self.sim_time - self._lat_print_timer < 1.0:
            return
        self._lat_print_timer = self.sim_time
        if len(self._lat_history) > 10:
            arr = np.array(self._lat_history)
            print(
                f"🌐 [串流延迟] "
                f"均值={arr.mean():.1f}ms  "
                f"P50={np.percentile(arr,50):.1f}ms  "
                f"P95={np.percentile(arr,95):.1f}ms  "
                f"最大={arr.max():.1f}ms"
            )

    # ==========================================================================
    def step(self):
        dt = 0.002
        self.sim_time += dt
        self.sim_steps += 1

        # ── 采样本帧完整管道延迟 ─────────────────────────────────────────────
        # stream_lat : Moonlight 串流网络延迟（编码 + 传输）
        # inf_delay  : 本地 YOLO 推理耗时
        # total_delay: 合计；决定 t_capture 的回溯量
        stream_lat, inf_delay, total_delay = self._sample_pipeline_delay()
        t_capture = self.sim_time - total_delay

        # ── 物理 & 视觉更新 ─────────────────────────────────────────────────
        self.tick_enemy(dt)
        self.tick_cv()

        # ── 构造检测结果并推送给 WorldModel ──────────────────────────────────
        dets = np.empty((0, 6), dtype=np.float32)
        if self.ctx.targets:
            det  = self.ctx.targets[0]
            dets = np.array(
                [[det.xyxy[0], det.xyxy[1], det.xyxy[2], det.xyxy[3], det.conf, 0.0]],
                dtype=np.float32
            )

        self.world_model.update_detections(
            detections=dets,
            frame_id=self.sim_steps,
            t_capture=t_capture,
            t_done=self.sim_time
        )
        self.world_model.step(self.ctx, self.ring_buffer)

        # ── 人类物理输入注入 ─────────────────────────────────────────────────
        if not hasattr(self, '_human_subpixel'):
            self._human_subpixel = np.zeros(2, dtype=np.float64)

        cur_flick_active = (
            not self.is_done
            and self.chase_mode == 'human_flick'
            and getattr(self, 'flick_timer', 999) < getattr(self, 'flick_duration', 0)
        )

        if cur_flick_active:
            progress      = self.flick_timer / self.flick_duration
            next_progress = min(1.0, (self.flick_timer + dt) / self.flick_duration)

            def ease_out(t): return 1 - (1 - t) ** 3

            curr_pos = self.flick_start_pos + (self.flick_target - self.flick_start_pos) * ease_out(progress)
            next_pos = self.flick_start_pos + (self.flick_target - self.flick_start_pos) * ease_out(next_progress)

            dx_px = next_pos[0] - curr_pos[0]
            dy_px = next_pos[1] - curr_pos[1]
            self.crosshair_pos[0] += dx_px
            self.crosshair_pos[1] += dy_px

            delta_x = dx_px + self._human_subpixel[0]
            delta_y = dy_px + self._human_subpixel[1]
            mx = int(np.floor(delta_x))
            my = int(np.floor(delta_y))
            self._human_subpixel[0] = delta_x - mx
            self._human_subpixel[1] = delta_y - my

            self.ring_buffer.add_event(mx, my, is_ai=False)
            self.flick_timer += dt

        # flick 结束边沿检测 → 通知控制器清除积分
        if self._prev_flick_active and not cur_flick_active:
            if hasattr(self.world_model.controller, 'notify_flick_end'):
                self.world_model.controller.notify_flick_end()
        self._prev_flick_active = cur_flick_active

        # ── 控制器联合发力 ────────────────────────────────────────────────────
        if self.ctx.p_predict is not None and self.ctx.v_real is not None:
            dx_h_inst, dy_h_inst = self.ring_buffer.get_pure_human_delta_sum(
                self.sim_time - dt, self.sim_time
            )
            human_vx_inst = dx_h_inst / dt if dt > 0 else 0.0
            human_vy_inst = dy_h_inst / dt if dt > 0 else 0.0

            bbox_w = self.ctx.targets[0].w if self.ctx.targets else 60.0

            try:
                noise_scale = 0.5
                amplitude   = 3.0
                drift_x = noise.pnoise1(self.sim_time * noise_scale + self.noise_offset_x) * amplitude
                drift_y = noise.pnoise1(self.sim_time * noise_scale + self.noise_offset_y) * amplitude
            except NameError:
                drift_x, drift_y = 0.0, 0.0

            drifted_p_x = self.ctx.p_predict[0] + drift_x
            drifted_p_y = self.ctx.p_predict[1] + drift_y

            intent_x, intent_y = self.world_model.strategy.calculate_mouse_move(
                drifted_p_x, drifted_p_y, bbox_w=bbox_w
            )
            v_real_pixels  = self.ctx.v_real
            intent_vx, intent_vy = self.world_model.strategy.calculate_velocity_move(
                v_real_pixels[0], v_real_pixels[1], bbox_w=bbox_w
            )
            a_real_pixels  = getattr(self.ctx, 'a_real', (0.0, 0.0))
            intent_ax, intent_ay = self.world_model.strategy.calculate_velocity_move(
                a_real_pixels[0], a_real_pixels[1], bbox_w=bbox_w
            )

            pixel_error_dist = np.linalg.norm([drifted_p_x, drifted_p_y])
            spatial_factor   = np.clip(1.0 - (pixel_error_dist / 800.0) ** 2, 0.1, 1.0)

            time_since_seen  = self.sim_time - self.target_first_seen_time
            # pure_ai 模式：AI 是软件层独立瞄准，没有"眼→脑→手"的 150ms 视觉反
            # 应延迟；reaction_factor 是人类生理特征，强加给 AI 会让 power_factor
            # 在前 150ms 里被压到 0~1 斜坡，等效于 BALLISTIC 尾段被硬压速度。
            # 实测这一条对 256px pure_ai TTK 的影响约 80~120ms。
            if self.chase_mode == 'pure_ai':
                reaction_factor = 1.0
            else:
                reaction_factor = float(np.clip(time_since_seen / 0.15, 0.0, 1.0))

            dx_h_recent, dy_h_recent = self.ring_buffer.get_pure_human_delta_sum(
                self.sim_time - 0.1, self.sim_time
            )
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
                human_override_factor = 1.0 - (
                    (human_speed - speed_thresh_min) / (speed_thresh_max - speed_thresh_min)
                )

            power_factor = spatial_factor * reaction_factor * human_override_factor

            ctrl = self.world_model.controller
            if hasattr(ctrl, "set_mouse_emit"):
                ctrl.set_mouse_emit(True)
            ctrl.compute(
                target_x=intent_x, target_y=intent_y, dt=dt,
                human_v=np.array([human_vx_inst, human_vy_inst]),
                v_real=np.array([intent_vx, intent_vy]),
                a_real=np.array([intent_ax, intent_ay]),
                power_factor=power_factor,
                bbox_w=bbox_w
            )
            self.last_ai_factor = power_factor
        else:
            # Bug 5 修复：p_predict is None（目标丢失 >150ms 且非 coasting 期）
            # 不再每帧调用 reset_target_state() —— 每次 reset 会把 arm_vel × 0.05，
            # 连续 6 帧变成 ~1e-8，高速追踪突然丢帧后要从"绝对静止"重建速度，
            # 造成 SteadyMAE 飙升。reset 的正确时机只在 spawn_new_target 时调用
            # 一次（已在 spawn_new_target 里做了）。
            # 这里只记录 "AI 没发力"，保留 controller 的 arm_vel 与 Kalman 状态。
            # 与实战一致：关 emit 避免 tick_mouse 对 OU/漂移 1kHz 积分，但不 freeze
            #（保留 arm_vel 供重锁时续上）。
            ctrl = self.world_model.controller
            if hasattr(ctrl, "set_mouse_emit"):
                ctrl.set_mouse_emit(False)
            self.last_ai_factor = 0.0

        self.tick_mouse()
        self._print_latency_stats()