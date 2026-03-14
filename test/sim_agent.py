# sim_agent.py
import time
import numpy as np
import random
import math
from config import config
# [必须安装] pip install noise (与实机保持一致)
try:
    import noise
except ImportError:
    print("⚠️ 警告: 未安装 noise 库，仿真中的 Perlin 噪点将失效，请使用 pip install noise 安装")

from perception.ring_buffer import RingBuffer
from utils.types import Detection, InferenceContext


class SimAIAgent:
    def __init__(self, noise_level: float = 5.0, dropout_rate: float = 0.08, base_delay: float = 0.015):
        self.base_delay = base_delay
        self.sim_time = 0.0
        self._original_perf_counter = time.perf_counter
        time.perf_counter = lambda: self.sim_time

        try:
            from world_model import WorldModel
        except ImportError:
            pass

        self.world_model = WorldModel()
        self.ring_buffer = RingBuffer()

        self.sim_steps = 0
        self.noise_level = noise_level
        self.dropout_rate = dropout_rate

        self.crosshair_pos = np.zeros(2, dtype=np.float64)
        self.center = np.array([self.world_model.crop_center, self.world_model.crop_center], dtype=np.float64)

        self.ctx = InferenceContext()
        self.ctx.center_pos = self.center

        self.pipeline_delay = 0.0
        self.burst_state = 0

        # ==================== 实战靶场状态机 ====================
        self.max_kills = 30
        self.kill_count = 0
        self.is_done = False
        self.current_target_id = 0

        # 🌟 3D 摄像机投影常量 (Valorant FOV = 103)
        self.fov = 103.0
        self.screen_w = config.getfloat("General", "screen_width", 1920.0)
        self.screen_height=config.getfloat("General", "screen_height", 1080.0)
        # 焦距计算公式: F = (W / 2) / tan(FOV / 2)
        self.focal_length = (self.screen_w / 2.0) / math.tan(math.radians(self.fov / 2.0))  # 约 763.5 px

        # 🌟 3D 物理空间变量
        self.target_z = 10.0  # 离玩家的距离 (米)
        self.enemy_vel_3d = np.zeros(2, dtype=np.float64)  # 3D空间速度 (m/s) [0]:水平, [1]:垂直
        self.target_vel_3d = np.zeros(2, dtype=np.float64)
        self.current_action = 'stop'
        self.height_3d = 0.0  # 离地高度 (处理跳跃抛物线)

        self.enemy_pos = np.zeros(2, dtype=np.float64)
        self.enemy_vel = np.zeros(2, dtype=np.float64)  # 最终映射到屏幕的 2D 像素速度

        self.tot_threshold = 0.08
        self.target_first_seen_time = 0.0
        self.noise_offset_x = random.uniform(0, 1000.0)
        self.noise_offset_y = random.uniform(0, 1000.0)

        # FIX-3：记录上一帧的 flick 活跃状态，用于检测 flick 结束边沿
        self._prev_flick_active = False

        self.spawn_new_target()

    def spawn_new_target(self):
        """生成新目标：初始化 3D 距离与霓虹物理参数"""
        if self.kill_count >= self.max_kills:
            self.is_done = True
            self.ctx.targets = []
            return

        self.current_target_id += 1
        self.world_model.targets.clear()
        if hasattr(self.world_model.controller, 'reset_target_state'):
            self.world_model.controller.reset_target_state()

        self.chase_mode = np.random.choice(['pure_ai', 'human_flick'])

        # 随机生成 3D 距离 (5米贴脸 ~ 35米大长枪距离)
        self.target_z = np.random.uniform(5.0, 35.0)

        # 计算该距离下的基础屏幕像素视野半径（原有逻辑）
        scale_factor = self.focal_length / self.target_z
        base_radius_px = 3.0 * scale_factor  # 原有 3.0 倍

        # ── 根据模式决定生成半径（像素单位） ───────────────────────────────
        if self.chase_mode == 'pure_ai':
            # 纯AI：附近小范围，初始误差通常 100~500 px
            spawn_radius_px = base_radius_px * np.random.uniform(0.8, 1.8)
        else:
            # 人机flick：故意拉远，但严格限制在屏幕内
            # 目标初始距离 600~1400 px 左右（黄金玩家大甩枪常见范围）
            spawn_radius_px = np.random.uniform(600, 1400)  # 直接用像素范围，更直观

        # 随机角度
        angle = np.random.uniform(0, 2 * np.pi)

        # 计算相对准星的偏移
        offset_x = np.cos(angle) * spawn_radius_px
        offset_y = np.sin(angle) * spawn_radius_px

        # ── 强制限制在屏幕边界内（以中心为原点） ───────────────────────────────
        half_w = self.screen_w / 2.0  # 960
        half_h = self.screen_height / 2.0  # 540（1080p）

        # 留一点边距，避免正好贴边导致视觉奇怪
        margin = 50
        offset_x = np.clip(offset_x, -half_w + margin, half_w - margin)
        offset_y = np.clip(offset_y, -half_h + margin, half_h - margin)

        # 应用到 enemy_pos（相对中心）
        self.enemy_pos = self.crosshair_pos + np.array([offset_x, offset_y])

        # 打印初始距离，便于调试
        initial_dist = np.linalg.norm(self.enemy_pos - self.crosshair_pos)
        print(
            f"New target spawned | mode: {self.chase_mode} | z: {self.target_z:.1f}m | initial_dist: {initial_dist:.0f} px")

        # 初始化 3D 速度 (模拟霓虹初始奔跑 8.5 m/s)
        self.enemy_vel_3d = np.array([np.random.choice([-1, 1]) * 8.5, 0.0])
        self.target_vel_3d = self.enemy_vel_3d.copy()
        self.height_3d = 0.0
        self.current_action = 'sprint'

        self.tot_timer = 0.0
        self.target_first_seen_time = 0.0
        self.move_timer = 0.5

        if self.chase_mode == 'human_flick':
            error_offset = np.random.randn(2) * 20.0
            self.flick_target = self.enemy_pos + error_offset
            self.flick_duration = np.random.uniform(0.15, 0.22)
            self.flick_timer = 0.0
            self.flick_start_pos = self.crosshair_pos.copy()

        # FIX-3：新目标出现时重置 flick 边沿检测状态
        self._prev_flick_active = False

    def tick_enemy(self, dt: float):
        if self.is_done: return

        # ==========================================
        # 🏃 核心增强：霓虹 (Neon) 3D 物理与身法模型
        # ==========================================
        if getattr(self, 'move_timer', 0) <= 0:
            self.move_timer = np.random.uniform(0.15, 0.4)
            # 动作权重: 30%大招狂奔, 20%滑铲, 30%步枪ADAD, 15%跳跃, 5%急停
            action = np.random.choice(
                ['sprint', 'slide', 'adad', 'jump', 'stop'],
                p=[0.30, 0.20, 0.30, 0.15, 0.05]
            )
            self.current_action = action

            if action == 'sprint':
                # 大招狂奔，极限移速 8.5 m/s
                self.target_vel_3d[0] = np.random.choice([-1, 1]) * 8.5
            elif action == 'adad':
                # 步枪对枪拉扯 5.4 m/s
                self.target_vel_3d[0] = np.random.choice([-1, 1]) * 5.4
            elif action == 'slide':
                # 霓虹滑铲 (极高初速，带摩擦力)
                dir_x = np.sign(self.enemy_vel_3d[0]) if self.enemy_vel_3d[0] != 0 else np.random.choice([-1, 1])
                self.enemy_vel_3d[0] = dir_x * 14.0  # 瞬间爆发 14m/s (超过常规速度)
                self.target_vel_3d[0] = 0.0  # 目标速度为0，摩擦力会拉停
                self.move_timer = 0.6  # 滑铲动画持续久一点
            elif action == 'jump':
                if self.height_3d <= 0.01:  # 只有在地上才能起跳
                    self.enemy_vel_3d[1] = -5.8  # 向上爆发 (屏幕Y轴向上为负)
            elif action == 'stop':
                self.target_vel_3d[0] = 0.0

        self.move_timer -= dt

        # ----- 1. 3D 水平引擎 (模拟极其狂暴的角色加速度) -----
        if self.current_action == 'slide':
            # 滑铲带有地面阻尼
            accel_x = (self.target_vel_3d[0] - self.enemy_vel_3d[0]) * 5.0
        else:
            # Valorant中原生提速几乎是瞬发的，给予 40m/s^2 的夸张加速度
            accel_x = (self.target_vel_3d[0] - self.enemy_vel_3d[0]) * 40.0

        self.enemy_vel_3d[0] += accel_x * dt

        # ----- 2. 3D 垂直抛物线引擎 (重力系统) -----
        if self.height_3d > 0.0 or self.enemy_vel_3d[1] < 0:
            self.enemy_vel_3d[1] += 16.0 * dt  # 瓦罗兰特重力极大 (~16m/s^2)
            self.height_3d -= self.enemy_vel_3d[1] * dt  # 高度积分 (Y速度为负代表上升)

            if self.height_3d <= 0.0:  # 落地重置
                self.height_3d = 0.0
                self.enemy_vel_3d[1] = 0.0

        # ----- 3. 3D 到 2D 摄像机投影 (核心视角转换) -----
        scale_factor = self.focal_length / self.target_z

        # 屏幕上的真实像素速度 = 3D速度 * (焦距 / 距离)
        self.enemy_vel[0] = self.enemy_vel_3d[0] * scale_factor
        self.enemy_vel[1] = self.enemy_vel_3d[1] * scale_factor

        # 积分得到屏幕坐标
        self.enemy_pos += self.enemy_vel * dt

        # ==========================================
        # 动态判定体积 (Hitbox 根据 3D 距离热胀冷缩)
        # 设定头部有效爆头半径为 0.15 米
        # ==========================================
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
                    f"🎯 击杀 {self.kill_count:2d}/30! [{mode_str}] | 距离: {self.target_z:.1f}m | 动作: {self.current_action}")
                self.spawn_new_target()
        else:
            self.tot_timer = max(0.0, self.tot_timer - dt * 2.0)

    def tick_cv(self):
        """带有真实 3D 尺寸计算的机器视觉"""
        if self.is_done: return

        relative_pos = self.enemy_pos - self.crosshair_pos
        dist_px = np.linalg.norm(relative_pos)

        # 超过物理视野盲区
        if dist_px > 1000.0:
            self.ctx.targets = []
            return

        if self.burst_state > 0 or np.random.random() < self.dropout_rate:
            self.burst_state = max(0, self.burst_state - 1) if self.burst_state > 0 else np.random.randint(2, 6)
            self.ctx.targets = []
            return

        if self.target_first_seen_time == 0.0:
            self.target_first_seen_time = self.sim_time

        screen_x = relative_pos[0] + self.center[0]
        screen_y = relative_pos[1] + self.center[1]

        # ==========================================
        # 动态 YOLO 框尺寸 (根据距离)
        # ==========================================
        scale_factor = self.focal_length / self.target_z
        box_w = 0.45 * scale_factor  # 真实肩宽大概 0.45 米

        # 🌟 细节：霓虹滑铲时，身体高度被急剧压缩！
        height_3d = 0.8 if self.current_action == 'slide' else 1.7
        box_h = height_3d * scale_factor

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

    def tick_mouse(self):
        mx, my = self.world_model.controller.tick_mouse()
        px, py = self.world_model.strategy.reverse_map(float(mx), float(my), bbox_w=60.0)
        self.crosshair_pos[0] += px
        self.crosshair_pos[1] += py

        self.ring_buffer.add_event(mx, my, is_ai=True)
        self.ring_buffer.add_event(mx, my, is_ai=False)

    def step(self):
        dt = 0.002
        self.sim_time += dt
        self.sim_steps += 1

        proc_delay = max(0.002,
                         np.random.uniform(self.base_delay - 0.005, self.base_delay + 0.015) + np.random.exponential(
                             0.008))
        t_capture = self.sim_time - proc_delay

        self.tick_enemy(dt)
        self.tick_cv()

        dets = np.empty((0, 6), dtype=np.float32)
        if self.ctx.targets:
            det = self.ctx.targets[0]
            # 🌟 修复：将动态 YOLO 边框传入模型，取代硬编码的 30/60
            dets = np.array([[det.xyxy[0], det.xyxy[1], det.xyxy[2], det.xyxy[3], det.conf, 0.0]], dtype=np.float32)

        self.world_model.update_detections(
            detections=dets, frame_id=self.sim_steps, t_capture=t_capture, t_done=self.sim_time
        )
        self.world_model.step(self.ctx, self.ring_buffer)

        # --- 人类物理输入注入 ---
        if not hasattr(self, '_human_subpixel'):
            self._human_subpixel = np.zeros(2, dtype=np.float64)

        # FIX-3：在执行人类 flick 前，记录本帧 flick 是否活跃
        cur_flick_active = (
            not self.is_done
            and self.chase_mode == 'human_flick'
            and getattr(self, 'flick_timer', 999) < getattr(self, 'flick_duration', 0)
        )

        if cur_flick_active:
            progress = self.flick_timer / self.flick_duration
            next_progress = min(1.0, (self.flick_timer + dt) / self.flick_duration)

            def ease_out(t): return 1 - (1 - t) ** 3

            curr_pos = self.flick_start_pos + (self.flick_target - self.flick_start_pos) * ease_out(progress)
            next_pos = self.flick_start_pos + (self.flick_target - self.flick_start_pos) * ease_out(next_progress)

            dx_px, dy_px = next_pos[0] - curr_pos[0], next_pos[1] - curr_pos[1]
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

        # FIX-3：检测 flick 结束的下降沿（上帧活跃，本帧不活跃）
        # 此时通知控制器清除人手惯性速度，防止 AI 接管时从高速状态硬切入振荡
        if self._prev_flick_active and not cur_flick_active:
            if hasattr(self.world_model.controller, 'notify_flick_end'):
                self.world_model.controller.notify_flick_end()

        self._prev_flick_active = cur_flick_active

        # --- 控制器联合发力与预测 ---
        if self.ctx.p_predict is not None and self.ctx.v_real is not None:
            dx_h_inst, dy_h_inst = self.ring_buffer.get_pure_human_delta_sum(self.sim_time - dt, self.sim_time)
            human_vx_inst = dx_h_inst / dt if dt > 0 else 0.0
            human_vy_inst = dy_h_inst / dt if dt > 0 else 0.0

            bbox_w = self.ctx.targets[0].w if self.ctx.targets else 60.0

            try:
                noise_scale = 0.5
                amplitude = 3.0
                drift_x = noise.pnoise1(self.sim_time * noise_scale + self.noise_offset_x) * amplitude
                drift_y = noise.pnoise1(self.sim_time * noise_scale + self.noise_offset_y) * amplitude
            except NameError:
                drift_x, drift_y = 0.0, 0.0

            drifted_p_x = self.ctx.p_predict[0] + drift_x
            drifted_p_y = self.ctx.p_predict[1] + drift_y

            intent_x, intent_y = self.world_model.strategy.calculate_mouse_move(drifted_p_x, drifted_p_y, bbox_w=bbox_w)

            v_real_pixels = self.ctx.v_real
            intent_vx, intent_vy = self.world_model.strategy.calculate_velocity_move(v_real_pixels[0], v_real_pixels[1],
                                                                                     bbox_w=bbox_w)

            a_real_pixels = getattr(self.ctx, 'a_real', (0.0, 0.0))
            intent_ax, intent_ay = self.world_model.strategy.calculate_velocity_move(a_real_pixels[0], a_real_pixels[1],
                                                                                     bbox_w=bbox_w)

            pixel_error_dist = np.linalg.norm([drifted_p_x, drifted_p_y])
            spatial_factor = np.clip(1.0 - (pixel_error_dist / 800.0) ** 2, 0.1, 1.0)

            time_since_seen = self.sim_time - self.target_first_seen_time
            reaction_factor = np.clip(time_since_seen / 0.15, 0.0, 1.0)

            dx_h_recent, dy_h_recent = self.ring_buffer.get_pure_human_delta_sum(self.sim_time - 0.1, self.sim_time)
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

            self.world_model.controller.compute(
                target_x=intent_x, target_y=intent_y, dt=dt,
                human_v=np.array([human_vx_inst, human_vy_inst]),
                v_real=np.array([intent_vx, intent_vy]),
                a_real=np.array([intent_ax, intent_ay]),
                power_factor=power_factor,
                bbox_w=bbox_w
            )
            self.last_ai_factor = power_factor
        else:
            if hasattr(self.world_model.controller, 'reset_target_state'):
                self.world_model.controller.reset_target_state()
            self.last_ai_factor = 0.0

        self.tick_mouse()