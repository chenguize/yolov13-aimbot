# test/scenarios/takeover/scenario.py
"""
AI 接管能力测试场景。

────────────────────────────────────────────────────────────────────────────
场景设计的核心问题:

  实战中，人类开始瞄准一个目标，但中途可能被 AI 接手。
  关键不是"人打完了 AI 再打"(ball_tracking 的 human_flick 模式)，
  而是"人在打的过程中 AI 就开始介入，最后平滑过渡到 AI 全控"。

  本场景专门测试这个过渡过程的质量:
    - 过渡是否平滑（速度无突变）
    - AI 反应是否及时（接管延迟）
    - 接管后精度是否达标
────────────────────────────────────────────────────────────────────────────

三阶段每目标:
  HUMAN_FLICK   (0 → t_flick)    人类 eased-out 甩枪 + 故意偏差
  TAKEOVER      (t_flick → t_lock) 人类停止，AI 接管
  AI_TRACKING   (t_lock → 击杀)    AI 全控微调，命中

________________ 接管模式 ________________
  undershoot:  人类打短 15-40%，AI 补完
  overshoot:   人类打过 5-20%，AI 拉回修正
  near_miss:   人类擦边 10-35px 误差，AI 微调入魂

________________ 与 ball_tracking 的关键差异 ________________
  - spawn 距离 200-600px（始终在 AI FOV 内）
  - HumanIntentTracker 能在人类减速期就检测到方向协作，提前介入
  - 人类故意不瞄准目标，测试 AI 的真实接管能力
"""
import math
import numpy as np

from typing import TYPE_CHECKING, Tuple, Dict, Optional
from test.scenarios.base import BaseScenario
from utils.human_intent import HumanIntentTracker

if TYPE_CHECKING:
    from test.sim_agent import SimAIAgent


# ==============================================================================
#  常量
# ==============================================================================
TOT_THRESHOLD  = 0.08       # 驻留击杀阈值 (s)
TOT_DECAY_RATE = 2.0        # 驻留衰减速率 (/s)

# 接管模式
MODE_UNDERSHOOT = "undershoot"
MODE_OVERSHOOT  = "overshoot"
MODE_NEAR_MISS  = "near_miss"
MODE_VISIBILITY = "visibility_entry"
TAKEOVER_MODES  = [MODE_UNDERSHOOT, MODE_OVERSHOOT, MODE_NEAR_MISS, MODE_VISIBILITY]


class TakeoverScenario(BaseScenario):
    """AI 接管能力测试场景。"""

    def __init__(self, max_kills: int = 25, outside_ratio: float = 0.5):
        self._max_kills = max_kills
        self._kill_count = 0
        self._outside_ratio = float(np.clip(outside_ratio, 0.0, 1.0))
        self.perception_radius_px = 128.0
        self.clock_paused = False
        self._visibility_latched = False

        # ── 目标物理 ──
        self._target_z: float = 10.0
        self._enemy_vel_3d = np.zeros(2, dtype=np.float64)
        self._target_vel_3d = np.zeros(2, dtype=np.float64)
        self._current_action: str = "adad"
        self._height_3d: float = 0.0
        self._move_timer: float = 0.5

        # ── 击杀判定 ──
        self._tot_timer: float = 0.0

        # ── 接管状态 ──
        self._takeover_mode: str = MODE_UNDERSHOOT
        self._takeover_phase: str = "idle"

        # ── 人类甩枪 ──
        self._flick_target: Optional[np.ndarray] = None    # 人类瞄准的实际位置(含偏差)
        self._flick_duration: float = 0.0
        self._flick_timer: float = 0.0
        self._flick_start_pos = np.zeros(2, dtype=np.float64)
        self._prev_flick_pos = np.zeros(2, dtype=np.float64)
        self._human_released: bool = False
        self._release_progress: float = 0.62               # 松手进度 (0~1), 0.55-0.70 保留适中残余速度
        self._last_flick_delta = np.zeros(2, dtype=np.float64)  # 最后一帧的准星位移 (用于算松手速率)
        self._human_tremor_vel = np.zeros(2, dtype=np.float64)

        # ── 接管指标 (每 kill) ──
        self._release_dist: float = 0.0          # 人类松手时距目标距离(px)
        self._release_time: float = 0.0          # 人类松手时间
        self._release_vel: float = 0.0           # 人类松手时准星速率(px/s)
        self._lock_time: float = 0.0             # AI 锁定时间
        self._takeover_duration: float = 0.0     # 接管耗时
        self._handoff_accel_peak: float = 0.0
        self._error_50ms: float = 0.0
        self._error_50ms_recorded: bool = False
        self._last_controller_velocity = np.zeros(2, dtype=np.float64)
        self._release_origin = np.zeros(2, dtype=np.float64)
        self._release_line = np.array([1.0, 0.0], dtype=np.float64)
        self._cross_track_peak: float = 0.0
        self._cross_track_rebound: float = 0.0
        self._cross_track_last: float = 0.0
        self._transverse_probe_px_s: float = 0.0

        # ── 接管指标历史 (所有 kill) ──
        self.metrics_history: list = []

    # ==========================================================================
    #  BaseScenario 接口
    # ==========================================================================

    def init(self, agent: "SimAIAgent") -> None:
        self._kill_count = 0
        self.metrics_history.clear()
        self.spawn(agent)

    def spawn(self, agent: "SimAIAgent") -> None:
        """生成目标: 近距离 (200-600px) + 人类故意偏差。"""
        if self._kill_count >= self._max_kills:
            return

        # 清理上一目标状态
        agent.world_model.track_manager.tracks.clear()

        # ── 接管模式: 均匀随机 ──
        visibility_entry = np.random.random() < self._outside_ratio
        self.clock_paused = False
        self._takeover_mode = (
            MODE_VISIBILITY if visibility_entry
            else np.random.choice(TAKEOVER_MODES[:-1])
        )
        self._visibility_latched = not visibility_entry
        if hasattr(agent.world_model.controller, "reset_target_state"):
            try:
                agent.world_model.controller.reset_target_state(mode="pure_ai")
            except TypeError:
                agent.world_model.controller.reset_target_state()

        if hasattr(agent, '_intent_tracker'):
            agent._intent_tracker = HumanIntentTracker()

        # ── 3D 投影 ──
        self._target_z = np.random.uniform(5.0, 25.0)
        scale_factor = agent.focal_length / self._target_z
        agent.target_hitbox_x = 0.15 * scale_factor
        agent.target_hitbox_y = 0.15 * scale_factor

        # Half the trials begin outside the actual 128px recognition radius.
        # The remaining trials start visible and exercise ordinary mid-flick
        # authority transfer without conflating it with target acquisition.
        spawn_radius_px = (
            np.random.uniform(260, 600)
            if visibility_entry else np.random.uniform(70, 120)
        )
        angle = np.random.uniform(0, 2 * np.pi)
        offset_x = np.cos(angle) * spawn_radius_px
        offset_y = np.sin(angle) * spawn_radius_px

        half_w = agent.screen_w / 2.0
        half_h = agent.screen_height / 2.0
        margin = 50
        offset_x = np.clip(offset_x, -half_w + margin, half_w - margin)
        offset_y = np.clip(offset_y, -half_h + margin, half_h - margin)

        agent.enemy_pos = agent.crosshair_pos + np.array([offset_x, offset_y])
        target_pos = agent.enemy_pos.copy()

        # ── 3D 物理 (轻量 adad 晃动, 增加接管难度) ──
        self._enemy_vel_3d = np.array([np.random.choice([-1, 1]) * 3.5, 0.0])
        self._target_vel_3d = self._enemy_vel_3d.copy()
        self._height_3d = 0.0
        self._current_action = "adad"
        self._move_timer = np.random.uniform(0.2, 0.5)
        self._tot_timer = 0.0

        # ── 人类甩枪参数 ──
        self._flick_duration = np.random.uniform(0.12, 0.25)  # 快速甩枪
        self._flick_timer = 0.0
        self._flick_start_pos = agent.crosshair_pos.copy()
        self._prev_flick_pos = self._flick_start_pos.copy()
        self._human_released = False
        self._takeover_phase = "HUMAN_FLICK"
        self._release_progress = np.random.uniform(0.55, 0.70)  # 松手时机: 中途松手, 保留适中残余速度
        self._last_flick_delta = np.zeros(2, dtype=np.float64)
        self._human_tremor_vel.fill(0.0)

        # 计算人类瞄准位置 (故意偏差)
        direction = target_pos - self._flick_start_pos
        dist_total = float(np.linalg.norm(direction))
        if dist_total < 1.0:
            dist_total = 1.0
        unit_dir = direction / dist_total

        if self._takeover_mode == MODE_VISIBILITY:
            # Pull only far enough to reveal the target. This camera-acquisition
            # phase uses wall time for intent dynamics but pauses score time.
            self._flick_target = target_pos - unit_dir * 96.0
            self._release_progress = 0.80
            self._flick_duration = np.random.uniform(0.16, 0.24)
            self._takeover_phase = "NO_TARGET_PULL"
            self.clock_paused = True
        elif self._takeover_mode == MODE_UNDERSHOOT:
            # 人类打到 60-85% 就停
            frac = np.random.uniform(0.60, 0.85)
            self._flick_target = self._flick_start_pos + direction * frac

        elif self._takeover_mode == MODE_OVERSHOOT:
            # 人类打过 5-20%
            frac = np.random.uniform(1.05, 1.20)
            self._flick_target = self._flick_start_pos + direction * frac

        elif self._takeover_mode == MODE_NEAR_MISS:
            # 人类基本到位，但有 10-35px 随机偏移
            noise_offset = np.random.randn(2) * 8.0
            noise_offset = np.clip(noise_offset, -35, 35)
            # 确保误差 > 10px
            if np.linalg.norm(noise_offset) < 10.0:
                noise_offset = noise_offset / max(np.linalg.norm(noise_offset), 1e-6) * 15.0
            self._flick_target = target_pos + noise_offset

        # ── 重置接管指标 ──
        self._release_dist = 0.0
        self._release_time = 0.0
        self._release_vel = 0.0
        self._lock_time = 0.0
        self._takeover_duration = 0.0
        self._handoff_accel_peak = 0.0
        self._error_50ms = 0.0
        self._error_50ms_recorded = False
        self._last_controller_velocity.fill(0.0)
        self._release_origin.fill(0.0)
        self._release_line[:] = (1.0, 0.0)
        self._cross_track_peak = 0.0
        self._cross_track_rebound = 0.0
        self._cross_track_last = 0.0
        self._transverse_probe_px_s = float(
            np.random.choice([-1.0, 1.0]) * np.random.uniform(180.0, 320.0)
        )
        agent.takeover_release_time = 0.0   # 每 kill 独立, 防止跨 kill 泄露

        agent.chase_mode = "human_flick"
        agent.target_first_seen_time = 0.0
        agent.stream_latency_model.reset()

    def tick_physics(self, agent: "SimAIAgent", dt: float) -> None:
        """轻量 adad 晃动 + 击杀检测。"""
        # 身法: 只保留 adad 小幅晃动和偶尔 stop
        if self._move_timer <= 0:
            self._move_timer = np.random.uniform(0.2, 0.5)
            action = np.random.choice(
                ["adad", "adad", "adad", "stop"],
                p=[0.35, 0.35, 0.15, 0.15],
            )
            self._current_action = action
            if action == "adad":
                self._target_vel_3d[0] = np.random.choice([-1, 1]) * 3.5
            elif action == "stop":
                self._target_vel_3d[0] = 0.0

        self._move_timer -= dt

        # 水平引擎
        accel_x = (self._target_vel_3d[0] - self._enemy_vel_3d[0]) * 25.0
        self._enemy_vel_3d[0] += accel_x * dt

        # 垂直 (简化: 不跳)
        self._enemy_vel_3d[1] = 0.0

        # 投影
        scale = agent.focal_length / self._target_z
        agent.enemy_vel[0] = self._enemy_vel_3d[0] * scale
        agent.enemy_vel[1] = self._enemy_vel_3d[1] * scale
        agent.enemy_pos += agent.enemy_vel * dt

        # 当前误差
        err_x = abs(agent.enemy_pos[0] - agent.crosshair_pos[0])
        err_y = abs(agent.enemy_pos[1] - agent.crosshair_pos[1])
        cur_dist = float(np.hypot(err_x, err_y))

        if self._release_time > 0.0:
            elapsed = agent.sim_time - self._release_time
            if elapsed <= 0.150:
                travel = agent.crosshair_pos - self._release_origin
                cross_track = float(
                    self._release_line[0] * travel[1]
                    - self._release_line[1] * travel[0]
                )
                cross_abs = abs(cross_track)
                self._cross_track_peak = max(self._cross_track_peak, cross_abs)
                self._cross_track_rebound = max(
                    self._cross_track_rebound,
                    self._cross_track_peak - cross_abs,
                )
                self._cross_track_last = cross_track
            controller_velocity = np.asarray(
                getattr(agent.world_model.controller, 'crosshair_velocity', (0.0, 0.0)),
                dtype=np.float64,
            )
            if elapsed <= 0.100:
                accel = float(np.linalg.norm(
                    controller_velocity - self._last_controller_velocity
                )) / max(dt, 1e-6)
                self._handoff_accel_peak = max(self._handoff_accel_peak, accel)
            if elapsed >= 0.050 and not self._error_50ms_recorded:
                self._error_50ms = cur_dist
                self._error_50ms_recorded = True
            self._last_controller_velocity[:] = controller_velocity

        # 击杀检测
        if err_x <= agent.target_hitbox_x and err_y <= agent.target_hitbox_y:
            self._tot_timer += dt
            if self._tot_timer >= TOT_THRESHOLD:
                self._finalize_kill(agent, cur_dist)
                return
        else:
            self._tot_timer = max(0.0, self._tot_timer - dt * TOT_DECAY_RATE)

        # 检测 AI 是否已锁定 (误差 < hitbox 视为进入 AI_TRACKING)
        if (self._takeover_phase == "TAKEOVER"
                and self._lock_time == 0.0
                and err_x <= agent.target_hitbox_x
                and err_y <= agent.target_hitbox_y):
            self._lock_time = agent.sim_time
            self._takeover_duration = self._lock_time - self._release_time
            self._takeover_phase = "AI_TRACKING"

    def _finalize_kill(self, agent: "SimAIAgent", final_dist: float):
        """记录接管指标并生成下一目标。"""
        self._kill_count += 1
        agent.kill_count = self._kill_count

        # 如果没记录到锁定时间(直接命中), 用当前时间
        if self._lock_time == 0.0:
            self._lock_time = agent.sim_time
            if self._release_time > 0:
                self._takeover_duration = self._lock_time - self._release_time

        self.metrics_history.append({
            "kill_id":          self._kill_count,
            "mode":             self._takeover_mode,
            "release_dist":     self._release_dist,
            "release_vel":      self._release_vel,
            "release_progress": self._release_progress,
            "takeover_dur":     self._takeover_duration,
            "handoff_accel_peak": self._handoff_accel_peak,
            "error_50ms":       self._error_50ms,
            "cross_track_peak": self._cross_track_peak,
            "cross_track_rebound": self._cross_track_rebound,
            "checkmark_violation": (
                self._cross_track_peak > 10.0
                and self._cross_track_rebound > 8.0
            ),
            "transverse_probe_px_s": self._transverse_probe_px_s,
            "visibility_entry": self._takeover_mode == MODE_VISIBILITY,
            "final_dist":       final_dist,
            "phase":            self._takeover_phase,
        })

        self.spawn(agent)

    def tick_human_input(self, agent: "SimAIAgent", dt: float) -> Tuple[float, float]:
        """
        人类甩枪: 向 _flick_target 做 eased-out 曲线，带震颤噪声。
        在 55-70% 进度时提前松手，保留适中残余速度 (~800-2000 px/s) 让 AI 接管。
        """
        # 已松手 → 返回零
        if self._human_released:
            return (0.0, 0.0)

        release_time = self._release_progress * self._flick_duration

        # 甩枪未结束（且未到松手点）
        if self._flick_timer < release_time and self._flick_target is not None:
            self._flick_timer += dt

            t = min(self._flick_timer / self._flick_duration, 1.0)
            eased = 1.0 - (1.0 - t) ** 3   # eased-out 曲线

            cur_pos = (self._flick_start_pos
                       + (self._flick_target - self._flick_start_pos) * eased)

            delta = cur_pos - self._prev_flick_pos
            self._prev_flick_pos = cur_pos.copy()

            # 记录最后一帧准星位移（用于计算松手速率）
            self._last_flick_delta = delta.copy()

            # Velocity-domain OU tremor. Position-domain white noise at 500 Hz
            # creates impossible multi-thousand-pixel/s jumps; integrating a
            # correlated velocity process keeps the motion bandwidth finite.
            speed = float(np.linalg.norm(delta)) / max(dt, 1e-6)
            theta = 35.0
            sigma = float(np.clip(250.0 + 0.08 * speed, 250.0, 550.0))
            self._human_tremor_vel += (
                -theta * self._human_tremor_vel * dt
                + sigma * math.sqrt(dt) * np.random.normal(0.0, 1.0, 2)
            )
            noise = self._human_tremor_vel * dt

            result = delta + noise
            return (float(result[0]), float(result[1]))

        # A moving target may invalidate the endpoint sampled at spawn. Keep
        # the unscored camera pull active until the live target truly enters
        # the recognition radius; never release against a stale endpoint.
        if self._takeover_mode == MODE_VISIBILITY and not self._visibility_latched:
            live_error = agent.enemy_pos - agent.crosshair_pos
            live_dist = float(np.linalg.norm(live_error))
            if live_dist > 1e-6:
                pull_speed = 1200.0
                travel = min(max(live_dist - 112.0, 0.0), pull_speed * dt)
                delta = live_error * (travel / live_dist)
                self._last_flick_delta = delta.copy()
                return float(delta[0]), float(delta[1])

        # 甩枪到松手点 → 松手，AI 接管
        self._human_released = True
        self._takeover_phase = "TAKEOVER"
        self.clock_paused = False
        self._prev_flick_active = False

        # 记录松手时刻指标
        if hasattr(agent, 'enemy_pos') and hasattr(agent, 'crosshair_pos'):
            self._release_dist = float(
                np.linalg.norm(agent.enemy_pos - agent.crosshair_pos)
            )
            self._release_time = max(agent.sim_time, 1e-6)
            agent.takeover_release_time = self._release_time  # 告知评分系统: AI 时钟从此开始
            self._release_origin[:] = agent.crosshair_pos
            release_error = agent.enemy_pos - agent.crosshair_pos
            release_error_norm = float(np.linalg.norm(release_error))
            if release_error_norm > 1e-6:
                self._release_line[:] = release_error / release_error_norm
            # 准星速率 = 最后一帧位移 / 帧间隔 (px/s)
            self._release_vel = float(
                np.linalg.norm(self._last_flick_delta) / max(dt, 1e-6)
            )
            bbox_w = agent.ctx.targets[0].w if agent.ctx.targets else 60.0
            strategy = agent.world_model.strategy
            try:
                hvx, hvy = strategy.calculate_velocity_move(
                    self._last_flick_delta[0] / max(dt, 1e-6),
                    self._last_flick_delta[1] / max(dt, 1e-6),
                    px_x=0.0, px_y=0.0, bbox_w=bbox_w,
                )
            except TypeError:
                hvx, hvy = strategy.calculate_velocity_move(
                    self._last_flick_delta[0] / max(dt, 1e-6),
                    self._last_flick_delta[1] / max(dt, 1e-6),
                    bbox_w=bbox_w,
                )
            self._last_controller_velocity[:] = (hvx, hvy)
            ctrl = agent.world_model.controller
            if hasattr(ctrl, 'begin_handoff'):
                tangent = np.array(
                    [-self._release_line[1], self._release_line[0]],
                    dtype=np.float64,
                )
                probe_velocity = (
                    agent.enemy_vel
                    + tangent * self._transverse_probe_px_s
                )
                tvx, tvy = strategy.calculate_velocity_move(
                    float(probe_velocity[0]), float(probe_velocity[1]),
                    px_x=0.0, px_y=0.0, bbox_w=bbox_w,
                )
                error_px = agent.enemy_pos - agent.crosshair_pos
                ex, ey = strategy.calculate_mouse_move(
                    float(error_px[0]), float(error_px[1]), bbox_w=bbox_w,
                )
                ctrl.begin_handoff(
                    human_velocity=np.array([hvx, hvy], dtype=np.float64),
                    target_velocity=np.array([tvx, tvy], dtype=np.float64),
                    error=np.array([ex, ey], dtype=np.float64),
                    reason='human_release',
                )

        if hasattr(agent, '_prev_flick_active'):
            agent._prev_flick_active = False
        return (0.0, 0.0)

    def check_kill(self, agent: "SimAIAgent", dt: float) -> bool:
        """击杀判定已集成在 tick_physics 中。"""
        return False

    def is_target_visible(self, distance_px: float) -> bool:
        """Recognition hysteresis: enter at 128px, tolerate acquisition motion."""
        if not self._visibility_latched and distance_px <= self.perception_radius_px:
            self._visibility_latched = True
        # This scenario isolates first-entry handoff. Once acquired, keep the
        # track observable until kill; repeated FOV exits belong to the separate
        # coast/occlusion scenarios.
        return self._visibility_latched

    # ==========================================================================
    #  状态查询
    # ==========================================================================

    @property
    def is_done(self) -> bool:
        return self._kill_count >= self._max_kills

    @property
    def chase_mode(self) -> str:
        return "human_flick"

    @property
    def takeover_phase(self) -> str:
        return self._takeover_phase

    @property
    def current_takeover_mode(self) -> str:
        return self._takeover_mode

    # ==========================================================================
    #  接管报告
    # ==========================================================================

    def print_takeover_report(self):
        """打印接管能力汇总报告。"""
        if not self.metrics_history:
            print("[接管测试] 无数据。")
            return

        arr = self.metrics_history

        def _m(key):
            vals = [m[key] for m in arr if m.get(key, 0) > 0]
            return np.mean(vals) if vals else 0.0

        def _shape_stats(key):
            vals = [float(m.get(key, 0.0)) for m in arr]
            return np.mean(vals), np.percentile(vals, 95), max(vals)

        by_mode = {}
        for m in arr:
            by_mode.setdefault(m["mode"], []).append(m)

        print("\n" + "=" * 78)
        print("  AI 接管能力测试报告")
        print("=" * 78)
        print(f"  总击杀: {len(arr)}")
        print(f"  平均松手进度: {_m('release_progress'):.1%} (1.0=甩完才松, <1.0=中途松手)")
        print(f"  平均接管距离: {_m('release_dist'):.1f} px")
        print(f"  平均接管耗时: {_m('takeover_dur'):.4f} s")
        print(f"  平均松手速率: {_m('release_vel'):.0f} px/s (准星甩枪速度)")
        print(f"  最终精度:     {_m('final_dist'):.2f} px")
        peak_mean, peak_p95, peak_max = _shape_stats("cross_track_peak")
        rebound_mean, rebound_p95, rebound_max = _shape_stats("cross_track_rebound")
        violations = sum(bool(m.get("checkmark_violation", False)) for m in arr)
        print(
            f"  横向峰值150ms: mean={peak_mean:.2f}px P95={peak_p95:.2f}px "
            f"max={peak_max:.2f}px"
        )
        print(
            f"  √回折量150ms: mean={rebound_mean:.2f}px P95={rebound_p95:.2f}px "
            f"max={rebound_max:.2f}px | violations={violations}/{len(arr)}"
        )
        print("-" * 78)

        for mode in TAKEOVER_MODES:
            items = by_mode.get(mode, [])
            if not items:
                continue
            label = {"undershoot": "打短补完", "overshoot": "打过头拉回", "near_miss": "擦边微调"}
            print(f"  [{label.get(mode, mode)}] ({len(items)} kills)")
            durs = [m["takeover_dur"] for m in items if m.get("takeover_dur", 0) > 0]
            dists = [m["release_dist"] for m in items]
            vels = [m["release_vel"] for m in items if m.get("release_vel", 0) > 0]
            print(f"    接管耗时: {np.mean(durs):.4f}s  (max {max(durs):.4f}s)" if durs
                  else "    接管耗时: N/A")
            print(f"    接管距离: {np.mean(dists):.1f}px  (范围 {min(dists):.0f}-{max(dists):.0f})")
            if vels:
                print(f"    松手速率: {np.mean(vels):.0f}px/s (范围 {min(vels):.0f}-{max(vels):.0f})")
        print("=" * 78)
