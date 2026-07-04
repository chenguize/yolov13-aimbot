# test/scenarios/directional_change/scenario.py
"""
动目标变向场景 — 每 0.3-0.8s 突然变向 90-180 度。

特性：
  - 目标以恒定 3D 速度运动 (3-8 m/s)
  - 每 0.3-0.8s 突然变向 90-180 度 (随机左/右)
  - 变向瞬间速度向量突变 → 加速度脉冲
  - 测试 IMM-Kalman CV / CA / CT 三模型的切换响应
  - 3D 距离 5-25m, 生成距离 100-400px
  - 通过 agent.enemy_list 持续生成检测 (单目标)
"""

import math
import numpy as np

from typing import TYPE_CHECKING, Tuple, Optional
from test.scenarios.base import BaseScenario

if TYPE_CHECKING:
    from test.sim_agent import SimAIAgent


# ==============================================================================
#  命中判定常量
# ==============================================================================
TOT_THRESHOLD  = 0.08
TOT_DECAY_RATE = 2.0


class DirectionalChangeScenario(BaseScenario):
    """动目标变向场景 — 测试 IMM CV/CA/CT 模型切换。"""

    def __init__(
        self,
        max_kills: int = 15,
        seed: Optional[int] = None,
        dist_range: Tuple[float, float] = (5.0, 25.0),
        speed_range: Tuple[float, float] = (3.0, 8.0),
        change_interval_range: Tuple[float, float] = (0.3, 0.8),
        change_angle_range: Tuple[float, float] = (90.0, 180.0),
        spawn_dist_px_range: Tuple[float, float] = (100.0, 400.0),
    ):
        """
        :param max_kills: 总击杀数到达即结束
        :param seed: 随机种子
        :param dist_range: 目标 3D 距离 z 范围 (m)
        :param speed_range: 目标 3D 速度范围 (m/s, 恒定)
        :param change_interval_range: 变向间隔 (s)
        :param change_angle_range: 变向角度范围 (度)
        :param spawn_dist_px_range: 生成距离范围 (px, 距准星)
        """
        self._max_kills = max_kills
        self._rng = np.random.RandomState(seed)
        self._dist_range = dist_range
        self._speed_range = speed_range
        self._change_interval_range = change_interval_range
        self._change_angle_range = change_angle_range
        self._spawn_dist_px_range = spawn_dist_px_range

        self._kill_count = 0

        # 目标物理 (2D 平面运动, 用 3D 速度 + z 投影)
        self._pos = np.zeros(2, dtype=np.float64)
        self._vel = np.zeros(2, dtype=np.float64)
        self._z = 10.0
        self._speed_3d = 5.0          # 恒定 3D 速度 (m/s)
        self._angle = 0.0             # 当前运动方向 (rad, 屏幕坐标系)
        self._vel_3d = np.zeros(2, dtype=np.float64)   # (vx_3d, vy_3d) m/s

        # 变向计时
        self._change_timer = 0.5

        # 击杀
        self._tot_timer = 0.0

    # ==========================================================================
    #  BaseScenario 接口
    # ==========================================================================

    def init(self, agent: "SimAIAgent") -> None:
        self._kill_count = 0
        self.spawn(agent)

    def spawn(self, agent: "SimAIAgent") -> None:
        """生成新目标：3D 距离 / 恒定速度 / 初始方向。"""
        if self._kill_count >= self._max_kills:
            return

        agent.world_model.track_manager.tracks.clear()

        if hasattr(agent.world_model.controller, 'reset_target_state'):
            try:
                agent.world_model.controller.reset_target_state(mode="pure_ai")
            except TypeError:
                agent.world_model.controller.reset_target_state()

        # 3D 距离 + 命中框
        self._z = float(self._rng.uniform(*self._dist_range))
        scale_factor = agent.focal_length / self._z
        agent.target_hitbox_x = 0.15 * scale_factor
        agent.target_hitbox_y = 0.15 * scale_factor

        # 生成位置 (100-400 px 距准星)
        spawn_dist = float(self._rng.uniform(*self._spawn_dist_px_range))
        angle = float(self._rng.uniform(0, 2 * np.pi))
        half_w = agent.screen_w / 2.0
        half_h = agent.screen_height / 2.0
        margin = 50
        ox = float(np.clip(np.cos(angle) * spawn_dist, -half_w + margin, half_w - margin))
        oy = float(np.clip(np.sin(angle) * spawn_dist, -half_h + margin, half_h - margin))
        self._pos = agent.crosshair_pos + np.array([ox, oy], dtype=np.float64)

        # 恒定 3D 速度 + 随机初始方向
        self._speed_3d = float(self._rng.uniform(*self._speed_range))
        self._angle = float(self._rng.uniform(0, 2 * np.pi))
        self._vel_3d = np.array(
            [np.cos(self._angle) * self._speed_3d, np.sin(self._angle) * self._speed_3d],
            dtype=np.float64,
        )

        # 投影到 2D
        scale = agent.focal_length / max(self._z, 1.0)
        self._vel = self._vel_3d * scale

        # 变向计时
        self._change_timer = float(self._rng.uniform(*self._change_interval_range))
        self._tot_timer = 0.0

        agent.chase_mode = "pure_ai"
        agent.target_first_seen_time = 0.0
        agent.enemy_pos = self._pos.copy()
        agent.enemy_vel = self._vel.copy()
        agent.stream_latency_model.reset()

    def tick_physics(self, agent: "SimAIAgent", dt: float) -> None:
        """变向状态机 + 恒速运动 + 击杀。"""
        # ── 变向计时 ──
        self._change_timer -= dt
        if self._change_timer <= 0:
            # 突然变向 90-180 度 (随机左/右)
            delta_deg = float(self._rng.uniform(*self._change_angle_range))
            delta_rad = math.radians(delta_deg) * int(self._rng.choice([-1, 1]))
            self._angle += delta_rad
            # 瞬时变向：速度向量直接跳变 (加速度脉冲, 测试 IMM CA/CT)
            self._vel_3d = np.array(
                [np.cos(self._angle) * self._speed_3d,
                 np.sin(self._angle) * self._speed_3d],
                dtype=np.float64,
            )
            self._change_timer = float(self._rng.uniform(*self._change_interval_range))

        # ── 恒速运动 (CV 段, 测试 IMM CV 模型) ──
        scale = agent.focal_length / max(self._z, 1.0)
        self._vel = self._vel_3d * scale
        self._pos += self._vel * dt

        # 屏幕边界软约束 (反射方向)
        half_w = agent.screen_w / 2.0
        half_h = agent.screen_height / 2.0
        cx = agent.crosshair_pos[0]
        cy = agent.crosshair_pos[1]
        bounced = False
        if self._pos[0] < cx - half_w + 50:
            self._pos[0] = cx - half_w + 50
            self._angle = np.pi - self._angle
            bounced = True
        elif self._pos[0] > cx + half_w - 50:
            self._pos[0] = cx + half_w - 50
            self._angle = np.pi - self._angle
            bounced = True
        if self._pos[1] < cy - half_h + 50:
            self._pos[1] = cy - half_h + 50
            self._angle = -self._angle
            bounced = True
        elif self._pos[1] > cy + half_h - 50:
            self._pos[1] = cy + half_h - 50
            self._angle = -self._angle
            bounced = True
        if bounced:
            self._vel_3d = np.array(
                [np.cos(self._angle) * self._speed_3d,
                 np.sin(self._angle) * self._speed_3d],
                dtype=np.float64,
            )
            self._vel = self._vel_3d * scale

        # ── 检测生成 (单目标持续可见) ──
        agent.enemy_list = [(self._pos, self._vel, self._z)]

        # 同步 agent 状态
        agent.enemy_pos = self._pos.copy()
        agent.enemy_vel = self._vel.copy()

        # ── 击杀检测 ──
        if self.check_kill(agent, dt):
            self._kill_count += 1
            agent.kill_count = self._kill_count
            self.spawn(agent)

    def check_kill(self, agent: "SimAIAgent", dt: float) -> bool:
        """准星在命中框内驻留 TOT_THRESHOLD 秒即击杀。"""
        err_x = abs(agent.enemy_pos[0] - agent.crosshair_pos[0])
        err_y = abs(agent.enemy_pos[1] - agent.crosshair_pos[1])
        if err_x <= agent.target_hitbox_x and err_y <= agent.target_hitbox_y:
            self._tot_timer += dt
            if self._tot_timer >= TOT_THRESHOLD:
                return True
        else:
            self._tot_timer = max(0.0, self._tot_timer - dt * TOT_DECAY_RATE)
        return False

    def tick_human_input(self, agent: "SimAIAgent", dt: float) -> Tuple[float, float]:
        return (0.0, 0.0)

    @property
    def is_done(self) -> bool:
        return self._kill_count >= self._max_kills

    @property
    def chase_mode(self) -> str:
        return "pure_ai"

    @property
    def current_angle(self) -> float:
        """当前运动方向 (rad, 供测试观测)。"""
        return self._angle
