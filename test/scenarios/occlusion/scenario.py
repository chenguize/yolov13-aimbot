# test/scenarios/occlusion/scenario.py
"""
遮挡场景 — 单目标周期性被遮挡 (烟雾/墙后)。

特性：
  - 单目标 Neon 风格身法 (sprint / adad)
  - 周期性遮挡：遮挡期间 tick_cv 不生成检测 (agent.enemy_list = [])
  - 遮挡持续 0.1-0.5s 随机
  - 遮挡期间目标继续运动 (物理不中断)
  - 遮挡结束后目标位置可能偏移 (模拟移动中遮挡, 测试 ReID 重关联)
  - 测试 Kalman coast 续跑 + 恢复检测后 ReID 重关联
"""

import numpy as np

from typing import TYPE_CHECKING, Tuple, Optional
from test.scenarios.base import BaseScenario

if TYPE_CHECKING:
    from test.sim_agent import SimAIAgent


# ==============================================================================
#  命中判定常量
# ==============================================================================
TOT_THRESHOLD  = 0.08      # 准星在命中框内累计秒数即击杀
TOT_DECAY_RATE = 2.0       # 离开命中框后驻留计时衰减速率 (/s)

# 身法
_ACTION_LIST  = ["sprint", "adad", "stop"]
_ACTION_PROBS = [0.35, 0.45, 0.20]


class OcclusionScenario(BaseScenario):
    """遮挡场景 — coast 续跑 + ReID 重关联测试。"""

    def __init__(
        self,
        max_kills: int = 15,
        seed: Optional[int] = None,
        dist_range: Tuple[float, float] = (5.0, 25.0),
        speed_range: Tuple[float, float] = (3.0, 8.5),
        occlude_interval_range: Tuple[float, float] = (1.5, 4.0),
        occlude_duration_range: Tuple[float, float] = (0.1, 0.5),
        drift_during_occlusion_range: Tuple[float, float] = (20.0, 80.0),
    ):
        """
        :param max_kills: 总击杀数到达即结束
        :param seed: 随机种子
        :param dist_range: 目标 3D 距离 z 范围 (m)
        :param speed_range: 目标 3D 速度范围 (m/s)
        :param occlude_interval_range: 两次遮挡之间间隔 (s)
        :param occlude_duration_range: 遮挡持续时长 (s)
        :param drift_during_occlusion_range: 遮挡结束时额外位置偏移 (px)
        """
        self._max_kills = max_kills
        self._rng = np.random.RandomState(seed)
        self._dist_range = dist_range
        self._speed_range = speed_range
        self._occlude_interval_range = occlude_interval_range
        self._occlude_duration_range = occlude_duration_range
        self._drift_during_occlusion_range = drift_during_occlusion_range

        self._kill_count = 0

        # 目标物理
        self._pos = np.zeros(2, dtype=np.float64)
        self._vel = np.zeros(2, dtype=np.float64)
        self._z = 10.0
        self._vel_3d = np.zeros(2, dtype=np.float64)
        self._target_vel_3d = np.zeros(2, dtype=np.float64)
        self._action = "sprint"
        self._move_timer = 0.5

        # 遮挡状态机
        self._occluded = False
        self._occlude_timer = 0.0      # 距下次遮挡的倒计时
        self._occlude_duration = 0.0   # 当前遮挡剩余时长

        # 击杀
        self._tot_timer = 0.0

    # ==========================================================================
    #  BaseScenario 接口
    # ==========================================================================

    def init(self, agent: "SimAIAgent") -> None:
        self._kill_count = 0
        self.spawn(agent)

    def spawn(self, agent: "SimAIAgent") -> None:
        """生成新目标：3D 距离 / 身法 / 遮挡参数。"""
        if self._kill_count >= self._max_kills:
            return

        agent.world_model.track_manager.tracks.clear()

        if hasattr(agent.world_model.controller, 'reset_target_state'):
            try:
                agent.world_model.controller.reset_target_state(mode="pure_ai")
            except TypeError:
                agent.world_model.controller.reset_target_state()

        # 3D 投影 + 命中框
        self._z = float(self._rng.uniform(*self._dist_range))
        scale_factor = agent.focal_length / self._z
        agent.target_hitbox_x = 0.15 * scale_factor
        agent.target_hitbox_y = 0.15 * scale_factor

        # 生成位置
        spawn_radius = float(self._rng.uniform(100, 400))
        angle = float(self._rng.uniform(0, 2 * np.pi))
        half_w = agent.screen_w / 2.0
        half_h = agent.screen_height / 2.0
        margin = 50
        ox = float(np.clip(np.cos(angle) * spawn_radius, -half_w + margin, half_w - margin))
        oy = float(np.clip(np.sin(angle) * spawn_radius, -half_h + margin, half_h - margin))
        self._pos = agent.crosshair_pos + np.array([ox, oy], dtype=np.float64)

        # 3D 物理
        speed = float(self._rng.uniform(*self._speed_range))
        self._vel_3d = np.array([int(self._rng.choice([-1, 1])) * speed, 0.0], dtype=np.float64)
        self._target_vel_3d = self._vel_3d.copy()
        self._action = "sprint"
        self._move_timer = float(self._rng.uniform(0.2, 0.5))
        self._tot_timer = 0.0

        # 遮挡状态机重置
        self._occluded = False
        self._occlude_timer = float(self._rng.uniform(*self._occlude_interval_range))
        self._occlude_duration = 0.0

        agent.chase_mode = "pure_ai"
        agent.target_first_seen_time = 0.0
        agent.enemy_pos = self._pos.copy()
        agent.enemy_vel = self._vel.copy()
        agent.stream_latency_model.reset()

    def tick_physics(self, agent: "SimAIAgent", dt: float) -> None:
        """遮挡状态机 + 目标物理 + 检测生成控制 + 击杀。"""
        # ── 遮挡状态机 ──
        if self._occluded:
            self._occlude_duration -= dt
            if self._occlude_duration <= 0:
                # 遮挡结束 → 恢复检测 + 位置偏移 (模拟移动中遮挡)
                self._occluded = False
                drift = float(self._rng.uniform(*self._drift_during_occlusion_range))
                drift_angle = float(self._rng.uniform(0, 2 * np.pi))
                self._pos += np.array(
                    [np.cos(drift_angle) * drift, np.sin(drift_angle) * drift],
                    dtype=np.float64,
                )
                self._occlude_timer = float(self._rng.uniform(*self._occlude_interval_range))
        else:
            self._occlude_timer -= dt
            if self._occlude_timer <= 0:
                # 进入遮挡
                self._occluded = True
                self._occlude_duration = float(self._rng.uniform(*self._occlude_duration_range))

        # ── 目标物理 (遮挡期间也继续运动) ──
        self._tick_target_physics(dt, agent)

        # ── 检测生成：遮挡时不生成检测 ──
        if self._occluded:
            agent.enemy_list = []
        else:
            agent.enemy_list = [(self._pos, self._vel, self._z)]

        # 同步 agent 状态 (check_kill 用)
        agent.enemy_pos = self._pos.copy()
        agent.enemy_vel = self._vel.copy()

        # ── 击杀检测 ──
        if self.check_kill(agent, dt):
            self._kill_count += 1
            agent.kill_count = self._kill_count
            self.spawn(agent)

    def _tick_target_physics(self, dt: float, agent: "SimAIAgent") -> None:
        """Neon 风格身法 (sprint / adad / stop)。"""
        if self._move_timer <= 0:
            self._move_timer = float(self._rng.uniform(0.15, 0.4))
            self._action = str(self._rng.choice(_ACTION_LIST, p=_ACTION_PROBS))
            speed = float(self._rng.uniform(*self._speed_range))
            if self._action == "sprint":
                self._target_vel_3d[0] = int(self._rng.choice([-1, 1])) * speed
            elif self._action == "adad":
                self._target_vel_3d[0] = int(self._rng.choice([-1, 1])) * speed * 0.6
            elif self._action == "stop":
                self._target_vel_3d[0] = 0.0

        self._move_timer -= dt

        # 水平引擎
        accel = (self._target_vel_3d[0] - self._vel_3d[0]) * 40.0
        self._vel_3d[0] += accel * dt
        self._vel_3d[1] = 0.0

        # 3D -> 2D 投影
        scale = agent.focal_length / max(self._z, 1.0)
        self._vel[0] = self._vel_3d[0] * scale
        self._vel[1] = self._vel_3d[1] * scale
        self._pos += self._vel * dt

        # 屏幕边界软约束
        half_w = agent.screen_w / 2.0
        half_h = agent.screen_height / 2.0
        cx = agent.crosshair_pos[0]
        cy = agent.crosshair_pos[1]
        if self._pos[0] < cx - half_w + 50:
            self._target_vel_3d[0] = abs(self._target_vel_3d[0])
        elif self._pos[0] > cx + half_w - 50:
            self._target_vel_3d[0] = -abs(self._target_vel_3d[0])
        if self._pos[1] < cy - half_h + 50:
            self._pos[1] = cy - half_h + 50
        elif self._pos[1] > cy + half_h - 50:
            self._pos[1] = cy + half_h - 50

    def check_kill(self, agent: "SimAIAgent", dt: float) -> bool:
        """准星在命中框内驻留 TOT_THRESHOLD 秒即击杀 (遮挡期间不累计)。"""
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
    def is_occluded(self) -> bool:
        """当前是否处于遮挡状态 (供测试观测)。"""
        return self._occluded
