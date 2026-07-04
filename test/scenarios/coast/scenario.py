# test/scenarios/coast/scenario.py
"""
Coast 场景 — 周期性丢框, 测试 Kalman 预测精度和 coast 续跑。

特性：
  - 单目标 Neon 风格身法 (sprint / adad)
  - 周期性丢框 (模拟检测失败): 持续 3-10 帧 (6-20ms @ 2ms/帧)
  - 丢框期间 tick_cv 不生成检测 (agent.enemy_list = [])
  - 丢框期间目标继续运动 (物理不中断)
  - 测试 Kalman 预测精度 (coast 续跑) + 恢复检测后重关联
  - 丢框恢复时记录真实位移 (供预测误差分析)
"""

import numpy as np

from typing import TYPE_CHECKING, Tuple, Optional, List, Dict
from test.scenarios.base import BaseScenario

if TYPE_CHECKING:
    from test.sim_agent import SimAIAgent


# ==============================================================================
#  命中判定常量
# ==============================================================================
TOT_THRESHOLD  = 0.08
TOT_DECAY_RATE = 2.0

# 身法
_ACTION_LIST  = ["sprint", "adad", "stop"]
_ACTION_PROBS = [0.35, 0.45, 0.20]


class CoastScenario(BaseScenario):
    """Coast 场景 — 丢框续跑 + Kalman 预测精度测试。"""

    def __init__(
        self,
        max_kills: int = 15,
        seed: Optional[int] = None,
        dist_range: Tuple[float, float] = (5.0, 25.0),
        speed_range: Tuple[float, float] = (3.0, 8.5),
        drop_interval_range: Tuple[float, float] = (0.8, 2.0),
        drop_frames_range: Tuple[int, int] = (3, 10),
    ):
        """
        :param max_kills: 总击杀数到达即结束
        :param seed: 随机种子
        :param dist_range: 目标 3D 距离 z 范围 (m)
        :param speed_range: 目标 3D 速度范围 (m/s)
        :param drop_interval_range: 两次丢框之间间隔 (s)
        :param drop_frames_range: 丢框持续帧数范围 (3-10 帧 = 6-20ms @ 2ms/帧)
        """
        self._max_kills = max_kills
        self._rng = np.random.RandomState(seed)
        self._dist_range = dist_range
        self._speed_range = speed_range
        self._drop_interval_range = drop_interval_range
        self._drop_frames_range = drop_frames_range

        self._kill_count = 0

        # 目标物理
        self._pos = np.zeros(2, dtype=np.float64)
        self._vel = np.zeros(2, dtype=np.float64)
        self._z = 10.0
        self._vel_3d = np.zeros(2, dtype=np.float64)
        self._target_vel_3d = np.zeros(2, dtype=np.float64)
        self._action = "sprint"
        self._move_timer = 0.5

        # 丢框状态机
        self._dropping = False
        self._drop_interval_timer = 0.0   # 距下次丢框倒计时 (s)
        self._drop_frames_left = 0        # 丢框剩余帧数
        self._drop_frames_total = 0       # 本次丢框总帧数 (误差分析用)
        self._drop_start_pos = np.zeros(2, dtype=np.float64)  # 丢框开始时真实位置

        # 击杀
        self._tot_timer = 0.0

        # 预测误差分析记录 (供测试 harness 读取)
        self.coast_events: List[Dict] = []

    # ==========================================================================
    #  BaseScenario 接口
    # ==========================================================================

    def init(self, agent: "SimAIAgent") -> None:
        self._kill_count = 0
        self.coast_events = []
        self.spawn(agent)

    def spawn(self, agent: "SimAIAgent") -> None:
        """生成新目标：3D 距离 / 身法 / 丢框参数。"""
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

        # 丢框状态机重置
        self._dropping = False
        self._drop_interval_timer = float(self._rng.uniform(*self._drop_interval_range))
        self._drop_frames_left = 0

        agent.chase_mode = "pure_ai"
        agent.target_first_seen_time = 0.0
        agent.enemy_pos = self._pos.copy()
        agent.enemy_vel = self._vel.copy()
        agent.stream_latency_model.reset()

    def tick_physics(self, agent: "SimAIAgent", dt: float) -> None:
        """丢框状态机 + 目标物理 + 检测生成控制 + 击杀。"""
        # ── 丢框状态机 (帧计数) ──
        if self._dropping:
            self._drop_frames_left -= 1
            if self._drop_frames_left <= 0:
                # 丢框恢复 → 记录预测误差分析数据
                self._dropping = False
                true_displacement = float(np.linalg.norm(self._pos - self._drop_start_pos))
                # 记录 Kalman 预测 (若可用)
                predicted = None
                if getattr(agent.ctx, 'p_predict', None) is not None:
                    # p_predict 是相对 center 的预测位置, 记录供 harness 比较
                    predicted = agent.ctx.p_predict.copy()
                self.coast_events.append({
                    "drop_frames": self._drop_frames_total,
                    "drop_duration_s": self._drop_frames_total * dt,
                    "true_displacement_px": true_displacement,
                    "true_end_pos": self._pos.copy(),
                    "predicted_pos": predicted,
                    "kill_id": self._kill_count,
                })
                self._drop_interval_timer = float(self._rng.uniform(*self._drop_interval_range))
        else:
            self._drop_interval_timer -= dt
            if self._drop_interval_timer <= 0:
                # 进入丢框
                self._dropping = True
                fmin, fmax = self._drop_frames_range
                self._drop_frames_left = int(self._rng.randint(fmin, fmax + 1))
                self._drop_frames_total = self._drop_frames_left
                self._drop_start_pos = self._pos.copy()

        # ── 目标物理 (丢框期间也继续运动) ──
        self._tick_target_physics(dt, agent)

        # ── 检测生成：丢框时不生成检测 ──
        if self._dropping:
            agent.enemy_list = []
        else:
            agent.enemy_list = [(self._pos, self._vel, self._z)]

        # 同步 agent 状态
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
    def is_dropping(self) -> bool:
        """当前是否处于丢框状态 (供测试观测)。"""
        return self._dropping
