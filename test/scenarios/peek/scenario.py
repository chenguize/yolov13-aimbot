# test/scenarios/peek/scenario.py
"""
Peek 场景 — 目标突然出现/消失 (从掩体后 peek)。

特性：
  - 目标从屏幕边缘 (左/右/上) 突然出现
  - 出现后快速 strafe 横向移动 (3D 3-8 m/s)
  - 出现到消失间隔 0.5-2.0s
  - 消失后 0.3-1.0s 在另一边缘重新出现
  - 测试 IMM-Kalman CT 模型对突然变向 (0→v, v→0) 的响应
  - 消失期间 tick_cv 不生成检测 (agent.enemy_list = [])
"""

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

# 出现边缘
_EDGE_LEFT = "left"
_EDGE_RIGHT = "right"
_EDGE_TOP = "top"
_EDGES = [_EDGE_LEFT, _EDGE_RIGHT, _EDGE_TOP]


class PeekScenario(BaseScenario):
    """Peek 场景 — 目标突然出现/消失, 测试 IMM CT 模型。"""

    def __init__(
        self,
        max_kills: int = 15,
        seed: Optional[int] = None,
        dist_range: Tuple[float, float] = (8.0, 20.0),
        strafe_speed_range: Tuple[float, float] = (3.0, 8.0),
        visible_duration_range: Tuple[float, float] = (0.5, 2.0),
        hidden_duration_range: Tuple[float, float] = (0.3, 1.0),
    ):
        """
        :param max_kills: 总击杀数到达即结束
        :param seed: 随机种子
        :param dist_range: 目标 3D 距离 z 范围 (m)
        :param strafe_speed_range: strafe 3D 速度范围 (m/s)
        :param visible_duration_range: 出现持续时长 (s)
        :param hidden_duration_range: 消失持续时长 (s)
        """
        self._max_kills = max_kills
        self._rng = np.random.RandomState(seed)
        self._dist_range = dist_range
        self._strafe_speed_range = strafe_speed_range
        self._visible_duration_range = visible_duration_range
        self._hidden_duration_range = hidden_duration_range

        self._kill_count = 0

        # 目标状态
        self._pos = np.zeros(2, dtype=np.float64)
        self._vel = np.zeros(2, dtype=np.float64)
        self._z = 10.0
        self._vel_3d_x = 0.0   # 3D 水平速度 (m/s, strafe 方向)
        self._edge = _EDGE_LEFT

        # 出现/消失状态机
        self._visible = False
        self._visible_timer = 0.0
        self._hidden_timer = 0.0

        # 击杀
        self._tot_timer = 0.0

    # ==========================================================================
    #  BaseScenario 接口
    # ==========================================================================

    def init(self, agent: "SimAIAgent") -> None:
        self._kill_count = 0
        # 初始为隐藏状态, 短暂后首次 peek
        self._visible = False
        self._hidden_timer = float(self._rng.uniform(0.1, 0.4))
        self._visible_timer = 0.0
        agent.enemy_list = []
        self.spawn(agent)

    def spawn(self, agent: "SimAIAgent") -> None:
        """重置击杀计时 + 进入隐藏状态 (下个 tick 重新 peek)。"""
        if self._kill_count >= self._max_kills:
            return

        agent.world_model.track_manager.tracks.clear()

        if hasattr(agent.world_model.controller, 'reset_target_state'):
            try:
                agent.world_model.controller.reset_target_state(mode="pure_ai")
            except TypeError:
                agent.world_model.controller.reset_target_state()

        self._tot_timer = 0.0
        self._visible = False
        self._hidden_timer = float(self._rng.uniform(*self._hidden_duration_range))
        self._visible_timer = 0.0

        agent.chase_mode = "pure_ai"
        agent.target_first_seen_time = 0.0
        agent.enemy_list = []
        agent.enemy_pos = self._pos.copy()
        agent.enemy_vel = self._vel.copy()
        agent.stream_latency_model.reset()

    def _peek_spawn(self, agent: "SimAIAgent") -> None:
        """目标从随机边缘 peek 出现。"""
        # 3D 距离 + 命中框
        self._z = float(self._rng.uniform(*self._dist_range))
        scale_factor = agent.focal_length / self._z
        agent.target_hitbox_x = 0.15 * scale_factor
        agent.target_hitbox_y = 0.15 * scale_factor

        # 选择边缘
        self._edge = str(self._rng.choice(_EDGES))
        half_w = agent.screen_w / 2.0
        half_h = agent.screen_height / 2.0
        cx = agent.crosshair_pos[0]
        cy = agent.crosshair_pos[1]

        if self._edge == _EDGE_LEFT:
            # 左边缘 → strafe 向右
            self._pos = np.array([cx - half_w + 60, cy + float(self._rng.uniform(-150, 150))], dtype=np.float64)
            self._vel_3d_x = float(self._rng.uniform(*self._strafe_speed_range))
        elif self._edge == _EDGE_RIGHT:
            # 右边缘 → strafe 向左
            self._pos = np.array([cx + half_w - 60, cy + float(self._rng.uniform(-150, 150))], dtype=np.float64)
            self._vel_3d_x = -float(self._rng.uniform(*self._strafe_speed_range))
        else:  # _EDGE_TOP
            # 上边缘 → 随机左右 strafe
            self._pos = np.array([cx + float(self._rng.uniform(-200, 200)), cy - half_h + 60], dtype=np.float64)
            self._vel_3d_x = float(self._rng.choice([-1.0, 1.0])) * float(self._rng.uniform(*self._strafe_speed_range))

        # 投影到 2D 速度
        scale = agent.focal_length / max(self._z, 1.0)
        self._vel = np.array([self._vel_3d_x * scale, 0.0], dtype=np.float64)

        self._visible = True
        self._visible_timer = float(self._rng.uniform(*self._visible_duration_range))
        self._tot_timer = 0.0

    def tick_physics(self, agent: "SimAIAgent", dt: float) -> None:
        """出现/消失状态机 + strafe 运动 + 击杀。"""
        if self._visible:
            self._visible_timer -= dt
            if self._visible_timer <= 0:
                # 消失
                self._visible = False
                self._hidden_timer = float(self._rng.uniform(*self._hidden_duration_range))
                self._vel_3d_x = 0.0
                self._vel = np.zeros(2, dtype=np.float64)
        else:
            self._hidden_timer -= dt
            if self._hidden_timer <= 0:
                # peek 出现
                self._peek_spawn(agent)

        # ── 可见时 strafe 运动 ──
        if self._visible:
            scale = agent.focal_length / max(self._z, 1.0)
            self._vel[0] = self._vel_3d_x * scale
            self._vel[1] = 0.0
            self._pos += self._vel * dt

            # 屏幕边界约束 (strafe 到对侧边缘附近也保持可见)
            half_w = agent.screen_w / 2.0
            cx = agent.crosshair_pos[0]
            if self._pos[0] < cx - half_w + 40:
                self._pos[0] = cx - half_w + 40
            elif self._pos[0] > cx + half_w - 40:
                self._pos[0] = cx + half_w - 40

        # ── 检测生成 ──
        if self._visible:
            agent.enemy_list = [(self._pos, self._vel, self._z)]
        else:
            agent.enemy_list = []

        # 同步 agent 状态
        agent.enemy_pos = self._pos.copy()
        agent.enemy_vel = self._vel.copy()

        # ── 击杀检测 ──
        if self._visible and self.check_kill(agent, dt):
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
    def is_visible(self) -> bool:
        """目标当前是否可见 (供测试观测)。"""
        return self._visible
