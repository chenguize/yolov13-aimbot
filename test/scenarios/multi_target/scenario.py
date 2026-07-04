# test/scenarios/multi_target/scenario.py
"""
多目标追踪场景 — 2-4 个目标同时活跃。

特性：
  - 每个目标独立 Neon 风格身法 (sprint / adad / slide / stop)
  - 不同 3D 距离 (5-35m) 和运动方向
  - 通过 agent.enemy_list 驱动 tick_cv 多目标检测
  - 击杀当前锁定目标后切换到最近目标 (威胁优先级 = 距离最近)
  - 目标颜色相近但有差异 (hue 偏移 0.02-0.08)，测试 TrackManager ReID 颜色直方图
  - 构造参数支持场景随机化 (目标数 / 距离 / 速度 / seed)
"""

import numpy as np

from typing import TYPE_CHECKING, Tuple, List, Optional
from test.scenarios.base import BaseScenario

if TYPE_CHECKING:
    from test.sim_agent import SimAIAgent


# ==============================================================================
#  命中判定常量
# ==============================================================================
TOT_THRESHOLD  = 0.10      # 准星在命中框内累计秒数即击杀 (多目标略宽)
TOT_DECAY_RATE = 2.0       # 离开命中框后驻留计时衰减速率 (/s)

# 身法状态机
_ACTION_LIST  = ["sprint", "adad", "slide", "stop"]
_ACTION_PROBS = [0.30, 0.40, 0.15, 0.15]


class _Target:
    """单个目标的物理 + 状态机 (Neon 风格简化版)。"""
    __slots__ = (
        'pos', 'vel', 'z', 'vel_3d', 'target_vel_3d', 'height_3d',
        'action', 'move_timer', 'color_hue', 'alive',
        'hitbox_x', 'hitbox_y', 'tot_timer',
    )

    def __init__(self):
        self.pos = np.zeros(2, dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)
        self.z = 10.0
        self.vel_3d = np.zeros(2, dtype=np.float64)
        self.target_vel_3d = np.zeros(2, dtype=np.float64)
        self.height_3d = 0.0
        self.action = "sprint"
        self.move_timer = 0.5
        self.color_hue = 0.0
        self.alive = True
        self.hitbox_x = 5.0
        self.hitbox_y = 5.0
        self.tot_timer = 0.0


class MultiTargetScenario(BaseScenario):
    """多目标追踪场景。"""

    def __init__(
        self,
        max_kills: int = 20,
        num_targets_range: Tuple[int, int] = (2, 4),
        dist_range: Tuple[float, float] = (5.0, 35.0),
        speed_range: Tuple[float, float] = (3.0, 8.5),
        seed: Optional[int] = None,
    ):
        """
        :param max_kills: 总击杀数到达即结束
        :param num_targets_range: 每批目标数量采样范围 (含两端)
        :param dist_range: 目标 3D 距离 z 采样范围 (m)
        :param speed_range: 目标 3D 速度采样范围 (m/s)
        :param seed: 随机种子 (None=不可复现)
        """
        self._max_kills = max_kills
        self._num_targets_range = num_targets_range
        self._dist_range = dist_range
        self._speed_range = speed_range
        self._rng = np.random.RandomState(seed)

        self._kill_count = 0
        self._targets: List[_Target] = []
        self._locked_idx: int = 0
        self._chase_mode: str = "pure_ai"

    # ==========================================================================
    #  BaseScenario 接口
    # ==========================================================================

    def init(self, agent: "SimAIAgent") -> None:
        self._kill_count = 0
        self._targets = []
        agent._multi_target_mode = True
        self._spawn_batch(agent)

    def spawn(self, agent: "SimAIAgent") -> None:
        """生成新一批目标 (当所有目标被击杀后调用)。"""
        self._targets = []
        self._spawn_batch(agent)

    def _spawn_batch(self, agent: "SimAIAgent") -> None:
        """生成一批目标：随机数量 / 距离 / 速度 / 颜色。"""
        nmin, nmax = self._num_targets_range
        n = int(self._rng.randint(nmin, nmax + 1))

        # 清理上一批轨迹
        agent.world_model.track_manager.tracks.clear()

        # 颜色基线：所有目标 hue 接近但有差异 (ReID 测试)
        base_hue = float(self._rng.uniform(0.0, 1.0))
        half_w = agent.screen_w / 2.0
        half_h = agent.screen_height / 2.0
        margin = 60

        for i in range(n):
            t = _Target()
            t.z = float(self._rng.uniform(*self._dist_range))
            scale = agent.focal_length / t.z
            t.hitbox_x = 0.15 * scale
            t.hitbox_y = 0.15 * scale

            # 分散生成位置
            angle = float(self._rng.uniform(0, 2 * np.pi))
            radius = float(self._rng.uniform(80, 400))
            ox = float(np.clip(np.cos(angle) * radius, -half_w + margin, half_w - margin))
            oy = float(np.clip(np.sin(angle) * radius, -half_h + margin, half_h - margin))
            t.pos = agent.crosshair_pos + np.array([ox, oy], dtype=np.float64)

            # 3D 物理：随机方向 + 速度
            speed = float(self._rng.uniform(*self._speed_range))
            direction = int(self._rng.choice([-1, 1]))
            t.vel_3d = np.array([direction * speed, 0.0], dtype=np.float64)
            t.target_vel_3d = t.vel_3d.copy()
            t.height_3d = 0.0
            t.action = "sprint"
            t.move_timer = float(self._rng.uniform(0.2, 0.5))

            # 颜色 hue：相近但有差异 (0.02-0.08 偏移)
            t.color_hue = (base_hue + float(self._rng.uniform(0.02, 0.08)) * (i + 1)) % 1.0
            t.alive = True
            t.tot_timer = 0.0
            self._targets.append(t)

        # 锁定最近目标 (威胁优先级)
        self._locked_idx = self._select_nearest(agent)

        # 同步 agent 状态
        agent.chase_mode = "pure_ai"
        agent.target_first_seen_time = 0.0
        if hasattr(agent.world_model.controller, 'reset_target_state'):
            try:
                agent.world_model.controller.reset_target_state(mode="pure_ai")
            except TypeError:
                agent.world_model.controller.reset_target_state()
        agent.stream_latency_model.reset()

    def _select_nearest(self, agent: "SimAIAgent") -> int:
        """选择距离准星最近的存活目标索引 (威胁优先级)。"""
        best_idx = -1
        best_dist = float('inf')
        for i, t in enumerate(self._targets):
            if not t.alive:
                continue
            d = float(np.linalg.norm(t.pos - agent.crosshair_pos))
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def tick_physics(self, agent: "SimAIAgent", dt: float) -> None:
        """更新所有存活目标物理 + 构建 enemy_list + 击杀检测。"""
        # 1. 更新每个存活目标的物理
        for t in self._targets:
            if t.alive:
                self._tick_target_physics(t, dt, agent)

        # 2. 构建 enemy_list 供 tick_cv 多目标检测
        enemy_list = []
        for t in self._targets:
            if t.alive:
                enemy_list.append((t.pos, t.vel, t.z))
        agent.enemy_list = enemy_list

        # 3. 重新选择锁定目标 (当前已死则切换)
        if self._locked_idx < 0 or self._locked_idx >= len(self._targets) \
                or not self._targets[self._locked_idx].alive:
            self._locked_idx = self._select_nearest(agent)

        # 4. 同步 agent.enemy_pos/vel/hitbox 到锁定目标 (供 check_kill / 评分)
        if 0 <= self._locked_idx < len(self._targets):
            locked = self._targets[self._locked_idx]
            agent.enemy_pos = locked.pos.copy()
            agent.enemy_vel = locked.vel.copy()
            agent.target_hitbox_x = locked.hitbox_x
            agent.target_hitbox_y = locked.hitbox_y

        # 5. 击杀检测
        if self.check_kill(agent, dt):
            self._on_kill(agent)

    def _tick_target_physics(self, t: _Target, dt: float, agent: "SimAIAgent") -> None:
        """Neon 风格身法状态机 (sprint / adad / slide / stop)。"""
        if t.move_timer <= 0:
            t.move_timer = float(self._rng.uniform(0.15, 0.4))
            t.action = str(self._rng.choice(_ACTION_LIST, p=_ACTION_PROBS))
            speed = float(self._rng.uniform(*self._speed_range))

            if t.action == "sprint":
                t.target_vel_3d[0] = int(self._rng.choice([-1, 1])) * speed
            elif t.action == "adad":
                t.target_vel_3d[0] = int(self._rng.choice([-1, 1])) * speed * 0.6
            elif t.action == "slide":
                dir_x = float(np.sign(t.vel_3d[0])) if t.vel_3d[0] != 0 else int(self._rng.choice([-1, 1]))
                t.vel_3d[0] = dir_x * speed * 1.6
                t.target_vel_3d[0] = 0.0
                t.move_timer = 0.6
            elif t.action == "stop":
                t.target_vel_3d[0] = 0.0

        t.move_timer -= dt

        # 水平引擎 (slide 慢减速, 其他快减速)
        if t.action == "slide":
            accel = (t.target_vel_3d[0] - t.vel_3d[0]) * 5.0
        else:
            accel = (t.target_vel_3d[0] - t.vel_3d[0]) * 40.0
        t.vel_3d[0] += accel * dt

        # 垂直 (简化: 无跳跃)
        t.vel_3d[1] = 0.0

        # 3D -> 2D 投影
        scale = agent.focal_length / max(t.z, 1.0)
        t.vel[0] = t.vel_3d[0] * scale
        t.vel[1] = t.vel_3d[1] * scale
        t.pos += t.vel * dt

        # 屏幕边界软约束 (防止飞出可视区域)
        half_w = agent.screen_w / 2.0
        half_h = agent.screen_height / 2.0
        cx = agent.crosshair_pos[0]
        cy = agent.crosshair_pos[1]
        if t.pos[0] < cx - half_w + 50:
            t.target_vel_3d[0] = abs(t.target_vel_3d[0])
        elif t.pos[0] > cx + half_w - 50:
            t.target_vel_3d[0] = -abs(t.target_vel_3d[0])
        if t.pos[1] < cy - half_h + 50:
            t.pos[1] = cy - half_h + 50
        elif t.pos[1] > cy + half_h - 50:
            t.pos[1] = cy + half_h - 50

    def check_kill(self, agent: "SimAIAgent", dt: float) -> bool:
        """基于当前锁定目标的命中框 + 驻留时间。"""
        if self._locked_idx < 0 or self._locked_idx >= len(self._targets):
            return False
        t = self._targets[self._locked_idx]
        if not t.alive:
            return False

        err_x = abs(agent.enemy_pos[0] - agent.crosshair_pos[0])
        err_y = abs(agent.enemy_pos[1] - agent.crosshair_pos[1])

        if err_x <= agent.target_hitbox_x and err_y <= agent.target_hitbox_y:
            t.tot_timer += dt
            if t.tot_timer >= TOT_THRESHOLD:
                return True
        else:
            t.tot_timer = max(0.0, t.tot_timer - dt * TOT_DECAY_RATE)
        return False

    def _on_kill(self, agent: "SimAIAgent") -> None:
        """击杀当前锁定目标 → 标记死亡 → 切换最近目标 → 全灭则新生成一批。"""
        t = self._targets[self._locked_idx]
        t.alive = False
        t.tot_timer = 0.0
        self._kill_count += 1
        agent.kill_count = self._kill_count

        # 切换到下一个最近目标 (目标切换测试)
        self._locked_idx = self._select_nearest(agent)

        # 全部死亡 → 生成新一批 (若未达 max_kills)
        if self._locked_idx < 0 and self._kill_count < self._max_kills:
            self._spawn_batch(agent)

        # 清零控制器惯性：Kill 后 arm_vel 残留会导致准星惯性飞出检测范围，
        # 丢失目标长达 1-2s。只清零 arm_vel/servo_accel，不做 full reset
        # （full reset 会重置反应渐变和锁定确认，每次 kill 多花 20ms+6ms）。
        ctrl = agent.world_model.controller
        if hasattr(ctrl, '_arm_vel'):
            ctrl._arm_vel[:] = 0.0
        if hasattr(ctrl, '_servo_accel'):
            ctrl._servo_accel[:] = 0.0

    def tick_human_input(self, agent: "SimAIAgent", dt: float) -> Tuple[float, float]:
        """纯 AI 场景, 无人类输入。"""
        return (0.0, 0.0)

    @property
    def is_done(self) -> bool:
        return self._kill_count >= self._max_kills

    @property
    def chase_mode(self) -> str:
        return "pure_ai"

    # ==========================================================================
    #  状态查询 (供测试 / 评分使用)
    # ==========================================================================

    @property
    def locked_target_idx(self) -> int:
        return self._locked_idx

    @property
    def alive_count(self) -> int:
        return sum(1 for t in self._targets if t.alive)

    @property
    def target_colors(self) -> List[float]:
        """返回各存活目标 hue (ReID 测试用)。"""
        return [t.color_hue for t in self._targets if t.alive]
