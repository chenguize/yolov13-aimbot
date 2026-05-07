# test/scenarios/ball_tracking/scenario.py
"""
单目标追踪场景 —— Valorant 霓虹 (Neon) 风格目标。

特性：
  - 3D 摄像机投影 (FOV 103)
  - 霓虹身法：sprint / slide / adad / jump / stop
  - 两种瞄准模式：pure_ai (AI 独立) / human_flick (人类甩枪后 AI 接)
  - 命中框 + 驻留时间判定击杀
  - 人类 eased-out 弹道甩枪模拟
"""

import math
import numpy as np

from typing import TYPE_CHECKING, Tuple
from test.scenarios.base import BaseScenario

from utils.human_intent import HumanIntentTracker

if TYPE_CHECKING:
    from test.sim_agent import SimAIAgent


# ==============================================================================
#  命中判定常量
# ==============================================================================
TOT_THRESHOLD   = 0.08      # 准星在命中框内累计秒数即击杀
TOT_DECAY_RATE  = 2.0       # 离开命中框后驻留计时衰减速率 (/s)


class BallTrackingScenario(BaseScenario):
    """Valorant 霓虹单目标追踪。"""

    def __init__(self, max_kills: int = 30, pure_ai_only: bool = False):
        """
        :param pure_ai_only: 为 True 时每局固定 pure_ai（Neon 身法不变），不模拟 human_flick。
        """
        self._max_kills = max_kills
        self._pure_ai_only = pure_ai_only
        self._kill_count = 0

        # 内部物理状态
        self._target_z: float = 10.0
        self._enemy_vel_3d = np.zeros(2, dtype=np.float64)
        self._target_vel_3d = np.zeros(2, dtype=np.float64)
        self._current_action: str = "stop"
        self._height_3d: float = 0.0

        # 移动计时器
        self._move_timer: float = 0.5

        # 驻留击杀
        self._tot_timer: float = 0.0

        # 人类甩枪
        self._flick_target: np.ndarray = None
        self._flick_duration: float = 0.0
        self._flick_timer: float = 0.0
        self._flick_start_pos: np.ndarray = np.zeros(2, dtype=np.float64)
        self._chase_mode: str = "pure_ai"
        self._prev_flick_active: bool = False
        self._prev_flick_pos: np.ndarray = None
        self._flick_ended: bool = False

    # ==========================================================================
    #  BaseScenario 接口
    # ==========================================================================

    def init(self, agent: "SimAIAgent") -> None:
        """模拟启动。只 spawn 第一个目标。"""
        self._kill_count = 0
        self.spawn(agent)

    def spawn(self, agent: "SimAIAgent") -> None:
        """生成新目标：3D 距离、霓虹物理、甩枪参数。"""
        if self._kill_count >= self._max_kills:
            return

        agent.world_model.track_manager.tracks.clear()

        if self._pure_ai_only:
            self._chase_mode = "pure_ai"
        else:
            self._chase_mode = np.random.choice(["pure_ai", "human_flick"])
        if hasattr(agent.world_model.controller, "reset_target_state"):
            try:
                agent.world_model.controller.reset_target_state(mode=self._chase_mode)
            except TypeError:
                agent.world_model.controller.reset_target_state()

        if hasattr(agent, '_intent_tracker'):
            agent._intent_tracker = HumanIntentTracker()

        # 3D 投影像素半径
        self._target_z = np.random.uniform(5.0, 35.0)
        scale_factor = agent.focal_length / self._target_z
        agent.target_hitbox_x = 0.15 * scale_factor
        agent.target_hitbox_y = 0.15 * scale_factor

        # 生成距离：pure_ai=近(60~256px) human_flick=远(600~1400px)
        if self._chase_mode == "pure_ai":
            spawn_radius_px = np.random.uniform(60, 256)
        else:
            spawn_radius_px = np.random.uniform(600, 1400)

        angle = np.random.uniform(0, 2 * np.pi)
        offset_x = np.cos(angle) * spawn_radius_px
        offset_y = np.sin(angle) * spawn_radius_px

        half_w = agent.screen_w / 2.0
        half_h = agent.screen_height / 2.0
        margin = 50
        offset_x = np.clip(offset_x, -half_w + margin, half_w - margin)
        offset_y = np.clip(offset_y, -half_h + margin, half_h - margin)

        agent.enemy_pos = agent.crosshair_pos + np.array([offset_x, offset_y])

        # 3D 物理初值
        self._enemy_vel_3d = np.array([np.random.choice([-1, 1]) * 8.5, 0.0])
        self._target_vel_3d = self._enemy_vel_3d.copy()
        self._height_3d = 0.0
        self._current_action = "sprint"
        self._move_timer = 0.5
        self._tot_timer = 0.0

        agent.chase_mode = self._chase_mode
        agent.target_first_seen_time = 0.0

        # 人类甩枪参数
        if self._chase_mode == "human_flick":
            error_offset = np.random.randn(2) * 20.0
            self._flick_target = agent.enemy_pos + error_offset
            self._flick_duration = np.random.uniform(0.15, 0.22)
            self._flick_timer = 0.0
            self._flick_start_pos = agent.crosshair_pos.copy()
            self._prev_flick_pos = self._flick_start_pos.copy()
            self._flick_ended = False
        else:
            self._flick_target = None
            self._flick_duration = 0.0
            self._flick_timer = 0.0

        self._prev_flick_active = False

        agent.stream_latency_model.reset()

    def tick_physics(self, agent: "SimAIAgent", dt: float) -> None:
        """霓虹 3D 身法 + 摄像机投影 + 击杀检测。"""
        # 身法状态机
        if self._move_timer <= 0:
            self._move_timer = np.random.uniform(0.15, 0.4)
            action = np.random.choice(
                ["sprint", "slide", "adad", "jump", "stop"],
                p=[0.30, 0.20, 0.30, 0.15, 0.05],
            )
            self._current_action = action

            if action == "sprint":
                self._target_vel_3d[0] = np.random.choice([-1, 1]) * 8.5
            elif action == "adad":
                self._target_vel_3d[0] = np.random.choice([-1, 1]) * 5.4
            elif action == "slide":
                dir_x = (np.sign(self._enemy_vel_3d[0])
                         if self._enemy_vel_3d[0] != 0
                         else np.random.choice([-1, 1]))
                self._enemy_vel_3d[0] = dir_x * 14.0
                self._target_vel_3d[0] = 0.0
                self._move_timer = 0.6
            elif action == "jump":
                if self._height_3d <= 0.01:
                    self._enemy_vel_3d[1] = -5.8
            elif action == "stop":
                self._target_vel_3d[0] = 0.0

        self._move_timer -= dt

        # 水平引擎
        if self._current_action == "slide":
            accel_x = (self._target_vel_3d[0] - self._enemy_vel_3d[0]) * 5.0
        else:
            accel_x = (self._target_vel_3d[0] - self._enemy_vel_3d[0]) * 40.0
        self._enemy_vel_3d[0] += accel_x * dt

        # 垂直抛物线
        if self._height_3d > 0.0 or self._enemy_vel_3d[1] < 0:
            self._enemy_vel_3d[1] += 16.0 * dt
            self._height_3d -= self._enemy_vel_3d[1] * dt
            if self._height_3d <= 0.0:
                self._height_3d = 0.0
                self._enemy_vel_3d[1] = 0.0

        # 3D -> 2D 投影
        scale = agent.focal_length / self._target_z
        agent.enemy_vel[0] = self._enemy_vel_3d[0] * scale
        agent.enemy_vel[1] = self._enemy_vel_3d[1] * scale
        agent.enemy_pos += agent.enemy_vel * dt

        # 击杀检测
        err_x = abs(agent.enemy_pos[0] - agent.crosshair_pos[0])
        err_y = abs(agent.enemy_pos[1] - agent.crosshair_pos[1])

        if err_x <= agent.target_hitbox_x and err_y <= agent.target_hitbox_y:
            self._tot_timer += dt
            if self._tot_timer >= TOT_THRESHOLD:
                self._kill_count += 1
                agent.kill_count = self._kill_count
                self.spawn(agent)
                return
        else:
            self._tot_timer = max(0.0, self._tot_timer - dt * TOT_DECAY_RATE)

    def tick_human_input(self, agent: "SimAIAgent", dt: float) -> Tuple[float, float]:
        """
        人类甩枪：eased-out 弹道曲线模拟人手从远处拉枪的过程。
        只在 chase_mode == 'human_flick' 时返回位移增量。
        """
        if self._chase_mode != "human_flick" or self._flick_target is None:
            return (0.0, 0.0)

        if self._flick_timer >= self._flick_duration:
            if not self._flick_ended:
                self._flick_ended = True
                self._prev_flick_active = False
                agent._prev_flick_active = False
            return (0.0, 0.0)

        self._flick_timer += dt
        self._prev_flick_active = True
        agent._prev_flick_active = True

        # eased-out: t^3 曲线（初速快 -> 末速慢，模拟人手减速）
        t = self._flick_timer / self._flick_duration
        eased = 1.0 - (1.0 - t) ** 3

        current_flick_pos = (self._flick_start_pos
                             + (self._flick_target - self._flick_start_pos) * eased)

        # 增量 = 当前位置 - 上一帧位置
        prev = getattr(self, '_prev_flick_pos', self._flick_start_pos.copy())
        delta = current_flick_pos - prev
        self._prev_flick_pos = current_flick_pos.copy()

        # perf 噪声：人手微抖
        noise = np.random.normal(0, 2.0, 2)
        result = delta + noise

        return (float(result[0]), float(result[1]))

    def check_kill(self, agent: "SimAIAgent", dt: float) -> bool:
        """已集成在 tick_physics 中。"""
        return False

    @property
    def is_done(self) -> bool:
        return self._kill_count >= self._max_kills

    @property
    def chase_mode(self) -> str:
        return self._chase_mode
