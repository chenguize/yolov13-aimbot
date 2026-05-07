# test/scenarios/multi_target/scenario.py
"""
多目标追踪场景 — 占位模块。

计划特性：
  - 2-5 个同时活跃目标，不同运动模式
  - TrackManager 多轨迹池真实测试
  - 目标切换延迟测量
  - 并发威胁优先级选择
"""

from typing import TYPE_CHECKING, Tuple
from test.scenarios.base import BaseScenario

if TYPE_CHECKING:
    from test.sim_agent import SimAIAgent


class MultiTargetScenario(BaseScenario):
    """多目标追踪 — 待实现。"""

    def __init__(self, max_kills: int = 30, num_targets: int = 3):
        self._max_kills = max_kills
        self._num_targets = num_targets

    def init(self, agent: "SimAIAgent") -> None:
        raise NotImplementedError("MultiTargetScenario 尚未实现")

    def spawn(self, agent: "SimAIAgent") -> None:
        raise NotImplementedError("MultiTargetScenario 尚未实现")

    def tick_physics(self, agent: "SimAIAgent", dt: float) -> None:
        raise NotImplementedError("MultiTargetScenario 尚未实现")

    def tick_human_input(self, agent: "SimAIAgent", dt: float) -> Tuple[float, float]:
        raise NotImplementedError("MultiTargetScenario 尚未实现")

    @property
    def is_done(self) -> bool:
        return False

    @property
    def chase_mode(self) -> str:
        return "pure_ai"
