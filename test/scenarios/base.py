# test/scenarios/base.py
"""
场景基类。每个测试场景继承此类，定义：
  - 目标如何生成
  - 目标如何运动
  - 人类如何操作
  - 何时判定击杀/结束
"""

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from test.sim_agent import SimAIAgent


class BaseScenario:
    """场景抽象接口。引擎 (SimAIAgent) 持有场景并按帧委托。"""

    def init(self, agent: "SimAIAgent") -> None:
        """
        模拟启动时调用一次。场景可在此初始化内部状态。
        agent 提供 crosshair_pos / world_model / 3D 摄像机参数等只读接口。
        """
        pass

    def spawn(self, agent: "SimAIAgent") -> None:
        """
        生成新目标。agent.kill_count 确定当前进度。
        场景负责设置 agent.enemy_pos / agent.chase_mode 等。
        """
        raise NotImplementedError

    def tick_physics(self, agent: "SimAIAgent", dt: float) -> None:
        """
        每帧调用：更新目标物理位置 / 运动状态。
        场景直接修改 agent.enemy_pos / agent.enemy_vel 等。
        """
        raise NotImplementedError

    def tick_human_input(self, agent: "SimAIAgent", dt: float) -> Tuple[float, float]:
        """
        每帧调用：计算人类鼠标位移 (dx, dy) px。
        返回 (0, 0) 表示纯 AI 场景（无人类输入）。
        """
        return (0.0, 0.0)

    def check_kill(self, agent: "SimAIAgent", dt: float) -> bool:
        """
        每帧调用：检测是否命中。默认基于命中框 + 驻留时间。
        返回 True 表示击杀成功。
        """
        return False

    @property
    def is_done(self) -> bool:
        """模拟是否已结束（达到目标击杀数等）。"""
        return False

    @property
    def chase_mode(self) -> str:
        """当前目标的瞄准模式 'pure_ai' | 'human_flick'。"""
        return "pure_ai"
