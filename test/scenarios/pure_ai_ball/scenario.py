# test/scenarios/pure_ai_ball/scenario.py
"""
纯 AI 瞄准场景 —— 与 Neon 小球追踪 **相同的机动与击杀规则**，
但 **每局固定 chase_mode=pure_ai**，且 **不注入人类鼠标输入**。

用途：
  - 对比 [WorldModel] predict_ahead / ctrl_lead / 延迟自适应 开闭时的 TTK 与精度
  - 排除 human_flick 干扰，专注闭环 + 预测管线
"""

from typing import TYPE_CHECKING

from test.scenarios.ball_tracking import BallTrackingScenario

if TYPE_CHECKING:
    from test.sim_agent import SimAIAgent


class PureAIBallScenario(BallTrackingScenario):
    """每击杀均为 pure_ai 出生（60~256px），无甩枪模拟。"""

    def __init__(self, max_kills: int = 30):
        super().__init__(max_kills=max_kills, pure_ai_only=True)
