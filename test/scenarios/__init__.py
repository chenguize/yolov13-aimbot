# test/scenarios/__init__.py
"""
测试场景包。

每个场景是一个自包含子包，实现 BaseScenario 接口：
  - init():       模拟启动初始化
  - spawn():      生成新目标 (位置 / 运动 / 模式)
  - tick_physics(): 每帧更新目标物理
  - tick_human_input(): 每帧计算人类输入
  - check_kill(): 每帧判定击杀
  - is_done():    模拟是否结束

引擎 (sim_agent.SimAIAgent) 持有场景实例，按帧委托。

场景目录结构:
  scenarios/
    base.py                  ← 场景基类
    ball_tracking/           ← 小球追踪 (Neon 单目标)
    pure_ai_ball/            ← 纯 AI 追踪 (同霓虹身法，无 human_flick)
    multi_target/            ← 多目标追踪 (待实现)
    ...
"""
from test.scenarios.base import BaseScenario
from test.scenarios.ball_tracking import BallTrackingScenario
from test.scenarios.multi_target import MultiTargetScenario
from test.scenarios.takeover import TakeoverScenario
from test.scenarios.pure_ai_ball import PureAIBallScenario

__all__ = [
    "BaseScenario",
    "BallTrackingScenario",
    "MultiTargetScenario",
    "TakeoverScenario",
    "PureAIBallScenario",
]
