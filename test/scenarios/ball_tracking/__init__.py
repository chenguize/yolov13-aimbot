# test/scenarios/ball_tracking/__init__.py
"""
小球追踪场景 —— Valorant 霓虹 (Neon) 风格单目标追踪。

包含:
  - scenario.py: BallTrackingScenario 场景定义
  - 后续可扩展: 配置文件、辅助函数、变体场景等
"""

from .scenario import BallTrackingScenario

__all__ = ["BallTrackingScenario"]
