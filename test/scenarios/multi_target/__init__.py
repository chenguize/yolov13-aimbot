# test/scenarios/multi_target/__init__.py
"""
多目标追踪场景 —— 2-4 个目标同时活跃。

特性:
  - 每个目标独立 Neon 风格身法 (sprint / adad / slide / stop)
  - 不同 3D 距离和运动方向
  - TrackManager 多轨迹池真实测试
  - 目标切换延迟测量 (击杀后切换最近目标)
  - 并发威胁优先级选择 (距离最近 = 最高优先级)
  - ReID 颜色直方图测试 (目标 hue 相近但有差异)

包含:
  - scenario.py: MultiTargetScenario 场景定义
"""

from .scenario import MultiTargetScenario

__all__ = ["MultiTargetScenario"]
