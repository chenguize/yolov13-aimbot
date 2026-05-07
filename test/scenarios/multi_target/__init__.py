# test/scenarios/multi_target/__init__.py
"""
多目标追踪场景 —— 占位模块。

计划特性：
  - 2-5 个同时活跃目标，不同运动模式
  - TrackManager 多轨迹池真实测试
  - 目标切换延迟测量
  - 并发威胁优先级选择

后续可扩展: 配置文件、目标模板、运动模式生成器等。
"""

from .scenario import MultiTargetScenario

__all__ = ["MultiTargetScenario"]
