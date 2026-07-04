# test/scenarios/peek/__init__.py
"""
Peek 场景 —— 目标从掩体后突然出现/消失, 测试 IMM-Kalman CT 模型响应。

包含:
  - scenario.py: PeekScenario 场景定义
"""

from .scenario import PeekScenario

__all__ = ["PeekScenario"]
