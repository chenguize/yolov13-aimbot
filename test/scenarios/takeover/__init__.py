# test/scenarios/takeover/__init__.py
"""
AI 接管能力测试场景。

测试 AI 在人类瞄准中途接手的能力:
  - undershoot: 人类故意打短，AI 补完最后一段
  - overshoot: 人类打过头，AI 反向修正
  - near_miss: 人类擦边，AI 微调入魂

核心指标:
  - 接管平滑度 (velocity continuity at handoff)
  - 接管延迟 (release → lock)
  - 最终精度 (post-takeover error)

与 ball_tracking 的差异:
  - 目标始终在 AI 视野内 (200-600px), 而非 FOV 外 (600-1400px)
  - 人类甩枪故意不瞄准 (15-40% 误差), 而非大致到位
  - AI 在人类减速阶段就开始混合介入 (HumanIntentTracker 方向融合)
"""

from .scenario import TakeoverScenario

__all__ = ["TakeoverScenario"]
