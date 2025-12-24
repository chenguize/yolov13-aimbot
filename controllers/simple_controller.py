# controllers/simple_controller.py - 简单比例控制器
#
# 核心职责：
# 1. 实现基础比例控制算法
# 2. 提供最基础的瞄准功能
# 3. 作为调试和低灵敏度场景的备选方案
#
# 架构特点：直接线性映射误差，最基础、最可预测
#

from .base_controller import BaseController
from typing import Tuple, Dict, Optional, Any


class SimpleController(BaseController):
    """
    简单比例控制器 - 直接线性映射误差
    最基础、最可预测的实现，适合调试和低灵敏度场景
    """

    def __init__(self):
        """初始化简单控制器"""
        super().__init__()
        # 从配置中读取灵敏度参数
        self.sensitivity = self.config.getfloat("Controller", "sensitivity", 0.42)

    def compute(
        self,
        target: Optional[Dict[str, Any]],
        current_mouse_pos: Tuple[float, float],
        dt: float
    ) -> Tuple[float, float]:
        """
        计算鼠标移动量 - 使用比例控制算法
        """
        if not target:
            return 0.0, 0.0  # 无目标时返回零移动

        # 获取目标屏幕坐标
        tx = target.get("screen_x", self.screen_center[0])
        ty = target.get("screen_y", self.screen_center[1])

        # 计算误差
        error_x = tx - current_mouse_pos[0]
        error_y = ty - current_mouse_pos[1]

        # 应用死区 - 小于死区的误差不处理
        if abs(error_x) <= self.deadzone and abs(error_y) <= self.deadzone:
            return 0.0, 0.0

        # 比例映射 - 直接乘以灵敏度系数
        move_x = error_x * self.sensitivity
        move_y = error_y * self.sensitivity

        return move_x, move_y