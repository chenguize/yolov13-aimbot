# aim_strategies/valorant/strategy.py
from typing import Tuple
from config import config


class ValorantStrategy:
    """
    Valorant 专用鼠标移动映射（2025年底实战版）
    - 仅实现：高速比例映射 + 死区
    - 无任何预测、历史速度、卡尔曼、外推等功能
    - 适合纯反应式瞄准（当前帧位置 → 直接映射）
    """

    def __init__(self):
        # 从 config 读取参数（推荐值 4.5~6.0）
        self.mapping_factor = config.getfloat("Controller", "sensitivity", 5.0)
        # 死区（像素），小移动不动，防抖 + 反检测
        self.deadzone = config.getfloat("Output", "deadzone_pixels", 1.2)

    def calculate_mouse_move(
            self,
            target_x: float,
            target_y: float,
            center_x: float,
            center_y: float,
            dt: float = 0.0  # dt 参数保留兼容性，但不使用
    ) -> Tuple[float, float]:
        """
        计算本次鼠标移动量（像素）

        参数:
            target_x, target_y: 目标在屏幕上的绝对坐标（中心点）
            center_x, center_y: 参考点（屏幕中心）
            dt: 时间间隔（当前不使用，保留接口兼容）

        返回:
            (dx, dy): 本次要移动的像素量
        """
        dx = target_x - center_x
        dy = target_y - center_y

        # 死区：太小不动，防微抖动 + 反作弊
        if abs(dx) < self.deadzone and abs(dy) < self.deadzone:
            return 0.0, 0.0

        # Valorant 高灵敏度比例映射（一帧到位，反应快）
        move_x = dx * self.mapping_factor
        move_y = dy * self.mapping_factor

        return move_x, move_y