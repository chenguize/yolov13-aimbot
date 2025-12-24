# controllers/base_controller.py - 控制器抽象基类
#
# 核心职责：
# 1. 定义控制器统一接口
# 2. 提供基本配置和屏幕参数
# 3. 实现扳机判断逻辑
#
# 架构特点：抽象基类，定义统一计算接口
#

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, Any
from config import config


class BaseController(ABC):
    """
    所有控制器的抽象基类
    定义了统一的计算接口和基本的屏幕中心计算
    """

    def __init__(self):
        """初始化基础控制器参数"""
        self.config = config  # 配置引用
        # 计算屏幕中心点
        self.screen_center = (
            self.config.getint("General", "screen_width", 1920) / 2,
            self.config.getint("General", "screen_height", 1080) / 2
        )
        # 死区阈值，小于此值的移动不执行
        self.deadzone = self.config.getfloat("Output", "deadzone_pixels", 1.2)

    @abstractmethod
    def compute(
        self,
        target: Optional[Dict[str, Any]],
        current_mouse_pos: Tuple[float, float],
        dt: float
    ) -> Tuple[float, float]:
        """
        计算本次鼠标移动量 (dx, dy) - 像素单位
        所有控制器必须实现此方法
        
        Args:
            target: 来自 world_model 的最佳目标字典，可能为 None
                    至少包含: screen_x, screen_y, conf
            current_mouse_pos: 当前鼠标屏幕绝对坐标 (x, y)
            dt: 本帧到上一帧的时间间隔（秒）

        Returns:
            Tuple[float, float]: 本次要移动的像素量 (dx, dy)
        """
        pass

    def should_trigger(self, target: Optional[Dict[str, Any]], current_mouse_pos: Tuple[float, float]) -> bool:
        """
        判断当前帧是否应该触发扳机（triggerbot独立判断）
        默认使用准星中心与目标中心的距离 + 置信度阈值
        """
        if not target:
            return False

        tx = target.get("screen_x", self.screen_center[0])
        ty = target.get("screen_y", self.screen_center[1])
        # 计算准星与目标的距离
        distance = ((tx - current_mouse_pos[0]) ** 2 + (ty - current_mouse_pos[1]) ** 2) ** 0.5

        # 检查距离是否在触发FOV内，且目标置信度是否足够
        return (
            distance <= self.config.getfloat("Triggerbot", "trigger_fov", 15.0) and
            target.get("conf", 0.0) >= self.config.getfloat("Triggerbot", "trigger_conf_threshold", 0.55)
        )