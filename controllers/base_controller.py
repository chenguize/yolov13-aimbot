# controllers/base_controller.py
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, Any
from config import config


class BaseController(ABC):
    """
    所有控制器的抽象基类
    定义了统一的计算接口和基本的屏幕中心计算
    """

    def __init__(self):
        self.config = config
        self.screen_center = (
            self.config.getint("General", "screen_width", 1920) / 2,
            self.config.getint("General", "screen_height", 1080) / 2
        )
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
        distance = ((tx - current_mouse_pos[0]) ** 2 + (ty - current_mouse_pos[1]) ** 2) ** 0.5

        return (
            distance <= self.config.getfloat("Triggerbot", "trigger_fov", 15.0) and
            target.get("conf", 0.0) >= self.config.getfloat("Triggerbot", "trigger_conf_threshold", 0.55)
        )