from abc import ABC, abstractmethod
from typing import Tuple
from config import config


class BaseController(ABC):
    def __init__(self):
        """
        初始化基础控制器
        Phase 5: 纯粹的运动控制基类，不再包含 Trigger 逻辑
        """
        self.config = config

        # 屏幕参数
        w = self.config.getint("General", "screen_width", 1920)
        h = self.config.getint("General", "screen_height", 1080)
        self.screen_center = (w / 2, h / 2)

    @abstractmethod
    def compute(self, intent_dx: float, intent_dy: float, dt: float) -> Tuple[float, float]:
        """
        核心计算接口
        Args:
            intent_dx/dy: Strategy 传来的几何意图 (Counts)
            dt: 距离上一帧的时间间隔
        Returns:
            (out_x, out_y): 最终硬件执行量
        """
        pass

    @abstractmethod
    def tick_mouse(self) -> Tuple[int, int]:
        """
        1000Hz 轮询接口
        """
        pass