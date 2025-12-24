# controllers/base_controller.py
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional


class BaseController(ABC):
    @abstractmethod
    def compute(self, target: Dict, current_pos: Tuple[float, float], dt: float) -> Tuple[float, float]:
        """返回本次要移动的 dx, dy (像素)"""
        pass