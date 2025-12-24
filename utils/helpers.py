# utils/helpers.py
import math
from typing import Union, Tuple


def clamp(value: Union[float, int], min_val: Union[float, int], max_val: Union[float, int]) -> float:
    """将值限制在 [min_val, max_val] 区间内"""
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """线性插值"""
    return a + (b - a) * t


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """平滑阶跃函数（Hermite插值）"""
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def inv_lerp(a: float, b: float, value: float) -> float:
    """反向线性插值，返回 t 值"""
    if abs(b - a) < 1e-6:
        return 0.0
    return clamp((value - a) / (b - a), 0.0, 1.0)