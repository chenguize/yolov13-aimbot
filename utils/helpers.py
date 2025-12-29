# utils/helpers.py - 通用工具函数
#
# 核心职责：
# 1. 提供数学计算辅助函数
# 2. 实现常用的数值处理算法
# 3. 为其他模块提供基础工具支持
#

import math
from typing import Union, Tuple


def clamp(value: Union[float, int], min_val: Union[float, int], max_val: Union[float, int]) -> float:
    """将值限制在 [min_val, max_val] 区间内"""
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """线性插值 - 在 a 和 b 之间根据 t 进行插值"""
    return a + (b - a) * t


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """平滑阶跃函数（Hermite插值）- 在 [edge0, edge1] 区间内提供平滑过渡"""
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def inv_lerp(a: float, b: float, value: float) -> float:
    """反向线性插值 - 返回 value 在 [a, b] 区间中的相对位置 [0, 1]"""
    if abs(b - a) < 1e-6:
        return 0.0
    return clamp((value - a) / (b - a), 0.0, 1.0)


def letterbox(img: any, new_shape: Tuple[int, int] = (256, 256), color: Tuple[int, int, int] = (114, 114, 114)):
    """
    实现 letterbox 预处理 - 将图像调整为指定尺寸并保持宽高比
    在 YOLO 检测中常用，确保图像不失真的同时适应模型输入
    """
    # 获取原图像尺寸
    if hasattr(img, 'shape'):
        h, w = img.shape[:2]
    else:
        # 如果 img 不是 numpy 数组，假定为 PIL 图像
        w, h = img.size if hasattr(img, 'size') else (256, 256)

    # 计算缩放比例
    ratio = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * ratio)), int(round(h * ratio)))

    # 计算填充尺寸
    dw = new_shape[1] - new_unpad[0]  # width padding
    dh = new_shape[0] - new_unpad[1]  # height padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    # 如果是 numpy 数组，执行实际的 letterbox 操作
    if hasattr(img, 'shape') and len(img.shape) >= 2:
        import cv2
        # 调整图像大小
        resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        # 创建新画布
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        result = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return result, ratio, (dw, dh)
    else:
        # 如果无法处理，返回原始尺寸信息
        return img, ratio, (dw, dh)