# utils/types.py
# Phase 3: 核心数据结构定义
#
# 包含：
# 1. FrameInfo (原子帧信息)
# 2. Detection (标准化检测目标)
# 3. StrategyResult (策略计算结果)
# 4. InferenceContext (Structure B - 单帧任务单 - 修复版)

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import numpy as np


@dataclass
class FrameInfo:
    """
    [Phase 3] 原子帧信息
    由 Capture 线程生成，通过 Bus 传递给 Inference。
    """
    frame: np.ndarray  # 图像数据 (BGR)
    frame_id: int  # 帧序列号
    t_cap: float  # [关键] 截图瞬间的时间锚点 (Time Anchor)
    center_pos: Tuple[int, int]  # 截图时的屏幕中心坐标 (cx, cy)


@dataclass
class Detection:
    """
    [Phase 3] 单目标检测结果
    WorldModel 内部使用的标准格式。
    """
    x: float  # 屏幕绝对坐标 X (物体中心)
    y: float  # 屏幕绝对坐标 Y (物体中心)
    w: float  # 宽度
    h: float  # 高度
    conf: float  # 置信度
    class_id: int  # 类别 ID (0=Head, 1=Body, etc.)

    # 原始检测框 (x1, y1, x2, y2)，通常是 numpy 数组切片
    xyxy: Optional[np.ndarray] = None


@dataclass
class StrategyResult:
    """
    [Phase 3] 策略层计算结果
    Strategy 只负责计算几何映射，不负责执行。
    """
    raw_counts_x: int  # 理论鼠标 X 轴移动量 (Mickeys)
    raw_counts_y: int  # 理论鼠标 Y 轴移动量

    # 以下字段用于 Trace Replay 分析，非必须
    aim_angle_yaw: float = 0.0
    aim_angle_pitch: float = 0.0


@dataclass
class InferenceContext:
    """
    [Structure B] 单帧推理上下文 / 任务单
    这是 Phase 3 的核心载体，贯穿 WorldModel -> Strategy -> Controller -> Agent。
    """
    # --- 1. 时序信息 (Timing) ---
    t_cap: float = 0.0  # 截图时间 (来自 FrameInfo)
    t_inference_done: float = 0.0  # 推理完成时间 (用于计算延迟)

    # --- 2. 感知结果 (Perception) ---
    targets: List[Detection] = field(default_factory=list)  # 本帧所有检测目标

    # --- 3. 认知状态 (Cognition / WorldModel) ---
    selected_id: int = -1  # 最终选定的目标 ID
    v_real: Tuple[float, float] = (0.0, 0.0)  # Kalman 估算的真实世界速度 (px/s)
    a_real: Tuple[float, float] = (0.0, 0.0)  # Kalman 估算的加速度 (px/s^2)
    # p_predict 用 Optional：None 代表"本帧无可用目标"。
    # 原 Tuple 默认 (0.0, 0.0) 导致 `if not ctx.p_predict` 判空失效（tuple 永真）。
    p_predict: Optional[Tuple[float, float]] = None
    is_valid: bool = False  # 本帧数据是否可信 (是否允许开火/瞄准)

    # [关键修复] 目标置信度 (用于 Triggerbot 判定)
    conf: float = 0.0

    # --- 4. 调试与反馈 (Debug / Feedback) ---
    dynamic_lag_ms: float = 0.0  # 当前系统计算出的动态延迟
    final_move: Optional[Tuple[float, float]] = None  # 本帧计算出的鼠标移动量 (dx, dy)

    # --- 5. 策略意图 (Strategy Intent - 可选) ---
    strategy_result: Optional[StrategyResult] = None