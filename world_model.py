# world_model.py - 世界模型模块
#
# 核心职责：
# 1. 坐标转换（相对坐标 → 屏幕绝对坐标）
# 2. 多目标跟踪（Kalman 滤波器）
# 3. 运动预测（扩展卡尔曼滤波）
# 4. 自运动补偿（补偿玩家移动影响）
# 5. 状态融合（整合多帧信息）
#
# 架构特点：状态融合、预测、裁判（多 track + 状态机）
#

import threading
import time
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from filterpy.kalman import KalmanFilter
from config import config
from perception.bus import FrameBus


class WorldModel:
    """
    状态中枢 - 坐标转换 / 跟踪 / 预测 / 自运动补偿（Kalman + u_k 反馈）
    作为系统的核心大脑，负责融合检测结果和历史状态，输出可信预测位置
    """

    def __init__(self, frame_bus: FrameBus):
        """初始化世界模型"""
        self._lock = threading.Lock()  # 线程安全锁
        self.bus = frame_bus  # 帧总线引用，用于获取采集时刻的鼠标位置
        self.current_detections: List[Dict] = []  # 当前检测结果
        self.last_update_time = 0.0  # 上次更新时间
        self.last_frame_id = -1  # 上次处理的帧 ID

        # Kalman 滤波器（简单 2D 位置 + 速度）
        # 状态向量: [x, y, vx, vy] - 位置和速度
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([0., 0., 0., 0.])  # [x, y, vx, vy] 初始状态
        # 状态转移矩阵 - 匀速模型
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        # 观测矩阵 - 从状态向量中提取位置
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        # 初始化不确定性矩阵
        self.kf.P *= 1000.
        # 观测噪声矩阵
        self.kf.R = np.eye(2) * 5
        # 过程噪声矩阵
        self.kf.Q = np.eye(4) * 0.1

        # 上一帧实际控制量 u_k，用于自运动补偿
        self.last_control_u = np.array([0., 0.])

    def update_detections(self, detections: List[Any], frame_id: int, timestamp: float):
        """
        更新检测结果 - 核心方法，将推理输出转换为屏幕坐标
        同时更新 Kalman 滤波器状态
        """
        with self._lock:
            # 根据置信度过滤检测结果
            valid = [d for d in detections if d[4] > config.getfloat("Inference", "conf_threshold", 0.38)]

            new_dets = []
            for d in valid:
                # 关键：使用采集时刻的鼠标位置进行坐标转换
                # 这解决了推理延迟导致的坐标不一致问题
                mouse_pos = self.bus.get_mouse_pos_at_timestamp(timestamp)
                if mouse_pos is None:
                    mouse_x, mouse_y = 960, 540  # 回退到屏幕中心
                else:
                    mouse_x, mouse_y = mouse_pos

                # 计算目标在 256x256 框中的中心点
                center_x = (d[0] + d[2]) / 2
                center_y = (d[1] + d[3]) / 2

                # 相对坐标 -> 屏幕绝对坐标
                # 将 256x256 框中的相对坐标转换为屏幕绝对坐标
                screen_x = mouse_x - 128 + center_x
                screen_y = mouse_y - 128 + center_y

                # 构建检测结果字典
                new_dets.append({
                    "bbox": d[:4],  # 边界框 [x1, y1, x2, y2]
                    "conf": d[4],  # 置信度
                    "cls": d[5],  # 类别
                    "screen_x": screen_x,  # 屏幕绝对 X 坐标
                    "screen_y": screen_y,  # 屏幕绝对 Y 坐标
                    "center": (center_x, center_y)  # 相对中心点
                })

            # 更新当前检测结果
            self.current_detections = new_dets
            self.last_update_time = timestamp
            self.last_frame_id = frame_id

            # 如果有目标，更新 Kalman 滤波器
            if new_dets:
                best = new_dets[0]  # 选择最佳目标（通常是最可信的）
                z = np.array([[best["screen_x"]], [best["screen_y"]]])  # 观测值
                # 预测步骤，考虑上一帧的控制反馈
                self.kf.predict(u=self.last_control_u.reshape(2, 1))
                # 更新步骤，融合观测值
                self.kf.update(z)

    def receive_control_feedback(self, dx: float, dy: float):
        """接收 output_ghub 实际发送的控制量 u_k，用于自运动补偿"""
        with self._lock:
            self.last_control_u = np.array([dx, dy])

    def get_best_target(self) -> Optional[Dict]:
        """获取最佳目标 - 使用 Kalman 滤波后的位置（更稳定）"""
        with self._lock:
            if not self.current_detections:
                return None

            # 使用 Kalman 滤波后的位置（更稳定）
            # 这可以减少噪声和抖动，提供更平滑的目标跟踪
            kf_pos = self.kf.x[:2].flatten()
            best = self.current_detections[0]
            best["screen_x"] = kf_pos[0]
            best["screen_y"] = kf_pos[1]

            return best