# world_model.py - 世界模型模块（已彻底停用预测与平滑）

import threading
import time
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
# from filterpy.kalman import KalmanFilter # 不再需要
from config import config
from perception.bus import FrameBus


class WorldModel:
    """
    状态中枢 - 仅负责坐标转换
    [已阉割] 删除了 Kalman 预测、平滑以及自运动补偿
    """

    def __init__(self, frame_bus: FrameBus):
        self._lock = threading.Lock()
        self.bus = frame_bus
        self.current_detections: List[Dict] = []
        self.last_update_time = 0.0
        self.last_frame_id = -1

        # [移除] 不再初始化 Kalman 滤波器
        # self.kf = ...

        # [保留] 尽管不再用于预测，保留反馈接口以防 main.py 报错
        self.last_control_u = np.array([0.0, 0.0])

    def update_detections(self, detections: List[List[float]], frame_id: int, timestamp: float):
        with self._lock:
            sw = config.getint("General", "screen_width", 1920)
            sh = config.getint("General", "screen_height", 1080)
            cap_size = config.getint("General", "capture_size", 256)

            ref_x, ref_y = sw // 2, sh // 2
            half_size = cap_size / 2

            new_dets = []
            for det in detections:
                # 安全检查：确保 det 至少包含 [x1, y1, x2, y2, conf, cls] 6个元素
                if not det or len(det) < 6:
                    continue

                x1, y1, x2, y2, conf, cls = det
                # 只有当类别 ID 是 7 时才进行坐标转换和添加
                # 这里使用 float(cls) 并转 int，可以处理 7.0 的情况
                if int(float(cls)) == 7:
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    screen_x = ref_x + (center_x - half_size)
                    screen_y = ref_y + (center_y - half_size)

                    new_dets.append({
                        "screen_x": screen_x,
                        "screen_y": screen_y,
                        "conf": conf,
                        "cls": 7,  # 统一存为整数 7
                        "width": x2 - x1,
                        "height": y2 - y1
                    })

            self.current_detections = new_dets
            self.last_update_time = timestamp
            self.last_frame_id = frame_id
    def receive_control_feedback(self, dx: float, dy: float):
        """空实现，仅为了兼容 main.py 的调用"""
        pass

    def get_best_target(self) -> Optional[Dict]:
        """直接从当前检测中选择距离屏幕中心最近的目标，无平滑"""
        with self._lock:
            if not self.current_detections:
                return None

            # 获取屏幕中心基准
            sw = config.getint("General", "screen_width", 1920)
            sh = config.getint("General", "screen_height", 1080)
            screen_center = (sw // 2, sh // 2)

            # [修改] 不再读取 kf_pos，直接基于 current_detections 寻找最佳目标
            best = min(
                self.current_detections,
                key=lambda d: (d["screen_x"] - screen_center[0]) ** 2 + (d["screen_y"] - screen_center[1]) ** 2
            )
            print("已选择目标：", best)
            # 返回原始检测数据，不经过任何平滑处理
            return best