# world_model.py
# Phase 3 核心组件：世界状态仲裁与因果对冲中心
#
# 职责：
# 1. 接收 Inference 的原始检测结果
# 2. 结合 RingBuffer 进行因果位移对冲 (Causal Hedging)
# 3. 使用 Kalman Filter 维护目标真实状态 (Pixel Coordinates)
# 4. (可选) 进行时延补偿预测

import time
import math
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from config import config

# 引入项目内依赖
from perception.ring_buffer import RingBuffer
from utils.types import Detection, InferenceContext


@dataclass
class TargetState:
    """
    单个目标的跟踪状态 (Kalman Filter 容器)
    """
    id: int
    first_seen: float
    last_seen: float

    # Kalman State: [x, y, vx, vy]
    # x, y: 屏幕绝对像素坐标
    state: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # Covariance Matrix: P
    covariance: np.ndarray = field(default_factory=lambda: np.eye(4) * 100.0)

    # 连续跟踪帧数 (用于置信度熔断)
    hit_streak: int = 0


class SimpleKalman:
    """
    轻量级 Kalman Filter (Constant Velocity Model)
    """

    def __init__(self):
        # 状态转移矩阵 F (dt 将在 predict 时动态注入)
        self.F = np.eye(4)

        # 观测矩阵 H (我们只能观测 x, y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # 测量噪声 R (YOLO 的抖动幅度)
        noise_pos = config.getfloat("WorldModel", "kalman_R_pos", 5.0)
        self.R = np.eye(2) * noise_pos

        # 过程噪声 Q (目标运动的不确定性)
        noise_proc = config.getfloat("WorldModel", "kalman_Q_proc", 0.5)
        self.Q = np.eye(4) * noise_proc

        # 预分配单位矩阵
        self.I = np.eye(4)

    def predict(self, state_obj: TargetState, dt: float):
        """预测步骤: X = F * X"""
        if dt <= 0: return
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        state_obj.state = self.F @ state_obj.state
        state_obj.covariance = self.F @ state_obj.covariance @ self.F.T + self.Q

    def update(self, state_obj: TargetState, measurement: np.ndarray):
        """更新步骤: 融合观测值"""
        z = measurement
        y = z - (self.H @ state_obj.state)
        S = self.H @ state_obj.covariance @ self.H.T + self.R
        try:
            K = state_obj.covariance @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        state_obj.state = state_obj.state + (K @ y)
        state_obj.covariance = (self.I - K @ self.H) @ state_obj.covariance


class WorldModel:
    """
    Phase 3: 世界模型核心
    """

    def __init__(self):
        self.kalman = SimpleKalman()
        self.current_target: Optional[TargetState] = None

        # --- 数据缓冲 (线程安全) ---
        self._data_lock = threading.Lock()
        self._latest_detection_data: Optional[Dict] = None

        # --- 配置参数 ---
        self.fov_x = config.getfloat("Triggerbot", "trigger_fov_x", 150.0)
        self.fov_y = config.getfloat("Triggerbot", "trigger_fov_y", 150.0)
        self.max_coast_frames = 5

        # [Phase 3] 预测开关与延迟参数
        # 建议先设为 False，跑通后再开
        self.enable_prediction = config.getbool("WorldModel", "enable_prediction", False)
        self.system_latency = config.getfloat("WorldModel", "system_latency", 0.045)

        # 屏幕中心
        self.screen_w = config.getint("General", "screen_width", 1920)
        self.screen_h = config.getint("General", "screen_height", 1080)
        self.center_x = self.screen_w / 2
        self.center_y = self.screen_h / 2

        self.last_t_cap = 0.0

        # 灵敏度系数 (Counts per Pixel) - 用于位移对冲
        self.k_factor_x = config.getfloat("AimStrategy", "k_factor_x", 1.9)
        self.k_factor_y = config.getfloat("AimStrategy", "k_factor_y", 1.2)

    def update_detections(self, detections: List[list], frame_id: int, t_cap: float, t_done: float):
        """[Inference Thread 调用]"""
        with self._data_lock:
            self._latest_detection_data = {
                "dets": detections,
                "fid": frame_id,
                "t_cap": t_cap,
                "t_done": t_done
            }

    def step(self, context: InferenceContext, ring_buffer: RingBuffer):
        """[Main Thread 调用]"""

        # 1. 提取数据
        data = None
        with self._data_lock:
            data = self._latest_detection_data

        if not data:
            context.is_valid = False
            return

        # 2. 填充 Context
        current_dets = []
        for d in data['dets']:
            cx = (d[0] + d[2]) / 2
            cy = (d[1] + d[3]) / 2
            w = d[2] - d[0]
            h = d[3] - d[1]
            current_dets.append(Detection(
                x=cx, y=cy, w=w, h=h,
                conf=d[4], class_id=int(d[5]),
                xyxy=(d[0], d[1], d[2], d[3])
            ))

        context.targets = current_dets
        context.t_cap = data['t_cap']
        context.t_inference_done = data['t_done']

        # 3. 目标仲裁
        best_det = self._select_target(current_dets)

        current_time = context.t_cap
        dt = current_time - self.last_t_cap

        if dt <= 0: dt = 0.001
        if dt > 0.5: dt = 0.1

        # --- A. Reset ---
        if not self.current_target and not best_det:
            context.is_valid = False
            self.last_t_cap = current_time
            return

        # --- B. Init ---
        if best_det and (not self.current_target or self.current_target.id != best_det.class_id):
            self.current_target = TargetState(
                id=best_det.class_id,
                first_seen=current_time,
                last_seen=current_time
            )
            self.current_target.state[:2] = [best_det.x, best_det.y]
            context.is_valid = False
            context.selected_id = best_det.class_id
            self.last_t_cap = current_time
            return

        # --- C. Tracking & Hedging ---
        if self.current_target and best_det:
            target = self.current_target

            # 1. 因果位移对冲 (Hedging)
            # 计算这一帧之间，鼠标动了多少，并转换回像素
            mouse_counts_x, mouse_counts_y = ring_buffer.get_cursor_delta_sum(self.last_t_cap, current_time)

            kx = self.k_factor_x if self.k_factor_x > 0.1 else 1.0
            ky = self.k_factor_y if self.k_factor_y > 0.1 else 1.0

            pixel_shift_x = mouse_counts_x / kx
            pixel_shift_y = mouse_counts_y / ky

            # 还原真实世界坐标：观测值 + 鼠标位移导致的视觉反向偏移
            z_measured = np.array([
                best_det.x + pixel_shift_x,
                best_det.y + pixel_shift_y
            ])

            # 2. Kalman 更新
            self.kalman.predict(target, dt)
            self.kalman.update(target, z_measured)

            target.last_seen = current_time
            target.hit_streak += 1

            vx, vy = target.state[2], target.state[3]

            # 熔断机制
            if target.hit_streak < 3:
                context.is_valid = False
            else:
                context.is_valid = True

            context.v_real = (vx, vy)
            context.selected_id = target.id

            # --- D. 时延预测 (Latency Prediction) [可开关] ---
            if self.enable_prediction:
                # 开启预测：P_final = P_curr + V * Latency
                t_predict_total = self.system_latency
                # 这里还可以加上 (time.perf_counter() - t_cap) 来补偿计算耗时
            else:
                # 关闭预测：只瞄准当前 Kalman 滤波后的位置
                t_predict_total = 0.0

            pred_x = target.state[0] + vx * t_predict_total
            pred_y = target.state[1] + vy * t_predict_total

            context.p_predict = (pred_x, pred_y)

        # --- E. Coasting ---
        elif self.current_target and not best_det:
            time_since_lost = current_time - self.current_target.last_seen
            if time_since_lost < (self.max_coast_frames * 0.02):
                target = self.current_target
                self.kalman.predict(target, dt)
                context.is_valid = False  # 盲预测暂时不瞄准
            else:
                self.current_target = None
                context.is_valid = False

        self.last_t_cap = current_time

    def _select_target(self, targets: List[Detection]) -> Optional[Detection]:
        if not targets: return None
        best_target = None
        min_dist = float('inf')
        for t in targets:
            dx = t.x - self.center_x
            dy = t.y - self.center_y
            if abs(dx) > self.fov_x or abs(dy) > self.fov_y: continue
            dist = dx * dx + dy * dy
            if dist < min_dist:
                min_dist = dist
                best_target = t
        return best_target