# world_model.py
# 修复版：绝对坐标系因果对冲 + 固定lead补偿 + 高响应卡尔曼滤波
# 核心改进：
# 1. 引入 ego_pos_px 记录相机（准星）的绝对位移，打破动坐标系死循环。
# 2. 视觉检测在绝对坐标系下滤波，彻底解决“越追越慢”的阻尼Bug。
# 3. 释放卡尔曼 Q 矩阵，极大提高对变向的响应速度。

import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict

from config import config
from perception.ring_buffer import RingBuffer
from utils.types import Detection, InferenceContext
from aim_strategies.factory import create_aim_strategy

try:
    from numba import jit

    HAS_NUMBA = True
except ImportError:
    def jit(*args, **kwargs):
        def deco(fn):
            return fn

        return deco


    HAS_NUMBA = False


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def kf6_predict(state, cov, Q, dt):
    F = np.eye(6)
    F[0, 2] = dt
    F[1, 3] = dt
    F[0, 4] = 0.5 * dt * dt
    F[1, 5] = 0.5 * dt * dt
    F[2, 4] = dt
    F[3, 5] = dt

    predicted_state = F @ state
    predicted_cov = F @ cov @ F.T + Q
    return predicted_state, predicted_cov


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def kf6_update(state, cov, R, meas):
    H = np.zeros((2, 6))
    H[0, 0] = 1.0
    H[1, 1] = 1.0

    innovation = meas - H @ state
    S = H @ cov @ H.T + R
    S_inv = np.linalg.inv(S)
    K = cov @ H.T @ S_inv

    updated_state = state + K @ innovation
    I_KH = np.eye(6) - K @ H
    updated_cov = I_KH @ cov @ I_KH.T + K @ R @ K.T
    return innovation, updated_state, updated_cov


@dataclass
class TargetState:
    id: int
    first_seen: float
    last_seen: float
    state: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(6, dtype=np.float64) * 100.0)
    confidence: float = 0.3


class SimpleKalman:
    def __init__(self):
        self.R_base = config.getfloat("Kalman", "R", 2.0)
        self.R = np.eye(2, dtype=np.float64) * self.R_base

        q_pos = config.getfloat("Kalman", "Q_pos", 2.0)
        q_vel = config.getfloat("Kalman", "Q_vel", 30.0)
        q_acc = config.getfloat("Kalman", "Q_acc", 150.0)

        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_acc, q_acc])

    def predict(self, target: TargetState, dt: float):
        target.state, target.covariance = kf6_predict(target.state, target.covariance, self.Q, dt)

    def update(self, target: TargetState, measurement: np.ndarray):
        innovation, target.state, target.covariance = kf6_update(target.state, target.covariance, self.R, measurement)
        return innovation


class WorldModel:
    def __init__(self):
        self.kalman = SimpleKalman()
        self.strategy = create_aim_strategy()

        from controllers.controller_factory import get_controller
        self.controller = get_controller()

        self.capture_size = config.getint("General", "capture_size", 256)
        self.crop_center = self.capture_size / 2.0

        self.targets: Dict[int, TargetState] = {}
        self.last_ts = 0.0
        self.last_ring_time = 0.0  # [修复小毛病] 在 init 明确初始化
        self.ring_buffer: Optional[RingBuffer] = None

        self.frame_ready_event = threading.Event()
        self._data_lock = threading.Lock()
        self._latest_detection_data: Optional[Dict] = None

        self.last_mode = "track"
        self.mode_threshold_high = config.getfloat("Controller", "mode_threshold_high", 50.0)
        self.mode_threshold_low = config.getfloat("Controller", "mode_threshold_low", 35.0)

        self.base_hardware_lag = config.getfloat("WorldModel", "base_hardware_lag", 0.015)

        self.ego_pos_px = np.zeros(2, dtype=np.float64)

    def update_detections(self, detections: np.ndarray, frame_id: int, t_capture: float, t_done: float):
        with self._data_lock:
            self._latest_detection_data = {
                "detections": detections,
                "frame_id": frame_id,
                "t_capture": t_capture
            }
        self.frame_ready_event.set()

    def step(self, context: InferenceContext, ring_buffer: RingBuffer):
        self.ring_buffer = ring_buffer
        now = time.perf_counter()

        # ========== 1. 时间管理 ==========
        if self.last_ts == 0:
            dt = 0.001
        else:
            raw_dt = now - self.last_ts
            dt = float(np.clip(raw_dt, 0.0005, 0.05))
        self.last_ts = now

        # ========== 2. 因果对冲：消除重复叠加 ==========
        if self.last_ring_time == 0.0:
            self.last_ring_time = now - dt

        dx_counts, dy_counts = ring_buffer.get_cursor_delta_sum(self.last_ring_time, now + 0.0001)
        self.last_ring_time = now

        px_dx, px_dy = self.strategy.reverse_map(float(dx_counts), float(dy_counts))

        self.ego_pos_px[0] += px_dx
        self.ego_pos_px[1] += px_dy

        # ========== 3. 检测结果处理 ==========
        best_detection = context.targets[0] if context.targets else None

        if not best_detection:
            context.p_predict = None
            context.v_real = (0.0, 0.0)
            context.conf = 0.0
            # [核心神级修复：清零绝对坐标系] 当屏幕内没敌人的瞬间，将因果积分归零，彻底消灭浮点数漂移危机！
            if not any(now - t.last_seen < 3.0 for t in self.targets.values()):
                if np.linalg.norm(self.ego_pos_px) < 50:  # 准星近中心
                    self.ego_pos_px[:] = 0.0
            return

        measurement = np.array([
            best_detection.x - context.center_pos[0],
            best_detection.y - context.center_pos[1]
        ], dtype=np.float64)

        abs_measurement = measurement + self.ego_pos_px

        # ========== 4. 卡尔曼滤波 (在绝对坐标系下进行) ==========
        target_id = int(best_detection.class_id)
        target = self.targets.get(target_id)

        if target is None:
            target = TargetState(id=target_id, first_seen=now, last_seen=now)
            target.state[:2] = abs_measurement
            self.targets[target_id] = target
        else:
            target.last_seen = now

        self.kalman.predict(target, dt)
        innovation = self.kalman.update(target, abs_measurement)

        # ========== 5. 状态提取 (绝对状态) ==========
        abs_position = target.state[:2].copy()
        abs_velocity = target.state[2:4].copy()

        # ========== 6. 动态预测未来位置 (补偿真实延迟) ==========
        t_capture = now
        with self._data_lock:
            if self._latest_detection_data:
                t_capture = self._latest_detection_data.get("t_capture", now)

        software_lag = now - t_capture
        dynamic_lead_time = software_lag + self.base_hardware_lag
        dynamic_lead_time = float(np.clip(dynamic_lead_time, 0.005, 0.080))

        pred_abs_pos = abs_position + abs_velocity * dynamic_lead_time
        context.dynamic_lag_ms = dynamic_lead_time * 1000.0

        # ========== 7. 映射回相对误差域给控制器 ==========
        rel_pred_pos = pred_abs_pos - self.ego_pos_px

        # ========== 8. 防抖模式切换 ==========
        error_distance = np.linalg.norm(measurement)
        threshold = self.mode_threshold_high if self.last_mode == "track" else self.mode_threshold_low
        self.controller.mode = "flick" if error_distance > threshold else "track"
        self.last_mode = self.controller.mode

        # ========== 9. 传递给上下文 ==========
        context.p_predict = (rel_pred_pos[0], rel_pred_pos[1])
        context.v_real = (abs_velocity[0], abs_velocity[1])
        context.conf = 1.0

    def cleanup_old_targets(self, max_age: float = 2.0):
        now = time.perf_counter()
        to_remove = [tid for tid, tgt in self.targets.items() if now - tgt.last_seen > max_age]
        for tid in to_remove:
            del self.targets[tid]