# world_model.py
# Tier S 修复版：Y轴松绑 + 加速度补偿自然 + 延迟自适应（保留全部原有结构）

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
        def deco(fn): return fn
        return deco
    HAS_NUMBA = False


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def kf6_predict(state, cov, Q, dt):
    F = np.eye(6)
    F[0, 2] = dt; F[1, 3] = dt
    F[0, 4] = 0.5 * dt * dt; F[1, 5] = 0.5 * dt * dt
    F[2, 4] = dt; F[3, 5] = dt
    predicted_state = F @ state
    predicted_cov = F @ cov @ F.T + Q
    return predicted_state, predicted_cov


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def kf6_update(state, cov, R, meas):
    H = np.zeros((2, 6)); H[0, 0] = 1.0; H[1, 1] = 1.0
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

        q_pos = config.getfloat("Kalman", "Q_pos", 5.0)
        q_vel = config.getfloat("Kalman", "Q_vel", 80.0)
        q_acc = config.getfloat("Kalman", "Q_acc", 200.0)

        self.Q_base = np.diag([q_pos, q_pos, q_vel, q_vel, q_acc, q_acc])
        self.Q = self.Q_base.copy()
        self.innov_ema = 0.0

    def predict(self, target: TargetState, dt: float):
        target.state, target.covariance = kf6_predict(target.state, target.covariance, self.Q, dt)

    def update(self, target: TargetState, measurement: np.ndarray):
        innovation, target.state, target.covariance = kf6_update(target.state, target.covariance, self.R, measurement)
        innov_mag = np.linalg.norm(innovation)

        # ── Fix: 连续平滑 Q 自适应，消灭阶跃跳变 ─────────────────────────────────
        # 旧代码在 innov_ema=12 处硬切换 Q_base*0.6 ↔ Q_base*3.5，产生每帧 5.8x 的
        # 过程噪声阶跃。卡尔曼在 innov_ema 徘徊 12 附近时反复跳变，直接在速度估计上
        # 注入高频噪声，是 Y 轴 ±7.5px 持续震荡的根本原因。
        # 修复：用 EMA 更慢地追踪 innov，再用平滑 sigmoid 映射到 [0.5, 4.0] 范围，
        # 彻底消除不连续跳变。
        # 【大幅度减弱卡尔曼的神经质响应】
        if innov_mag > self.innov_ema:
            # 降低 Fast Attack 的权重，不让 Q 飙升得太快
            self.innov_ema = 0.8 * self.innov_ema + 0.2 * innov_mag
        else:
            # 稍微加快 Decay，让它更快回落到平滑状态
            self.innov_ema = 0.9 * self.innov_ema + 0.1 * innov_mag

            # 压缩 q_scale 的膨胀上限：从最高 4.5 倍压制到最高 2.0 倍
        q_scale = 0.5 + 1.5 * float(np.clip(self.innov_ema / 30.0, 0.0, 1.0))
        self.Q = self.Q_base * q_scale

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
        self.last_ring_time = 0.0
        self.ring_buffer: Optional[RingBuffer] = None

        self.frame_ready_event = threading.Event()
        self._data_lock = threading.Lock()
        self._latest_detection_data: Optional[Dict] = None

        self.last_mode = "track"
        self.mode_threshold_high = config.getfloat("Controller", "mode_threshold_high", 50.0)
        self.mode_threshold_low = config.getfloat("Controller", "mode_threshold_low", 35.0)

        self.base_hardware_lag = config.getfloat("WorldModel", "base_hardware_lag", 0.015)
        self.ego_pos_px = np.zeros(2, dtype=np.float64)
        self.smoothed_lead_time: float = self.base_hardware_lag

        # ── 远程链路延迟补偿（Moonlight + VirtualHere 架构专用）────────────────────
        #
        # [问题 A] Moonlight 视频延迟：
        #   capture.py 的 t_cap 是 DXCam 在笔记本上收到解码帧的时间，
        #   但该帧对应的游戏状态实际发生在 t_cap - moonlight_latency 时刻。
        #   未补偿时，lead_time 被低估，准星预测落后于实际位置。
        #
        # [问题 B] VirtualHere 鼠标输出延迟：
        #   output.py 的 mouse_xy 在笔记本立即写入 ring_buffer 并发送 SendInput，
        #   但 VirtualHere 将该输入通过网络转发到台式机，游戏实际移动准星的时刻
        #   是 send_time + virtualhere_latency。
        #   未补偿时，ego_pos 每帧都比真实准星超前 virtualhere_latency 的位移量，
        #   worldmodel 误认为准星已经过冲，产生系统性反向修正。
        #
        # 配置建议：
        #   moonlight_latency_ms = 测量值（ping + 编解码，通常 8~25ms）
        #   virtualhere_latency_ms = 测量值（通常比 Moonlight 小，约 3~12ms）
        #   两者均可从 Moonlight 的延迟统计 HUD 和 ping 估算。
        self._moonlight_latency = config.getfloat("WorldModel", "moonlight_latency_ms", 15.0) / 1000.0
        self._virtualhere_latency = config.getfloat("WorldModel", "virtualhere_latency_ms", 8.0) / 1000.0

        # ── Fix: ego 运动追踪 depth_scale 对称性 ─────────────────────────────────
        # ring_buffer counts 由 calculate_velocity_move(bbox_w) 生成，
        # 逆变换必须用同一 bbox_w，否则 ego_pos 系统性虚高。
        self.current_bbox_w: float = 60.0

        # ── Fix: 按 target_id 存储 prev_meas，避免跨目标速度污染 ──────────────────
        # 旧的单 _prev_meas 在目标消失后重现时，用旧位置估算初速度，
        # 引入 500~2000px/s 的虚假速度冲击。
        self._prev_meas_by_id: Dict[int, np.ndarray] = {}
        self._prev_meas_time_by_id: Dict[int, float] = {}

    def update_detections(self, detections: np.ndarray, frame_id: int, t_capture: float, t_done: float):
        with self._data_lock:
            self._latest_detection_data = {"detections": detections, "frame_id": frame_id, "t_capture": t_capture}
        self.frame_ready_event.set()

    def step(self, context: InferenceContext, ring_buffer: RingBuffer):
        self.ring_buffer = ring_buffer
        now = time.perf_counter()

        if self.last_ts == 0:
            dt = 0.001
        else:
            raw_dt = now - self.last_ts
            dt = float(np.clip(raw_dt, 0.0005, 0.05))
        self.last_ts = now

        if self.last_ring_time == 0.0:
            self.last_ring_time = now - dt

        vh_lag = self._virtualhere_latency
        dx_counts, dy_counts = ring_buffer.get_cursor_delta_sum(
            self.last_ring_time - vh_lag, now - vh_lag + 0.0001
        )
        self.last_ring_time = now

        px_dx, px_dy = self.strategy.reverse_map_velocity(
            float(dx_counts), float(dy_counts), bbox_w=self.current_bbox_w
        )
        self.ego_pos_px[0] += px_dx
        self.ego_pos_px[1] += px_dy

        with self._data_lock:
            data = self._latest_detection_data
            self._latest_detection_data = None

        if data is not None:
            if len(data["detections"]) > 0:
                dets = []
                for box in data["detections"]:
                    x1, y1, x2, y2, conf, cls = box
                    if conf < 0.3: continue
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    dets.append(Detection(x=cx, y=cy, w=x2 - x1, h=y2 - y1, conf=conf, class_id=int(cls)))
                dets.sort(key=lambda d: d.conf, reverse=True)
                context.targets = dets
                if dets:
                    self.current_bbox_w = float(dets[0].w)
            else:
                context.targets = []

        best_detection = context.targets[0] if context.targets else None
        target = None
        is_coasting = False
        measurement = np.zeros(2, dtype=np.float64)

        if best_detection:
            target_id = int(best_detection.class_id)
            target = self.targets.get(target_id)
            measurement = np.array([best_detection.x - self.crop_center,
                                    best_detection.y - self.crop_center], dtype=np.float64)
            abs_measurement = measurement + self.ego_pos_px

            if target is None:
                target = TargetState(id=target_id, first_seen=now, last_seen=now)
                target.state[:2] = abs_measurement
                prev_m = self._prev_meas_by_id.get(target_id)
                prev_t = self._prev_meas_time_by_id.get(target_id, 0.0)
                if prev_m is not None and 0 < now - prev_t < 0.08:
                    v_init = (abs_measurement - prev_m) / (now - prev_t)
                    speed = float(np.linalg.norm(v_init))
                    if speed < 800.0:
                        target.state[2:4] = v_init
                        target.covariance[2, 2] = 300.0
                        target.covariance[3, 3] = 300.0
                self.targets[target_id] = target
            else:
                target.last_seen = now

            self._prev_meas_by_id[target_id] = abs_measurement.copy()
            self._prev_meas_time_by_id[target_id] = now

            self.kalman.predict(target, dt)
            self.kalman.update(target, abs_measurement)
        else:
            if self.targets:
                target = max(self.targets.values(), key=lambda t: t.last_seen)
                if now - target.last_seen < 0.150:
                    is_coasting = True
                    self.kalman.predict(target, dt)
                else:
                    target = None

        if not target:
            context.p_predict = None
            context.v_real = (0.0, 0.0)
            context.conf = 0.0
            context.is_valid = False
            if not any(now - t.last_seen < 2.0 for t in self.targets.values()):
                self.ego_pos_px[:] = 0.0
            return

        abs_position = target.state[:2].copy()
        abs_velocity = target.state[2:4].copy()
        abs_accel = target.state[4:6].copy()

        t_capture = now
        if not is_coasting and data is not None:
            t_capture = data.get("t_capture", now)

        software_lag = now - t_capture
        raw_lead_time = self._moonlight_latency + software_lag + self.base_hardware_lag
        raw_lead_time = float(np.clip(raw_lead_time, 0.005, 0.120))
        self.smoothed_lead_time = 0.85 * self.smoothed_lead_time + 0.15 * raw_lead_time

        if is_coasting:
            abs_velocity *= 0.85

        base_lead = self.smoothed_lead_time

        # ==========================================
        # 🌟 修复四：移除带有歧义的 dist_to_target，完全依赖协方差惩罚
        # ==========================================
        accel_norm = np.linalg.norm(abs_accel)
        accel_penalty = 1.0 / (1.0 + 0.001 * accel_norm)

        cov_trace = np.trace(target.covariance)
        cov_penalty = np.clip(1.0 - (cov_trace - 450.0) / 1400.0, 0.45, 1.0)

        # 结合协方差惩罚，目标疯狂扭动时主动放弃预测长度
        smart_lead = base_lead * accel_penalty * cov_penalty
        # ==========================================

        accel_bonus = 0.5 * abs_accel * (smart_lead ** 2)
        bonus_norm = float(np.linalg.norm(accel_bonus))
        if bonus_norm > 28.0:
            accel_bonus = accel_bonus * (28.0 / bonus_norm)

        pred_abs_pos = abs_position + abs_velocity * smart_lead + accel_bonus

        context.dynamic_lag_ms = smart_lead * 1000.0
        rel_pred_pos = pred_abs_pos - self.ego_pos_px

        if is_coasting:
            self.controller.mode = "track"
        else:
            error_distance = np.linalg.norm(measurement)
            threshold = self.mode_threshold_high if self.last_mode == "track" else self.mode_threshold_low
            self.controller.mode = "flick" if error_distance > threshold else "track"
        self.last_mode = self.controller.mode

        context.p_predict = (rel_pred_pos[0], rel_pred_pos[1])
        context.v_real = (abs_velocity[0], abs_velocity[1])
        context.a_real = (abs_accel[0], abs_accel[1])
        context.conf = 0.5 if is_coasting else 1.0
        context.is_valid = True
    def cleanup_old_targets(self, max_age: float = 2.0):
        now = time.perf_counter()
        to_remove = [tid for tid, tgt in self.targets.items() if now - tgt.last_seen > max_age]
        for tid in to_remove:
            del self.targets[tid]