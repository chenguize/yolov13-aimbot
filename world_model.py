# world_model.py
# Tier S 修复版：Y轴松绑 + 加速度补偿自然 + 延迟自适应进化引擎

import logging
import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict

from config import config
from perception.ring_buffer import RingBuffer

_log = logging.getLogger("WorldModel")
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
    F[0, 2] = dt;
    F[1, 3] = dt
    F[0, 4] = 0.5 * dt * dt;
    F[1, 5] = 0.5 * dt * dt
    F[2, 4] = dt;
    F[3, 5] = dt
    predicted_state = F @ state
    predicted_cov = F @ cov @ F.T + Q
    return predicted_state, predicted_cov


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def kf6_update(state, cov, R, meas):
    H = np.zeros((2, 6));
    H[0, 0] = 1.0;
    H[1, 1] = 1.0
    innovation = meas - H @ state
    S = H @ cov @ H.T + R
    S_inv = np.linalg.inv(S)
    K = cov @ H.T @ S_inv
    updated_state = state + K @ innovation
    I_KH = np.eye(6) - K @ H
    updated_cov = I_KH @ cov @ I_KH.T + K @ R @ K.T
    return innovation, updated_state, updated_cov


def _init_covariance() -> np.ndarray:
    """
    Kalman 协方差的合理初值。

    原 `np.eye(6) * 100.0` 严重虚高：
      · 新 target 的 state[:2] 是直接从 measurement 赋值的 —— 位置不确定性
        应等于 R（测量噪声），而非 100。原值导致 trace=600 → cov_penalty=0.89
        本不应该有的前馈衰减。
      · 速度、加速度确实未知，但也不是 100 —— 设成 Q 的量级更合理，
        让 Kalman 第一次 update 后快速收敛。

    这里用与 SimpleKalman 的 Q/R 量级相匹配的值：
      σ_pos² = 2   (= R_base)
      σ_vel² = 400 (~ 5·Q_vel, 让速度在 2-3 帧内收敛)
      σ_acc² = 800 (~ 4·Q_acc)
    trace = 2·2 + 2·400 + 2·800 = 2404 → cov_penalty 最初压到 0.45（下限），
    但这是必要的：刚 spawn 时速度估计不可信。一旦 Kalman 更新几次、协方差
    下降到 ~600，cov_penalty 恢复到 1.0，前馈正常工作。
    """
    return np.diag([2.0, 2.0, 400.0, 400.0, 800.0, 800.0]).astype(np.float64)


@dataclass
class TargetState:
    id: int
    first_seen: float
    last_seen: float
    state: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    covariance: np.ndarray = field(default_factory=_init_covariance)
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

    def reset_adaptive_state(self):
        """
        在目标切换 / 重生时调用。清除 innov_ema 与 Q 的自适应放大，避免上一
        目标（如高速 adad / slide）的 innov 尾巴污染下一个 target 的前 100ms。

        症状：上一局高速目标推高 innov_ema → Q_scale≈2.0 → P 协方差膨胀快
        → 新 target 前 100ms 的 v_est 抖动被放大 → controller 收到错误的
        ff_vel → BALLISTIC 着陆偏差 → precision 方差变大（某些 seed Prec=0）。
        """
        self.innov_ema = 0.0
        self.Q = self.Q_base.copy()

    def predict(self, target: TargetState, dt: float):
        target.state, target.covariance = kf6_predict(target.state, target.covariance, self.Q, dt)

    def update(self, target: TargetState, measurement: np.ndarray):
        innovation, target.state, target.covariance = kf6_update(target.state, target.covariance, self.R, measurement)
        innov_mag = np.linalg.norm(innovation)

        if innov_mag > self.innov_ema:
            self.innov_ema = 0.8 * self.innov_ema + 0.2 * innov_mag
        else:
            self.innov_ema = 0.9 * self.innov_ema + 0.1 * innov_mag

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
        self.ego_pos_px = np.array([self.crop_center, self.crop_center], dtype=np.float64)
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

        self.smoothed_lead_time: float = self.base_hardware_lag

        self._moonlight_latency = config.getfloat("WorldModel", "moonlight_latency_ms", 15.0) / 1000.0

        # ==============================================================================
        # 🧬 自适应延迟进化引擎 (Adaptive Latency Engine)
        # ==============================================================================
        self.dynamic_vh_latency = 0.008
        self.ego_velocity_ema = np.zeros(2, dtype=np.float64)
        self._latency_print_timer = 0.0

        self.current_bbox_w: float = 60.0
        self._prev_meas_by_id: Dict[int, np.ndarray] = {}
        self._prev_meas_time_by_id: Dict[int, float] = {}

        # 推理耗时 EMA（毫秒），用于诊断与可选的 lead 自适应
        self.inference_ms_ema: float = 0.0

    def update_detections(self, detections: np.ndarray, frame_id: int, t_capture: float, t_done: float):
        with self._data_lock:
            self._latest_detection_data = {
                "detections": detections,
                "frame_id": frame_id,
                "t_capture": t_capture,
                "t_done": t_done,
            }
        # 推理耗时 EMA，供监控 / 诊断（非关键路径）
        infer_ms = (t_done - t_capture) * 1000.0
        if self.inference_ms_ema == 0.0:
            self.inference_ms_ema = infer_ms
        else:
            self.inference_ms_ema = 0.9 * self.inference_ms_ema + 0.1 * infer_ms
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

        # ── 1. 物理时钟：维持准星的“绝对实时位置” ──────────────────────────────
        dx_counts, dy_counts = ring_buffer.get_cursor_delta_sum(self.last_ring_time, now)
        self.last_ring_time = now

        px_dx, px_dy = self.strategy.reverse_map_velocity(
            float(dx_counts), float(dy_counts), bbox_w=self.current_bbox_w
        )
        self.ego_pos_px[0] += px_dx
        self.ego_pos_px[1] += px_dy

        # 平滑记录当前的鼠标物理初速度 (用于残差分析)
        if dt > 0:
            v_ego_x = px_dx / dt
            v_ego_y = px_dy / dt
            self.ego_velocity_ema[0] = 0.85 * self.ego_velocity_ema[0] + 0.15 * v_ego_x
            self.ego_velocity_ema[1] = 0.85 * self.ego_velocity_ema[1] + 0.15 * v_ego_y

        with self._data_lock:
            data = self._latest_detection_data
            self._latest_detection_data = None

        if data is None:
            # 本帧无新推理包：禁止沿用旧 context.targets，否则无检测帧仍会拿上一框当 best
            context.targets = []

        # 单主目标槽位。原先用 class_id 作 dict 键，多同类别目标（多敌人均为 class 0）
        # 会共用一个 Kalman，量测在目标间乱切 → 实战「完全不锁人」。
        PRIMARY_TRACK = 0

        # ── 2. 时空回溯：计算拍照瞬间的准星位置 ────────────────────────────────
        ego_at_capture = self.ego_pos_px.copy()

        if data is not None:
            t_capture = data.get("t_capture", now)

            # 【核心】：扣除从 (t_capture - 动态延迟) 到 现在 之间产生的所有位移
            past_dx, past_dy = ring_buffer.get_cursor_delta_sum(
                t_capture - self.dynamic_vh_latency, now
            )
            r_px_dx, r_px_dy = self.strategy.reverse_map_velocity(
                float(past_dx), float(past_dy), bbox_w=self.current_bbox_w
            )
            ego_at_capture[0] -= r_px_dx
            ego_at_capture[1] -= r_px_dy

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
            target_id = PRIMARY_TRACK
            target = self.targets.get(target_id)

            # 使用回溯坐标与画面残差进行合成
            measurement = np.array([best_detection.x - self.crop_center,
                                    best_detection.y - self.crop_center], dtype=np.float64)
            abs_measurement = measurement + ego_at_capture

            if target is None:
                self.kalman.reset_adaptive_state()
                target = TargetState(id=int(best_detection.class_id), first_seen=now, last_seen=now)
                target.state[:2] = abs_measurement
                prev_m = self._prev_meas_by_id.get(PRIMARY_TRACK)
                prev_t = self._prev_meas_time_by_id.get(PRIMARY_TRACK, 0.0)
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
                # 类别可随框跳变，保持语义
                try:
                    target.id = int(best_detection.class_id)
                except (TypeError, ValueError):
                    pass

            self._prev_meas_by_id[PRIMARY_TRACK] = abs_measurement.copy()
            self._prev_meas_time_by_id[PRIMARY_TRACK] = now

            self.kalman.predict(target, dt)

            # ── 3. 卡尔曼残差提取与延迟自适应进化 ──────────────────────────────
            innovation = self.kalman.update(target, abs_measurement)

            vx, vy = self.ego_velocity_ema
            speed_sq = vx ** 2 + vy ** 2

            if speed_sq > 6400.0:
                time_error = (innovation[0] * vx + innovation[1] * vy) / speed_sq
                time_error = np.clip(time_error, -0.015, 0.015)

                self.dynamic_vh_latency += 0.02 * time_error
                self.dynamic_vh_latency = np.clip(self.dynamic_vh_latency, 0.002, 0.050)

                if now - self._latency_print_timer > 1.0:
                    _log.info("自适应 vh 延迟: %.1f ms", self.dynamic_vh_latency * 1000.0)
                    self._latency_print_timer = now

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
                self.ego_pos_px[:] = self.crop_center
            return

        # 丢框后的 150ms 纯预测 (coast)：不输出 p_predict 给主循环，避免盲飞阶段乱吸鼠标 /
        # 「无目标也在动」；Kalman 已在上面 predict 过，保持内部状态即可。
        if is_coasting:
            context.p_predict = None
            context.v_real = (0.0, 0.0)
            context.a_real = (0.0, 0.0)
            context.conf = 0.0
            context.is_valid = False
            return

        abs_position = target.state[:2].copy()
        abs_velocity = target.state[2:4].copy()
        abs_accel = target.state[4:6].copy()

        t_capture = now
        if data is not None:
            t_capture = data.get("t_capture", now)

        software_lag = now - t_capture
        raw_lead_time = self._moonlight_latency + software_lag + self.base_hardware_lag
        raw_lead_time = float(np.clip(raw_lead_time, 0.005, 0.120))
        self.smoothed_lead_time = 0.85 * self.smoothed_lead_time + 0.15 * raw_lead_time

        base_lead = self.smoothed_lead_time

        accel_norm = np.linalg.norm(abs_accel)
        accel_penalty = 1.0 / (1.0 + 0.001 * accel_norm)
        cov_trace = np.trace(target.covariance)
        cov_penalty = np.clip(1.0 - (cov_trace - 450.0) / 1400.0, 0.45, 1.0)

        # ── 架构改进：复合 lead = 管道延迟 + 控制器剩余执行时长 ───────────────
        # 原 smart_lead 只覆盖 "拍照 → 命令发出" 的管道延迟（~15-30ms），但
        # BALLISTIC 本身是 80~200ms 的开环 min-jerk 轨迹；期间目标会持续移动，
        # p_predict 若不把这段时间算进去，BALLISTIC 着陆时天然产生 ~V_target·T
        # 级别的系统偏差（e.g. 10m/8.5m/s → ~60-90px），逼 TRACKING 多花 150-300ms
        # 把这段偏差磨掉 —— 这就是 pure_ai 0.617s TTK 的核心源头。
        #
        # ctrl_lead 默认 0（保持对无此接口的控制器的兼容）；CIPHER 的
        # get_expected_lead() 在 BALLISTIC 返回 remain·0.5（质心时刻），
        # 在 TRACKING/空转时返回 0。
        ctrl_lead = 0.0
        if hasattr(self.controller, 'get_expected_lead'):
            try:
                ctrl_lead = float(self.controller.get_expected_lead())
            except Exception:
                ctrl_lead = 0.0
        # ctrl_lead 上限 150ms，防极端 Fitts 尾巴把 p_predict 推太远造成过冲
        ctrl_lead = float(np.clip(ctrl_lead, 0.0, 0.150))

        # accel/cov penalty 只作用于管道 lead 部分（物理噪声相关），不惩罚
        # 控制器自身的确定性剩余时间 —— 否则 BALLISTIC 会永远"差一截"。
        pipeline_lead = base_lead * accel_penalty * cov_penalty
        smart_lead    = pipeline_lead + ctrl_lead

        accel_bonus = 0.5 * abs_accel * (smart_lead ** 2)
        bonus_norm = float(np.linalg.norm(accel_bonus))
        if bonus_norm > 28.0:
            accel_bonus = accel_bonus * (28.0 / bonus_norm)

        pred_abs_pos = abs_position + abs_velocity * smart_lead + accel_bonus

        context.dynamic_lag_ms = smart_lead * 1000.0
        rel_pred_pos = pred_abs_pos - self.ego_pos_px

        # 此处 is_coasting 已提前 return，仅保留有量测的帧
        error_distance = np.linalg.norm(measurement)
        threshold = self.mode_threshold_high if self.last_mode == "track" else self.mode_threshold_low
        self.controller.mode = "flick" if error_distance > threshold else "track"
        self.last_mode = self.controller.mode

        context.p_predict = (rel_pred_pos[0], rel_pred_pos[1])
        context.v_real = (abs_velocity[0], abs_velocity[1])
        context.a_real = (abs_accel[0], abs_accel[1])
        # 实战触发与 gate 用真实检测置信；旧逻辑写死 1.0 时 Triggerbot/调试全失真
        context.conf = float(best_detection.conf) if best_detection is not None else 0.0
        context.is_valid = True

    def cleanup_old_targets(self, max_age: float = 2.0):
        now = time.perf_counter()
        to_remove = [tid for tid, tgt in self.targets.items() if now - tgt.last_seen > max_age]
        for tid in to_remove:
            del self.targets[tid]