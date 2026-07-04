# world_model.py
# Tier S 修复版：Y轴松绑 + 加速度补偿自然 + 延迟自适应进化引擎

import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

from config import config
from utils.aim_class_filter import get_aim_target_class_set
from perception.ring_buffer import RingBuffer
from perception.track_manager import TrackManager, Track

from utils.logger import get_logger

_log = get_logger("WorldModel")
from utils.types import Detection, InferenceContext


def _ring_intent_human_delta(ring: RingBuffer, t_start: float, t_end: float) -> Tuple[int, int]:
    """人手意图增量：委托 RingBuffer 自动处理后端差异。"""
    return ring.get_intent_delta(t_start, t_end)


def _reorder_dets_by_sticky_prev(
    dets: List[Detection],
    sticky: Optional[Tuple[float, float, float, float]],
    conf_margin: float,
    max_center_px: float,
) -> List[Detection]:
    """
    多目标时若仅按 conf 排序，两人 conf 接近时 best 会在两框间来回切 → 量测跳变 → Kalman/前馈穿零。
    在「与上一帧首选中心距离近 + conf 仍在 top 一带」时保持同一物理目标优先。
    """
    if sticky is None or len(dets) < 2:
        return dets
    pcx, pcy = float(sticky[0]), float(sticky[1])
    out = sorted(dets, key=lambda d: d.conf, reverse=True)
    best_c = out[0].conf
    in_band = [d for d in out if d.conf >= best_c - conf_margin]
    if not in_band:
        return out
    closest = min(in_band, key=lambda d: float(np.hypot(d.x - pcx, d.y - pcy)))
    ddist = float(np.hypot(closest.x - pcx, closest.y - pcy))
    if ddist >= max_center_px:
        return out
    if out[0] is closest:
        return out
    others = [d for d in out if d is not closest]
    others.sort(key=lambda d: d.conf, reverse=True)
    return [closest] + others
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
        self.strategy = create_aim_strategy()
        from controllers.controller_factory import get_controller
        self.controller = get_controller()
        if hasattr(self.controller, "set_pixel_to_count_scale"):
            calibration = getattr(self.strategy, "calib", None)
            self.controller.set_pixel_to_count_scale(
                getattr(calibration, "k_x", 1.0),
                getattr(calibration, "k_y", 1.0),
            )

        self.capture_size = config.getint("Hardware", "capture_size", 256)
        self.crop_center = self.capture_size / 2.0
        self.ego_pos_px = np.array([self.crop_center, self.crop_center], dtype=np.float64)
        self.track_manager = TrackManager()
        self.last_ts = 0.0
        self.last_ring_time = 0.0
        self.ring_buffer: Optional[RingBuffer] = None

        self.frame_ready_event = threading.Event()
        self._data_lock = threading.Lock()
        self._latest_detection_data: Optional[Dict] = None

        self.last_mode = "track"
        self.mode_threshold_high = config.getfloat("Controller", "mode_threshold_high", 50.0)
        self.mode_threshold_low = config.getfloat("Controller", "mode_threshold_low", 35.0)

        self.base_hardware_lag = config.getfloat("Hardware", "base_hardware_lag", 0.015)

        self.smoothed_lead_time: float = self.base_hardware_lag

        # 画面“内容”相对真实游戏状态的一程滞后（与 t_cap 是否含得无关，必须外填）：
        #   Moonlight/串流/云：约 20–50ms，填 Moonlight 统计或 “主机→本机” 观感延迟。
        #   virtualhere / KMV：USB 等额外 2–8ms 可在此叠。
        self._stream_ingress_s = (
            max(0.0, config.getfloat("Latency", "moonlight_latency_ms", 0.0))
            + max(0.0, config.getfloat("Latency", "virtualhere_latency_ms", 0.0))
        ) / 1000.0

        # ==============================================================================
        # 🧬 自适应延迟进化引擎 (Adaptive Latency Engine)
        # ==============================================================================
        self.ego_velocity_ema = np.zeros(2, dtype=np.float64)
        self._latency_print_timer = 0.0

        # ── WAN 模式（Sunshine+Moonlight 远程串流）────────────────────────────
        self._wan_mode = config.getbool("Latency", "wan_mode", False)
        if self._wan_mode:
            self._wan_min_latency_s = max(
                0.010,
                config.getfloat("Latency", "wan_min_latency_ms", 60.0) / 1000.0,
            )
            self._wan_jitter_s = max(
                0.005,
                config.getfloat("Latency", "wan_jitter_ms", 25.0) / 1000.0,
            )
            # WAN: 自适应延迟范围放宽到 [5ms, 150ms]
            self._vh_lat_min = 0.005
            self._vh_lat_max = 0.150
            self._vh_lat_alpha = 0.03  # 更慢的适应速率（不追逐帧 jitter）
            self._lead_clamp_max = 0.300  # pipeline lead 上限放开
            # 初始化到预估延迟的 60%，避免前 0.3s 回退不足
            self.dynamic_vh_latency = max(0.008, self._wan_min_latency_s * 0.6)
            # jitter-safe latency EMA（比 dynamic_vh_latency 更稳定，用于 lead 计算）
            self._vh_lat_smoothed: float = self._wan_min_latency_s
        else:
            self._vh_lat_min = 0.002
            self._vh_lat_max = 0.050
            self._vh_lat_alpha = 0.02
            self._lead_clamp_max = 0.200
            self.dynamic_vh_latency = 0.008

        self.current_bbox_w: float = 60.0

        # 推理耗时 EMA（毫秒），用于诊断与可选的 lead 自适应
        self.inference_ms_ema: float = 0.0

        # ── Controller speed calibration EMA ──
        # controller 的解析 lead（get_expected_lead）假设理想阻抗响应，但实际
        # arm_vel 受 LPF / speed_cap / anti-orbit / 离散化影响，响应偏慢。
        # 此 EMA 学习实际响应速度与目标速度的比率，反比缩放 ctrl_lead。
        self._ctrl_speed_scale: float = 1.0
        self._ctrl_speed_print_timer: float = 0.0

        # ── Arrival-based calibration (混合校准第二路) ──
        # 用实际 AI 位移 vs 预期 arm_vel·dt 的差异，校准控制管道的有效传输率。
        # 比纯 speed_scale 更直接：不依赖 Kalman 速度估计，直接从 SendInput 结果反推。
        self._prev_arm_vel_cts = np.zeros(2, dtype=np.float64)  # 上帧 controller arm_vel
        self._arrival_calib_timer: float = 0.0

        # 多目标时按上一帧首选框 (cx,cy,w,h) 粘滞 best，减轻 conf 微差导致 cls/框切换 → 量测跳变
        self._sticky_crop: Optional[Tuple[float, float, float, float]] = None

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
        context.is_coasting = False
        now = time.perf_counter()

        # ── 延迟模式判定（每帧读 config，兼容运行时 reload）────────────────
        _predict_ahead = config.getbool("WorldModel", "predict_ahead", True)
        # 零延迟：本机无硬件延迟 + 无串流 + 非 WAN → 跳过全部延迟补偿管线
        _zero_latency = (
            not config.getbool("Latency", "wan_mode", False)
            and self._stream_ingress_s <= 0.0
            and self.base_hardware_lag <= 0.0
        )

        if self.last_ts == 0:
            dt = 0.001
        else:
            raw_dt = now - self.last_ts
            dt = float(np.clip(raw_dt, 0.0005, 0.05))
        self.last_ts = now

        if self.last_ring_time == 0.0:
            self.last_ring_time = now - dt

        # ── 1. 物理时钟：维持准星的“绝对实时位置” ──────────────────────────────
        # 用 get_total_delta_sum（人+AI）而非 get_cursor_delta_sum（仅人）：
        # AI SendInput 同样转动游戏相机，只有全量位移才能正确追踪真实相机旋转，
        # 避免 AI 造成的相机旋转泄漏进卡尔曼速度估计。
        if config.getbool("WorldModel", "use_total_delta", True):
            dx_counts, dy_counts = ring_buffer.get_total_delta_sum(self.last_ring_time, now)
        else:
            dx_counts, dy_counts = ring_buffer.get_cursor_delta_sum(self.last_ring_time, now)
        self.last_ring_time = now

        # ego_pos_px is an accumulated world coordinate, not a screen-space
        # offset. Incremental camera rotation is applied around screen center;
        # using the unbounded world coordinate as a FOV Jacobian anchor makes
        # the inverse gain explode as the camera keeps moving.
        px_dx, px_dy = self.strategy.reverse_map_velocity(
            float(dx_counts), float(dy_counts),
            px_x=0.0, px_y=0.0, bbox_w=self.current_bbox_w
        )
        self.ego_pos_px[0] += px_dx
        self.ego_pos_px[1] += px_dy

        # 平滑记录当前的鼠标物理初速度 (用于残差分析；仅自适应延迟引擎需要)
        if _predict_ahead and not _zero_latency and dt > 0:
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

        # ── 2. 时空回溯：计算拍照瞬间的准星位置 ────────────────────────────────
        ego_at_capture = self.ego_pos_px.copy()

        track_dets: List[dict] = []  # TrackManager 格式检测列表

        if data is not None:
            t_capture = data.get("t_capture", now)

            if _zero_latency:
                # 零延迟：无需回溯，拍照时刻视角 = 当前视角
                pass
            else:
                # 【核心】：扣除从 (t_capture - 延迟) 到 现在 之间产生的所有位移
                # 延迟来源：predict_ahead 开启时用自适应 dynamic_vh_latency；
                #          关闭时用固定 base_hardware_lag + stream_ingress_s
                _backtrack_lat = (
                    self.dynamic_vh_latency if _predict_ahead
                    else self.base_hardware_lag + self._stream_ingress_s
                )
                if config.getbool("WorldModel", "use_total_delta", True):
                    past_dx, past_dy = ring_buffer.get_total_delta_sum(
                        t_capture - _backtrack_lat, now
                    )
                else:
                    past_dx, past_dy = ring_buffer.get_cursor_delta_sum(
                        t_capture - _backtrack_lat, now
                    )
                # Backtracking uses the same center-anchored incremental map
                # as the forward ego integration above.
                r_px_dx, r_px_dy = self.strategy.reverse_map_velocity(
                    float(past_dx), float(past_dy),
                    px_x=0.0, px_y=0.0, bbox_w=self.current_bbox_w
                )
                ego_at_capture[0] -= r_px_dx
                ego_at_capture[1] -= r_px_dy

            if len(data["detections"]) > 0:
                dets: List[Detection] = []
                cls_allow = get_aim_target_class_set()
                for box in data["detections"]:
                    x1, y1, x2, y2, conf, cls = box
                    if conf < 0.3:
                        continue
                    ci = int(cls)
                    if cls_allow is not None and ci not in cls_allow:
                        continue
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    dets.append(Detection(
                        x=cx, y=cy, w=x2 - x1, h=y2 - y1, conf=conf, class_id=ci,
                        xyxy=np.array([x1, y1, x2, y2], dtype=np.float64),
                    ))

                if not dets:
                    context.targets = []
                    self._sticky_crop = None
                else:
                    # 粘滞排序：多目标 conf 接近时保持同一物理目标优先
                    dets.sort(key=lambda d: d.conf, reverse=True)
                    if config.getbool("WorldModel", "sticky_target_enable", True) and len(dets) >= 2:
                        dets = _reorder_dets_by_sticky_prev(
                            dets,
                            self._sticky_crop,
                            config.getfloat("WorldModel", "sticky_conf_margin", 0.10),
                            config.getfloat("WorldModel", "sticky_max_center_px", 120.0),
                        )
                    context.targets = dets
                    self.current_bbox_w = float(dets[0].w)
                    self._sticky_crop = (
                        float(dets[0].x), float(dets[0].y), float(dets[0].w), float(dets[0].h),
                    )

                    # ── 构建 TrackManager 格式的检测列表 ──
                    for det in dets:
                        w2, h2 = det.w / 2.0, det.h / 2.0
                        bbox = (det.x - w2, det.y - h2, det.x + w2, det.y + h2)
                        meas = np.array([det.x - self.crop_center,
                                         det.y - self.crop_center], dtype=np.float64)
                        abs_meas = meas + ego_at_capture
                        track_dets.append({
                            "abs_meas": abs_meas,
                            "bbox": bbox,
                            "conf": det.conf,
                            "class_id": det.class_id,
                        })
            else:
                context.targets = []
                self._sticky_crop = None

        # ── 3. TrackManager：多目标 IoU 匹配 + 独立 Kalman 池 ─────────────────
        if track_dets:
            self.track_manager.match_and_update(track_dets, now, dt)
        else:
            self.track_manager.predict_all(dt)

        best_track = self.track_manager.select_best(self.ego_pos_px, self.crop_center)

        # 无任何活跃轨迹
        if best_track is None:
            context.p_predict = None
            context.v_real = (0.0, 0.0)
            context.conf = 0.0
            context.is_valid = False
            if not self.track_manager.tracks:
                self.ego_pos_px[:] = self.crop_center
            return

        # A committed handoff motor program is intentionally open-loop. Keep
        # its estimator alive through a short detector burst instead of cutting
        # actuator power halfway through the transfer.
        coast_limit = 5
        takeover_state = str(getattr(
            self.controller, 'takeover_state', 'ACTIVE_LOCK'
        ))
        handoff_reason = str(getattr(
            self.controller, '_handoff_reason', ''
        ))
        if (
            takeover_state in ('HUMAN_LEAD', 'REACTION', 'PRIMING')
            and handoff_reason in ('human_override', 'human_release', 'legacy_flick_end')
        ):
            coast_limit = max(5, config.getint(
                'Controller', 'cipher_handoff_coast_frames', 12
            ))
        if best_track.coast_count > coast_limit:
            context.is_coasting = True
            context.p_predict = None
            context.v_real = (0.0, 0.0)
            context.a_real = (0.0, 0.0)
            context.conf = 0.0
            context.is_valid = False
            return

        # ── 3.5 卡尔曼残差提取与延迟自适应进化 ──────────────────────────────
        # 仅 predict_ahead 开启且非零延迟时运行；关闭预测时锁死 dynamic_vh_latency=0
        # 防止历史自适应值残留污染 ego 回溯（症状：关预测后准星仍偏→关不干净）。
        if _predict_ahead and not _zero_latency:
            innov = getattr(best_track, 'last_innovation', None)
            if innov is not None and config.getbool("WorldModel", "adaptive_latency_enable", True):
                vx, vy = self.ego_velocity_ema
                speed_sq = vx ** 2 + vy ** 2

                # P1 修复：原门控 speed>80px/s 过松，把目标真实加速度带来的 innov 误
                # 当作时间错位 → vh_latency 抖动。要求 ego 速度更高(>200px/s)且 innov
                # 与 ego_v 强对齐(cos>0.85)，才认为是真的延迟误差。
                _ADAPT_VEL_SQ = 40000.0  # 200 px/s 平方
                if speed_sq > _ADAPT_VEL_SQ:
                    innov_mag = float(np.linalg.norm(innov))
                    v_mag = float(np.sqrt(speed_sq))
                    cos_align = (innov[0] * vx + innov[1] * vy) / max(innov_mag * v_mag, 1e-9)
                    if cos_align > 0.85:
                        time_error = (innov[0] * vx + innov[1] * vy) / speed_sq
                        if self._wan_mode:
                            time_error = np.clip(time_error, -0.025, 0.025)
                            self.dynamic_vh_latency += self._vh_lat_alpha * time_error
                            self.dynamic_vh_latency = float(np.clip(
                                self.dynamic_vh_latency,
                                max(self._vh_lat_min, self._wan_min_latency_s * 0.3),
                                self._vh_lat_max,
                            ))
                            self._vh_lat_smoothed = (
                                0.95 * self._vh_lat_smoothed + 0.05 * self.dynamic_vh_latency
                            )
                        else:
                            time_error = np.clip(time_error, -0.015, 0.015)
                            self.dynamic_vh_latency += 0.02 * time_error
                            self.dynamic_vh_latency = np.clip(self.dynamic_vh_latency, self._vh_lat_min, self._vh_lat_max)

                        if now - self._latency_print_timer > 1.0:
                            _log.kalman_adaptive_latency(
                                self.dynamic_vh_latency * 1000.0, wan_mode=self._wan_mode,
                            )
                            if self._wan_mode:
                                _log.kalman_wan_jitter_safe(
                                    self._vh_lat_smoothed * 1000.0,
                                    self._wan_min_latency_s * 1000.0,
                                )
                            self._latency_print_timer = now
        elif not _predict_ahead:
            # 关闭预测：锁死自适应延迟为 0，并用固定硬件延迟做 ego 回溯
            self.dynamic_vh_latency = 0.0
            # 同时清理校准状态，防止残留值在下一次开启预测时造成跳变
            self.smoothed_lead_time = 0.0
            self._ctrl_speed_scale = 1.0
            self._prev_arm_vel_cts[:] = 0.0

        abs_position = best_track.abs_position.copy()
        abs_velocity = best_track.abs_velocity.copy()
        abs_accel = best_track.abs_accel.copy()

        if not _predict_ahead:
            # 关「时间前视」：瞄准误差 = 当前 KF 平滑后的物面位置 − 准星（不做 v·T/½aT²/ctrl_lead）。
            # 动目标/高延迟下会**滞后**，可用来对照过冲/穿零是否由 lead 引起；v_real/a_real 仍给控制器前馈。
            rel_pred_pos = abs_position - self.ego_pos_px
            context.dynamic_lag_ms = 0.0
        elif _zero_latency:
            # ── 零延迟快速路径：跳过平滑/惩罚/校准，但保留 ctrl_lead ──
            # 控制器执行时间（BALLISTIC 剩余时长 / TRACKING settling）仍需补偿，
            # 否则动目标场景准星落点时球已跑远 → 顶着 SteadyMAE 高拖尾 → 空枪。
            #
            # 近距须削弱 v·T 项：本机 Aimlab/色块 KF 速度噪声大；BALLISTIC 内 ff 已部分
            # 补偿动目标，WM 再用满 lead 乘 v 易「标点飞过」过冲（日志里 lead≈150ms 典型）。
            t_capture = data.get("t_capture", now) if data is not None else now
            software_lag = max(0.0, now - t_capture)

            ctrl_lead = 0.0
            if config.getbool("WorldModel", "ctrl_lead_enable", True):
                if hasattr(self.controller, 'get_expected_lead'):
                    try:
                        ctrl_lead = float(self.controller.get_expected_lead())
                    except Exception:
                        ctrl_lead = 0.0
            _zl_cap = config.getfloat("WorldModel", "zero_latency_ctrl_lead_cap_s", 0.085)
            ctrl_lead = float(np.clip(ctrl_lead, 0.0, max(0.0, _zl_cap)))

            _zl_smart_max = config.getfloat("WorldModel", "zero_latency_max_smart_lead_s", 0.110)
            smart_lead = float(np.clip(software_lag + ctrl_lead, 0.0, max(0.005, _zl_smart_max)))

            px_err_kf = float(np.linalg.norm(abs_position - self.ego_pos_px))
            _d0 = config.getfloat("WorldModel", "zero_latency_vel_damp_start_px", 24.0)
            _d1 = config.getfloat("WorldModel", "zero_latency_vel_damp_full_px", 140.0)
            _dlo = config.getfloat("WorldModel", "zero_latency_vel_damp_min", 0.20)
            if _d1 > _d0 + 1.0:
                vel_lead_scale = float(
                    np.clip((px_err_kf - _d0) / (_d1 - _d0), _dlo, 1.0)
                )
            else:
                vel_lead_scale = 1.0

            pred_abs_pos = abs_position + abs_velocity * (smart_lead * vel_lead_scale)
            context.dynamic_lag_ms = smart_lead * vel_lead_scale * 1000.0
            rel_pred_pos = pred_abs_pos - self.ego_pos_px
        else:
            t_capture = now
            if data is not None:
                t_capture = data.get("t_capture", now)

            software_lag = now - t_capture
            # 串流 ingress + 本机 t_cap→本步 的软件排队/推理/线程间隙 + 键鼠/显示刚性延迟
            raw_lead_time = self._stream_ingress_s + software_lag + self.base_hardware_lag
            # WAN 模式：施加最小延迟地板 + 上限放宽到 300ms
            if self._wan_mode:
                raw_lead_time = max(raw_lead_time, self._wan_min_latency_s)
                raw_lead_time = float(np.clip(raw_lead_time, 0.010, self._lead_clamp_max))
            else:
                raw_lead_time = float(np.clip(raw_lead_time, 0.005, self._lead_clamp_max))
            # 网络/调度抖动大时加快跟上，减小编码抖动的预测滞后
            lead_gap = abs(raw_lead_time - self.smoothed_lead_time)
            # 大跳变（>25ms）时快速跟上，避免延迟突变后 3-4 帧才响应
            ema = 0.80 if lead_gap > 0.025 else 0.20
            self.smoothed_lead_time = (1.0 - ema) * self.smoothed_lead_time + ema * raw_lead_time

            base_lead = self.smoothed_lead_time

            accel_norm = np.linalg.norm(abs_accel)
            # 目标加速度越大，越需要更多预测提前量（加速度意味着方向/速度在变，
            # 管道延迟叠加收敛时间会让准星落点大幅滞后）。原 1/(1+a) 是反向惩罚。
            accel_penalty = float(np.clip(1.0 + 0.0003 * accel_norm, 1.0, 1.4))
            cov_trace = np.trace(best_track.target.covariance)
            # 协方差 trace 初值 2404 → 原下限 0.45 意味着新目标前几百毫秒预测被
            # 打 55% 折扣。下限提到 0.75，新目标至少保留 75% 预测容量。
            cov_penalty = np.clip(1.0 - (cov_trace - 450.0) / 1400.0, 0.75, 1.0)

            # ── 架构改进：复合 lead = 管道延迟 + 控制器剩余执行时长 ───────────────
            # 原 smart_lead 只覆盖 "拍照 → 命令发出" 的管道延迟（~15-30ms），但
            # BALLISTIC 本身是 80~200ms 的开环 min-jerk 轨迹；期间目标会持续移动，
            # p_predict 若不把这段时间算进去，BALLISTIC 着陆时天然产生 ~V_target·T
            # 级别的系统偏差（e.g. 10m/8.5m/s → ~60-90px），逼 TRACKING 多花 150-300ms
            # 把这段偏差磨掉 —— 这就是 pure_ai 0.617s TTK 的核心源头。
            #
            # ctrl_lead 默认 0（保持对无此接口的控制器的兼容）；CIPHER 的
            # get_expected_lead() 用二阶阻抗动力学解析求解 ——
            # BALLISTIC: remain（全剩余时长，因 ff_vel 已补偿目标位移）
            # TRACKING:  4/(ζ·ω_n) + 饱和段 err/v_max
            ctrl_lead = 0.0
            if config.getbool("WorldModel", "ctrl_lead_enable", True):
                if hasattr(self.controller, 'get_expected_lead'):
                    try:
                        ctrl_lead = float(self.controller.get_expected_lead())
                    except Exception:
                        ctrl_lead = 0.0
            # ctrl_lead 上限 150ms，防极端 Fitts 尾巴把 p_predict 推太远造成过冲
            ctrl_lead = float(np.clip(ctrl_lead, 0.0, 0.150))

            # ── Controller speed calibration: 闭环校 control latency ───────────
            # ctrl_lead 基于阻抗解析解（理想响应），实际 arm_vel 受 LPF / speed_cap
            # / anti-orbit / 离散化影响而偏慢。EMA 学习 (target_speed / arm_speed)
            # 的比率，反比放大 ctrl_lead 补偿控制响应不足。
            ctrl_lead *= self._ctrl_speed_scale
            ctrl_lead = float(np.clip(ctrl_lead, 0.0, 0.250))

            # 更新 speed_scale: 用 Kalman 目标速度 vs 控制器上一帧 arm 速度
            # P1 修复：原钳制 [0.80, 1.40] 太宽 + 无衰减回归 → 正反馈风险
            # (tgt_spd 估计偏大 → scale 增大 → lead 增大 → 下帧 tgt_spd 更偏)。
            # 改为：1) 缩窄钳制到 [0.90, 1.20]；2) 每帧向 1.0 衰减 0.5%，让长期无数据时回归中性。
            arm_vel = getattr(self.controller, 'crosshair_velocity', None)
            if arm_vel is not None:
                arm_spd = float(np.linalg.norm(arm_vel))
                tgt_spd = float(np.linalg.norm(abs_velocity))
                if tgt_spd > 10.0 and arm_spd > 0.5:
                    # ratio > 1: 目标快但 arm 慢 → 响应不足 → 需要更大 lead
                    ratio = tgt_spd / max(arm_spd, 1.0)
                    ratio_clamped = float(np.clip(ratio, 0.7, 1.5))
                    self._ctrl_speed_scale = (
                        0.95 * self._ctrl_speed_scale + 0.05 * ratio_clamped
                    )
                    # P1 修复：缩窄钳制范围，抑制正反馈
                    self._ctrl_speed_scale = float(np.clip(self._ctrl_speed_scale, 0.90, 1.20))
                    if now - self._ctrl_speed_print_timer > 2.0:
                        _log.ctrl_speed_scale_diag(self._ctrl_speed_scale, arm_spd, tgt_spd)
                        self._ctrl_speed_print_timer = now
                else:
                    # P1 修复：无显著运动时缓慢回归 1.0，避免历史偏差长期残留
                    self._ctrl_speed_scale = 0.995 * self._ctrl_speed_scale + 0.005 * 1.0

            # ── Arrival-based calibration: 实测 AI 位移 vs 预期 ──────────────
            # 上一帧 controller 输出 arm_vel，MouseWorker 1000Hz 循环发送 SendInput。
            # 本帧从 ring_buffer 取出实际 AI 位移（总位移 − 人手位移），比较预期位移。
            # 若比例系统性地偏小 → 管道传输率低于预期 → 反比放大 ctrl_speed_scale。
            # P1 修复：作为主校准源（不依赖 Kalman 估计），缩窄钳制范围与 speed_scale 一致
            prev_arm = self._prev_arm_vel_cts
            prev_arm_spd = float(np.linalg.norm(prev_arm))
            if prev_arm_spd > 50.0 and dt > 0.001:
                # 预期 AI 位移（counts）：arm_vel（counts/s）× dt（s）
                expected_dx = prev_arm[0] * dt
                expected_dy = prev_arm[1] * dt
                expected_spd = float(np.hypot(expected_dx, expected_dy))
                if expected_spd > 5.0:
                    # 实际 AI 位移：总位移 − 人手意图（pynput 时人位移需减 AI 重影）
                    human_dx, human_dy = _ring_intent_human_delta(ring_buffer, now - dt, now)
                    total_dx, total_dy = ring_buffer.get_total_delta_sum(
                        now - dt, now
                    )
                    ai_dx = float(total_dx - human_dx)
                    ai_dy = float(total_dy - human_dy)
                    ai_spd = float(np.hypot(ai_dx, ai_dy))
                    if ai_spd > 3.0:
                        arrival_ratio = ai_spd / max(expected_spd, 1.0)
                        # arrival < 1: 输出未完全到达（管道吞吐不足）→ 需要更大 lead
                        arrival_correction = 1.0 / max(arrival_ratio, 0.30)
                        self._ctrl_speed_scale = (
                            0.97 * self._ctrl_speed_scale
                            + 0.03 * float(np.clip(arrival_correction, 0.7, 1.5))
                        )
                        # P1 修复：缩窄钳制范围 [0.90, 1.20]，与 speed_scale 路径一致
                        _lo, _hi = (0.85, 1.30) if self._wan_mode else (0.90, 1.20)
                        self._ctrl_speed_scale = float(np.clip(self._ctrl_speed_scale, _lo, _hi))
                        if now - self._arrival_calib_timer > 3.0:
                            _log.arrival_calib_diag(
                                arrival_ratio, expected_spd, ai_spd,
                                self._ctrl_speed_scale,
                            )
                            self._arrival_calib_timer = now

            # 保存本帧 arm_vel 供下帧 arrival calibration 使用
            arm_now = getattr(self.controller, 'crosshair_velocity', None)
            if arm_now is not None:
                self._prev_arm_vel_cts[0] = float(arm_now[0])
                self._prev_arm_vel_cts[1] = float(arm_now[1])

            # accel/cov penalty 只作用于管道 lead 部分（物理噪声相关），不惩罚
            # 控制器自身的确定性剩余时间 —— 否则 BALLISTIC 会永远"差一截"。
            pipeline_lead = base_lead * accel_penalty * cov_penalty
            smart_lead    = pipeline_lead + ctrl_lead

            accel_bonus = 0.5 * abs_accel * (smart_lead ** 2)
            bonus_norm = float(np.linalg.norm(accel_bonus))
            # 上限从 28px 提到 60px，配合更大的 smart_lead（含收敛时间）
            # ½at² 在 lead=120ms, accel=3000px/s² 时 ≈ 22px，60px 留有足够余量
            if bonus_norm > 60.0:
                accel_bonus = accel_bonus * (60.0 / bonus_norm)

            pred_abs_pos = abs_position + abs_velocity * smart_lead + accel_bonus

            context.dynamic_lag_ms = smart_lead * 1000.0
            rel_pred_pos = pred_abs_pos - self.ego_pos_px

        # 此处 is_coasting 已提前 return，仅保留有量测的帧
        rel_pos = abs_position - self.ego_pos_px
        error_distance = float(np.linalg.norm(rel_pos))
        threshold = self.mode_threshold_high if self.last_mode == "track" else self.mode_threshold_low
        new_mode = "flick" if error_distance > threshold else "track"
        if new_mode != self.last_mode:
            _log.controller_mode(self.last_mode, new_mode,
                                error_distance, threshold)
        self.controller.mode = new_mode
        self.last_mode = new_mode

        context.p_predict = (rel_pred_pos[0], rel_pred_pos[1])
        context.v_real = (abs_velocity[0], abs_velocity[1])
        context.a_real = (abs_accel[0], abs_accel[1])
        # 实战触发与 gate 用真实检测置信；旧逻辑写死 1.0 时 Triggerbot/调试全失真
        context.conf = float(best_track.conf_ema)
        context.is_valid = True

    def cleanup_old_targets(self, max_age: float = 2.0):
        """手动清理长期 coast 的轨迹（通常 TrackManager 自动处理，此方法为兼容旧调用）。"""
        now = time.perf_counter()
        dead_ids = [
            tid for tid, t in self.track_manager.tracks.items()
            if now - t.last_matched > max_age
        ]
        for tid in dead_ids:
            del self.track_manager.tracks[tid]
