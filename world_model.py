import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from config import config
from perception.ring_buffer import RingBuffer
from utils.types import Detection, InferenceContext
from aim_strategies.factory import create_aim_strategy


@dataclass
class TargetState:
    id: int
    first_seen: float
    last_seen: float
    state: np.ndarray = field(default_factory=lambda: np.zeros(4))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(4) * 100.0)
    confidence: float = 0.3


class SimpleKalman:
    def __init__(self):
        self.F = np.eye(4)
        self.R = np.eye(2) * config.getfloat("WorldModel", "kalman_R_pos", 5.0)
        self.Q = np.eye(4) * config.getfloat("WorldModel", "kalman_Q_proc", 0.5)

    def predict(self, state_obj: TargetState, dt: float):
        if dt <= 0: return
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        state_obj.state = self.F @ state_obj.state
        state_obj.covariance = self.F @ state_obj.covariance @ self.F.T + self.Q

    def update(self, state_obj: TargetState, measurement: np.ndarray) -> np.ndarray:
        if np.any(np.isnan(measurement)): return np.zeros(2)
        y = measurement - state_obj.state[:2]
        P = state_obj.covariance
        S = P[:2, :2] + self.R
        det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        if abs(det) < 1e-6: return y
        S_inv = np.array([[S[1, 1] / det, -S[0, 1] / det], [-S[1, 0] / det, S[0, 0] / det]])
        K = P[:, :2] @ S_inv
        state_obj.state = state_obj.state + (K @ y)
        state_obj.covariance = P - (K @ P[:2, :])
        return y


class WorldModel:
    def __init__(self):
        self.kalman = SimpleKalman()
        self.current_target: Optional[TargetState] = None

        # [Sync Optimization] 使用 Event 代替 sleep 轮询
        self.frame_ready_event = threading.Event()
        self._data_lock = threading.Lock()
        self._latest_detection_data: Optional[Dict] = None

        self.capture_size = config.getint("General", "capture_size", 256)
        self.crop_center = self.capture_size / 2
        self.search_fov = config.getfloat("WorldModel", "search_fov", 100.0)

        self.last_t_cap = 0.0
        self.last_processed_fid = -1
        self.strategy = create_aim_strategy()
        from controllers.controller_factory import get_controller
        self.controller = get_controller()

        self.last_raw_pos: Optional[Tuple[float, float]] = None
        self.last_calib_t: float = 0.0
        self.dynamic_lag = config.getfloat("WorldModel", "perception_lag_ms", 20.0) / 1000.0
        self.lag_lr = 0.01

    def update_detections(self, detections: np.ndarray, frame_id: int, t_cap: float, t_done: float):
        new_data = {"dets": detections, "fid": frame_id, "t_cap": t_cap}
        with self._data_lock:
            self._latest_detection_data = new_data
        # [Signal] 唤醒 Main 线程
        self.frame_ready_event.set()

    def wait_for_frame(self, timeout: float = 0.1) -> bool:
        """阻塞等待新帧，返回是否成功获取"""
        return self.frame_ready_event.wait(timeout)

    def step(self, context: InferenceContext, ring_buffer: RingBuffer) -> Optional[Tuple[float, float]]:
        # [Reset] 清除信号，准备下一次等待
        self.frame_ready_event.clear()

        current_time = time.perf_counter()
        data = None
        with self._data_lock:
            if self._latest_detection_data is not None:
                data = self._latest_detection_data

        if not data or (current_time - data['t_cap'] > 0.1):
            context.is_valid = False
            self.last_raw_pos = None
            return None

        is_new_frame = (data['fid'] != self.last_processed_fid)
        corrected_t_cap = data['t_cap'] - self.dynamic_lag
        mouse_x, mouse_y = ring_buffer.get_cursor_delta_sum(corrected_t_cap, current_time)
        px_shift_x, px_shift_y = self.strategy.reverse_map(mouse_x, mouse_y)
        vis_shift = (-px_shift_x, -px_shift_y)

        best_det = self._select_target(data['dets'])
        dt = max(0.001, min(current_time - self.last_t_cap, 0.1))

        if is_new_frame and best_det and self.current_target and self.current_target.id == best_det.class_id:
            if self.last_raw_pos is not None:
                raw_dx, raw_dy = best_det.x - self.last_raw_pos[0], best_det.y - self.last_raw_pos[1]
                human_cx, human_cy = ring_buffer.get_human_delta_sum(self.last_calib_t, data['t_cap'])
                if abs(human_cx) > 3 or abs(human_cy) > 3:
                    self.strategy.feedback_update(human_cx, -raw_dx, human_cy, -raw_dy)
            self.last_raw_pos, self.last_calib_t = (best_det.x, best_det.y), data['t_cap']
        elif not best_det:
            self.last_raw_pos = None

        active_data, last_residual = self._process_tracking(best_det, vis_shift[0], vis_shift[1], dt, data['t_cap'],
                                                            is_new_frame)

        if is_new_frame and self.current_target and last_residual is not None:
            v = self.current_target.state[2:4]
            v_sq = v[0] ** 2 + v[1] ** 2
            # [Fix] 增加最小衰减，防止在静止目标上 lag 锁死
            if v_sq > 25.0 and abs(last_residual[0]) < 50:
                delta_tau = -(last_residual[0] * v[0] + last_residual[1] * v[1]) / v_sq
                self.dynamic_lag = np.clip(self.dynamic_lag + delta_tau * self.lag_lr, 0.005, 0.150)
            else:
                # 缓慢回归默认值，适应环境变化
                self.dynamic_lag = self.dynamic_lag * 0.999 + 0.020 * 0.001

        final_move = None
        if active_data["is_valid"]:
            pred_x, pred_y = active_data["p_predict"]
            intent_x, intent_y = self.strategy.calculate_mouse_move(pred_x - self.crop_center,
                                                                    pred_y - self.crop_center)
            self.controller.compute(intent_x, intent_y, dt)
            final_move = (intent_x, intent_y)
            self._apply_to_context(context, active_data)
            context.dynamic_lag_ms = self.dynamic_lag * 1000
            context.final_move = final_move

        self.last_t_cap = current_time
        if is_new_frame: self.last_processed_fid = data['fid']
        return final_move

    # _process_tracking, _apply_to_context, _select_target 保持不变 (已包含之前的 Fix)
    def _process_tracking(self, best_det, vs_x, vs_y, dt, t_cap, is_new_frame):
        res = {"is_valid": False, "p_predict": (0, 0), "v_real": (0, 0), "conf": 0.0, "id": -1, "t_cap": t_cap}
        if not best_det:
            if self.current_target:
                self.kalman.predict(self.current_target, dt)
                if (time.perf_counter() - self.current_target.last_seen) > 0.1: self.current_target = None
            return res, None
        if not self.current_target or self.current_target.id != best_det.class_id:
            self.current_target = TargetState(id=best_det.class_id, first_seen=t_cap, last_seen=t_cap)
            self.current_target.state[:2] = [best_det.x + vs_x, best_det.y + vs_y]
        target = self.current_target
        self.kalman.predict(target, dt)
        residual = self.kalman.update(target,
                                      np.array([best_det.x + vs_x, best_det.y + vs_y])) if is_new_frame else None
        if is_new_frame: target.last_seen = t_cap
        res.update({"is_valid": True, "p_predict": (target.state[0], target.state[1]),
                    "v_real": (target.state[2], target.state[3]), "conf": 1.0, "id": target.id})
        return res, residual

    def _apply_to_context(self, context, data):
        context.is_valid, context.p_predict, context.v_real = data["is_valid"], data["p_predict"], data["v_real"]
        context.selected_id, context.t_cap, context.conf = data["id"], data["t_cap"], data["conf"]
        context.targets = [Detection(x=(d[0] + d[2]) / 2, y=(d[1] + d[3]) / 2, w=d[2] - d[0], h=d[3] - d[1], conf=d[4],
                                     class_id=int(d[5]), xyxy=d[:4]) for d in data["raw_dets"]] if data.get(
            "raw_dets") is not None else []

    def _select_target(self, dets: np.ndarray) -> Optional[Detection]:
        if dets is None or len(dets) == 0: return None
        if np.isnan(dets).any(): return None
        candidates = dets[dets[:, 5] == 7]
        if len(candidates) == 0: return None
        cxs, cys = (candidates[:, 0] + candidates[:, 2]) * 0.5, (candidates[:, 1] + candidates[:, 3]) * 0.5
        dxs, dys = cxs - self.crop_center, cys - self.crop_center
        fov_mask = (np.abs(dxs) <= self.search_fov) & (np.abs(dys) <= self.search_fov)
        if not np.any(fov_mask): return None
        best_idx = np.argmin(dxs[fov_mask] ** 2 + dys[fov_mask] ** 2)
        idx = np.where(fov_mask)[0][best_idx]
        return Detection(x=float(cxs[idx]), y=float(cys[idx]), w=float(candidates[idx, 2] - candidates[idx, 0]),
                         h=float(candidates[idx, 3] - candidates[idx, 1]), conf=float(candidates[idx, 4]),
                         class_id=int(candidates[idx, 5]), xyxy=candidates[idx, :4])