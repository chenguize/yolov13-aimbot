# world_model.py
# Phase 4 最终版：动态延迟补偿 + 智能搜索
import time
import math
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
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
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.R = np.eye(2) * config.getfloat("WorldModel", "kalman_R_pos", 5.0)
        self.Q = np.eye(4) * config.getfloat("WorldModel", "kalman_Q_proc", 0.5)
        self.I = np.eye(4)

    def predict(self, state_obj: TargetState, dt: float):
        if dt <= 0: return
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        state_obj.state = self.F @ state_obj.state
        state_obj.covariance = self.F @ state_obj.covariance @ self.F.T + self.Q

    def update(self, state_obj: TargetState, measurement: np.ndarray):
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
    def __init__(self):
        self.kalman = SimpleKalman()
        self.current_target: Optional[TargetState] = None
        self._data_lock = threading.Lock()
        self._latest_detection_data: Optional[Dict] = None

        self.capture_size = config.getint("General", "capture_size", 256)
        self.crop_center_x = self.capture_size / 2
        self.crop_center_y = self.capture_size / 2
        self.search_fov = config.getfloat("WorldModel", "search_fov", 256.0)

        self.k_x = config.getfloat("AimStrategy", "k_factor_x", 1.0)
        self.k_y = config.getfloat("AimStrategy", "k_factor_y", 1.0)

        # [动态延迟配置]
        self.enable_prediction = config.getbool("WorldModel", "enable_prediction", True)
        # 硬件延迟是物理死值 (USB + 显示器响应)，通常 10-20ms，不会变
        self.hardware_latency = config.getfloat("WorldModel", "hardware_latency", 0.015)

        # 平均延迟平滑器 (防止某一帧波动太大)
        self.avg_pipeline_latency = 0.02

        self.lag_queue = deque()
        self.last_t_cap = 0.0
        self.last_processed_fid = -1
        self.strategy = create_aim_strategy()

    def update_detections(self, detections: List[list], frame_id: int, t_cap: float, t_done: float):
        with self._data_lock:
            self._latest_detection_data = {
                "dets": detections,
                "fid": frame_id,
                "t_cap": t_cap
            }

    def step(self, context: InferenceContext, ring_buffer: RingBuffer):
        current_time = time.perf_counter()

        data = None
        with self._data_lock:
            data = self._latest_detection_data

        if not data:
            context.is_valid = False
            return

        # [核心逻辑] 动态计算这一帧处理了多久
        # Latency = 当前时刻 - 截图时刻
        current_pipeline_latency = current_time - data['t_cap']

        # 熔断：如果延迟超过 200ms，说明卡死了，不要预测了
        if current_pipeline_latency > 0.2:
            current_pipeline_latency = 0.0
            context.is_valid = False
            return

        # 平滑处理 (EMA)
        self.avg_pipeline_latency = 0.8 * self.avg_pipeline_latency + 0.2 * current_pipeline_latency

        # 总预测时间 = 动态软件延迟 + 静态硬件延迟
        total_latency = current_pipeline_latency + self.hardware_latency

        is_new_frame = (data['fid'] != self.last_processed_fid)
        if is_new_frame:
            self.last_processed_fid = data['fid']

        mouse_x, mouse_y = ring_buffer.get_cursor_delta_sum(data['t_cap'], current_time)
        px_shift_x = mouse_x / max(0.1, self.k_x)
        px_shift_y = mouse_y / max(0.1, self.k_y)
        vis_shift_x, vis_shift_y = -px_shift_x, -px_shift_y

        best_det = self._select_target(data['dets'])
        dt = max(0.001, min(current_time - self.last_t_cap, 0.1))

        # 将计算好的 total_latency 传进去
        active_context_data = self._process_tracking(
            best_det, vis_shift_x, vis_shift_y, dt,
            data['t_cap'], is_new_frame, total_latency
        )
        active_context_data["raw_dets"] = data["dets"]

        self.lag_queue.append({"timestamp": current_time, "data": active_context_data})

        # 既然我们已经手动做了延迟补偿，Perception Lag 就可以设为 0 了
        lag_seconds = 0.0
        ready_context = None
        while self.lag_queue and (current_time - self.lag_queue[0]["timestamp"]) >= lag_seconds:
            ready_context = self.lag_queue.popleft()["data"]

        if ready_context:
            self._apply_to_context(context, ready_context)
        else:
            context.is_valid = False

        self.last_t_cap = current_time

    def _process_tracking(self, best_det: Optional[Detection], vs_x: float, vs_y: float, dt: float,
                          t_cap: float, is_new_frame: bool, pred_dt: float) -> Dict:
        res = {"is_valid": False, "p_predict": (0, 0), "v_real": (0, 0), "conf": 0.0, "id": -1, "t_cap": t_cap}

        if not best_det:
            if self.current_target:
                self.kalman.predict(self.current_target, dt)
                if (time.perf_counter() - self.current_target.last_seen) > 0.1:
                    self.current_target = None
            return res

        if not self.current_target or self.current_target.id != best_det.class_id:
            self.current_target = TargetState(
                id=best_det.class_id, first_seen=t_cap, last_seen=t_cap,
                confidence=0.3
            )
            self.current_target.state[:2] = [best_det.x + vs_x, best_det.y + vs_y]

        target = self.current_target
        self.kalman.predict(target, dt)

        if is_new_frame:
            z = np.array([best_det.x + vs_x, best_det.y + vs_y])
            self.kalman.update(target, z)
            target.last_seen = t_cap

        # --- 真正的动态预测 ---
        curr_x, curr_y = target.state[0], target.state[1]
        vel_x, vel_y = target.state[2], target.state[3]

        if self.enable_prediction:
            # 这里的 pred_dt 就是刚才动态算出来的 (软件耗时 + 硬件耗时)
            pred_x = curr_x + vel_x * pred_dt
            pred_y = curr_y + vel_y * pred_dt
        else:
            pred_x, pred_y = curr_x, curr_y

        res.update({"is_valid": True, "p_predict": (pred_x, pred_y),
                    "v_real": (vel_x, vel_y), "conf": 1.0, "id": target.id})
        return res

    def _apply_to_context(self, context, data):
        context.is_valid = data["is_valid"]
        context.p_predict = data["p_predict"]
        context.v_real = data["v_real"]
        context.selected_id = data["id"]
        context.t_cap = data["t_cap"]
        context.conf = data["conf"]
        if "raw_dets" in data and data["raw_dets"]:
            context.targets = []
            for d in data["raw_dets"]:
                cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
                context.targets.append(
                    Detection(x=cx, y=cy, w=d[2] - d[0], h=d[3] - d[1], conf=d[4], class_id=int(d[5]),
                              xyxy=(d[0], d[1], d[2], d[3])))
        else:
            context.targets = []

    def _select_target(self, dets_raw: List[list]) -> Optional[Detection]:
        if not dets_raw: return None
        best_det, min_dist_sq = None, float('inf')

        current_search_radius = self.search_fov
        if self.current_target:
            current_search_radius = min(self.search_fov, 80.0)

        for d in dets_raw:
            if int(d[5]) != 7: continue  # 必须是头

            cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
            dx, dy = cx - self.crop_center_x, cy - self.crop_center_y

            if abs(dx) > current_search_radius or abs(dy) > current_search_radius: continue

            dist_sq = dx ** 2 + dy ** 2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_det = Detection(x=cx, y=cy, w=d[2] - d[0], h=d[3] - d[1], conf=d[4], class_id=int(d[5]),
                                     xyxy=(d[0], d[1], d[2], d[3]))

        return best_det