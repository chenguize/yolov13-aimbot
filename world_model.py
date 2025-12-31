# world_model.py
# Phase 3 最终版：强制因果对冲 + 独立闭环调参双轨制
import time
import math
import threading
import random
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
    target_offset: Tuple[float, float] = (0.0, 0.0)


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

        self.search_fov = config.getfloat("WorldModel", "search_fov", 100.0)

        self.lag_queue = deque()
        self.last_t_cap = 0.0
        self.last_processed_fid = -1
        self.strategy = create_aim_strategy()

        # [新增] 轨道 B：独立调参专用状态记录
        self.last_raw_pos: Optional[Tuple[float, float]] = None
        self.last_calib_t: float = 0.0

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

        if current_time - data['t_cap'] > 0.1:
            context.is_valid = False
            return

        is_new_frame = (data['fid'] != self.last_processed_fid)

        # --- 轨道 A: 因果对冲 (Hedging) ---
        # 使用当前策略中的 K 值进行对冲，用于卡尔曼更新和当前瞄准决策
        mouse_x, mouse_y = ring_buffer.get_cursor_delta_sum(data['t_cap'], current_time)
        px_shift_x, px_shift_y = self.strategy.reverse_map(mouse_x, mouse_y)
        vis_shift_x, vis_shift_y = -px_shift_x, -px_shift_y

        best_det = self._select_target(data['dets'])
        dt = max(0.001, min(current_time - self.last_t_cap, 0.1))

        # --- 轨道 B: 独立闭环调参 (Calibration) ---
        # 仅在新帧到达、目标存在且 ID 匹配时，对比原始物理坐标差
        if is_new_frame and best_det and self.current_target and self.current_target.id == best_det.class_id:
            if self.last_raw_pos is not None:
                # 1. 算出两帧间的原始屏幕位移 (不带对冲补偿的物理差值)
                raw_dx = best_det.x - self.last_raw_pos[0]
                raw_dy = best_det.y - self.last_raw_pos[1]

                # 2. 算出账本总支出 (从上帧截图到本帧截图期间发送的所有原始指令)
                total_counts_x, total_counts_y = ring_buffer.get_cursor_delta_sum(self.last_calib_t, data['t_cap'])

                # 3. 闭环结算：指令 vs 原始像素位移
                # (注意：鼠标指令方向与画面位移方向相反，传入负号以对齐物理量)
                self.strategy.feedback_update(total_counts_x, -raw_dx, total_counts_y, -raw_dy)

            # 更新下一轮调参的参考基准
            self.last_raw_pos = (best_det.x, best_det.y)
            self.last_calib_t = data['t_cap']
        elif not best_det:
            # 丢失目标时重置调参基准
            self.last_raw_pos = None

        # 4. 状态更新与认知处理 (维持原本逻辑，使用对冲后的 vs 坐标)
        active_context_data = self._process_tracking(best_det, vis_shift_x, vis_shift_y, dt, data['t_cap'],
                                                     is_new_frame)
        active_context_data["raw_dets"] = data["dets"]

        # 5. 压入感知滞后链
        self.lag_queue.append({"timestamp": current_time, "data": active_context_data})

        # 6. 弹出已成熟的决策
        lag_seconds = config.getfloat("WorldModel", "perception_lag_ms", 0.0) / 1000.0
        ready_context = None
        while self.lag_queue and (current_time - self.lag_queue[0]["timestamp"]) >= lag_seconds:
            ready_context = self.lag_queue.popleft()["data"]

        if ready_context:
            self._apply_to_context(context, ready_context)
        else:
            context.is_valid = False

        self.last_t_cap = current_time
        if is_new_frame:
            self.last_processed_fid = data['fid']

    def _process_tracking(self, best_det: Optional[Detection], vs_x: float, vs_y: float, dt: float,
                          t_cap: float, is_new_frame: bool) -> Dict:
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
                target_offset=(0.0, 0.0), confidence=0.3
            )
            self.current_target.state[:2] = [best_det.x + vs_x, best_det.y + vs_y]

        target = self.current_target
        self.kalman.predict(target, dt)

        if is_new_frame:
            z_measured = np.array([best_det.x + vs_x, best_det.y + vs_y])
            self.kalman.update(target, z_measured)
            target.last_seen = t_cap

        res.update({"is_valid": True, "p_predict": (target.state[0], target.state[1]),
                    "v_real": (target.state[2], target.state[3]), "conf": 1.0, "id": target.id})
        return res

    def _apply_to_context(self, context: InferenceContext, data: Dict):
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

        for d in dets_raw:
            cls_id = int(d[5])
            if cls_id != 7: continue

            cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
            dx, dy = cx - self.crop_center_x, cy - self.crop_center_y

            if abs(dx) > self.search_fov or abs(dy) > self.search_fov: continue

            dist_sq = dx ** 2 + dy ** 2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_det = Detection(x=cx, y=cy, w=d[2] - d[0], h=d[3] - d[1], conf=d[4], class_id=cls_id,
                                     xyxy=(d[0], d[1], d[2], d[3]))

        return best_det