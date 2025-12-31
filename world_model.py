# world_model.py
# Phase 4 最终版：强制因果对冲 + 目标ID过滤
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

        # 读取配置中的灵敏度用于因果对冲计算
        self.k_x = config.getfloat("AimStrategy", "k_factor_x", 1.9)
        self.k_y = config.getfloat("AimStrategy", "k_factor_y", 1.2)

        self.lag_queue = deque()
        self.last_t_cap = 0.0
        self.last_processed_fid = -1  # [安全] 防止重复消费同一帧
        self.strategy = create_aim_strategy()

    def update_detections(self, detections: List[list], frame_id: int, t_cap: float, t_done: float):
        """由 InferenceThread 调用，推送最新视觉数据"""
        with self._data_lock:
            self._latest_detection_data = {
                "dets": detections,
                "fid": frame_id,
                "t_cap": t_cap
            }

    def step(self, context: InferenceContext, ring_buffer: RingBuffer):
        """主循环调用，执行世界状态仲裁"""
        current_time = time.perf_counter()

        # 1. 提取数据（线程安全）
        data = None
        with self._data_lock:
            data = self._latest_detection_data

        # [安全熔断 1] 数据为空
        if not data:
            context.is_valid = False
            return

        # [安全熔断 2] 数据超时 (>100ms) -> 说明 Capture 卡死，立即停止瞄准
        if current_time - data['t_cap'] > 0.1:
            context.is_valid = False
            return

        # [逻辑修正] 判断是否是新的一帧
        is_new_frame = (data['fid'] != self.last_processed_fid)
        if is_new_frame:
            self.last_processed_fid = data['fid']

        # 2. 因果对冲 (Causal Hedging)
        mouse_x, mouse_y = ring_buffer.get_cursor_delta_sum(data['t_cap'], current_time)
        px_shift_x = mouse_x / max(0.1, self.k_x)
        px_shift_y = mouse_y / max(0.1, self.k_y)
        vis_shift_x, vis_shift_y = -px_shift_x, -px_shift_y

        # 3. 目标选择 (这里会执行 ID=7 的过滤)
        best_det = self._select_target(data['dets'])
        dt = max(0.001, min(current_time - self.last_t_cap, 0.1))

        # 4. 状态更新与认知处理
        # 仅在新帧时 Update 卡尔曼观测值，否则只做 Predict
        active_context_data = self._process_tracking(best_det, vis_shift_x, vis_shift_y, dt, data['t_cap'],
                                                     is_new_frame)
        active_context_data["raw_dets"] = data["dets"]

        # 5. 压入感知滞后链
        self.lag_queue.append({"timestamp": current_time, "data": active_context_data})

        # 6. 弹出已成熟的决策
        # 注意：如果您在 config.ini 里 perception_lag_ms 设为了 0，这里会立即弹出
        lag_seconds = config.getfloat("WorldModel", "perception_lag_ms", 0.0) / 1000.0
        ready_context = None
        while self.lag_queue and (current_time - self.lag_queue[0]["timestamp"]) >= lag_seconds:
            ready_context = self.lag_queue.popleft()["data"]

        # 7. 填充输出上下文
        if ready_context:
            self._apply_to_context(context, ready_context)
        else:
            context.is_valid = False

        self.last_t_cap = current_time

    def _process_tracking(self, best_det: Optional[Detection], vs_x: float, vs_y: float, dt: float,
                          t_cap: float, is_new_frame: bool) -> Dict:
        # 初始化基础响应结构
        res = {"is_valid": False, "p_predict": (0, 0), "v_real": (0, 0), "conf": 0.0, "id": -1, "t_cap": t_cap}

        # Case A: 没有检测到目标 (或者目标不是 ID 7 被过滤掉了)
        if not best_det:
            if self.current_target:
                # 惯性维持：虽然这帧没看见，但根据速度继续预测一小会儿
                self.kalman.predict(self.current_target, dt)
                # 如果丢失超过 100ms，清除目标
                if (time.perf_counter() - self.current_target.last_seen) > 0.1:
                    self.current_target = None
            return res

        # Case B: 发现新目标或切换目标
        if not self.current_target or self.current_target.id != best_det.class_id:
            self.current_target = TargetState(
                id=best_det.class_id, first_seen=t_cap, last_seen=t_cap,
                target_offset=(0.0, 0.0), confidence=0.3
            )
            self.current_target.state[:2] = [best_det.x + vs_x, best_det.y + vs_y]

        target = self.current_target

        # 总是进行预测
        self.kalman.predict(target, dt)

        # 只有是新的一帧图像时，才用观测值修正卡尔曼状态
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
                # d[5] 是 class_id
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
            # d[5] 是 Class ID
            cls_id = int(d[5])

            # [新增] 强制过滤：如果不是 ID 7，直接跳过
            if cls_id != 7:
                continue

            cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
            dx, dy = cx - self.crop_center_x, cy - self.crop_center_y

            # FOV 检查
            if abs(dx) > self.search_fov or abs(dy) > self.search_fov: continue

            # 距离排序
            dist_sq = dx ** 2 + dy ** 2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_det = Detection(x=cx, y=cy, w=d[2] - d[0], h=d[3] - d[1], conf=d[4], class_id=cls_id,
                                     xyxy=(d[0], d[1], d[2], d[3]))

        return best_det