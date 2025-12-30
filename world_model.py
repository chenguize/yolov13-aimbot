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
# [新增] 引入工厂方法以创建 Strategy
from aim_strategies.factory import create_aim_strategy


@dataclass
class TargetState:
    """
    单个目标的跟踪状态 (Kalman Filter 容器)
    """
    id: int
    first_seen: float
    last_seen: float

    # Kalman State: [x, y, vx, vy]
    # x, y: 屏幕绝对像素坐标 (经过位移对冲后的“当前时刻”坐标)
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

        # 测量噪声 R
        noise_pos = config.getfloat("WorldModel", "kalman_R_pos", 5.0)
        self.R = np.eye(2) * noise_pos

        # 过程噪声 Q
        noise_proc = config.getfloat("WorldModel", "kalman_Q_proc", 0.5)
        self.Q = np.eye(4) * noise_proc

        self.I = np.eye(4)

    def predict(self, state_obj: TargetState, dt: float):
        if dt <= 0: return

        # x = x + vx * dt
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
    """
    Phase 3: 世界模型核心
    """

    def __init__(self):
        self.kalman = SimpleKalman()
        self.current_target: Optional[TargetState] = None

        # --- 数据缓冲 ---
        self._data_lock = threading.Lock()
        self._latest_detection_data: Optional[Dict] = None

        # --- 配置参数 ---
        self.fov_x = config.getfloat("Triggerbot", "trigger_fov_x", 150.0)
        self.fov_y = config.getfloat("Triggerbot", "trigger_fov_y", 150.0)
        self.max_coast_frames = 5
        self.enable_prediction = config.getbool("WorldModel", "enable_prediction", False)
        self.system_latency = config.getfloat("WorldModel", "system_latency", 0.045)
        self.screen_w = config.getint("General", "screen_width", 1920)
        self.screen_h = config.getint("General", "screen_height", 1080)
        self.center_x = self.screen_w / 2
        self.center_y = self.screen_h / 2

        self.last_t_cap = 0.0

        # [Phase 3 修改]
        # 持有 Strategy 实例，确保"瞄准"和"对冲"使用同一套数学逻辑
        self.strategy = create_aim_strategy()

    def update_detections(self, detections: List[list], frame_id: int, t_cap: float, t_done: float):
        """
        [Inference Thread 调用]
        """
        with self._data_lock:
            self._latest_detection_data = {
                "dets": detections,
                "fid": frame_id,
                "t_cap": t_cap,
                "t_done": t_done
            }

    def step(self, context: InferenceContext, ring_buffer: RingBuffer):
        """
        [Main Thread 调用]
        主循环步进函数
        """

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

        # 计算 dt
        current_time = time.perf_counter()
        dt = current_time - self.last_t_cap
        if dt <= 0: dt = 0.001
        if dt > 0.5: dt = 0.1

        # =========================================================
        # [核心] 因果位移对冲 (Causal Hedging) - 使用 Strategy 映射
        # =========================================================

        # 1. 查账：获取鼠标计数 (Mickeys)
        mouse_counts_x, mouse_counts_y = ring_buffer.get_cursor_delta_sum(context.t_cap, current_time)

        # 2. 逆向映射：Mickeys -> Pixels (利用 Strategy 的逆函数)
        pixel_shift_x, pixel_shift_y = self.strategy.reverse_map(mouse_counts_x, mouse_counts_y)

        # 3. 计算视觉反向偏移 (Visual Shift)
        # 物理规律：鼠标右移(+)，摄像机右转，物体在屏幕上相对左移(-)
        # 所以视觉偏移量 = - (物理位移量)
        vis_shift_x = -pixel_shift_x
        vis_shift_y = -pixel_shift_y

        # =========================================================
        # 逻辑分支 A: 重置 (Reset)
        # =========================================================
        if not self.current_target and not best_det:
            context.is_valid = False
            self.last_t_cap = current_time
            return

        # =========================================================
        # 逻辑分支 B: 初始化 (Init) - 立即应用对冲
        # =========================================================
        if best_det and (not self.current_target or self.current_target.id != best_det.class_id):
            self.current_target = TargetState(
                id=best_det.class_id,
                first_seen=context.t_cap,
                last_seen=context.t_cap
            )
            # [修正] 即使是第一帧，也要把"过去"的坐标拉到"现在"
            current_x = best_det.x + vis_shift_x
            current_y = best_det.y + vis_shift_y

            self.current_target.state[:2] = [current_x, current_y]

            # 第一帧直接给预测坐标 (无速度)
            context.p_predict = (current_x, current_y)
            context.is_valid = True
            context.selected_id = best_det.class_id

            self.last_t_cap = current_time
            return

        # =========================================================
        # 逻辑分支 C: 追踪与对冲 (Tracking & Hedging)
        # =========================================================
        if self.current_target and best_det:
            target = self.current_target

            # [修正] 喂给 Kalman 的观测值 = 旧观测 + 视觉偏移
            z_measured = np.array([
                best_det.x + vis_shift_x,
                best_det.y + vis_shift_y
            ])

            # --- Kalman 迭代 ---
            self.kalman.predict(target, dt)
            self.kalman.update(target, z_measured)

            target.last_seen = context.t_cap
            target.hit_streak += 1

            vx, vy = target.state[2], target.state[3]

            # 稳定性熔断
            if target.hit_streak < 3:
                context.is_valid = True
            else:
                context.is_valid = True

            context.v_real = (vx, vy)
            context.selected_id = target.id

            # --- 时延预测 (Latency Prediction) ---
            if self.enable_prediction:
                t_predict_total = self.system_latency
            else:
                t_predict_total = 0.0

            pred_x = target.state[0] + vx * t_predict_total
            pred_y = target.state[1] + vy * t_predict_total

            context.p_predict = (pred_x, pred_y)

        # =========================================================
        # 逻辑分支 D: 惯性导航 (Coasting)
        # =========================================================
        elif self.current_target and not best_det:
            # 使用真实流逝时间判断丢失时长
            time_since_lost = current_time - self.last_t_cap

            if time_since_lost < (self.max_coast_frames * 0.02):
                target = self.current_target
                self.kalman.predict(target, dt)
                # 盲预测时不建议瞄准
                context.is_valid = False
            else:
                self.current_target = None
                context.is_valid = False

        self.last_t_cap = current_time

    def _select_target(self, targets: List[Detection]) -> Optional[Detection]:
        """简单的最近邻策略"""
        if not targets:
            return None

        best_target = None
        min_dist = float('inf')

        for t in targets:
            dx = t.x - self.center_x
            dy = t.y - self.center_y

            if abs(dx) > self.fov_x or abs(dy) > self.fov_y:
                continue

            dist = dx * dx + dy * dy
            if dist < min_dist:
                min_dist = dist
                best_target = t

        return best_target