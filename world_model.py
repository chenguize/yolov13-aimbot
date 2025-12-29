# world_model.py
# Phase 3 核心组件：世界状态仲裁与因果对冲中心
#
# 职责：
# 1. 接收 Inference 的原始检测结果
# 2. 结合 RingBuffer 进行因果位移对冲 (Causal Hedging)
# 3. 使用 Kalman Filter 维护目标真实状态 (Pixel Coordinates)
# 4. 进行时延补偿预测

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
    F: 状态转移矩阵
    H: 观测矩阵
    R: 测量噪声
    Q: 过程噪声
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
        # 调大此值会使轨迹更平滑，但响应变慢
        noise_pos = config.getfloat("WorldModel", "kalman_R_pos", 5.0)
        self.R = np.eye(2) * noise_pos

        # 过程噪声 Q (目标运动的不确定性)
        # 调大此值能跟上急转弯，但直线运动会抖动
        noise_proc = config.getfloat("WorldModel", "kalman_Q_proc", 0.5)
        self.Q = np.eye(4) * noise_proc

        # 预分配单位矩阵
        self.I = np.eye(4)

    def predict(self, state_obj: TargetState, dt: float):
        """
        预测步骤: X = F * X
        """
        if dt <= 0:
            return

        # 更新 F 矩阵中的 dt 部分
        # x = x + vx * dt
        # y = y + vy * dt
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        # X_pred = F @ X
        state_obj.state = self.F @ state_obj.state

        # P_pred = F @ P @ F.T + Q
        state_obj.covariance = self.F @ state_obj.covariance @ self.F.T + self.Q

    def update(self, state_obj: TargetState, measurement: np.ndarray):
        """
        更新步骤: 融合观测值
        measurement: [x, y]
        """
        # y = z - H @ x (残差)
        z = measurement
        y = z - (self.H @ state_obj.state)

        # S = H @ P @ H.T + R (残差协方差)
        S = self.H @ state_obj.covariance @ self.H.T + self.R

        # K = P @ H.T @ inv(S) (卡尔曼增益)
        try:
            K = state_obj.covariance @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return  # 矩阵奇异，跳过更新

        # x = x + K @ y
        state_obj.state = state_obj.state + (K @ y)

        # P = (I - K @ H) @ P
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
        # 目标筛选 FOV
        self.fov_x = config.getfloat("Triggerbot", "trigger_fov_x", 150.0)
        self.fov_y = config.getfloat("Triggerbot", "trigger_fov_y", 150.0)

        # 目标丢失后的维持帧数
        self.max_coast_frames = 5

        # 系统总延迟 (用于预测补偿)
        # 包含：输入延迟 + 采集延迟 + 推理延迟 + 网络延迟
        self.system_latency = config.getfloat("WorldModel", "system_latency", 0.045)

        # 屏幕中心
        self.screen_w = config.getint("General", "screen_width", 1920)
        self.screen_h = config.getint("General", "screen_height", 1080)
        self.center_x = self.screen_w / 2
        self.center_y = self.screen_h / 2

        # 记录上一帧的时间锚点，用于计算 dt
        self.last_t_cap = 0.0

        # [关键] 灵敏度系数 (Counts per Pixel)
        # 用于将 RingBuffer 的鼠标计数转换为像素位移
        # 理想情况应从 Strategy 动态获取，这里先读配置或使用经验值
        # 默认值应与 Strategy 中的初始 K 值保持一致
        self.k_factor_x = config.getfloat("AimStrategy", "k_factor_x", 1.9)
        self.k_factor_y = config.getfloat("AimStrategy", "k_factor_y", 1.2)

    def update_detections(self, detections: List[list], frame_id: int, t_cap: float, t_done: float):
        """
        [Inference Thread 调用]
        接收最新的检测结果并暂存
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
        主循环步进函数：执行完整的 仲裁 -> 对冲 -> 预测 流程
        就地修改传入的 context 对象
        """

        # 1. 从缓冲区提取数据
        data = None
        with self._data_lock:
            data = self._latest_detection_data

        # 如果没有新数据，或者数据过旧（理论上应该由 FrameID 判断），这里简化处理
        if not data:
            context.is_valid = False
            return

        # 2. 填充 Context 基础信息
        # 将原始 List 转为 Detection 对象
        # data['dets'] 格式: [x1, y1, x2, y2, conf, cls]
        current_dets = []
        for d in data['dets']:
            # 转换为中心点坐标
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

        # 3. 目标仲裁 (Select Best Target)
        best_det = self._select_target(current_dets)

        # 计算 dt (距离上一帧的时间)
        current_time = context.t_cap
        dt = current_time - self.last_t_cap

        # 异常时间处理
        if dt <= 0: dt = 0.001
        if dt > 0.5: dt = 0.1  # 掉帧保护，避免 Kalman 飞出宇宙

        # =========================================================
        # 逻辑分支 A: 重置 (Reset)
        # =========================================================
        if not self.current_target and not best_det:
            context.is_valid = False
            self.last_t_cap = current_time
            return

        # =========================================================
        # 逻辑分支 B: 初始化 (Init)
        # =========================================================
        # 如果 ID 变了 (Switch) 或者 新目标出现 (New)
        if best_det and (not self.current_target or self.current_target.id != best_det.class_id):
            # Phase 3 简化：直接重置 Kalman，不尝试平滑过渡
            self.current_target = TargetState(
                id=best_det.class_id,
                first_seen=current_time,
                last_seen=current_time
            )
            self.current_target.state[:2] = [best_det.x, best_det.y]

            # 第一帧因为没有速度信息，标为无效
            context.is_valid = False
            context.selected_id = best_det.class_id
            self.last_t_cap = current_time
            return

        # =========================================================
        # 逻辑分支 C: 追踪与对冲 (Tracking & Hedging)
        # =========================================================
        if self.current_target and best_det:
            target = self.current_target

            # --- [Phase 3 核心] 因果位移对冲 ---
            # 1. 查账：从上一帧截图时间(last_t_cap) 到 这一帧截图时间(current_time)
            #    这段时间内，鼠标移动了多少 Counts?
            mouse_counts_x, mouse_counts_y = ring_buffer.get_cursor_delta_sum(self.last_t_cap, current_time)

            # 2. 换算：Mickeys -> Pixels
            # Pixels = Counts / K_Factor
            # 必须防止除零
            kx = self.k_factor_x if self.k_factor_x > 0.1 else 1.0
            ky = self.k_factor_y if self.k_factor_y > 0.1 else 1.0

            pixel_shift_x = mouse_counts_x / kx
            pixel_shift_y = mouse_counts_y / ky

            # 3. 对冲 (Compensation)
            # 如果鼠标向右移(+)，画面里的物体会向左跑(-)。
            # 我们想知道物体"原本"在哪，所以要"加回"这个位移。
            # z_measured = 观测坐标 + 鼠标造成的视觉位移
            z_measured = np.array([
                best_det.x + pixel_shift_x,
                best_det.y + pixel_shift_y
            ])

            # --- Kalman 迭代 ---
            self.kalman.predict(target, dt)
            self.kalman.update(target, z_measured)

            # 更新状态元数据
            target.last_seen = current_time
            target.hit_streak += 1

            # --- 输出构建 ---
            vx, vy = target.state[2], target.state[3]

            # 稳定性熔断：前 3 帧数据不稳定，不开枪
            if target.hit_streak < 3:
                context.is_valid = False
            else:
                context.is_valid = True

            context.v_real = (vx, vy)
            context.selected_id = target.id

            # --- 时延预测 (Latency Prediction) ---
            # 预测时间 = 画面年龄 + 系统延迟
            # 画面年龄 = 当前真实时间(perf_counter) - 截图时间(t_cap)
            # 但在回放或仿真中，我们通常用配置的固定延迟

            t_predict_total = self.system_latency

            # P_predict = P_current + V * t
            # 注意：这里的 P_current 是 Kalman 滤波后的"绝对世界坐标"
            # 这一步计算的是"假设我鼠标不动，N毫秒后他在屏幕哪里"

            pred_x = target.state[0] + vx * t_predict_total
            pred_y = target.state[1] + vy * t_predict_total

            context.p_predict = (pred_x, pred_y)

        # =========================================================
        # 逻辑分支 D: 惯性导航 (Coasting)
        # =========================================================
        elif self.current_target and not best_det:
            # 如果丢失时间不长，做线性外推
            time_since_lost = current_time - self.current_target.last_seen

            if time_since_lost < (self.max_coast_frames * 0.02):  # 约 100ms
                target = self.current_target
                self.kalman.predict(target, dt)
                # 盲预测时不建议设置为 valid，除非近战
                context.is_valid = False
            else:
                # 彻底丢失
                self.current_target = None
                context.is_valid = False

        self.last_t_cap = current_time

    def _select_target(self, targets: List[Detection]) -> Optional[Detection]:
        """
        简单的最近邻策略
        """
        if not targets:
            return None

        best_target = None
        min_dist = float('inf')

        for t in targets:
            # 计算到屏幕中心的距离
            dx = t.x - self.center_x
            dy = t.y - self.center_y

            # FOV 过滤
            if abs(dx) > self.fov_x or abs(dy) > self.fov_y:
                continue

            dist = dx * dx + dy * dy
            if dist < min_dist:
                min_dist = dist
                best_target = t

        return best_target