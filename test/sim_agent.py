# sim_agent.py
import time
import numpy as np

from perception.ring_buffer import RingBuffer
from utils.types import Detection, InferenceContext


class SimAIAgent:
    def __init__(self, noise_level: float = 5.0, dropout_rate: float = 0.05):
        # [时钟霸权]：彻底接管系统底层时钟！
        # 强制让所有使用 time.perf_counter() 的卡尔曼滤波和缓冲环与仿真同步
        self.sim_time = 0.0
        self._original_perf_counter = time.perf_counter
        time.perf_counter = lambda: self.sim_time

        try:
            from world_model import WorldModel
        except ImportError:
            pass

        self.world_model = WorldModel()
        self.ring_buffer = RingBuffer()

        self.sim_steps = 0
        self.noise_level = noise_level
        self.dropout_rate = dropout_rate

        self.enemy_pos = np.array([200.0, 0.0], dtype=np.float64)
        self.enemy_vel = np.array([0.0, 0.0], dtype=np.float64)
        self.crosshair_pos = np.zeros(2, dtype=np.float64)
        self.center = np.array([self.world_model.crop_center, self.world_model.crop_center], dtype=np.float64)

        self.ctx = InferenceContext()
        self.ctx.center_pos = self.center

    def tick_enemy(self, t: float):
        A1, w1 = 140.0, 2.2
        A2, w2 = 30.0, 5.5
        self.enemy_pos[0] = A1 * np.sin(w1 * t) + A2 * np.sin(w2 * t)
        self.enemy_vel[0] = A1 * w1 * np.cos(w1 * t) + A2 * w2 * np.cos(w2 * t)

        B1, w3, phase = 80.0, 1.8, 0.5
        B2, w4 = 20.0, 4.0
        self.enemy_pos[1] = B1 * np.sin(w3 * t + phase) + B2 * np.sin(w4 * t)
        self.enemy_vel[1] = B1 * w3 * np.cos(w3 * t + phase) + B2 * w4 * np.cos(w4 * t)

    def tick_cv(self):
        if self.dropout_rate > 0 and np.random.random() < self.dropout_rate:
            self.ctx.targets = []
            return

        relative_pos = self.enemy_pos - self.crosshair_pos
        screen_x = relative_pos[0] + self.center[0]
        screen_y = relative_pos[1] + self.center[1]

        if self.noise_level > 0:
            noise = np.random.randn(2) * self.noise_level
            screen_x += noise[0]
            screen_y += noise[1]
            confidence = np.clip(0.9 + np.random.randn() * 0.05, 0.5, 1.0)
        else:
            confidence = 0.9

        detection = Detection(
            x=screen_x, y=screen_y, w=60, h=120, conf=confidence, class_id=0,
            xyxy=np.array([screen_x - 30, screen_y - 60, screen_x + 30, screen_y + 60])
        )
        self.ctx.targets = [detection]

    def tick_mouse(self):
        mx, my = self.world_model.controller.tick_mouse()
        px, py = self.world_model.strategy.reverse_map(float(mx), float(my))
        self.crosshair_pos[0] += px
        self.crosshair_pos[1] += py
        self.ring_buffer.add_event(mx, my, is_ai=True)

    def step(self):
        dt = 0.002

        # 同步推进所有虚拟时间
        self.sim_time += dt
        relative_time = self.sim_time
        self.sim_steps += 1

        self.tick_enemy(relative_time)
        self.tick_cv()
        self.world_model.step(self.ctx, self.ring_buffer)

        if self.ctx.p_predict is not None and self.ctx.v_real is not None:
            err_x, err_y = self.ctx.p_predict
            v_x, v_y = self.ctx.v_real
            bbox_w = 60.0

            intent_x, intent_y = self.world_model.strategy.calculate_mouse_move(err_x, err_y, bbox_w=bbox_w)

            scale = self.world_model.strategy.depth_scale(bbox_w)
            intent_vx = v_x * self.world_model.strategy.calib.k_x * scale
            intent_vy = v_y * self.world_model.strategy.calib.k_y * scale

            self.world_model.controller.compute(
                target_x=intent_x, target_y=intent_y, dt=dt,
                human_v=np.array([0.0, 0.0]),
                v_real=np.array([intent_vx, intent_vy])
            )

        self.tick_mouse()