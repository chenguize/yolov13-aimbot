# pro_controller.py
import threading
import numpy as np
from typing import Tuple, Optional
from config import config
from .base_controller import BaseController

try:
    from numba import jit

    HAS_NUMBA = True
except ImportError:
    def jit(*args, **kwargs):
        return lambda fn: fn


    HAS_NUMBA = False


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def rollout_mpc_cost(error_pos, crosshair_vel, target_vel_abs, accel_seq, dt, w_pos, w_vel, w_acc):
    e_pos = error_pos.copy()
    v_cross = crosshair_vel.copy()
    total_cost = 0.0
    for i in range(accel_seq.shape[0]):
        accel = accel_seq[i]
        v_cross = v_cross + accel * dt
        error_vel = target_vel_abs - v_cross
        e_pos = e_pos + error_vel * dt
        cost_pos = w_pos * (e_pos[0] ** 2 + e_pos[1] ** 2)
        cost_vel = w_vel * (error_vel[0] ** 2 + error_vel[1] ** 2)
        cost_acc = w_acc * (accel[0] ** 2 + accel[1] ** 2)
        total_cost += cost_pos + cost_vel + cost_acc
    return total_cost


class PROController(BaseController):
    def __init__(self):
        super().__init__()
        self.max_speed = config.getfloat("Controller", "max_speed", 15000.0)
        self.max_accel = config.getfloat("Controller", "max_accel", 250000.0)

        self.horizon = config.getfloat("Controller", "horizon",18)
        self.samples = config.getfloat("Controller", "samples",80)

        self.w_pos = config.getfloat("Controller", "mpc_w_pos", 35.0)
        self.w_vel = config.getfloat("Controller", "mpc_w_vel", 0.001)
        self.w_acc = config.getfloat("Controller", "mpc_w_acc", 1e-5)

        self.crosshair_velocity = np.zeros(2, dtype=np.float64)
        self.current_dt = 0.001
        self._lock = threading.Lock()
        self._subpixel = np.zeros(2, dtype=np.float64)

        # [核心优化] 预分配 MPC 采样内存，解决 1000Hz 下 Python GC 导致的性能卡顿
        self._candidate_accels_buffer = np.zeros((self.samples, 2), dtype=np.float64)

    def compute(self, target_x: float, target_y: float, dt: float, human_v: Optional[np.ndarray] = None,
                v_real: Optional[np.ndarray] = None, power_factor: float = 1.0) -> Tuple[float, float]:
        with self._lock:
            self.current_dt = max(dt, 0.0005)
            error_pos = np.array([target_x, target_y], dtype=np.float64)
            error_dist = np.linalg.norm(error_pos)
            target_vel_absolute = v_real if v_real is not None else np.zeros(2)

            if error_dist < 1.0:
                self.crosshair_velocity *= 0.5
                return 0.0, 0.0

            best_cost = 1e18
            best_accel = np.zeros(2)

            # [核心优化] 复用预分配的数组，并清零
            candidate_accels = self._candidate_accels_buffer
            candidate_accels.fill(0)

            w_pos_dynamic = self.w_pos * power_factor

            # Kp: 随 power_factor 动态变化的弹簧拉力
            Kp = config.getfloat("Controller", "Kp",1200.0) * power_factor
            Kd = config.getfloat("Controller", "Kd",35.0)
            optimal_pd_accel = Kp * error_pos + Kd * (target_vel_absolute - self.crosshair_velocity)

            # 保底最优选项库
            candidate_accels[0] = optimal_pd_accel
            candidate_accels[1] = np.zeros(2)
            candidate_accels[2] = optimal_pd_accel * 1.2
            candidate_accels[3] = optimal_pd_accel * 0.8

            err_dir = error_pos / (error_dist + 1e-6)
            rel_vel = target_vel_absolute - self.crosshair_velocity
            rel_dir = rel_vel / (np.linalg.norm(rel_vel) + 1e-6)

            idx = 4
            for i in range(1, 10):
                scale = (i / 10.0) ** 2 * self.max_accel
                if idx < self.samples: candidate_accels[idx] = err_dir * scale; idx += 1
                if idx < self.samples: candidate_accels[idx] = rel_dir * scale; idx += 1

            while idx < self.samples:
                angle = np.random.uniform(0, 2 * np.pi)
                mag = (np.random.uniform(0, 1.0) ** 2) * self.max_accel
                candidate_accels[idx] = np.array([np.cos(angle), np.sin(angle)]) * mag
                idx += 1

            for accel in candidate_accels:
                norm = np.linalg.norm(accel)
                if norm > self.max_accel: accel *= self.max_accel / norm

                accel_seq = np.repeat(accel[None, :], self.horizon, axis=0)

                cost = rollout_mpc_cost(error_pos, self.crosshair_velocity, target_vel_absolute, accel_seq, dt,
                                        w_pos_dynamic, self.w_vel, self.w_acc)
                if cost < best_cost:
                    best_cost = cost
                    best_accel = accel

            self.crosshair_velocity += best_accel * dt
            speed = np.linalg.norm(self.crosshair_velocity)
            if speed > self.max_speed: self.crosshair_velocity *= self.max_speed / speed

            return self.crosshair_velocity[0], self.crosshair_velocity[1]

    def tick_mouse(self) -> Tuple[int, int]:
        with self._lock:
            delta = self.crosshair_velocity * self.current_dt + self._subpixel
            mx = int(np.floor(delta[0]))
            my = int(np.floor(delta[1]))
            self._subpixel[0] = delta[0] - mx
            self._subpixel[1] = delta[1] - my
            return mx, my