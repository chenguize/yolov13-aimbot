# pro_controller.py
# ═══════════════════════════════════════════════════════════════════════════════
# Tier S+++ │ 解析 LQR + 制动轮廓控制器  v5.1b (Enhanced Biomimetic - Hotfix)
# ═══════════════════════════════════════════════════════════════════════════════
#
# ┌─ v4 → v5 升级与重构清单 ────────────────────────────────────────────────────┐
# │                                                                             │
# │  [BIOMIMETIC] 动作切片与影子目标 (Action Chunking & Ghost Target)            │
# │  ─────────────────────────────────────────────────────────────────────────  │
# │  原理：打破 LQR 的 500Hz 连续伺服监听，引入间歇性控制理论。                      │
# │        AI 每隔 120ms 观察一次真实卡尔曼状态，生成“影子目标”并进行路径规划。       │
# │        在节拍间隙内，闭眼开环追逐影子目标的物理惯性投影。                       │
# │  效果：赋予 AI 真实的人类反应盲区。面对敌人 AD 碎步急停时，会产生极其真实的人类  │
# │        过冲 (Overshoot) 与拉回，彻底摧毁基于频域追踪的反作弊检测。              │
# │                                                                             │
# │  [ARCH] 深度增益调度 (Depth Gain Scheduling) [已在 v4 继承]                  │
# │  ─────────────────────────────────────────────────────────────────────────  │
# │  原理：作为动态增益直接注入到 LQR 中。确保核心计算使用的是纯粹的物理 Counts。     │
# │                                                                             │
# │  [SAFEGUARDS] 生命周期拦截                                                   │
# │  ─────────────────────────────────────────────────────────────────────────  │
# │  重置：reset_target_state 与 notify_flick_end 已接管规划器，强行打断节拍。   │
# └─────────────────────────────────────────────────────────────────────────────┘

import threading
import numpy as np
from typing import Tuple, Optional
from config import config
from .base_controller import BaseController

try:
    from scipy.linalg import solve_discrete_are

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from numba import jit

    HAS_NUMBA = True
except ImportError:
    def jit(*args, **kwargs):
        return lambda fn: fn


    HAS_NUMBA = False


# ══════════════════════════════════════════════════════════════════════════════
# § 1 │ DARE 求解器
# ══════════════════════════════════════════════════════════════════════════════

def _solve_dare_1d(q_pos: float, q_vel: float, r_acc: float, dt: float):
    """1D 双积分器 DARE。返回 (k_pos, k_vel)。"""
    A = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
    B = np.array([[0.5 * dt * dt], [dt]], dtype=np.float64)
    Q = np.diag([q_pos, q_vel]).astype(np.float64)
    R = np.array([[r_acc]], dtype=np.float64)

    if HAS_SCIPY:
        try:
            P = solve_discrete_are(A, B, Q, R)
        except Exception:
            P = _dare_iterate(A, B, Q, R)
    else:
        P = _dare_iterate(A, B, Q, R)

    BtP = B.T @ P
    K = np.linalg.inv(R + BtP @ B) @ (BtP @ A)
    return float(K[0, 0]), float(K[0, 1])


def _dare_iterate(A, B, Q, R, n_iter: int = 1000):
    P = Q.copy()
    for _ in range(n_iter):
        BtP = B.T @ P
        K_tmp = np.linalg.inv(R + BtP @ B) @ (BtP @ A)
        P_new = Q + A.T @ P @ A - A.T @ P @ B @ K_tmp
        if np.max(np.abs(P_new - P)) < 1e-12:
            break
        P = P_new
    return P


# ══════════════════════════════════════════════════════════════════════════════
# § 2 │ Numba 核心内核 v4
# ══════════════════════════════════════════════════════════════════════════════

@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def _lqr_kernel_v4(
        e_pos_x: float, e_pos_y: float,
        crosshair_vx: float, crosshair_vy: float,
        target_vx: float, target_vy: float,
        target_ax: float, target_ay: float,
        k_pos_t: float, k_vel_t: float,
        k_pos_f: float, k_vel_f: float,
        ff_acc: float,
        blend_dist: float,
        deadzone_r: float,
        max_accel: float,
        max_speed: float,
        spf: float,
        integral_x: float,
        integral_y: float,
        ki: float,
) -> Tuple[float, float]:
    e_dist = (e_pos_x ** 2 + e_pos_y ** 2) ** 0.5 + 1e-9

    # 1. 双模增益插值
    z = e_dist / (blend_dist + 1e-9) - 1.0
    w_flick = 0.5 + 0.5 * z / (1.0 + abs(z))
    w_track = 1.0 - w_flick

    k_pos_base = k_pos_t * w_track + k_pos_f * w_flick
    k_vel_base = k_vel_t * w_track + k_vel_f * w_flick

    k_pos = k_pos_base * (0.78 + 0.22 * spf)
    k_vel = k_vel_base * (1.20 - 0.20 * spf)

    # 2. 软化制动轮廓
    v_safe = (2.0 * max_accel * (e_dist + 15.0)) ** 0.5 - (2.0 * max_accel * 15.0) ** 0.5
    v_chase = v_safe if v_safe < max_speed else max_speed

    inv_dist = 1.0 / e_dist
    v_ref_x = (e_pos_x * inv_dist) * v_chase + target_vx
    v_ref_y = (e_pos_y * inv_dist) * v_chase + target_vy

    e_vel_x = v_ref_x - crosshair_vx
    e_vel_y = v_ref_y - crosshair_vy

    # 3. 死区软化
    if e_dist < deadzone_r:
        eff_e_pos_x = 0.0
        eff_e_pos_y = 0.0
    else:
        shrink = (e_dist - deadzone_r) / e_dist
        eff_e_pos_x = e_pos_x * shrink
        eff_e_pos_y = e_pos_y * shrink

    # 4. LQI 核心方程
    u_x = k_pos * eff_e_pos_x + k_vel * e_vel_x + ff_acc * target_ax + ki * integral_x
    u_y = k_pos * eff_e_pos_y + k_vel * e_vel_y + ff_acc * target_ay + ki * integral_y

    # 5. 加速度限幅
    u_norm = (u_x ** 2 + u_y ** 2) ** 0.5
    if u_norm > max_accel:
        scale = max_accel / u_norm
        u_x *= scale
        u_y *= scale

    return u_x, u_y


# ══════════════════════════════════════════════════════════════════════════════
# § 3 │ PROController 主类
# ══════════════════════════════════════════════════════════════════════════════

class PROController(BaseController):
    def __init__(self):
        super().__init__()
        self.mode = "track"
        self._capture_fps = config.getfloat("General", "capture_fps_target", 240.0)
        self._frame_time = 1.0 / max(self._capture_fps, 30.0)  # 兜底限制，防止除零或过低帧率
        self.max_speed = config.getfloat("Controller", "max_speed", 8250.0)
        # 【优化点 1】降低极限加速度：拉长物理加减速的过渡带，避免机器式的瞬间爆发
        self.max_accel = config.getfloat("Controller", "max_accel", 75000.0)
        self.integral = np.zeros(2, dtype=np.float64)

        # 恢复适中刚性，让瞄准像磁铁一样吸附，但不僵硬
        q_pos_track = config.getfloat("Controller", "q_pos_track", 35.0)
        q_vel_track = config.getfloat("Controller", "q_vel_track", 0.50)
        r_track = config.getfloat("Controller", "r_track", 5.0e-4)

        # 稍微柔化 Flick 模式，防止大甩时用力过猛
        q_pos_flick = config.getfloat("Controller", "q_pos_flick", 2.0)
        q_vel_flick = config.getfloat("Controller", "q_vel_flick", 8e-5)
        r_flick = config.getfloat("Controller", "r_flick", 2e-6)

        _dt = 0.002
        self._kp_track, self._kv_track = _solve_dare_1d(q_pos_track, q_vel_track, r_track, _dt)
        self._kp_flick, self._kv_flick = _solve_dare_1d(q_pos_flick, q_vel_flick, r_flick, _dt)

        print(f"[LQR v5.2] TRACK: k_pos={self._kp_track:.1f}, k_vel={self._kv_track:.2f}  (Biomimetic Tuned)")
        print(f"[LQR v5.2] FLICK: k_pos={self._kp_flick:.1f}, k_vel={self._kv_flick:.2f}")

        self._ff_acc_gain = config.getfloat("Controller", "ff_acc_gain", 0.35)
        self._blend_dist = config.getfloat("Controller", "blend_dist", 50.0)
        self._deadzone_sc = config.getfloat("Controller", "deadzone_scale", 0.10)
        self._friction = config.getfloat("Controller", "velocity_friction", 0.993)
        self._ki = config.getfloat("Controller", "ki_gain", 0.025)

        # 【优化点 3】拉长冷启动斜坡时间：将启动延迟从 60ms 拉长至 140ms，解决 TTK < 0.12s 的瞬锁判定
        self._ramp_ms = config.getfloat("Controller", "coldstart_ramp_ms", 40.0)
        self._ramp_min = config.getfloat("Controller", "coldstart_ramp_min", 0.85)
        self._target_age_ms: float = self._ramp_ms

        # ── FIX-3 Flick 衔接阻尼参数 ─────────────────────────────────────────
        self._flick_end_damp = config.getfloat("Controller", "flick_end_damp", 0.30)

        # ── BIOMIMETIC 动作切片与影子目标机制 ────────────────────────────────
        self._plan_interval = config.getfloat("Controller", "plan_interval", 0.120)
        self._plan_timer = self._plan_interval

        self._ghost_e_pos = np.zeros(2, dtype=np.float64)
        self._ghost_vel = np.zeros(2, dtype=np.float64)
        self._ghost_acc = np.zeros(2, dtype=np.float64)

        # ── 控制状态 ─────────────────────────────────────────────────────────
        self.crosshair_velocity = np.zeros(2, dtype=np.float64)
        self.current_dt = 0.001
        self._subpixel = np.zeros(2, dtype=np.float64)
        self._lock = threading.Lock()

        self._spf = 1.0
        self._spf_prev = 1.0
        self.last_error_dist = 0.0

        # ── FIX-6 噪声相位（各频率独立积分） ─────────────────────────────────
        self._ph_3_2 = 0.0
        self._ph_11_7 = 1.3
        self._ph_2_8 = 0.8
        self._ph_10_4 = 2.1
        self._perlin_t = 0.0
        self._perlin_off_x = np.random.uniform(0.0, 100.0)
        self._perlin_off_y = np.random.uniform(200.0, 300.0)

        # 【新增】物理手腕速度容器与配置文件读取
        self._wrist_velocity = np.zeros(2, dtype=np.float64)
        # 从配置读取手腕灵敏度，默认给 0.45 保证爆发力
        self._wrist_alpha = config.getfloat("Controller", "wrist_alpha", 0.45)

    # ──────────────────────────────────────────────────────────────────────────
    def reset_target_state(self):
        """切换目标时调用：重置残差、离合、积分、冷启动，并强制触发重新规划。"""
        with self._lock:
            self.crosshair_velocity *= 0.08
            self._wrist_velocity[:] = 0.0  # <--- 新增这行，清空手腕动能
            self._spf = 1.0
            self._spf_prev = 1.0
            self.integral[:] = 0.0
            self._target_age_ms = 0.0
            # 【核心安全锁】换目标必须立刻睁眼观察世界，拒绝发呆
            self._plan_timer = self._plan_interval

    # ──────────────────────────────────────────────────────────────────────────
    def notify_flick_end(self):
        """人机 Flick 结束时调用：清除惯性并强制触发 AI 重新规划。"""
        with self._lock:
            self.crosshair_velocity *= self._flick_end_damp
            self._wrist_velocity[:] = 0.0  # <--- 新增这行，清空手腕动能
            self.integral[:] = 0.0
            self._target_age_ms = 0.0
            # 【核心安全锁】人手接管完毕，立刻睁眼定位残差
            self._plan_timer = self._plan_interval

    def compute(
            self,
            target_x: float,
            target_y: float,
            dt: float,
            human_v: Optional[np.ndarray] = None,
            v_real: Optional[np.ndarray] = None,
            a_real: Optional[np.ndarray] = None,
            power_factor: float = 1.0,
            bbox_w: float = 60.0,
    ) -> Tuple[float, float]:
        with self._lock:
            dt = max(dt, 0.0005)
            self.current_dt = dt

            # ── § 3.0 深度增益调度 (Depth Gain Scheduling) ──────────────────
            depth_gain = float(np.clip(80.0 / max(bbox_w, 10.0), 0.45, 2.5))

            # ── § 3.1 连续离合平滑 ────────────────────────────────────────────
            if self._spf_prev < 0.15 and power_factor > 0.65:
                self._spf = power_factor
            else:
                alpha = 0.65 if power_factor < self._spf else 0.50
                self._spf = (1.0 - alpha) * self._spf + alpha * power_factor

            self._spf_prev = power_factor
            spf = self._spf

            # ── § 3.2 BIOMIMETIC 动态动作切片与影子目标 (Dynamic Action Chunking) ──
            real_e_pos = np.array([target_x, target_y], dtype=np.float64)
            real_vel = v_real if v_real is not None else np.zeros(2, dtype=np.float64)
            real_acc = a_real if a_real is not None else np.zeros(2, dtype=np.float64)

            human_vel = human_v if human_v is not None else np.zeros(2, dtype=np.float64)
            human_speed = float(np.linalg.norm(human_vel))
            human_w = min(1.0, human_speed / 100.0)
            eff_human = human_vel * human_w

            total_crosshair_vel = self.crosshair_velocity + eff_human
            real_dist = float(np.linalg.norm(real_e_pos))
            self._head_radius = max(5.0, bbox_w * 0.28)
            # 【帧率自适应节拍器】：严格对齐捕获帧周期，杜绝相位撕裂
            # 远距离观察间隔基准：约 50ms (比如 240Hz 下取 12 帧，120Hz 下取 6 帧)
            frames_far = max(3, int(round(0.050 / self._frame_time)))
            interval_far = frames_far * self._frame_time

            # 近距离观察间隔基准：约 25ms (比如 240Hz 下取 6 帧，120Hz 下取 3 帧)
            frames_close = max(1, int(round(0.025 / self._frame_time)))
            interval_close = frames_close * self._frame_time

            if real_dist > 100.0:
                current_interval = interval_far
            elif real_dist < 30.0:
                current_interval = interval_close
            else:
                progress = (real_dist - 30.0) / 70.0
                current_interval = interval_close + progress * (interval_far - interval_close)

            self._plan_timer += dt

            # 【惊跳反射 (Startle Response) 动态收紧】向动态半径看齐
            startle_thresh = max(self._head_radius, real_dist * 0.3)
            if np.linalg.norm(real_e_pos - self._ghost_e_pos) > startle_thresh:
                self._plan_timer = current_interval

            # 【新增：触膛唤醒】向动态半径看齐
            if real_dist < self._head_radius and self.last_error_dist >= self._head_radius:
                self._plan_timer = current_interval

            # 【阶段 1：观察足够，重新规划路径】
            if self._plan_timer >= current_interval:
                # [路线 B 真实拟人化]：高斯分布的肌肉欠冲，向动态半径看齐
                # 只有距离大于头部半径，且小于一定范围时才制造欠冲
                if self._head_radius < real_dist < (self._head_radius + 65.0):
                    # 欠冲落点均值设为刚好停在头部边缘外围一点点
                    short_dist = max(1.5, min(18.0, np.random.normal(loc=self._head_radius * 0.6, scale=3.5)))
                    direction = real_e_pos / real_dist
                    self._ghost_e_pos = real_e_pos - direction * short_dist
                else:
                    self._ghost_e_pos = real_e_pos.copy()

                # 【物理合理性保留】：人类视觉确实无法感知毫秒级的高频加速度
                self._ghost_acc = np.zeros(2, dtype=np.float64)

                # 视觉延迟带来的速度平滑保留
                alpha_v = 0.6
                self._ghost_vel = alpha_v * real_vel + (1.0 - alpha_v) * self._ghost_vel

                self._plan_timer = 0.0
            # 【阶段 2：开环执行（模拟人类生理延迟盲区）】
            else:
                # 预测衰减：由于我们干掉了加速度，速度的衰减需要稍微平缓一点
                decay_factor = 0.992
                self._ghost_vel *= decay_factor

                # 仅靠速度积分进行盲区预测
                self._ghost_e_pos += (self._ghost_vel - total_crosshair_vel) * dt

            # 瞒天过海：给下方所有系统（含 LQR 内核）喂食“影子目标”
            e_pos = self._ghost_e_pos
            target_vel = self._ghost_vel
            target_acc = self._ghost_acc

            error_dist = float(np.linalg.norm(e_pos))
            self.last_error_dist = error_dist

            # ── § 3.3 动态死区 ──────────────────────────────────────────────
            target_speed = float(np.linalg.norm(target_vel))
            deadzone_r = max(2.0, bbox_w * self._deadzone_sc * (1.0 + 0.0005 * target_speed))

            # ── § 3.4 LQI 积分抗风卷 ────────────────────────────────────────
            if error_dist < 100.0:
                self.integral[0] += e_pos[0] * dt
                self.integral[1] += e_pos[1] * dt
                max_integral = 800.0 + 700.0 * (1.0 - error_dist / 100.0)
                self.integral = np.clip(self.integral, -max_integral, max_integral)
            else:
                self.integral *= 0.8

            # ── § 3.5 冷启动缓入与深度增益融合 ────────────────────────────────
            self._target_age_ms += dt * 1000.0
            age_ramp = float(np.clip(
                self._ramp_min + (1.0 - self._ramp_min) * (self._target_age_ms / self._ramp_ms),
                self._ramp_min,
                1.0
            ))

            effective_depth_gain = depth_gain * (0.75 + 0.25 / max(age_ramp, 0.75))

            k_pos_t_eff = self._kp_track * age_ramp * effective_depth_gain
            k_pos_f_eff = self._kp_flick * age_ramp * effective_depth_gain
            k_vel_t_eff = self._kv_track * effective_depth_gain
            k_vel_f_eff = self._kv_flick * effective_depth_gain
            ff_acc_eff = self._ff_acc_gain * effective_depth_gain
            ki_eff = self._ki * effective_depth_gain

            # ── § 3.6 LQI 内核计算 ──────────────────────────────────────────
            u_x, u_y = _lqr_kernel_v4(
                e_pos[0], e_pos[1],
                total_crosshair_vel[0], total_crosshair_vel[1],
                target_vel[0], target_vel[1],
                target_acc[0], target_acc[1],
                k_pos_t_eff, k_vel_t_eff,
                k_pos_f_eff, k_vel_f_eff,
                ff_acc_eff,
                self._blend_dist,
                deadzone_r,
                self.max_accel,
                self.max_speed,
                spf,
                self.integral[0],
                self.integral[1],
                ki_eff,
            )

            # ── § 3.7 人手夺权时泄压离合 ────────────────────────────────────
            if power_factor < spf and spf < 0.95:
                clutch_tau = 0.012 + 0.11 * (spf ** 2)
                self.crosshair_velocity *= np.exp(-dt / clutch_tau) * (0.92 + 0.08 * spf)

            # ── § 3.8 速度积分 ───────────────────────────────────────────────
            self.crosshair_velocity += np.array([u_x, u_y]) * dt

            # ── § 3.9 轨道阻尼（Anti-orbit） ─────────────────────────────────
            if error_dist < 30.0:
                dynamic_friction = self._friction * (0.985 + 0.01 * (error_dist / 30.0))
            else:
                dynamic_friction = self._friction
            self.crosshair_velocity *= dynamic_friction

            # ── § 3.10 速度限幅 ──────────────────────────────────────────────
            speed = float(np.linalg.norm(self.crosshair_velocity))
            if speed > self.max_speed:
                self.crosshair_velocity *= self.max_speed / speed

            # ── § 3.11 真实的生理反应死区 (Biological Reaction Deadzone) ──────────
            # 人类不是“缓慢起步”，而是“绝对硬直 110ms -> 瞬间满负荷爆发”
            if self._target_age_ms < 110.0:
                # 视觉信号还在视神经传递，大脑未下达指令，手部肌肉强制锁死 (保留 2% 允许极微小游离)
                self.crosshair_velocity *= 0.02

            # ── § 3.12 mode 标志 ─────────────────────────────────────────────
            thresh_high = config.getfloat("Controller", "mode_threshold_high", 50.0)
            thresh_low = config.getfloat("Controller", "mode_threshold_low", 35.0)
            threshold = thresh_high if self.mode == "track" else thresh_low
            self.mode = "flick" if error_dist > threshold else "track"

            return float(self.crosshair_velocity[0]), float(self.crosshair_velocity[1])

        # ──────────────────────────────────────────────────────────────────────────
    def tick_mouse(self) -> Tuple[int, int]:
        with self._lock:
            # ================= 核心修复 1：分离物理时钟 =================
            import time  # 局部导入，防止文件头部漏导
            now = time.perf_counter()
            if not hasattr(self, '_last_mouse_time'):
                self._last_mouse_time = now
                return 0, 0

            # 获取真实的鼠标循环 dt (1000Hz 线程下通常在 0.001 秒左右)
            dt = now - self._last_mouse_time
            self._last_mouse_time = now

            # 安全锁：如果系统卡顿，限制最大积分步长，防止准星瞬间跨越屏幕
            if dt > 0.01:
                dt = 0.001
            if dt <= 0:
                return 0, 0
            # =========================================================

            # ================= 核心修复 2：清理重复的手腕滤波器 =================
            # 统一使用 config 中读取的 _wrist_alpha (约 0.22)，这是抑制 12 万加速度高频颤抖的核心
            alpha_w = getattr(self, '_wrist_alpha', 0.22)
            self._wrist_velocity = (1.0 - alpha_w) * self._wrist_velocity + alpha_w * self.crosshair_velocity

            # ================= 核心修复 3：统一量纲 (物理 Counts 还原为 Pixels) =================
            kx = config.getfloat("AimStrategy", "k_factor_x", 1.0)
            # 将 LQR 里的 counts 距离还原为真实的屏幕像素，用于拟人化阈值判定
            err_px = self.last_error_dist / max(kx, 0.01)
            # 兼容读取动态头部大小，如果没有则回退到 15px
            head_r = getattr(self, '_head_radius_px', getattr(self, '_head_radius', 15.0))

            # ── 噪声相位积分 ──
            self._ph_3_2 = (self._ph_3_2 + 2.0 * np.pi * 3.2 * dt) % (2.0 * np.pi)
            self._ph_2_8 = (self._ph_2_8 + 2.0 * np.pi * 2.8 * dt) % (2.0 * np.pi)
            self._perlin_t += dt

            try:
                import noise as _noise
                # Perlin 噪声天生就是连续的低频波，非常适合模拟呼吸
                px = _noise.pnoise1(self._perlin_t * 2.8 + self._perlin_off_x, octaves=2) * 0.05
                py = _noise.pnoise1(self._perlin_t * 3.1 + self._perlin_off_y, octaves=2) * 0.05
            except Exception:
                px, py = 0.0, 0.0

            sx = 0.012 * np.sin(self._ph_3_2)
            sy = 0.012 * np.sin(self._ph_2_8)

            # [路线 B 真实拟人化]：根据真实的屏幕像素距离决定噪音振幅
            if err_px < head_r:
                target_noise_amp = 0.035
            elif err_px < head_r * 2.5:
                target_noise_amp = 0.12
            else:
                target_noise_amp = 0.40

            # 【终极防抖细节】：平滑过渡噪音振幅，消除 if/else 带来的阶跃高频毛刺
            if not hasattr(self, '_smoothed_noise_amp'):
                self._smoothed_noise_amp = target_noise_amp
            self._smoothed_noise_amp = 0.95 * self._smoothed_noise_amp + 0.05 * target_noise_amp

            nx = (px + sx) * self._smoothed_noise_amp
            ny = (py + sy) * self._smoothed_noise_amp

            # ── 最终位移积分 ──
            delta = self._wrist_velocity * dt + self._subpixel
            delta[0] += nx
            delta[1] += ny

            mx = int(np.floor(delta[0]))
            my = int(np.floor(delta[1]))

            # 亚像素级残差保留，防止低速时被 int() 吃掉精度导致无法移动
            self._subpixel[0] = delta[0] - mx
            self._subpixel[1] = delta[1] - my

            return mx, my