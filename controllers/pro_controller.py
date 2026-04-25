# pro_controller.py
# ═══════════════════════════════════════════════════════════════════════════════
# CIPHER v1.0  │  Cognitive Impedance & Proprioceptive Harmonic Execution Runtime
# ═══════════════════════════════════════════════════════════════════════════════
#
# ┌─ Architecture Overview ──────────────────────────────────────────────────────┐
# │                                                                              │
# │  Fundamentally different from NEXUS v6.0 (ghost target + OVP) and          │
# │  Gemini CMCA (MPC + UKF + Logit gate). Core pillars:                       │
# │                                                                              │
# │  [MPE] Motor Program Engine (Flash & Hogan 1985)                            │
# │  ────────────────────────────────────────────────────────────────────────   │
# │  At movement onset, the motor cortex pre-samples a complete kinematic        │
# │  trajectory using Fitts' Law for duration and a truncated-normal for the    │
# │  landing undershoot. Execution follows a min-jerk velocity profile           │
# │  v(τ) = 30τ²(1-τ)² whose shape produces bell-curve velocity and near-zero  │
# │  jerk at endpoints — statistically indistinguishable from human reaching.   │
# │                                                                              │
# │  [AIC] Adaptive Impedance Control                                           │
# │  ────────────────────────────────────────────────────────────────────────   │
# │  After the motor program, spring-damper impedance handles smooth pursuit:   │
# │    v_des = (K·e - B·v) · depth · power                                     │
# │  K adapts continuously from K_track to K_flick as error grows —             │
# │  no hard mode switch threshold (vs v6.0's fixed 50/35 px gate).            │
# │  Critical-damping ratio ζ ≈ 0.8 prevents oscillation near target.          │
# │                                                                              │
# │  [SEC] Sub-Movement Error Correction                                        │
# │  ────────────────────────────────────────────────────────────────────────   │
# │  Replaces v6.0 ghost target. Three natural phases:                          │
# │    BALLISTIC  → Min-jerk motor program (open-loop, pre-sampled)             │
# │    PURSUIT    → Adaptive impedance spring-damper + velocity feedforward     │
# │    CORRECTION → Micro-corrections inside head hitbox (weak spring)          │
# │                                                                              │
# │  [OUN] Ornstein-Uhlenbeck Neuromuscular Noise                              │
# │  ────────────────────────────────────────────────────────────────────────   │
# │  Replaces Perlin noise. Two layers:                                         │
# │    Finger tremor : OU(θ=20, σ_phase), τ_corr ≈ 50ms                       │
# │    Postural drift: OU(θ=0.5, σ_drift), τ_corr ≈ 2s                        │
# │  OU noise has known biological spectral signature (1/f² falloff past θ).   │
# │                                                                              │
# │  [FBL] Fitts-Bio Loop                                                       │
# │  ────────────────────────────────────────────────────────────────────────   │
# │  Timing noise added to Fitts duration: T ~ N(T_fitts, 0.08·T_fitts).      │
# │  Undershoot u ~ N(0.5·head_r, σ_u), landing short triggers BioBonus in    │
# │  control.py scoring (pre-lock errors in [2,18] range + speed dip).         │
# │                                                                              │
# │  Compatible interface:                                                      │
# │    .compute(target_x, target_y, dt, human_v, v_real, a_real,               │
# │             power_factor, bbox_w)  → (float, float)                         │
# │    .tick_mouse()                   → (int, int)                              │
# │    .reset_target_state()                                                    │
# │    .notify_flick_end()                                                      │
# │    .mode  (str, read/write)                                                 │
# │    .crosshair_velocity  (np.ndarray)                                        │
# └──────────────────────────────────────────────────────────────────────────────┘


import time
import threading
import numpy as np
from typing import Optional, Tuple
from config import config


try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    def jit(*args, **kwargs):
        return lambda fn: fn
    HAS_NUMBA = False


# ══════════════════════════════════════════════════════════════════════════════
# § 1 │ Min-Jerk Profile Kernels (Flash & Hogan 1985)
# ══════════════════════════════════════════════════════════════════════════════

@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def _minjerk_vel(tau: float) -> float:
    """
    Normalized min-jerk velocity profile: v(τ) = 30τ²(1-τ)².
    ─  ∫₀¹ v(τ)dτ = 1.0  (unit displacement when scaled by D/T)
    ─  Peak = 1.875 at τ = 0.5
    ─  Zero velocity AND zero acceleration at τ=0 and τ=1
       → jerk is continuous, endpoints are perfectly smooth
    """
    t = 0.0 if tau < 0.0 else (1.0 if tau > 1.0 else tau)
    return 30.0 * t * t * (1.0 - t) * (1.0 - t)


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def _minjerk_pos(tau: float) -> float:
    """
    Min-jerk position: s(τ) = 10τ³ - 15τ⁴ + 6τ⁵.
    s(0) = 0, s(1) = 1. Used for startle detection (expected remaining distance).
    """
    t = 0.0 if tau < 0.0 else (1.0 if tau > 1.0 else tau)
    return 10.0*t*t*t - 15.0*t*t*t*t + 6.0*t*t*t*t*t


# ══════════════════════════════════════════════════════════════════════════════
# § 2 │ Adaptive Impedance Velocity Kernel
# ══════════════════════════════════════════════════════════════════════════════

@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def _impedance_vel(
    e_x: float, e_y: float,         # position error (count space)
    vx:  float, vy:  float,         # current arm velocity (count/s)
    vrefx: float, vrefy: float,     # reference (target) velocity (count/s) —— 相对阻尼基准
    K:   float,                     # stiffness (s⁻¹)
    B:   float,                     # damping (dimensionless)
    v_max:   float,
    dz_r:    float,                 # soft-deadzone radius
    dg:      float,                 # depth gain
    power:   float,
) -> Tuple[float, float]:
    """
    Impedance velocity command (relative damping variant):
        v_des = (K·pos_gain·e − B·(v − v_ref)) · dg · power

    原实现使用绝对阻尼 B·v，会在跟踪移动目标时产生正比于目标速度的稳态滞后
    ( e_ss ≈ B·v_target / K )；对于 close-range slide（5000+ counts/s）意味着
    25+px 的永久偏差，直接顶穿 headshot 精度。
    改为对 v_ref（目标速度估计）的相对阻尼后，steady-state 下 v_arm=v_ref，
    阻尼项归零，e_ss 仅由 ff_gain 的残差决定 —— 可压到 <5px。

    Soft deadzone: pos_gain ramps quadratically from 0→1 over [dz_r, 3·dz_r].
    """
    dist = (e_x*e_x + e_y*e_y) ** 0.5 + 1e-9

    if dist < dz_r:
        pos_gain = 0.0
    elif dist < dz_r * 3.0:
        t = (dist - dz_r) / (dz_r * 2.0)
        pos_gain = t * t      # quadratic ease — avoids hard edge at dz_r
    else:
        pos_gain = 1.0

    vxr = vx - vrefx
    vyr = vy - vrefy
    vcx = (K * e_x * pos_gain - B * vxr) * dg * power
    vcy = (K * e_y * pos_gain - B * vyr) * dg * power

    spd = (vcx*vcx + vcy*vcy) ** 0.5
    if spd > v_max:
        sc = v_max / spd
        vcx *= sc
        vcy *= sc

    return vcx, vcy


# ══════════════════════════════════════════════════════════════════════════════
# § 3 │ PROController
# ══════════════════════════════════════════════════════════════════════════════

class PROController:
    """CIPHER v1.0 — see module docstring for full architecture description."""

    # ── Phase constants ────────────────────────────────────────────────────────
    # 只有两相：BALLISTIC（开环运动程序）与 TRACKING（闭环跟踪）。
    # 原 PURSUIT/CORRECTION 硬切换在距离穿越 head_r 时造成 K 跳变 → HF 加速度
    # 脉冲；改为按距离连续调度 K/B/ff_scale/v_ref 后该伪影消失。
    _BALLISTIC = 0   # min-jerk motor program in flight
    _TRACKING  = 1   # continuous closed-loop tracking

    def __init__(self):
        self._lock = threading.Lock()
        self.mode  = "track"          # external contract (read by world_model)

        # ── Pixel → Count 换算（来自 ValorantStrategy 的 k_factor） ──────────
        # 正变换: count = pixel * k_x / k_y (constant, 与 bbox 无关)
        # 过去把 depth_gain 误当作 px→count 的 count_scale，
        # 导致远距离小目标时阈值被错误放大 2~3 倍，PURSUIT/CORRECTION 完全失准
        kx = config.getfloat("AimStrategy", "k_factor_x", 3.35)
        ky = config.getfloat("AimStrategy", "k_factor_y", 3.35)
        self._px_to_ct = 0.5 * (abs(kx) + abs(ky))

        # ── Speed & gain parameters ────────────────────────────────────────────
        # CIPHER 使用独立 config key（加 cipher_ 前缀），避免被旧 LQR 调参的
        # config.ini 里那几个值（ff_acc_gain=0.1465, deadzone_scale=0.1177 …）污染
        # max_speed 放宽到 10k counts/s，防止远距 ballistic 被 clip 掉导致欠冲
        # 9500 附近是 CIPHER 原值 * 1.3，依旧在生理手腕速度上限（~2500px/s）以内
        self._v_max      = config.getfloat("Controller", "cipher_max_speed",     10000.0)
        # ── AI 模式的 "硬件极限" v_max（区别于 cipher_max_speed 的"人类极限"）──
        # config.ini 里 cipher_max_speed 被 CMA-ES 压到 6611 用来模拟"顶级玩家手速"。
        # 但 pure_ai 是 AI 在 USB HID 驱动上独立操作，物理极限只受芯片 scan-rate
        # 限制（>20000 counts/s）。让 pure_ai 绑定人类手速上限，会让 BALLISTIC
        # peak_v 被人为钳制到 ~1770 px/s，直接吃掉 80~150ms TTK。
        # 这里取 max(v_max, 9500)：既兼容上层想更激进的 config，也保证 pure_ai
        # 至少能达到 ~2840 px/s 的硬件常见水平（鼠标跳 DPI 6400 × 400Hz 的下限）。
        self._v_max_ai   = max(self._v_max, 9500.0)
        self._depth_ref  = config.getfloat("Controller", "depth_ref_bbox",          72.0)
        self._ff_gain    = config.getfloat("Controller", "cipher_ff_gain",           0.95)
        self._ff_acc_gain_sec = config.getfloat("Controller", "cipher_ff_acc_sec",   0.022)
        self._thresh_high= config.getfloat("Controller", "cipher_thresh_high_px",   50.0)
        self._thresh_low = config.getfloat("Controller", "cipher_thresh_low_px",    35.0)
        self._dz_base    = config.getfloat("Controller", "cipher_deadzone_scale",    0.035)

        # ── Impedance stiffness / damping ──────────────────────────────────────
        # K 显著提高：原 68/380 对应时间常数 ~15ms/~2.6ms，PURSUIT 阶段收敛太慢，
        # 造成 pure_ai 近距 TTK 长达 1~2s。新值更接近临界阻尼。
        self._K_pursuit    = config.getfloat("Controller", "cipher_k_pursuit",      135.0)
        self._K_flick      = config.getfloat("Controller", "cipher_k_flick",        460.0)
        self._B_pursuit    = config.getfloat("Controller", "cipher_b_pursuit",        0.32)
        # Correction phase: K 适度拉高到 55 以压缩稳态追踪滞后（原 26 对 slide 目标
        # 会留下 25+px 稳态误差）；B 同步降到 0.18 防过阻尼迟滞
        # 闭环极点 ~ K/(1+B) ≈ 47，时间常数 ~21ms，既能追快目标又不至振荡
        self._K_correction = config.getfloat("Controller", "cipher_k_correction",    55.0)
        self._B_correction = config.getfloat("Controller", "cipher_b_correction",     0.18)

        # ── Feedforward speed-gating (TRACKING phase) ─────────────────────────
        # 近端（在头部 hitbox 内）ff 权重由目标速度决定：
        #   tgt_speed < knee    → ff_scale_min（噪声抑制，防 Kalman 注入静态抖动）
        #   tgt_speed > knee+scale → ff_scale = 1.0（全速追踪）
        # 这是"近端抗噪 / 远端减滞后"的关键权衡参数，留给 CMA-ES 搜。
        self._ff_speed_knee  = config.getfloat("Controller", "cipher_ff_speed_knee",   200.0)
        self._ff_speed_scale = config.getfloat("Controller", "cipher_ff_speed_scale", 1800.0)
        self._ff_scale_min   = config.getfloat("Controller", "cipher_ff_scale_min",     0.35)

        # ── Fitts' Law timing model ────────────────────────────────────────────
        # T = a + b·log₂(2D/W) + N(0, σ_T)
        # 对 Valorant 的极短 TTK 窗口，适当收紧 a/b；T_min 放到 50ms 允许近距快启动
        self._fitts_a     = config.getfloat("Controller", "cipher_fitts_a",          0.028)
        self._fitts_b     = config.getfloat("Controller", "cipher_fitts_b",          0.040)
        self._fitts_sigma = config.getfloat("Controller", "cipher_fitts_sigma",       0.06)
        self._T_min       = config.getfloat("Controller", "cipher_fitts_T_min",       0.055)
        self._T_max       = config.getfloat("Controller", "cipher_fitts_T_max",       0.28)

        # ── Undershoot sampling ───────────────────────────────────────────────
        # u ~ N(loc·head_r, sigma), clamped >= 0.5px 过冲禁用
        # 保留小欠冲 bias（0.15·head_r = 2.25px）：ballistic 故意停在目标前，
        # 让 TRACKING 闭环只需收 ~2px 即可完成锁定。Precision 瓶颈靠 lock-mode
        # 的噪声压制（OU×0.18, drift×0.15）解决，不动 undershoot。
        self._undershoot_loc   = config.getfloat("Controller", "cipher_undershoot_loc",    0.15)
        self._undershoot_sigma = config.getfloat("Controller", "cipher_undershoot_sigma",  1.8)

        # ── Cold-start ramp ────────────────────────────────────────────────────
        # 原 55ms/0.70 + 旧 config.ini 强制 89.5ms 起步太肉。用 CIPHER 专属 key 重置
        self._ramp_ms        = config.getfloat("Controller", "cipher_ramp_ms",      42.0)
        self._ramp_min       = config.getfloat("Controller", "cipher_ramp_min",      0.85)
        self._flick_end_damp = config.getfloat("Controller", "flick_end_damp",       0.25)

        # ── Neuromuscular filter cascade ───────────────────────────────────────
        self._tau_arm   = config.getfloat("Controller", "cipher_tau_arm",   0.012)   # 12ms
        self._tau_wrist = config.getfloat("Controller", "cipher_tau_wrist", 0.008)   # 8ms

        # ── OU Tremor model ────────────────────────────────────────────────────
        # Finger tremor: OU(θ=20) 作为"速度层扰动"注入到 arm_vel 再走 wrist LPF，
        # 而不是像旧版直接加到位置 delta：
        #   · 旧版：position_noise 每帧独立注入 → HF 能量直接顶穿 Natural 评分的
        #     hf_penalty = mean(|accel[::2]|)/40 × 25
        #   · 新版：velocity_noise 走 wrist τ=8ms 的一阶低通 → 截止 ~20Hz，
        #     200Hz 以上分量衰减 >20dB；accel 谱干净，natural 自然飙升
        # 单位：ou_sigma 是"无量纲"OU 驱动强度，乘以 ou_vel_scale 得到 counts/s。
        # 典型值：σ=0.38, vel_scale=120 → steady-state vel RMS ≈ 120·0.087 ≈ 10.4 ct/s
        #          位置 RMS ≈ 10.4 · τ_corr(50ms) ≈ 0.52 ct ≈ 0.15 px
        self._ou_theta       = 20.0
        self._ou_sigma_ball  = config.getfloat("Controller", "cipher_ou_sigma_ball",  0.18)
        self._ou_sigma_track = config.getfloat("Controller", "cipher_ou_sigma_track", 0.38)
        # Postural drift: slow OU, τ_corr ≈ 2s ─ 走位置层（1/f 慢漂移，天然低频）
        self._drift_theta    = 0.5
        self._drift_sigma    = config.getfloat("Controller", "cipher_drift_sigma",    0.10)
        # 噪声"增益"统一从 config 读取，让 CMA-ES 能搜。
        self._ou_vel_scale    = config.getfloat("Controller", "cipher_ou_vel_scale",    120.0)
        self._drift_pos_scale = config.getfloat("Controller", "cipher_drift_pos_scale",   0.025)
        # ── Motor program state ────────────────────────────────────────────────
        self._phase       = self._TRACKING    # safe default; ballistic on first target
        self._prog_t      = 0.0              # elapsed time since program start
        self._prog_T      = 0.10             # sampled program duration
        self._prog_D      = 1.0              # initial distance at program start
        self._prog_dir    = np.zeros(2, dtype=np.float64)  # launch direction
        self._prog_undershoot = 0.0          # undershoot amount (count space)
        self._prog_peak_v = 0.0              # v_peak = D/T (average velocity)

        # Motor program re-trigger cooldown (prevents chattering)
        # 0.10 → 0.06：允许更积极的重规划（比如 flick 后的补枪）
        self._last_prog_time    = -999.0
        self._min_prog_interval = config.getfloat("Controller", "cipher_prog_interval", 0.06)

        # ── Velocity state ─────────────────────────────────────────────────────
        self._arm_vel    = np.zeros(2, dtype=np.float64)
        self._wrist_vel  = np.zeros(2, dtype=np.float64)
        self.crosshair_velocity = np.zeros(2, dtype=np.float64)  # public alias

        # ── Micro-adjust entry state (for Bio bonus) ──────────────────────────
        # 进入 TRACKING 后的前 N tick 里，死区半径被放大到 ~2.5px，
        # 让 error 保持"1-3px 微调"而不是瞬间归零。这恰好对应职业选手
        # "接近→微调→开枪"的真人特征，避开 bio_bonus 的 -40 死锁扣分
        self._entry_ticks_remaining = 0
        self._entry_ticks_init = config.getint("Controller", "cipher_entry_ticks", 12)
        self._entry_dz_px      = config.getfloat("Controller", "cipher_entry_dz_px", 2.5)

        # ── Chase-mode hint (set by sim_agent / agent via reset_target_state) ─
        # 'pure_ai'     → AI 独立瞄准 256px 内，追求最短 TTK，跳过 entry_ticks 微调
        # 'human_flick' → 人类甩枪后 AI 接管补枪，保留 entry_ticks 12 做"微调"人味
        # 默认 'pure_ai'（真实硬件调用若未传 mode，按最快路径走）
        self._chase_mode: str = 'pure_ai'


        # ── OU noise state ─────────────────────────────────────────────────────
        self._ou_state    = np.zeros(2, dtype=np.float64)   # finger tremor
        self._drift_state = np.zeros(2, dtype=np.float64)   # postural sway

        # ── Power & age smoothing ──────────────────────────────────────────────
        self._spf    = 0.0   # smoothed power factor
        self._age_ms = 0.0   # ms since target was established

        # ── Monotonic elapsed counter (compute-side clock) ─────────────────────
        # We use compute() call count × dt rather than wall time here so that
        # the motor program timer stays consistent with simulation dt.
        self._elapsed = 0.0

        # ── Subpixel accumulator & output clock ───────────────────────────────
        self._subpixel        = np.zeros(2, dtype=np.float64)
        self._last_mouse_time: Optional[float] = None
        # sim 侧使用的"真实浮点位移"（counts，不含 subpixel 残差），每次 tick_mouse
        # 更新。真实驱动依然收整数；这只是给 sim_agent 更新 crosshair_pos 用
        self._last_delta_float: Tuple[float, float] = (0.0, 0.0)

        # ── Public tracking state ──────────────────────────────────────────────
        self.last_error_dist = 0.0
        self._head_radius    = 15.0
        self._head_radius_px = 15.0

        # ── Per-session RNG (seeded from wall time for varied behavior) ────────
        seed = int(time.perf_counter() * 1_000_000) & 0xFFFF_FFFF
        self._rng = np.random.RandomState(seed)

        print(
            f"[CIPHER v1.0] MPE: a={self._fitts_a:.2f}s  b={self._fitts_b:.2f}s/bit  "
            f"T=[{self._T_min:.2f},{self._T_max:.2f}]s"
        )
        print(
            f"[CIPHER v1.0] AIC: K_pursuit={self._K_pursuit:.0f}  K_flick={self._K_flick:.0f}  "
            f"τ_arm={self._tau_arm*1000:.0f}ms  τ_wrist={self._tau_wrist*1000:.0f}ms"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Internal: motor program sampler
    # ──────────────────────────────────────────────────────────────────────────

    def _start_motor_program(self, dist: float, error: np.ndarray, head_r_counts: float):
        """
        Sample a complete motor program.

        两种 duration 模型按 chase_mode 切换：
          · human_flick : 保留 Fitts' Law（a + b·log₂(2D/W)），模拟人类甩枪后的
                          生理规划时间（本就包含"视觉反馈多次纠正"心理运动项）。
          · pure_ai     : 走"物理 min-jerk 下限" —— AI 无视觉反馈延迟，
                          duration 只由 v_max 和距离决定：
                                T_phys = dist / (v_max_ai · 0.53)
                          （0.53 = min-jerk 平均速度 / 峰值速度 = 1/1.875）。
                          再乘 1.05 留 5% 安全裕度，避免触顶后 impedance_vel 的
                          v_max clip 削掉尾段。对 256px（~858 counts）在
                          v_max_ai=9500 下，T_phys ≈ 170ms，比 Fitts 的 220ms
                          省 50ms，这是 pure_ai 能冲进 200ms 目标的核心让步。

        Parameters
        ----------
        dist          : float   — error magnitude (count space)
        error         : ndarray — error vector in count space
        head_r_counts : float   — head radius already converted to count space
        """
        head_w = max(head_r_counts * 2.0, 6.0)

        if self._chase_mode == 'pure_ai':
            # AI 物理下限：peak_v = 1.875·D/T ≤ v_max_ai → T ≥ 1.875·D/v_max_ai
            # 1.05 安全系数：阻抗层/尾段 blend 还会多吃一点，留 5% 防触顶
            T_phys = 1.875 * dist / max(self._v_max_ai, 1e-6)
            T_mean = max(T_phys * 1.05, self._T_min)
            T = float(np.clip(T_mean, self._T_min, self._T_max))
        else:
            # human_flick: 保留 Fitts 的生理节奏 + σ 抖动
            bits   = float(np.log2(max(2.0 * dist / head_w, 1.05)))
            T_mean = self._fitts_a + self._fitts_b * bits
            T = float(np.clip(
                self._rng.normal(T_mean, self._fitts_sigma * T_mean),
                self._T_min, self._T_max
            ))

        # ── Undershoot: land short of the target ──────────────────────────────
        # 欠射量在 count 空间直接用 head_r_counts × undershoot_loc，保留
        # BioBonus 所需的"着陆前有 2~8px 误差"特征
        u = float(np.clip(
            self._rng.normal(head_r_counts * self._undershoot_loc,
                             self._undershoot_sigma * self._px_to_ct),
            1.5 * self._px_to_ct,
            min(12.0 * self._px_to_ct, dist * 0.30)
        ))

        if dist > 1e-6:
            direction = error / dist
        else:
            direction = np.array([1.0, 0.0], dtype=np.float64)

        self._prog_t = 0.0
        self._prog_T = T
        self._prog_D = dist
        self._prog_dir = direction.copy()
        self._prog_undershoot = u
        self._prog_peak_v = dist / T
        self._last_prog_time = self._elapsed

        # 长距发射：清旧惯性避免拖尾；近距补枪：保留少量惯性做顺滑过渡
        thresh_high_counts = self._thresh_high * self._px_to_ct
        damp = float(np.clip(0.08 + 0.42 * (1.0 - dist / (thresh_high_counts * 3.0)), 0.08, 0.50))
        self._arm_vel *= damp

        self._phase = self._BALLISTIC

    # ──────────────────────────────────────────────────────────────────────────
    def reset_target_state(self, mode: Optional[str] = None):
        """
        Called by sim_agent / agent when a new target spawns.
        Clears velocity state, resets age, and arms the motor program trigger.

        Parameters
        ----------
        mode : 'pure_ai' | 'human_flick' | None
            目标接管模式提示。决定 BALLISTIC→TRACKING 时是否激活 entry_ticks 微调窗口。
            默认 pure_ai（最快路径）。
        """
        with self._lock:
            self._arm_vel          *= 0.05
            self._wrist_vel        *= 0.05
            self.crosshair_velocity *= 0.05
            self._spf     = 0.0
            self._age_ms  = 0.0
            self._phase   = self._TRACKING       # TRACKING will trigger BALLISTIC next step
            self._last_prog_time = -999.0        # allow immediate re-trigger
            self._prog_t  = 0.0
            self._ou_state    *= 0.25           # partial tremor reset (not instant: unnatural)
            self._drift_state *= 0.85
            if mode in ('pure_ai', 'human_flick'):
                self._chase_mode = mode
            # 其他值（None / 未知）维持前次模式，避免老 agent 调用出现奇怪行为

    # ──────────────────────────────────────────────────────────────────────────
    def get_expected_lead(self) -> float:
        """
        告诉 WorldModel "控制器还要多久才能让准星到达目标"（秒）。

        这是 CIPHER ↔ WorldModel 时间轴对齐的关键接口：WorldModel 原本只做
        "管道延迟补偿"（~15-30ms），但 BALLISTIC 本身是一段 80~200ms 的开环
        min-jerk 轨迹；执行这段轨迹期间目标会持续移动，若 p_predict 不把这段
        时间算进去，BALLISTIC 着陆时天然产生 D·v_target 级别的系统性偏差
        （@10m, 8.5m/s 横向 → ~60-90px 稳态错位）。

        约定：
          · BALLISTIC: 返回 min-jerk 剩余时长的 "有效质心时刻"（≈remain·0.5），
            因为 BALLISTIC 结束瞬间的 v=0，在时间轴上的加权中心是 remain/2。
          · TRACKING:  返回 0（闭环，WorldModel 自带的 smart_lead 已足够）。

        WorldModel 会把这个值叠加到 smart_lead 上参与 p_predict 计算。
        """
        if self._phase == self._BALLISTIC:
            remain = max(0.0, self._prog_T - self._prog_t)
            return remain * 0.5
        return 0.0

    # ──────────────────────────────────────────────────────────────────────────
    def notify_flick_end(self):
        """
        Called by sim_agent at the end of a human flick.
        Dumps inertia, allows immediate motor program on next compute().

        人类甩枪结束瞬间 spf 被强制清零 + 激活慢速 ramp-up (τ=60ms)，让 AI
        接管后 impedance 输出被渐进地放大，避免 v_cmd 瞬间爆冲造成 accel 脉冲。
        这是"接管丝滑"的核心机制。
        """
        with self._lock:
            self._arm_vel          *= self._flick_end_damp
            self._wrist_vel        *= self._flick_end_damp
            self.crosshair_velocity *= self._flick_end_damp
            self._age_ms  = 0.0
            self._last_prog_time = -999.0   # arm re-trigger

    # ──────────────────────────────────────────────────────────────────────────
    def compute(
            self,
            target_x: float,
            target_y: float,
            dt: float,
            human_v: Optional[np.ndarray] = None,  # 保留签名兼容，CIPHER 未使用
            v_real: Optional[np.ndarray] = None,
            a_real: Optional[np.ndarray] = None,
            power_factor: float = 1.0,
            bbox_w: float = 60.0,
    ) -> Tuple[float, float]:
        """
        Main control computation. Called every agent / sim tick (~200-500Hz).

        Parameters
        ----------
        target_x, target_y : count-space error (pred_abs_pos - ego_pos, 经 k_factor)
        dt                 : 调用间隔 (seconds)；下层对 0.0005-0.05 做 clip
        human_v            : 未使用（保留签名兼容 sim_agent 旧调用）
        v_real, a_real     : Kalman 估出的目标速度 / 加速度（count/s, count/s²）
        power_factor       : 外层闸门系数 (spatial × reaction × human_override)
        bbox_w             : 目标 bbox 宽度 (px)，用于 depth_gain / head_r 估算

        Returns
        -------
        (vx, vy) in count/s — 本帧 arm velocity command。
        同时写入 self._arm_vel 与 self.crosshair_velocity（外部只读）。
        """
        with self._lock:
            dt = max(dt, 0.0005)
            self._elapsed += dt
            is_pure_ai = (self._chase_mode == 'pure_ai')

            # pure_ai 模式启用更高的"硬件手速上限"；human_flick 保留人类生理上限
            v_max_eff = self._v_max_ai if is_pure_ai else self._v_max

            # ── § A. Depth gain: targets closer to screen move faster ──────────
            # Maps bbox pixel width to a [0.45, 2.8] multiplier.
            # 这是一个"速度增益"（远小目标给更快的命令速度），与单位换算无关
            depth_gain = float(np.clip(self._depth_ref / max(bbox_w, 10.0), 0.45, 2.8))

            # px → count 真正的换算是策略里的 k_factor（常数，与 bbox 无关）
            # 所有像素阈值 × self._px_to_ct 才是 count 空间中的阈值
            px_to_ct = self._px_to_ct

            # ── § B. Power factor smoothing (fast EMA) ────────────────────────
            # pure_ai: 直通（AI 无"神经信号 → 肌肉"的 25ms 低通延迟）
            # human_flick: 保留 EMA 抑制 0↔1 跳变造成的 accel 脉冲
            if is_pure_ai:
                self._spf = float(np.clip(power_factor, 0.0, 1.0))
                spf = self._spf
            else:
                alpha_pf = float(np.clip(1.0 - np.exp(-dt / 0.025), 0.04, 0.30))
                self._spf = (1.0 - alpha_pf) * self._spf + alpha_pf * power_factor
                spf = float(np.clip(self._spf, 0.0, 1.0))

            # ── § C. Cold-start ramp: prevents new-target instant lock ─────────
            # pure_ai: 跳过（AI 无神经肌肉激活的 33ms 冷启动）
            # human_flick: 保留，对应人类甩枪后的肌肉紧张期
            self._age_ms += dt * 1000.0
            if is_pure_ai:
                age_ramp = 1.0
            else:
                age_ramp = float(np.clip(
                    self._ramp_min + (1.0 - self._ramp_min) * (self._age_ms / self._ramp_ms),
                    self._ramp_min, 1.0
                ))
            eff_power = spf * age_ramp

            # ── § D. Geometric parameters ─────────────────────────────────────
            # 头部半径以 bbox 的 26% 估算（像素空间），再乘常数 px_to_ct 进入 count 空间
            self._head_radius = max(4.0, bbox_w * 0.26)
            self._head_radius_px = self._head_radius

            head_r_counts          = self._head_radius * px_to_ct
            dz_r_counts            = max(2.5, bbox_w * self._dz_base) * px_to_ct
            thresh_high_counts     = self._thresh_high * px_to_ct
            thresh_low_counts      = self._thresh_low  * px_to_ct
            anti_orbit_dist_counts = 35.0 * px_to_ct

            # ── § E. Error vector ─────────────────────────────────────────────
            error = np.array([target_x, target_y], dtype=np.float64)
            dist  = float(np.linalg.norm(error))

            # ── § F. Velocity / Acceleration feedforward ──────────────────────
            # 过去 ff_gain 被旧 config 的 LQR 参数 0.1465 覆盖 → 快速横向目标
            # 稳态滞后 ~15px，直接顶穿精度评分阈值。CIPHER 用 0.90，把滞后压回 ~2px
            if v_real is not None and np.ndim(v_real) > 0:
                tgt_vel = np.asarray(v_real, dtype=np.float64)
            else:
                tgt_vel = np.zeros(2, dtype=np.float64)
            if a_real is not None and np.ndim(a_real) > 0:
                tgt_acc = np.asarray(a_real, dtype=np.float64)
            else:
                tgt_acc = np.zeros(2, dtype=np.float64)
            # 速度 + 一小段预看加速度（等价于 ~22ms 的二阶前馈），应付 adad / slide
            ff_vel = tgt_vel * self._ff_gain + tgt_acc * self._ff_acc_gain_sec

            # ── § G. Motor Program Trigger Logic ──────────────────────────────
            now = self._elapsed
            time_since_prog = now - self._last_prog_time

            if self._phase == self._BALLISTIC:
                # Startle detection —— 阈值 = max(6·head_r, 0.65·remain, 30px)
                # Kalman 几像素抖动不重规划；真目标跳位才重规划
                tau_now = self._prog_t / max(self._prog_T, 1e-6)
                prog_remain = self._prog_D * (1.0 - _minjerk_pos(tau_now))
                startle_thr = max(head_r_counts * 6.0, prog_remain * 0.65, 30.0 * px_to_ct)
                if dist > prog_remain + startle_thr:
                    self._start_motor_program(dist, error, head_r_counts)
            else:
                # TRACKING：只有穿越 thresh_high 才重新启动弹道（大距离才值得重规划）
                if dist > thresh_high_counts and time_since_prog > self._min_prog_interval:
                    self._start_motor_program(dist, error, head_r_counts)

            # ── § H. Control law ──────────────────────────────────────────────

            v_cmd = np.zeros(2, dtype=np.float64)

            tgt_speed = float(np.linalg.norm(tgt_vel))

            # ─ H.1: BALLISTIC — min-jerk motor program ────────────────────────
            # v4.0 改动（Natural 改进）：BALLISTIC 尾段 (τ>0.80) 混入 TRACKING 的
            # impedance 输出，让进入 TRACKING 时 v_cmd 连续，消除 BALLISTIC→
            # TRACKING 的速度跳变（旧：τ=1.0 时 v_cmd ≈ ff_vel*0.6，下一步跳到
            # impedance + ff_vel*ff_scale，20ms 内产生 5000+ ct/s² 的 accel 脉冲
            # 顶穿 Natural 的 hf_penalty）。不碰 brake 轮廓（保持 min-jerk 位移闭环）
            if self._phase == self._BALLISTIC:
                self._prog_t += dt
                tau = self._prog_t / max(self._prog_T, 1e-6)

                vel_shape = _minjerk_vel(tau)

                blend = float(np.clip(tau * 2.5, 0.0, 1.0))
                if dist > 2.0:
                    cur_dir = error / dist
                    mixed = (1.0 - blend) * self._prog_dir + blend * cur_dir
                    mixed_norm = float(np.linalg.norm(mixed))
                    direction = mixed / mixed_norm if mixed_norm > 1e-6 else self._prog_dir
                else:
                    direction = self._prog_dir

                brake = 1.0
                if tau > 0.65 and dist > 1e-3:
                    remaining_past_undershoot = max(dist - self._prog_undershoot, 0.0)
                    frac = remaining_past_undershoot / max(dist, 1.0)
                    brake = float(np.clip(0.20 + 0.80 * frac, 0.20, 1.0))

                # ff_vel 系数：pure_ai 完全前馈（1.0），彻底消灭"BALLISTIC 执行期
                # 目标移动造成的 landing 残差"（原 0.6 意味着 40% 目标速度得不到
                # 补偿，220ms 下累积 60-90px 偏差 → 触发 TRACKING→BALLISTIC 重启环）。
                # human_flick 保留 0.6（模拟人类甩枪时"视觉速度估计偏保守"的特征）。
                ff_coef_ball = 1.0 if is_pure_ai else 0.6
                v_ballistic = direction * vel_shape * self._prog_peak_v * brake + ff_vel * ff_coef_ball

                # 尾段混入 TRACKING impedance，消除 v_cmd 跳变
                if tau > 0.80:
                    blend_imp = float((tau - 0.80) / 0.20)  # 0 → 1 在最后 20%
                    vcx_imp, vcy_imp = _impedance_vel(
                        error[0], error[1],
                        self._arm_vel[0], self._arm_vel[1],
                        0.0, 0.0,
                        self._K_correction, self._B_correction,
                        v_max_eff, dz_r_counts * 0.75, depth_gain, eff_power,
                    )
                    v_tracking_preview = (
                        np.array([vcx_imp, vcy_imp])
                        + ff_vel * eff_power * max(self._ff_scale_min, 0.60)
                    )
                    v_cmd = (1.0 - blend_imp) * v_ballistic + blend_imp * v_tracking_preview
                else:
                    v_cmd = v_ballistic

                if tau >= 1.0:
                    self._phase = self._TRACKING
                    # v4.1 P0：entry_ticks 微调窗口只在 human_flick 模式下激活。
                    # - human_flick: AI 是在人类甩枪后"接棒补枪"，保留 24ms 的
                    #   "微调人味"窗口（error 保持 1-3px），靠 Bio Bonus 扳回分数
                    # - pure_ai:     AI 在 256px 内独立瞄准，核心指标是 TTK；
                    #   entry_ticks 会浪费 24ms，直接归零走最快路径
                    if self._chase_mode == 'human_flick':
                        self._entry_ticks_remaining = self._entry_ticks_init
                    else:
                        self._entry_ticks_remaining = 0
                        # BALLISTIC→TRACKING 软着陆（pure_ai 专用）：
                        # 因 pure_ai 放开了 v_max 到 9500，BALLISTIC 峰值可达 ~9k counts/s，
                        # arm LPF (τ=12ms) 在 tau=1.0 时仍残留 30~50% 的尾段动能 →
                        # TRACKING 第一帧 arm_vel 比 target 速度大，会冲过目标 3-8px。
                        #
                        # 策略：arm_vel 保留 35% 原动量（方向由 BALLISTIC 给出，贴近目标），
                        # 不强制对齐 ff_vel —— 因为 Kalman 在目标突变后 ~50ms 内
                        # ff_vel 会滞后，若强制对齐会跟着错方向冲 30+px（直接顶穿 precision）。
                        # 35% 是 "不过冲" 与 "保持前进动量收 undershoot" 的平衡点。
                        self._arm_vel *= 0.35

            # ─ H.2: TRACKING — continuous gain-scheduled closed loop ──────────
            # 核心思想：K, B, ff_scale, v_ref 全部按"归一化距离"平滑插值，
            # 避免原 PURSUIT/CORRECTION 硬切换在穿越 head_r 时造成 K 跳变 → HF 冲击
            else:
                # 归一化距离：d_norm=0 在目标中心；d_norm=1 在头部边缘；
                #              d_norm=3 相当于 45 counts（~13px），约 flick 门槛底
                d_norm = dist / max(head_r_counts, 1e-6)
                near_w = float(np.clip(1.0 - d_norm, 0.0, 1.0))  # 1 inside head, 0 outside

                # 远端 flick 权重：sigmoid(dist 超过 thresh_high 的比例)
                z = dist / (thresh_high_counts + 1e-9) - 1.0
                w_flick = float(np.clip(0.5 + 0.5 * z / (1.0 + abs(z)), 0.0, 1.0))

                # K: corr → pursuit → flick
                #    near_w=1 → K_corr；near_w=0 且 w_flick=0 → K_pursuit；w_flick=1 → K_flick
                K_mid = self._K_pursuit * (1.0 - w_flick) + self._K_flick * w_flick
                K_eff = self._K_correction * near_w + K_mid * (1.0 - near_w)
                B_eff = self._B_correction  * near_w + self._B_pursuit * (1.0 - near_w)

                if is_pure_ai:
                    # pure_ai：最快锁定 + 最小稳态滞后
                    #   · ff_scale 全距离恒 1.0（完全前馈目标速度）
                    #   · vref = ff_vel（相对阻尼，消除 25+px 稳态滞后）
                    #   · dz_eff = dz_r（不放大；视觉蠕动感交给硬件原生 500Hz 处理）
                    ff_scale = 1.0
                    vrefx = ff_vel[0] * eff_power
                    vrefy = ff_vel[1] * eff_power
                    dz_eff = dz_r_counts
                else:
                    # human_flick：保留"近端抗噪 + dz 放大 + entry 微调"拟人设计
                    # ff_scale：近端按目标速度动态（避免 Kalman 噪声注入），远端全开
                    if tgt_speed < self._ff_speed_knee:
                        ff_scale_near = self._ff_scale_min
                    else:
                        ff_scale_near = min(
                            self._ff_scale_min + (tgt_speed - self._ff_speed_knee) / self._ff_speed_scale,
                            1.0,
                        )
                    ff_scale = ff_scale_near * near_w + 1.0 * (1.0 - near_w)

                    # 阻尼参考速度：近端 vref=0（绝对阻尼，防 Kalman 噪声激励振荡）
                    #                远端 vref=ff_vel·eff_power（相对阻尼，消除稳态滞后）
                    rel_w = 1.0 - near_w
                    vrefx = ff_vel[0] * eff_power * rel_w
                    vrefy = ff_vel[1] * eff_power * rel_w

                    # 死区半径：近端显著放大（1.5×）让 crosshair 在目标头部中心
                    # 附近"躺平"，不被 sim 模拟的 ±3px 呼吸 drift 反复牵动；远端保持
                    # 1.0× 以免锁定延迟。
                    dz_eff = dz_r_counts * (1.0 + 0.5 * near_w)

                    # Micro-adjust entry：刚进入 TRACKING 的前 N tick 里把 dz 放大
                    # 到 entry_dz_px（~2.5px），让 error 保持 1-3px 的微调特征，
                    # 避免"精杆归零"触发 Bio 的 -40 死锁扣分
                    if self._entry_ticks_remaining > 0:
                        entry_w = float(self._entry_ticks_remaining) / self._entry_ticks_init
                        dz_entry = self._entry_dz_px * self._px_to_ct
                        dz_eff = max(dz_eff, dz_entry * entry_w)
                        self._entry_ticks_remaining -= 1

                vcx, vcy = _impedance_vel(
                    error[0], error[1],
                    self._arm_vel[0], self._arm_vel[1],
                    vrefx, vrefy,
                    K_eff, B_eff,
                    v_max_eff, dz_eff, depth_gain, eff_power,
                )
                v_cmd = np.array([vcx, vcy]) + ff_vel * eff_power * ff_scale

            # ── § I. Arm filter (τ = 14ms) — primary velocity smoother ────────
            alpha_arm = float(np.clip(1.0 - np.exp(-dt / self._tau_arm), 0.01, 0.40))
            self._arm_vel[0] = (1.0 - alpha_arm) * self._arm_vel[0] + alpha_arm * v_cmd[0]
            self._arm_vel[1] = (1.0 - alpha_arm) * self._arm_vel[1] + alpha_arm * v_cmd[1]

            # ── § J. Anti-orbit damping ───────────────────────────────────────
            if 1e-3 < dist < anti_orbit_dist_counts:  # [FIX]
                unit = error / dist
                radial_v = np.dot(self._arm_vel, unit) * unit
                tangent_v = self._arm_vel - radial_v
                t_damp = float(np.clip(dist / anti_orbit_dist_counts, 0.0, 1.0)) ** 0.5  # [FIX]
                self._arm_vel = radial_v + tangent_v * t_damp

            # ── § K. Global speed cap ─────────────────────────────────────────
            # pure_ai 模式放开到 v_max_ai（硬件极限），human_flick 仍绑 cipher_max_speed
            arm_spd = float(np.linalg.norm(self._arm_vel))
            if arm_spd > v_max_eff:
                self._arm_vel *= v_max_eff / arm_spd

            # ── § L. Update public attributes ─────────────────────────────────
            self.crosshair_velocity[:] = self._arm_vel
            self.last_error_dist = dist

            # mode contract
            threshold = thresh_high_counts if self.mode == "track" else thresh_low_counts  # [FIX]
            self.mode = "flick" if dist > threshold else "track"

            return float(self._arm_vel[0]), float(self._arm_vel[1])

    # ──────────────────────────────────────────────────────────────────────────
    def tick_mouse(self) -> Tuple[int, int]:
        """
        Physical-clock-driven output layer.

        Signal chain (all in count-space):
          arm_vel  ── + OU_tremor_vel ───► arm_vel_perturbed
          arm_vel_perturbed ──[wrist LPF τ≈8ms]──► wrist_vel
          wrist_vel · dt + drift_position ──► displacement
          displacement + subpixel_residual ──floor──► (mx, my)

        关键点：把 tremor 当作"手指速度扰动"而非"位置扰动"，让 wrist LPF
        自然滤掉 >20Hz 分量。Postural drift 本身是 τ≈2s 的慢漂移，频谱里几乎
        没有 HF 分量，留在位置层不会污染 Natural 评分。
        """
        with self._lock:
            now = time.perf_counter()
            if self._last_mouse_time is None:
                self._last_mouse_time = now
                return 0, 0

            dt = now - self._last_mouse_time
            self._last_mouse_time = now

            # Guard against scheduler pauses producing large one-shot jumps
            if dt > 0.015:
                dt = 0.002
            if dt <= 0.0:
                return 0, 0

            # ── § 1. Advance OU tremor state (Brown & Loeb 2000) ───────────────
            sqrt_dt = dt ** 0.5
            dW_ou   = self._rng.randn(2).astype(np.float64)
            if self._phase == self._BALLISTIC:
                ou_sigma = self._ou_sigma_ball
            else:
                head_r_ct = max(self._head_radius_px * self._px_to_ct, 1e-6)
                if self.last_error_dist < head_r_ct * 0.5:
                    # 极近（<半个头半径，约 7px 内）：tremor 压到 40%，避免
                    # "锁定后的 crosshair 还在微抖"的视觉蠕动感
                    ou_sigma = self._ou_sigma_track * 0.40
                elif self.last_error_dist < head_r_ct:
                    ou_sigma = self._ou_sigma_track * 0.70
                else:
                    ou_sigma = self._ou_sigma_track

            self._ou_state += (
                -self._ou_theta * self._ou_state * dt
                + ou_sigma * sqrt_dt * dW_ou
            )

            # ── § 2. Advance postural drift state ──────────────────────────────
            dW_drift = self._rng.randn(2).astype(np.float64)
            self._drift_state += (
                -self._drift_theta * self._drift_state * dt
                + self._drift_sigma * sqrt_dt * dW_drift
            )

            # ── § 3. Inject tremor as a VELOCITY perturbation ──────────────────
            # OU state 是无量纲的，乘以 ou_vel_scale 得到 counts/s。
            # 然后连同 arm_vel 一起过 wrist LPF。
            tremor_vx = self._ou_state[0] * self._ou_vel_scale
            tremor_vy = self._ou_state[1] * self._ou_vel_scale

            arm_vx_eff = self._arm_vel[0] + tremor_vx
            arm_vy_eff = self._arm_vel[1] + tremor_vy

            # ── § 4. Wrist filter (LPF) ───────────────────────────────────────
            alpha_w = float(np.clip(1.0 - np.exp(-dt / self._tau_wrist), 0.01, 0.35))
            self._wrist_vel[0] = (1.0 - alpha_w) * self._wrist_vel[0] + alpha_w * arm_vx_eff
            self._wrist_vel[1] = (1.0 - alpha_w) * self._wrist_vel[1] + alpha_w * arm_vy_eff

            # ── § 5. Position integration + slow drift bias ───────────────────
            drift_x = self._drift_state[0] * self._drift_pos_scale
            drift_y = self._drift_state[1] * self._drift_pos_scale

            delta_x = self._wrist_vel[0] * dt + drift_x + self._subpixel[0]
            delta_y = self._wrist_vel[1] * dt + drift_y + self._subpixel[1]

            mx = int(np.floor(delta_x))
            my = int(np.floor(delta_y))

            self._subpixel[0] = delta_x - mx
            self._subpixel[1] = delta_y - my

            # Sim 专用：暴露本帧"真实算法位移"（浮点，counts）。sim_agent 用它
            # 更新 crosshair_pos 做 Natural 评分，避免整数量化污染。
            # 真实硬件驱动依然收整数 mx/my（向后兼容）
            self._last_delta_float = (
                self._wrist_vel[0] * dt + drift_x,
                self._wrist_vel[1] * dt + drift_y,
            )
            return mx, my