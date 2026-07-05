"""Professional intermittent cognitive-motor controller."""
import math
import threading
import time
from typing import Optional, Tuple

import numpy as np

from config import config
from utils.logger import get_logger

_log_cipher = get_logger("PROController")

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    def jit(*args, **kwargs):
        return lambda fn: fn
    HAS_NUMBA = False

@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def _minjerk_pos(tau):
    if tau < 0.0:
        t = 0.0
    elif tau > 1.0:
        t = 1.0
    else:
        t = tau
    return 10.0 * t * t * t - 15.0 * t * t * t * t + 6.0 * t * t * t * t * t

@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def _minjerk5_coeffs(d, T):
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    T5 = T4 * T
    T3 = max(T3, 1e-12)
    T4 = max(T4, 1e-12)
    T5 = max(T5, 1e-12)
    return np.array([0.0, 0.0, 0.0, 10.0 * d / T3, -15.0 * d / T4, 6.0 * d / T5
        ], dtype=np.float64)

@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def _impedance_vel(e_x, e_y, vx, vy, vrefx, vrefy, K, B, v_max, dz_r, dg, power):
    dist = (e_x * e_x + e_y * e_y) ** 0.5 + 1e-09
    if dist < dz_r:
        pos_gain = 0.0
    elif dist < dz_r * 3.0:
        t = (dist - dz_r) / (dz_r * 2.0)
        pos_gain = t * t
    else:
        pos_gain = 1.0
    vxr = vx - vrefx
    vyr = vy - vrefy
    vcx = (K * e_x * pos_gain - B * vxr) * dg * power
    vcy = (K * e_y * pos_gain - B * vyr) * dg * power
    spd = (vcx * vcx + vcy * vcy) ** 0.5
    if spd > v_max:
        sc = v_max / spd
        vcx *= sc
        vcy *= sc
    return vcx, vcy

try:
    from utils.rust_engine import CipherEngine
except ImportError:
    CipherEngine = None

_USE_RUST = config.getbool("Controller", "rust_enable", False) and CipherEngine is not None
if _USE_RUST:
    _minjerk_vel = CipherEngine.minjerk_vel
    _minjerk_pos = CipherEngine.minjerk_pos
    _impedance_vel = CipherEngine.impedance_vel

CIPHER_KERNEL_BACKEND = "rust" if _USE_RUST else "numba"


class PROController:
    _BALLISTIC = 0
    _TRACKING = 1

    def __init__(self):
        self._lock = threading.Lock()
        self.mode = 'track'
        if config.getbool('Strategy', 'bypass_strategy_mapping', False):
            self._px_to_ct = 1.0
            _log_cipher.warning(
                'bypass_strategy_mapping=True → _px_to_ct=1.0 (调试模式, 阈值按 1px≡1ct)')
        else:
            kx = config.getfloat('Strategy', 'k_factor_x', 3.35)
            ky = config.getfloat('Strategy', 'k_factor_y', 3.35)
            self._px_to_ct = 0.5 * (abs(kx) + abs(ky))
        self._v_max = config.getfloat('Controller', 'cipher_max_speed', 10000.0)
        self._v_max_ai = max(self._v_max, 9500.0)
        self._target_speed_limit = config.getfloat('Controller',
            'cipher_target_speed_limit', 5000.0)
        self._target_accel_limit = config.getfloat('Controller',
            'cipher_target_accel_limit', 250000.0)
        self._depth_ref = config.getfloat('Controller', 'depth_ref_bbox', 72.0)
        self._ff_gain = config.getfloat('Controller', 'cipher_ff_gain', 0.95)
        self._ff_acc_gain_sec = config.getfloat('Controller', 'cipher_ff_acc_sec',
            0.022)
        self._thresh_high = config.getfloat('Controller', 'cipher_thresh_high_px', 50.0
            )
        self._thresh_low = config.getfloat('Controller', 'cipher_thresh_low_px', 35.0)
        self._dz_base = config.getfloat('Controller', 'cipher_deadzone_scale', 0.035)
        self._K_pursuit = config.getfloat('Controller', 'cipher_k_pursuit', 135.0)
        self._K_flick = config.getfloat('Controller', 'cipher_k_flick', 460.0)
        self._B_pursuit = config.getfloat('Controller', 'cipher_b_pursuit', 0.70)
        self._K_correction = config.getfloat('Controller', 'cipher_k_correction', 55.0)
        self._B_correction = config.getfloat('Controller', 'cipher_b_correction', 0.70)
        self._ff_speed_knee = config.getfloat('Controller', 'cipher_ff_speed_knee',
            200.0)
        self._ff_speed_scale = config.getfloat('Controller',
            'cipher_ff_speed_scale', 1800.0)
        self._ff_scale_min = config.getfloat('Controller', 'cipher_ff_scale_min', 0.35)
        self._ff_predict_dt = config.getfloat('Controller', 'cipher_ff_predict_dt',
            0.03)
        self._ff_smooth_alpha = config.getfloat('Controller',
            'cipher_ff_smooth_alpha', 0.3)
        self._ff_scale_near_min = config.getfloat('Controller',
            'cipher_ff_scale_near_min', 0.45)
        self._ff_predict_dt_near = config.getfloat('Controller',
            'cipher_ff_predict_dt_near', 0.015)
        self._ff_predict_dt_far = config.getfloat('Controller',
            'cipher_ff_predict_dt_far', 0.035)
        self._ff_predict_adaptive = config.getbool('Controller',
            'cipher_ff_predict_adaptive', True)
        self._fitts_a = config.getfloat('Controller', 'cipher_fitts_a', 0.028)
        self._fitts_b = config.getfloat('Controller', 'cipher_fitts_b', 0.04)
        self._fitts_sigma = config.getfloat('Controller', 'cipher_fitts_sigma', 0.06)
        self._T_min = config.getfloat('Controller', 'cipher_fitts_T_min', 0.055)
        self._T_max = config.getfloat('Controller', 'cipher_fitts_T_max', 0.28)
        self._undershoot_loc = config.getfloat('Controller',
            'cipher_undershoot_loc', 0.15)
        self._undershoot_sigma = config.getfloat('Controller',
            'cipher_undershoot_sigma', 1.8)
        self._ramp_ms = config.getfloat('Controller', 'cipher_ramp_ms', 42.0)
        self._ramp_min = config.getfloat('Controller', 'cipher_ramp_min', 0.85)
        self._flick_end_damp = config.getfloat('Controller', 'flick_end_damp', 0.25)
        self._tau_arm = config.getfloat('Controller', 'cipher_tau_arm', 0.012)
        self._tau_wrist = config.getfloat('Controller', 'cipher_tau_wrist', 0.008)
        self._tau_arm_dyn_gain = config.getfloat('Controller',
            'cipher_tau_arm_dyn_gain', 0.6)
        self._motor_endpoint_ema_alpha = config.getfloat('Controller',
            'cipher_motor_endpoint_ema_alpha', 0.15)
        self._ou_theta = 20.0
        self._ou_sigma_ball = config.getfloat('Controller', 'cipher_ou_sigma_ball',
            0.18)
        self._ou_sigma_track = config.getfloat('Controller',
            'cipher_ou_sigma_track', 0.38)
        self._drift_theta = 0.5
        self._drift_sigma = config.getfloat('Controller', 'cipher_drift_sigma', 0.1)
        self._ou_vel_scale = config.getfloat('Controller', 'cipher_ou_vel_scale', 120.0
            )
        self._drift_pos_scale = config.getfloat('Controller',
            'cipher_drift_pos_scale', 0.025)
        _zeta_corr = config.getfloat('Controller', 'cipher_zeta_correction', 0.0)
        if _zeta_corr > 0:
            self._B_correction = max(0.05, 2.0 * _zeta_corr * np.sqrt(self.
                _K_correction * self._tau_arm) - 1.0)
            _log_cipher.debug('B_correction derived from ζ=%.3f: %.4f', _zeta_corr,
                self._B_correction)
            _zeta_purs = config.getfloat('Controller', 'cipher_zeta_pursuit', 0.0)
        else:
            _zeta_purs = config.getfloat('Controller', 'cipher_zeta_pursuit', 0.0)
        if _zeta_purs > 0:
            self._B_pursuit = max(0.05, 2.0 * _zeta_purs * np.sqrt(self._K_pursuit *
                self._tau_arm) - 1.0)
            _log_cipher.debug('B_pursuit derived from ζ=%.3f: %.4f', _zeta_purs,
                self._B_pursuit)
            _hyst_ratio = config.getfloat('Controller', 'cipher_thresh_hyst_ratio', 0.0
                )
        else:
            _hyst_ratio = config.getfloat('Controller', 'cipher_thresh_hyst_ratio', 0.0
                )
        if _hyst_ratio > 0:
            self._thresh_low = self._thresh_high * _hyst_ratio
            _log_cipher.debug('thresh_low derived from ratio=%.3f: %.1f',
                _hyst_ratio, self._thresh_low)
            _wrist_ratio = config.getfloat('Controller', 'cipher_tau_wrist_ratio', 0.0)
        else:
            _wrist_ratio = config.getfloat('Controller', 'cipher_tau_wrist_ratio', 0.0)
        if _wrist_ratio > 0:
            self._tau_wrist = self._tau_arm * _wrist_ratio
            _log_cipher.debug('tau_wrist derived from ratio=%.3f: %.4f',
                _wrist_ratio, self._tau_wrist)
            self._phase = self._TRACKING
        else:
            self._phase = self._TRACKING
        self._prog_t = 0.0
        self._prog_T = 0.1
        self._prog_D = 1.0
        self._prog_dir = np.zeros(2, dtype=np.float64)
        self._prog_undershoot = 0.0
        self._prog_peak_v = 0.0
        self._prog_cx = np.zeros(6, dtype=np.float64)
        self._prog_cy = np.zeros(6, dtype=np.float64)
        self._prog_init_dx = 0.0
        self._prog_init_dy = 0.0
        self._prev_control_error = np.zeros(2, dtype=np.float64)
        self._has_prev_control_error = False
        self._last_prog_time = -999.0
        self._min_prog_interval = config.getfloat('Controller',
            'cipher_prog_interval', 0.06)
        self._arm_vel = np.zeros(2, dtype=np.float64)
        self._wrist_vel = np.zeros(2, dtype=np.float64)
        self._servo_accel = np.zeros(2, dtype=np.float64)
        self.crosshair_velocity = np.zeros(2, dtype=np.float64)
        self._entry_ticks_remaining = 0
        self._entry_ticks_init = config.getint('Controller', 'cipher_entry_ticks', 12)
        self._entry_dz_px = config.getfloat('Controller', 'cipher_entry_dz_px', 2.5)
        self._chase_mode = 'pure_ai'
        self._ou_state = np.zeros(2, dtype=np.float64)
        self._drift_state = np.zeros(2, dtype=np.float64)
        self._ff_vel_smoothed = np.zeros(2, dtype=np.float64)
        self._spf = 0.0
        self._age_ms = 0.0
        self._elapsed = 0.0
        self._subpixel = np.zeros(2, dtype=np.float64)
        self._emit_mouse = False
        self._last_mouse_time = None
        self._last_delta_float = 0.0, 0.0
        self._emit_ix = 0
        self._emit_iy = 0
        self._emit_fx = 0.0
        self._emit_fy = 0.0
        self._move_diag_prev_int = 0, 0
        self._move_diag_prev_flt = 0.0, 0.0
        self._move_diag_cvx = 0.0
        self._move_diag_cvy = 0.0
        self._move_diag_dt = 0.001
        self.last_error_dist = 0.0
        self._head_radius = 15.0
        self._head_radius_px = 15.0
        self._last_bbox_w = 60.0
        # NumPy is explicitly seeded by the simulator, making controller noise
        # reproducible there. In production NumPy's process seed remains random.
        seed = int(np.random.randint(0, 2147483647))
        self._rng = np.random.RandomState(seed)
        self._ic_reaction_mean = config.getfloat('Controller',
            'cipher_reaction_mean_ms', 122.0) / 1000.0
        self._ic_reaction_sd = config.getfloat('Controller',
            'cipher_reaction_sd_ms', 10.0) / 1000.0
        self._ic_reaction_min = config.getfloat('Controller',
            'cipher_reaction_min_ms', 95.0) / 1000.0
        self._ic_reaction_max = config.getfloat('Controller',
            'cipher_reaction_max_ms', 150.0) / 1000.0
        self._ic_state = 'reaction'
        self._ic_reaction_elapsed = 0.0
        self._ic_reaction_delay = self._ic_reaction_mean
        self._pro_hold_error = np.zeros(2, dtype=np.float64)
        self._pro_hold_target_velocity = np.zeros(2, dtype=np.float64)
        self._pro_last_event = -1000000000.0
        self._pro_event_count = 0
        self._pro_refractory = config.getfloat('Controller',
            'cipher_pro_refractory_ms', 58.0) / 1000.0
        self._landing_decel = config.getfloat(
            'Controller', 'cipher_landing_decel', 9000.0
        )
        self._landing_radius_mul = config.getfloat(
            'Controller', 'cipher_landing_radius_mul', 3.0
        )
        self._capture_static_speed = config.getfloat(
            'Controller', 'cipher_capture_static_speed', 80.0
        )
        _log_cipher.info('CIPHER v1.0 MPE: a=%.2fs b=%.2fs/bit T=[%.2f,%.2f]s',
            self._fitts_a, self._fitts_b, self._T_min, self._T_max)
        _log_cipher.info(
            'CIPHER v1.0 AIC: K_pursuit=%.0f K_flick=%.0f τ_arm=%.0fms τ_wrist=%.0fms',
            self._K_pursuit, self._K_flick, self._tau_arm * 1000.0, self._tau_wrist *
            1000.0)
        self._alpha = 1.0
        self._alpha_target = 1.0
        self._alpha_tau = 0.03
        self._takeover_state = 'ACTIVE_LOCK'
        self._target_visible = False
        self._target_confidence = 0.0
        self._handoff_reason = 'startup'
        self._handoff_generation = 0
        self._handoff_elapsed = 0.0
        self._handoff_motor_elapsed = 0.0
        self._handoff_blend_duration = config.getfloat(
            'Controller', 'cipher_handoff_blend_ms', 20.0
        ) / 1000.0
        self._handoff_guard_duration = config.getfloat(
            'Controller', 'cipher_handoff_guard_ms', 100.0
        ) / 1000.0
        self._handoff_release_hold = config.getfloat(
            'Controller', 'cipher_handoff_release_hold_ms', 10.0
        ) / 1000.0
        self._handoff_max_accel = config.getfloat(
            'Controller', 'cipher_handoff_max_accel', 1500.0
        )
        self._handoff_residual_limit = config.getfloat(
            'Controller', 'cipher_handoff_residual_limit', 5000.0
        )
        self._handoff_target_speed_limit = config.getfloat(
            'Controller', 'cipher_handoff_target_speed_limit', 1800.0
        )
        self._handoff_open_loop_horizon = config.getfloat(
            'Controller', 'cipher_handoff_open_loop_horizon_ms', 500.0
        ) / 1000.0
        self._handoff_open_loop_max = config.getfloat(
            'Controller', 'cipher_handoff_open_loop_max', 750.0
        )
        self._handoff_open_loop_fraction = config.getfloat(
            'Controller', 'cipher_handoff_open_loop_fraction', 0.75
        )
        self._handoff_open_loop_far_max_boost = config.getfloat(
            'Controller', 'cipher_handoff_open_loop_far_max_boost', 50.0
        )
        self._handoff_open_loop_far_fraction_boost = config.getfloat(
            'Controller', 'cipher_handoff_open_loop_far_fraction_boost', 0.05
        )
        self._handoff_open_loop_far_ratio = config.getfloat(
            'Controller', 'cipher_handoff_open_loop_far_ratio', 0.60
        )
        self._handoff_ff_revision_gain = config.getfloat(
            'Controller', 'cipher_handoff_ff_revision_gain', 1.0
        )
        self._handoff_ff_acc_sec = config.getfloat(
            'Controller', 'cipher_handoff_ff_acc_sec', 0.020
        )
        self._handoff_target_coast_gain = config.getfloat(
            'Controller', 'cipher_handoff_target_coast_gain', 0.85
        )
        self._handoff_residual_velocity = np.zeros(2, dtype=np.float64)
        self._handoff_target_velocity = np.zeros(2, dtype=np.float64)
        self._handoff_seed_target_velocity = np.zeros(2, dtype=np.float64)
        self._handoff_seed_radial = np.zeros(2, dtype=np.float64)
        self._handoff_closing_speed = 0.0
        self._handoff_last_output = np.zeros(2, dtype=np.float64)
        self._handoff_last_final_velocity = np.zeros(2, dtype=np.float64)
        self._human_speed_prev = 0.0
        self._human_speed_ema = 0.0
        self._human_accel_ema = 0.0
        self._pre_warm_blend = 0.0
        self._pre_warm_target_v = np.zeros(2, dtype=np.float64)
        self._attention_state = 1.0
        self._attention_tau = 5.0
        self._noise_finger = np.zeros(2, dtype=np.float64)
        self._noise_hand = np.zeros(2, dtype=np.float64)
        self._noise_fatigue = np.zeros(2, dtype=np.float64)
        self._micro_pause_enable = config.getbool('Controller',
            'cipher_micro_pause_enable', True)
        _log_cipher.info(
            'CIPHER v2.0 重构: alpha 连续融合 + 1/f 噪声 + attention 调制 + 预测性 warm_start')
        return None

    @property
    def chase_mode(self):
        if self._alpha > 0.5:
            return 'pure_ai'
        return 'human_flick'

    @property
    def alpha(self):
        return self._alpha

    @property
    def takeover_state(self):
        return self._takeover_state

    def set_pixel_to_count_scale(self, kx, ky):
        """Synchronize pixel-domain thresholds with the active aim strategy."""
        with self._lock:
            if config.getbool('Strategy', 'bypass_strategy_mapping', False):
                self._px_to_ct = 1.0
            else:
                sx = abs(float(kx))
                sy = abs(float(ky))
                if sx > 1e-6 and sy > 1e-6:
                    self._px_to_ct = 0.5 * (sx + sy)
            _log_cipher.info(
                'Controller mapping synchronized: px_to_ct=%.4f', self._px_to_ct
            )

    def set_blend_alpha(self, alpha_target, takeover_state=None):
        with self._lock as __temp_157:
            self._alpha_target = float(np.clip(alpha_target, 0.0, 1.0))
            if takeover_state is not None:
                requested_state = str(takeover_state).upper()
                if requested_state == 'POST_TAKEOVER':
                    requested_state = 'REACTION'
                if requested_state == 'HUMAN_LEAD' and self._takeover_state != 'HUMAN_LEAD':
                    self._start_reaction_locked('human_override', preserve_elapsed=False)
                # The external authority scheduler may request ACTIVE_LOCK while
                # the motor system is still reacting or priming. Do not let that
                # bypass the physiological movement envelope.
                if (
                    requested_state == 'ACTIVE_LOCK'
                    and self._takeover_state in ('REACTION', 'PRIMING')
                ):
                    requested_state = self._takeover_state
                if requested_state != self._takeover_state:
                    _log_cipher.info('takeover_state: %s → %s (alpha_target=%.2f)',
                        self._takeover_state, requested_state, self._alpha_target)
                    self._takeover_state = requested_state
            if self._alpha_target > 0.95:
                self._pre_warm_blend = 0.0
            return None

    def _sample_reaction_delay_locked(self):
        sampled = self._rng.normal(self._ic_reaction_mean, self._ic_reaction_sd)
        self._ic_reaction_delay = float(np.clip(
            sampled, self._ic_reaction_min, self._ic_reaction_max
        ))

    def _start_reaction_locked(self, reason, preserve_elapsed=False):
        if not preserve_elapsed:
            self._ic_reaction_elapsed = 0.0
            self._sample_reaction_delay_locked()
        self._ic_state = 'reaction'
        self._phase = self._TRACKING
        self._prog_t = 0.0
        self._servo_accel.fill(0.0)
        self._handoff_elapsed = 0.0
        self._handoff_motor_elapsed = 0.0
        self._handoff_reason = str(reason)
        self._handoff_generation += 1
        self._takeover_state = 'REACTION'

    def observe_target(self, visible, target_velocity=None, confidence=0.0):
        """Publish target visibility without delaying perception or planning."""
        with self._lock:
            visible = bool(visible)
            if target_velocity is not None:
                velocity = np.nan_to_num(
                    np.asarray(target_velocity, dtype=np.float64),
                    nan=0.0, posinf=0.0, neginf=0.0,
                )
                self._handoff_target_velocity[:] = self._clip_vector_norm(
                    velocity, self._handoff_target_speed_limit
                )
                self._pre_warm_target_v[:] = self._handoff_target_velocity
            self._target_confidence = float(np.clip(confidence, 0.0, 1.0))

            if not visible:
                self._target_visible = False
                self._takeover_state = 'NO_TARGET'
                self._ic_state = 'reaction'
                self._ic_reaction_elapsed = 0.0
                self._handoff_elapsed = 0.0
                self._handoff_motor_elapsed = 0.0
                return

            if not self._target_visible:
                self._target_visible = True
                self._start_reaction_locked('target_visible', preserve_elapsed=False)

    def begin_handoff(self, human_velocity=None, target_velocity=None,
                      error=None, reason='human_release'):
        """Commit a human-to-AI handoff while preserving velocity continuity.

        The authority transition itself may remove human camera motion that is
        no longer useful. We retain only the human component that closes the
        current error and use target velocity for the remaining axes. The
        resulting primitive then coasts open-loop through cognitive reaction.
        """
        with self._lock:
            if (
                reason == 'human_release'
                and self._handoff_reason == 'human_release'
                and self._takeover_state != 'NO_TARGET'
            ):
                return
            human = np.zeros(2, dtype=np.float64) if human_velocity is None else np.nan_to_num(
                np.asarray(human_velocity, dtype=np.float64),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            target = self._handoff_target_velocity.copy()
            if target_velocity is not None:
                target = np.nan_to_num(
                    np.asarray(target_velocity, dtype=np.float64),
                    nan=0.0, posinf=0.0, neginf=0.0,
                )
                target = self._clip_vector_norm(
                    target, self._handoff_target_speed_limit
                )
                self._handoff_target_velocity[:] = target
                self._pre_warm_target_v[:] = target

            if error is not None:
                error_vec = np.nan_to_num(
                    np.asarray(error, dtype=np.float64),
                    nan=0.0, posinf=0.0, neginf=0.0,
                )
            else:
                error_vec = self._prev_control_error.copy()
            error_norm = float(np.linalg.norm(error_vec))
            if error_norm > 1e-6:
                radial = error_vec / error_norm
                human_radial = float(np.dot(human, radial))
                target_radial = float(np.dot(target, radial))
                inherited_closing = human_radial - target_radial
                far_weight = float(np.clip(
                    (error_norm / max(self._px_to_ct, 1e-6) - 60.0) / 60.0,
                    0.0, 1.0,
                ))
                open_loop_max = (
                    self._handoff_open_loop_max
                    + far_weight * self._handoff_open_loop_far_max_boost
                )
                open_loop_fraction = float(np.clip(
                    self._handoff_open_loop_fraction
                    + far_weight * self._handoff_open_loop_far_fraction_boost,
                    0.0, 0.9,
                ))
                scheduled_horizon = self._handoff_open_loop_horizon * (
                    1.0
                    - far_weight
                    * (1.0 - self._handoff_open_loop_far_ratio)
                )
                planned_closing = min(
                    error_norm / max(scheduled_horizon, 1e-6),
                    open_loop_max,
                )
                reaction_left = max(
                    self._ic_reaction_delay - self._ic_reaction_elapsed,
                    0.050,
                )
                budget_speed = (
                    error_norm
                    * open_loop_fraction
                    / reaction_left
                )
                closing_speed = min(
                    max(0.0, inherited_closing, planned_closing),
                    open_loop_max,
                    budget_speed,
                )
                residual = (
                    target * self._handoff_target_coast_gain
                    + radial * closing_speed
                )
                self._handoff_seed_radial[:] = radial
                self._handoff_closing_speed = closing_speed
            else:
                residual = target.copy()
                self._handoff_seed_radial.fill(0.0)
                self._handoff_closing_speed = 0.0
            residual = self._clip_vector_norm(residual, self._handoff_residual_limit)
            self._handoff_residual_velocity[:] = residual
            self._handoff_seed_target_velocity[:] = target
            self._handoff_last_output[:] = residual
            self._handoff_last_final_velocity[:] = residual
            self._arm_vel[:] = residual
            self._wrist_vel[:] = residual
            self.crosshair_velocity[:] = residual
            self._target_visible = True

            preserve = self._ic_state == 'reaction'
            self._start_reaction_locked(reason, preserve_elapsed=preserve)

    def _apply_handoff_envelope(self, candidate, dt):
        handoff_program = self._handoff_reason in (
            'human_release', 'notify_flick_end'
        )
        if not handoff_program:
            return candidate
        envelope_end = (
            self._handoff_guard_duration
            + self._handoff_release_hold
        )
        envelope_active = self._handoff_motor_elapsed < envelope_end
        if self._takeover_state != 'PRIMING' and not envelope_active:
            return candidate

        self._handoff_motor_elapsed += dt
        shaped = candidate
        if self._takeover_state == 'PRIMING':
            blend = float(np.clip(
                self._handoff_motor_elapsed / max(self._handoff_blend_duration, 1e-6),
                0.0, 1.0,
            ))
            shaped = (
                (1.0 - blend) * self._handoff_residual_velocity
                + blend * candidate
            )
        if self._handoff_motor_elapsed <= self._handoff_guard_duration:
            delta = shaped - self._handoff_last_output
            delta = self._clip_vector_norm(
                delta, self._handoff_max_accel * max(dt, 1e-6)
            )
            shaped = self._handoff_last_output + delta
        self._handoff_last_output[:] = shaped
        return shaped

    def _enforce_final_handoff_slew(self, candidate, dt):
        handoff_program = self._handoff_reason in (
            'human_release', 'notify_flick_end'
        )
        release_guarded = (
            self._ic_state == 'reaction'
            and self._handoff_elapsed <= self._handoff_guard_duration + 2.0 * dt
        )
        motor_guarded = self._handoff_motor_elapsed < self._handoff_guard_duration
        if self._ic_state == 'reaction':
            envelope_elapsed = self._handoff_elapsed
        else:
            envelope_elapsed = self._handoff_motor_elapsed
        extended_guard = (
            handoff_program
            and envelope_elapsed
            < self._handoff_guard_duration + self._handoff_release_hold
        )
        if not handoff_program or not (
            release_guarded or motor_guarded or extended_guard
        ):
            self._handoff_last_final_velocity[:] = candidate
            return candidate
        delta = self._clip_vector_norm(
            candidate - self._handoff_last_final_velocity,
            self._handoff_max_accel * max(dt, 1e-6),
        )
        final_velocity = self._handoff_last_final_velocity + delta
        self._handoff_last_final_velocity[:] = final_velocity
        return final_velocity

    def _advance_alpha(self, dt):
        a = float(np.clip(1.0 - np.exp(-dt / self._alpha_tau), 0.01, 0.4))
        self._alpha = (1.0 - a) * self._alpha + a * self._alpha_target
        return None

    def _advance_attention(self, dt):
        theta_att = 1.0 / self._attention_tau
        sigma_att = 0.15
        dW = self._rng.randn(1)[0]
        self._attention_state += -theta_att * (self._attention_state - 1.0
            ) * dt + sigma_att * np.sqrt(dt) * dW
        self._attention_state = self._attention_state
        self._attention_state = float(np.clip(self._attention_state, 0.7, 1.3))
        return None

    def _advance_1f_noise(self, dt):
        theta_hand = 0.5
        sigma_hand = 0.08
        dW_hand = self._rng.randn(2).astype(np.float64)
        self._noise_hand += -theta_hand * self._noise_hand * dt + sigma_hand * np.sqrt(
            dt) * dW_hand
        self._noise_hand = self._noise_hand
        theta_fatigue = 0.05
        sigma_fatigue = 0.04
        dW_fatigue = self._rng.randn(2).astype(np.float64)
        self._noise_fatigue += (-theta_fatigue * self._noise_fatigue * dt +
            sigma_fatigue * np.sqrt(dt) * dW_fatigue)
        self._noise_fatigue = self._noise_fatigue
        return None

    def pre_warm(self, target_velocity, blend_factor=0.3):
        with self._lock as __temp_175:
            self._pre_warm_target_v = np.asarray(target_velocity, dtype=np.float64
                ).copy()
            self._pre_warm_blend = float(np.clip(blend_factor, 0.0, 1.0))
            if self._pre_warm_blend > 0.05:
                _log_cipher.debug(
                    'pre_warm activated: blend=%.2f, target_v=[%.0f, %.0f]', self.
                    _pre_warm_blend, self._pre_warm_target_v[0], self.
                    _pre_warm_target_v[1])
                return None
            else:
                return None

    def warm_start_from_velocity_directional(self, human_vx, human_vy, target_vx, target_vy, error_x, error_y):
        with self._lock as __temp_183:
            human_v = np.array([human_vx, human_vy], dtype=np.float64)
            target_v = np.array([target_vx, target_vy], dtype=np.float64)
            err = np.array([error_x, error_y], dtype=np.float64)
            human_mag = float(np.linalg.norm(human_v))
            err_mag = float(np.linalg.norm(err))
            if not human_mag < 100.0:
                if err_mag < 1.0:
                    __temp_194, __temp_195 = 0.7, 0.3
                    w_human = __temp_194
                    w_target = __temp_195
                else:
                    human_dir = human_v / human_mag
                    err_dir = err / err_mag
                    alignment = float(np.dot(human_dir, err_dir))
                    if alignment > 0.7:
                        __temp_198, __temp_199 = 0.9, 0.1
                        w_human = __temp_198
                        w_target = __temp_199
                    elif alignment > 0.0:
                        __temp_200, __temp_201 = 0.6, 0.4
                        w_human = __temp_200
                        w_target = __temp_201
                    else:
                        __temp_202, __temp_203 = 0.2, 0.8
                        w_human = __temp_202
                        w_target = __temp_203
            else:
                __temp_204, __temp_205 = 0.7, 0.3
                w_human = __temp_204
                w_target = __temp_205
            seed = human_v * w_human + target_v * w_target
            speed = float(np.linalg.norm(seed))
            if speed < 5000.0:
                self._arm_vel[None:None] = seed * 0.7
                self._wrist_vel[None:None] = self._arm_vel
                self.crosshair_velocity[None:None] = self._arm_vel
                self._spf = 0.3
                self._age_ms = 20.0
                self._phase = self._TRACKING
                self._prog_t = 0.0
            if human_mag > 1.0:
                if err_mag > 1.0:
                    _log_cipher.debug(
                        'warm_start_directional: align=%.2f, w_h=%.1f, w_t=%.1f, seed_speed=%.0f'
                        , float(np.dot(human_v / max(human_mag, 1e-06), err / max(
                        err_mag, 1e-06))), w_human, w_target, speed)
                    return None
                else:
                    _log_cipher.debug(
                        'warm_start_directional: align=%.2f, w_h=%.1f, w_t=%.1f, seed_speed=%.0f'
                        , 0.0, w_human, w_target, speed)
                    return None
            else:
                _log_cipher.debug(
                    'warm_start_directional: align=%.2f, w_h=%.1f, w_t=%.1f, seed_speed=%.0f'
                    , 0.0, w_human, w_target, speed)
                return None

    def _start_motor_program(self, dist, error, head_r_counts):
        head_w = max(head_r_counts * 2.0, 6.0)
        T_phys = 1.875 * dist / max(self._v_max_ai, 1e-06)
        T_phys_mean = max(T_phys * 1.05, self._T_min)
        T_phys_clipped = float(np.clip(T_phys_mean, self._T_min, self._T_max))
        bits = float(np.log2(max(2.0 * dist / head_w, 1.05)))
        T_fitts_mean = self._fitts_a + self._fitts_b * bits
        T_fitts = float(np.clip(self._rng.normal(T_fitts_mean, self._fitts_sigma *
            T_fitts_mean), self._T_min, self._T_max))
        T = self._alpha * T_phys_clipped + (1.0 - self._alpha) * T_fitts
        T = float(np.clip(T, self._T_min, self._T_max))
        u = float(np.clip(self._rng.normal(head_r_counts * self._undershoot_loc,
            self._undershoot_sigma * self._px_to_ct), 1.5 * self._px_to_ct, min(
            12.0 * self._px_to_ct, dist * 0.3)))
        if dist > 1e-06:
            direction = error / dist
        else:
            direction = np.array([1.0, 0.0], dtype=np.float64)
        self._prog_t = 0.0
        self._prog_T = T
        self._prog_D = dist
        self._prog_dir = direction.copy()
        self._prog_undershoot = u
        self._prog_peak_v = dist / T
        self._prog_init_dx = float(error[0])
        self._prog_init_dy = float(error[1])
        self._prog_cx = _minjerk5_coeffs(self._prog_init_dx, T)
        self._prog_cy = _minjerk5_coeffs(self._prog_init_dy, T)
        self._last_prog_time = self._elapsed
        thresh_high_counts = self._thresh_high * self._px_to_ct
        damp = float(np.clip(0.08 + 0.42 * (1.0 - dist / (thresh_high_counts * 3.0)
            ), 0.08, 0.5))
        self._arm_vel *= damp
        self._arm_vel = self._arm_vel
        self._phase = self._BALLISTIC
        _log_cipher.phase_switch('TRACKING', 'BALLISTIC', dist=round(dist), T_ms=
            round(T * 1000, 1), peak_v=round(self._prog_peak_v))
        return None

    def reset_target_state(self, mode=None, hard=True):
        with self._lock as __temp_248:
            if not hard:
                if mode == 'pure_ai':
                    self._alpha_target = 1.0
                elif mode == 'human_flick':
                    self._alpha_target = 0.0
                return None
            self._arm_vel *= 0.05
            self._arm_vel = self._arm_vel
            self._wrist_vel *= 0.05
            self._wrist_vel = self._wrist_vel
            self._servo_accel.fill(0.0)
            self.crosshair_velocity *= 0.05
            self.crosshair_velocity = self.crosshair_velocity
            self._spf = 0.0
            self._age_ms = 0.0
            self._phase = self._TRACKING
            self._last_prog_time = -999.0
            self._prog_t = 0.0
            self._has_prev_control_error = False
            self._ic_state = 'reaction'
            self._ic_reaction_elapsed = 0.0
            if hasattr(self, '_rng'):
                sampled = self._rng.normal(self._ic_reaction_mean, self._ic_reaction_sd
                    )
                self._ic_reaction_delay = float(np.clip(sampled, self.
                    _ic_reaction_min, self._ic_reaction_max))
            self._pro_hold_error.fill(0.0)
            self._pro_hold_target_velocity.fill(0.0)
            self._pro_last_event = -1000000000.0
            self._pro_event_count = 0
            self._target_visible = False
            self._target_confidence = 0.0
            self._handoff_reason = 'target_reset'
            self._handoff_elapsed = 0.0
            self._handoff_motor_elapsed = 0.0
            self._handoff_residual_velocity.fill(0.0)
            self._handoff_target_velocity.fill(0.0)
            self._handoff_seed_target_velocity.fill(0.0)
            self._handoff_seed_radial.fill(0.0)
            self._handoff_closing_speed = 0.0
            self._handoff_last_output.fill(0.0)
            self._handoff_last_final_velocity.fill(0.0)
            self._ou_state *= 0.25
            self._ou_state = self._ou_state
            self._drift_state *= 0.85
            self._drift_state = self._drift_state
            if mode == 'pure_ai':
                self._alpha_target = 1.0
                self._takeover_state = 'NO_TARGET'
            elif mode == 'human_flick':
                self._alpha_target = 0.0
                self._takeover_state = 'NO_TARGET'
            return None

    def _effective_power(self):
        if self._alpha > 0.95:
            age_ramp = 1.0
        else:
            age_ramp_full = float(np.clip(self._ramp_min + (1.0 - self._ramp_min) *
                (self._age_ms / self._ramp_ms), self._ramp_min, 1.0))
            age_ramp = self._alpha * 1.0 + (1.0 - self._alpha) * age_ramp_full
        return float(np.clip(self._spf * age_ramp, 0.0, 1.0))

    def get_expected_lead(self):
        if self._ic_state == 'reaction':
            reaction_left = max(
                0.0, self._ic_reaction_delay - self._ic_reaction_elapsed
            )
            return float(np.clip(reaction_left + 0.11, 0.0, 0.25))
        if self._phase == self._BALLISTIC:
            return max(0.0, self._prog_T - self._prog_t)
        err = self.last_error_dist
        if err < 1.0:
            return 0.0
        v_max = self._v_max + (self._v_max_ai - self._v_max) * self._alpha
        dg = float(np.clip(self._depth_ref / max(self._last_bbox_w, 10.0), 0.45, 2.8))
        K = self._K_pursuit
        tau = self._tau_arm
        power = self._effective_power()
        gain = K * dg * max(power, 0.01)
        omega_n = np.sqrt(gain / tau)
        zeta = 1.0 / (2.0 * np.sqrt(gain * tau))
        zeta = float(np.clip(zeta, 0.1, 1.0))
        e_sat = v_max / max(gain, 1e-06)
        if err <= e_sat:
            ts = 4.0 / max(zeta * omega_n, 1e-06)
        else:
            t_sat = (err - e_sat) / max(v_max, 1e-06)
            t_linear = 4.0 / max(zeta * omega_n, 1e-06)
            ts = t_sat + t_linear
        return float(np.clip(ts, 0.0, 0.25))

    def notify_flick_end(self):
        # Compatibility entrypoint for callers without measured hand velocity.
        # It enters the same handoff state machine and never selects another
        # control algorithm.
        if (
            self._handoff_reason == 'human_release'
            and self._takeover_state in ('REACTION', 'PRIMING', 'ACTIVE_LOCK')
            and (
                self._takeover_state != 'ACTIVE_LOCK'
                or self._handoff_motor_elapsed < self._handoff_guard_duration
            )
        ):
            return
        self.begin_handoff(
            human_velocity=self._arm_vel.copy(),
            target_velocity=self._handoff_target_velocity.copy(),
            error=self._prev_control_error.copy(),
            reason='notify_flick_end',
        )

    def freeze_output_integrators(self):
        with self._lock as __temp_298:
            self._arm_vel[None:None] = 0.0
            self._wrist_vel[None:None] = 0.0
            self._servo_accel[None:None] = 0.0
            self.crosshair_velocity[None:None] = 0.0
            self._ou_state[None:None] = 0.0
            self._drift_state[None:None] = 0.0
            self._ff_vel_smoothed[None:None] = 0.0
            self._subpixel[None:None] = 0.0
            self._last_delta_float = 0.0, 0.0
            self._ic_state = 'reaction'
            self._ic_reaction_elapsed = 0.0
            self._pro_hold_error.fill(0.0)
            self._pro_hold_target_velocity.fill(0.0)
            self._pro_last_event = -1000000000.0
            self._pro_event_count = 0
            self._last_mouse_time = None
            self._prog_t = 0.0
            self._spf = 0.0
            self._phase = self._TRACKING
            self._last_prog_time = -999.0
            self._emit_mouse = False
            return None

    def yield_output_to_human(self):
        """Stop the actuator without restarting the cognitive controller.

        The safety gate still blocks every synthetic event.  Keeping the
        reaction and trajectory state alive lets perception continue planning
        while the human is moving, so release does not pay a second reaction
        delay.
        """
        with self._lock:
            self._arm_vel[:] = 0.0
            self._wrist_vel[:] = 0.0
            self._servo_accel[:] = 0.0
            self.crosshair_velocity[:] = 0.0
            self._ou_state[:] = 0.0
            self._drift_state[:] = 0.0
            self._subpixel[:] = 0.0
            self._last_delta_float = 0.0, 0.0
            self._last_mouse_time = None

    def set_mouse_emit(self, enable):
        with self._lock as __temp_312:
            self._emit_mouse = bool(enable)
            return None

    def warm_start_from_velocity(self, vx, vy):
        with self._lock as __temp_315:
            seed = np.array([vx, vy], dtype=np.float64)
            speed = float(np.linalg.norm(seed))
            if speed < 5000.0:
                self._arm_vel[None:None] = seed * 0.7
                self._wrist_vel[None:None] = self._arm_vel
                self._servo_accel[None:None] = 0.0
                self.crosshair_velocity[None:None] = self._arm_vel
                self._spf = 0.3
                self._age_ms = 20.0
                self._phase = self._TRACKING
                self._prog_t = 0.0
            return None

    def soft_freeze(self, dt):
        tau_hold = 0.012
        alpha = float(np.clip(1.0 - np.exp(-dt / tau_hold), 0.01, 0.4))
        with self._lock as __temp_324:
            self._arm_vel *= 1.0 - alpha
            self._arm_vel = self._arm_vel
            self._wrist_vel *= 1.0 - alpha
            self._wrist_vel = self._wrist_vel
            self._servo_accel *= 1.0 - alpha
            self._servo_accel = self._servo_accel
            self.crosshair_velocity[None:None] = self._arm_vel
            return None

    def _accum_emit_tick(self, mx, my, fdx, fdy):
        self._emit_ix += mx
        self._emit_ix = self._emit_ix
        self._emit_iy += my
        self._emit_iy = self._emit_iy
        self._emit_fx += fdx
        self._emit_fx = self._emit_fx
        self._emit_fy += fdy
        self._emit_fy = self._emit_fy
        return None

    def get_move_emit_diag(self):
        with self._lock as __temp_326:
            __temp_327, __temp_328 = self._move_diag_prev_int
            eix = __temp_327
            eiy = __temp_328
            __temp_329, __temp_330 = self._move_diag_prev_flt
            efx = __temp_329
            efy = __temp_330
            {prev_emit_int: (eix, eiy), prev_emit_flt: (efx, efy), cmd_vel_ct_s: (
                self._move_diag_cvx, self._move_diag_cvy), dt_s: self._move_diag_dt}

    def _adaptive_predict_dt(self, dist_counts, head_r_counts):
        if not self._ff_predict_adaptive:
            return self._ff_predict_dt
        near_thr = max(3.0 * head_r_counts, 30.0 * self._px_to_ct)
        far_thr = max(30.0 * head_r_counts, 300.0 * self._px_to_ct)
        if far_thr <= near_thr:
            return self._ff_predict_dt_near
        if dist_counts <= near_thr:
            return self._ff_predict_dt_near
        if dist_counts >= far_thr:
            return self._ff_predict_dt_far
        t = (math.log(dist_counts) - math.log(near_thr)) / (math.log(far_thr) -
            math.log(near_thr))
        return self._ff_predict_dt_near + (self._ff_predict_dt_far - self.
            _ff_predict_dt_near) * t

    def compute(self, target_x, target_y, dt, human_v=None, v_real=None, a_real=None, power_factor=1.0, bbox_w=60.0):
        with self._lock as __temp_339:
            p_iy = self._emit_iy
            p_ix = self._emit_ix
            p_fy = self._emit_fy
            p_fx = self._emit_fx
            self._emit_ix = 0
            self._emit_iy = 0
            self._emit_fx = 0.0
            self._emit_fy = 0.0
            self._move_diag_prev_int = int(p_ix), int(p_iy)
            self._move_diag_prev_flt = float(p_fx), float(p_fy)
            dt = max(dt, 0.0005)
            self._elapsed += dt
            self._elapsed = self._elapsed
            self._advance_alpha(dt)
            self._advance_attention(dt)
            self._advance_1f_noise(dt)
            alpha = self._alpha
            is_pure_ai = alpha > 0.5
            v_max_eff = self._v_max + (self._v_max_ai - self._v_max) * alpha
            if self._pre_warm_blend > 0.01:
                blend = self._pre_warm_blend * 0.1
                target_seed = self._pre_warm_target_v
                if float(np.linalg.norm(self._arm_vel)) < 3000.0:
                    self._arm_vel = (1.0 - blend
                        ) * self._arm_vel + blend * target_seed * 0.5
                self._pre_warm_blend *= 1.0 - 0.05 * dt * 1000.0
                self._pre_warm_blend = self._pre_warm_blend
                self._pre_warm_blend = float(max(0.0, self._pre_warm_blend - 0.001))
            depth_gain = float(np.clip(self._depth_ref / max(bbox_w, 10.0), 0.45, 2.8))
            px_to_ct = self._px_to_ct
            alpha_pf = float(np.clip(1.0 - np.exp(-dt / 0.025), 0.04, 0.3))
            spf_raw = float(np.clip(power_factor, 0.0, 1.0))
            spf_ema = (1.0 - alpha_pf) * self._spf + alpha_pf * power_factor
            self._spf = alpha * spf_raw + (1.0 - alpha) * spf_ema
            spf = float(np.clip(self._spf, 0.0, 1.0))
            self._age_ms += dt * 1000.0
            self._age_ms = self._age_ms
            if alpha < 0.95:
                eff_power = self._effective_power()
            else:
                eff_power = float(np.clip(spf, 0.0, 1.0))
            self._head_radius = max(4.0, bbox_w * 0.26)
            self._head_radius_px = self._head_radius
            self._last_bbox_w = float(bbox_w)
            head_r_counts = self._head_radius * px_to_ct
            head_r_gain_sched = float(max(head_r_counts, np.clip(12.0 * px_to_ct,
                5.0, 80.0)))
            dz_r_counts = max(2.5, bbox_w * self._dz_base) * px_to_ct
            thresh_high_counts = self._thresh_high * px_to_ct
            thresh_low_counts = self._thresh_low * px_to_ct
            anti_orbit_dist_counts = 35.0 * px_to_ct
            error = np.array([target_x, target_y], dtype=np.float64)
            dist = float(np.linalg.norm(error))
            if v_real is not None:
                if np.ndim(v_real) > 0:
                    tgt_vel = np.nan_to_num(np.asarray(v_real, dtype=np.float64),
                        nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    tgt_vel = np.zeros(2, dtype=np.float64)
            else:
                tgt_vel = np.zeros(2, dtype=np.float64)
            if a_real is not None:
                if np.ndim(a_real) > 0:
                    tgt_acc = np.nan_to_num(np.asarray(a_real, dtype=np.float64),
                        nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    tgt_acc = np.zeros(2, dtype=np.float64)
            else:
                tgt_acc = np.zeros(2, dtype=np.float64)
            tgt_vel_norm = float(np.linalg.norm(tgt_vel))
            if tgt_vel_norm > self._target_speed_limit:
                tgt_vel *= self._target_speed_limit / max(tgt_vel_norm, 1e-09)
            tgt_acc_norm = float(np.linalg.norm(tgt_acc))
            if tgt_acc_norm > self._target_accel_limit:
                tgt_acc *= self._target_accel_limit / max(tgt_acc_norm, 1e-09)
            ff_vel = tgt_vel * self._ff_gain + tgt_acc * self._ff_acc_gain_sec
            self._handoff_target_velocity[:] = tgt_vel
            self._pre_warm_target_v[:] = tgt_vel
            if self._has_prev_control_error:
                expected_error = self._prev_control_error + (tgt_vel - self._arm_vel
                    ) * dt
                error_innovation = float(np.linalg.norm(error - expected_error))
            else:
                error_innovation = 0.0
            adaptive_dt = self._adaptive_predict_dt(dist, head_r_counts)
            ff_vel_pred = tgt_vel + tgt_acc * adaptive_dt
            self._ff_vel_smoothed = (1.0 - self._ff_smooth_alpha
                ) * self._ff_vel_smoothed + self._ff_smooth_alpha * ff_vel_pred
            ff_vel_track = self._ff_vel_smoothed
            if self._ic_state == 'reaction':
                self._ic_reaction_elapsed = min(
                    self._ic_reaction_elapsed + dt,
                    self._ic_reaction_delay,
                )
                # During human lead the full visual state is still updated, but
                # autonomous motion cannot start. Reaction time accumulates as
                # covert planning, allowing an immediate handoff after release.
                human_has_authority = self._takeover_state == 'HUMAN_LEAD'
                if human_has_authority:
                    self._arm_vel *= math.exp(-dt / 0.018)
                    self._servo_accel.fill(0.0)
                    self._arm_vel[:] = self._enforce_final_handoff_slew(
                        self._arm_vel, dt
                    )
                    self.crosshair_velocity[:] = self._arm_vel
                    self.last_error_dist = dist
                    self._prev_control_error[:] = error
                    self._has_prev_control_error = True
                    self._move_diag_dt = dt
                    self._move_diag_cvx = float(self._arm_vel[0])
                    self._move_diag_cvy = float(self._arm_vel[1])
                    self._handoff_last_output[:] = self._arm_vel
                    return float(self._arm_vel[0]), float(self._arm_vel[1])
                if self._ic_reaction_elapsed < self._ic_reaction_delay:
                    handoff_reaction = self._handoff_reason in (
                        'human_release', 'notify_flick_end'
                    )
                    # Intermittent control holds the already selected motor
                    # primitive during the cognitive open-loop interval. An
                    # exponential stop here discarded useful human momentum
                    # and turned reaction latency into avoidable phase lag.
                    if handoff_reaction:
                        predicted_target_velocity = self._clip_vector_norm(
                            self._handoff_target_velocity
                            + tgt_acc * self._handoff_ff_acc_sec,
                            self._handoff_target_speed_limit,
                        )
                        ff_revision = (
                            predicted_target_velocity
                            - self._handoff_seed_target_velocity
                        ) * self._handoff_ff_revision_gain
                        if dist > 1e-6:
                            current_radial = error / dist
                            current_far_weight = float(np.clip(
                                (
                                    dist / max(self._px_to_ct, 1e-6) - 60.0
                                ) / 60.0,
                                0.0, 1.0,
                            ))
                            current_open_loop_max = (
                                self._handoff_open_loop_max
                                + current_far_weight
                                * self._handoff_open_loop_far_max_boost
                            )
                            current_horizon = self._handoff_open_loop_horizon * (
                                1.0
                                - current_far_weight
                                * (1.0 - self._handoff_open_loop_far_ratio)
                            )
                            current_closing_speed = min(
                                max(
                                    self._handoff_closing_speed,
                                    dist / max(current_horizon, 1e-6),
                                ),
                                current_open_loop_max,
                            )
                            radial_revision = (
                                current_radial * current_closing_speed
                                - self._handoff_seed_radial
                                * self._handoff_closing_speed
                            )
                        else:
                            radial_revision = np.zeros(2, dtype=np.float64)
                        reaction_velocity = self._clip_vector_norm(
                            self._handoff_residual_velocity
                            + ff_revision
                            + radial_revision,
                            self._handoff_residual_limit,
                        )
                    else:
                        reaction_velocity = self._arm_vel * math.exp(-dt / 0.018)
                    reaction_guarded = (
                        handoff_reaction
                        and self._handoff_elapsed < self._handoff_guard_duration
                    )
                    self._handoff_elapsed += dt
                    if reaction_guarded:
                        delta = self._clip_vector_norm(
                            reaction_velocity - self._handoff_last_output,
                            self._handoff_max_accel * max(dt, 1e-6),
                        )
                        self._arm_vel[:] = self._handoff_last_output + delta
                    else:
                        self._arm_vel[:] = reaction_velocity
                    self._arm_vel = self._arm_vel
                    self._arm_vel[:] = self._enforce_final_handoff_slew(
                        self._arm_vel, dt
                    )
                    self._servo_accel.fill(0.0)
                    self.crosshair_velocity[None:None] = self._arm_vel
                    self.last_error_dist = dist
                    self._prev_control_error[None:None] = error
                    self._has_prev_control_error = True
                    self._move_diag_dt = dt
                    self._move_diag_cvx = float(self._arm_vel[0])
                    self._move_diag_cvy = float(self._arm_vel[1])
                    self._handoff_last_output[:] = self._arm_vel
                    return float(self._arm_vel[0]), float(self._arm_vel[1])
                # Freeze the latest planned endpoint only when motor commitment
                # begins. Perception was unrestricted throughout REACTION.
                self._takeover_state = 'PRIMING'
                if self._handoff_reason in ('human_release', 'notify_flick_end'):
                    self._ic_state = 'primary'
                    self._handoff_motor_elapsed = 0.0
                    residual_at_commit = self._arm_vel.copy()
                    self._start_motor_program(dist, error, head_r_counts)
                    self._handoff_residual_velocity[:] = residual_at_commit
                    self._handoff_last_output[:] = residual_at_commit
                    self._arm_vel[:] = residual_at_commit
                    self._wrist_vel[:] = residual_at_commit
                    self.crosshair_velocity[:] = self._arm_vel
                else:
                    # Autonomous acquisition keeps the original distance-aware
                    # choice: far targets launch a program below, near targets
                    # enter impedance tracking directly.
                    self._ic_state = 'hold'
            tgt_speed = float(np.linalg.norm(tgt_vel))
            now = self._elapsed
            time_since_prog = now - self._last_prog_time
            crossed_program_endpoint = False
            if self._phase == self._BALLISTIC:
                # A minimum-jerk primitive is only open-loop while the measured
                # target remains ahead of its launch direction. Plant-gain
                # error or an output stall can otherwise make the timed
                # program keep accelerating after the crosshair has crossed.
                along_program = float(np.dot(error, self._prog_dir))
                if (
                    self._prog_t > 0.0
                    and tgt_speed < self._capture_static_speed
                    and along_program <= 0.0
                ):
                    crossed_program_endpoint = True
                    self._phase = self._TRACKING
                    self._prog_t = 0.0
                    self._pro_hold_error[:] = error
                    self._pro_hold_target_velocity[:] = tgt_vel
                    self._pro_last_event = self._elapsed
                    _log_cipher.phase_switch(
                        'BALLISTIC', 'TRACKING', reason='measured_endpoint_crossing'
                    )
            if self._phase == self._BALLISTIC:
                tau_now = self._prog_t / max(self._prog_T, 1e-06)
                prog_remain = self._prog_D * (1.0 - _minjerk_pos(tau_now))
                startle_thr = max(head_r_counts * 6.0, prog_remain * 0.65, 30.0 *
                    px_to_ct)
                innovation_thr = max(head_r_counts * 1.5, 12.0 * px_to_ct)
                can_replan = time_since_prog >= self._min_prog_interval
                distance_startle = dist > prog_remain + startle_thr
                if alpha > 0.95:
                    if distance_startle:
                        if error_innovation > innovation_thr:
                            should_replan = can_replan
                        else:
                            should_replan = error_innovation > innovation_thr
                    else:
                        should_replan = distance_startle
                else:
                    should_replan = distance_startle
                if should_replan:
                    _log_cipher.phase_switch('BALLISTIC', 'BALLISTIC(re-plan)',
                        reason='startle', dist=round(dist), remain=round(
                        prog_remain), thr=round(startle_thr), innovation=round(
                        error_innovation))
                    self._start_motor_program(dist, error, head_r_counts)
                    v_cmd = np.zeros(2, dtype=np.float64)
                else:
                    v_cmd = np.zeros(2, dtype=np.float64)
            elif dist > thresh_high_counts and not crossed_program_endpoint:
                if time_since_prog > self._min_prog_interval:
                    self._start_motor_program(dist, error, head_r_counts)
                    v_cmd = np.zeros(2, dtype=np.float64)
                else:
                    v_cmd = np.zeros(2, dtype=np.float64)
            else:
                v_cmd = np.zeros(2, dtype=np.float64)
            if self._phase == self._BALLISTIC:
                if eff_power < 0.02:
                    _log_cipher.phase_switch('BALLISTIC', 'TRACKING', reason=
                        'emergency_yield', eff_power=float(eff_power))
                    self._phase = self._TRACKING
                    self._prog_t = 0.0
                    self._arm_vel *= 0.15
                    self._arm_vel = self._arm_vel
                    v_cmd = np.zeros(2, dtype=np.float64)
                else:
                    self._prog_t += dt
                    self._prog_t = self._prog_t
                    tau = self._prog_t / max(self._prog_T, 1e-06)
                    v_x_poly = 3.0 * self._prog_cx[3
                        ] * self._prog_t * self._prog_t + 4.0 * self._prog_cx[4
                        ] * self._prog_t ** 3 + 5.0 * self._prog_cx[5
                        ] * self._prog_t ** 4
                    v_y_poly = 3.0 * self._prog_cy[3
                        ] * self._prog_t * self._prog_t + 4.0 * self._prog_cy[4
                        ] * self._prog_t ** 3 + 5.0 * self._prog_cy[5
                        ] * self._prog_t ** 4
                    brake = 1.0
                    if tau > 0.65:
                        if dist > 0.001:
                            remaining_past_undershoot = max(dist - self.
                                _prog_undershoot, 0.0)
                            frac = remaining_past_undershoot / max(dist, 1.0)
                            brake = float(np.clip(0.2 + 0.8 * frac, 0.2, 1.0))
                    if self._micro_pause_enable:
                        if 0.85 < tau:
                            if tau < 0.95:
                                pause_factor = 1.0 - 0.15 * (1.0 - abs(tau - 0.9) /
                                    0.05)
                                brake *= float(np.clip(pause_factor, 0.85, 1.0))
                    v_x_poly *= brake
                    v_y_poly *= brake
                    ff_coef_ball = 0.6 + 0.4 * alpha
                    v_ballistic = np.array([v_x_poly + ff_vel[0] * ff_coef_ball,
                        v_y_poly + ff_vel[1] * ff_coef_ball], dtype=np.float64)
                    if tau > 0.8:
                        blend_imp = float((tau - 0.8) / 0.2)
                        __temp_434, __temp_435 = _impedance_vel(error[0], error[1],
                            self._arm_vel[0], self._arm_vel[1], 0.0, 0.0, self.
                            _K_correction, self._B_correction, v_max_eff,
                            dz_r_counts * 0.75, depth_gain, eff_power)
                        vcx_imp = __temp_434
                        vcy_imp = __temp_435
                        v_tracking_preview = np.array([vcx_imp, vcy_imp]
                            ) + ff_vel * eff_power * max(self._ff_scale_min, 0.6)
                        v_cmd = (1.0 - blend_imp
                            ) * v_ballistic + blend_imp * v_tracking_preview
                    else:
                        v_cmd = v_ballistic
                    if tau >= 1.0:
                        _log_cipher.phase_switch('BALLISTIC', 'TRACKING', prog_T_ms
                            =round(self._prog_T * 1000, 1), elapsed_ms=round(self.
                            _prog_t * 1000, 1))
                        self._phase = self._TRACKING
                        entry_ticks_eff = int(round(self._entry_ticks_init * (1.0 -
                            alpha)))
                        self._entry_ticks_remaining = entry_ticks_eff
                        land_damp = 0.5 - 0.24 * alpha
                        if alpha > 0.95:
                            self._arm_vel = (self._arm_vel * land_damp +
                                ff_vel_track * (1.0 - land_damp))
                        else:
                            self._arm_vel *= land_damp
                            self._arm_vel = self._arm_vel
            else:
                smooth_pursuit = tgt_speed > 450.0
                capture_zone = (
                    tgt_speed < self._capture_static_speed
                    and dist <= anti_orbit_dist_counts
                )
                if smooth_pursuit or capture_zone or crossed_program_endpoint:
                    control_error = error
                    self._pro_hold_error[:] = error
                    self._pro_hold_target_velocity[:] = tgt_vel
                else:
                    predicted_hold_error = self._pro_hold_error + (self.
                        _pro_hold_target_velocity - self._arm_vel) * dt
                    event_threshold = max(3.0, min(10.0, float(bbox_w) * 0.1)
                        ) * px_to_ct
                    hold_mismatch = float(np.linalg.norm(error - predicted_hold_error))
                    if not self._pro_last_event < 0.0:
                        if hold_mismatch >= event_threshold:
                            if (self._elapsed - self._pro_last_event >= self.
                                _pro_refractory):
                                self._pro_hold_error[None:None] = error
                                self._pro_hold_target_velocity[None:None] = tgt_vel
                                self._pro_last_event = self._elapsed
                                self._pro_event_count += 1
                                self._pro_event_count = self._pro_event_count
                            else:
                                self._pro_hold_error[None:None] = predicted_hold_error
                        else:
                            self._pro_hold_error[None:None] = predicted_hold_error
                    else:
                        self._pro_hold_error[None:None] = error
                        self._pro_hold_target_velocity[None:None] = tgt_vel
                        self._pro_last_event = self._elapsed
                        self._pro_event_count += 1
                        self._pro_event_count = self._pro_event_count
                    control_error = self._pro_hold_error
                d_norm = dist / max(head_r_gain_sched, 1e-06)
                near_w = float(np.clip(1.0 - d_norm, 0.0, 1.0))
                # True hysteretic gain schedule: below thresh_low there must
                # be no residual flick gain. The old rational curve retained
                # 25% flick gain even at zero error, making 30px corrections
                # behave like ballistic moves and repeatedly cross the target.
                gain_span = max(thresh_high_counts - thresh_low_counts, 1e-6)
                gain_u = float(np.clip(
                    (dist - thresh_low_counts) / gain_span, 0.0, 1.0
                ))
                w_flick = gain_u * gain_u * (3.0 - 2.0 * gain_u)
                K_mid = self._K_pursuit * (1.0 - w_flick) + self._K_flick * w_flick
                K_eff = self._K_correction * near_w + K_mid * (1.0 - near_w)
                B_eff = self._B_correction * near_w + self._B_pursuit * (1.0 - near_w)
                if is_pure_ai:
                    near_ff = float(np.clip(1.0 - 0.5 * near_w, self.
                        _ff_scale_near_min, 1.0))
                    ff_scale = near_ff
                    vrefx = ff_vel_track[0] * eff_power * near_ff
                    vrefy = ff_vel_track[1] * eff_power * near_ff
                    dz_eff = dz_r_counts * (1.0 + 0.42 * near_w * near_w)
                else:
                    if tgt_speed < self._ff_speed_knee:
                        ff_scale_near = self._ff_scale_min
                    else:
                        ff_scale_near = min(self._ff_scale_min + (tgt_speed - self.
                            _ff_speed_knee) / self._ff_speed_scale, 1.0)
                    ff_scale = ff_scale_near * near_w + 1.0 * (1.0 - near_w)
                    rel_w = 1.0 - near_w
                    vrefx = ff_vel_track[0] * eff_power * rel_w
                    vrefy = ff_vel_track[1] * eff_power * rel_w
                    dz_eff = dz_r_counts * (1.0 + 0.5 * near_w)
                    if self._entry_ticks_remaining > 0:
                        entry_w = float(self._entry_ticks_remaining) / max(self.
                            _entry_ticks_init, 1)
                        dz_entry = self._entry_dz_px * self._px_to_ct
                        dz_eff = max(dz_eff, dz_entry * entry_w)
                        self._entry_ticks_remaining -= 1
                        self._entry_ticks_remaining = self._entry_ticks_remaining
                if 0.05 < alpha:
                    if alpha < 0.95:
                        if is_pure_ai:
                            dz_eff_other = dz_r_counts * (1.0 + 0.5 * near_w)
                        else:
                            dz_eff_other = dz_r_counts * (1.0 + 0.42 * near_w * near_w)
                        blend_strength = 1.0 - abs(alpha - 0.5) * 2.0
                        dz_eff = (1.0 - 0.5 * blend_strength
                            ) * dz_eff + 0.5 * blend_strength * dz_eff_other
                __temp_463, __temp_464 = _impedance_vel(control_error[0],
                    control_error[1], self._arm_vel[0], self._arm_vel[1], vrefx,
                    vrefy, K_eff, B_eff, v_max_eff, dz_eff, depth_gain, eff_power)
                vcx = __temp_463
                vcy = __temp_464
                v_cmd = np.array([vcx, vcy]) + ff_vel_track * eff_power * ff_scale
            d_norm_i = float(np.clip(dist / max(head_r_gain_sched, 1e-06), 0.0, 2.0))
            tau_arm_eff = self._tau_arm * (1.0 - self._tau_arm_dyn_gain + self.
                _tau_arm_dyn_gain * d_norm_i)
            alpha_arm = float(np.clip(1.0 - np.exp(-dt / tau_arm_eff), 0.01, 0.4))
            self._arm_vel[0] = (1.0 - alpha_arm) * self._arm_vel[0
                ] + alpha_arm * v_cmd[0]
            self._arm_vel[1] = (1.0 - alpha_arm) * self._arm_vel[1
                ] + alpha_arm * v_cmd[1]
            if 0.001 < dist:
                unit = error / dist
                relative_velocity = self._arm_vel - ff_vel_track
                radial_closing = float(np.dot(relative_velocity, unit))
                stopping_distance = max(dist - 0.35 * head_r_counts, 0.0)
                safe_closing = math.sqrt(
                    2.0 * self._landing_decel * stopping_distance
                )
                # The safe-speed test already encodes a dynamic braking
                # radius v^2/(2a). Applying it globally starts deceleration at
                # the physically correct distance instead of waiting for a
                # fixed 35px landing circle that can be much too late.
                if radial_closing > safe_closing:
                    if tgt_speed < self._capture_static_speed:
                        reduction = radial_closing - safe_closing
                    else:
                        reduction = min(
                            radial_closing - safe_closing,
                            self._landing_decel * dt,
                        )
                    self._arm_vel -= unit * reduction
                if dist < anti_orbit_dist_counts:
                    radial_v = np.dot(self._arm_vel, unit) * unit
                    tangent_v = self._arm_vel - radial_v
                    t_damp = float(np.clip(dist / anti_orbit_dist_counts, 0.0, 1.0)
                        ) ** 0.5
                    self._arm_vel = radial_v + tangent_v * t_damp
            arm_spd = float(np.linalg.norm(self._arm_vel))
            if arm_spd > v_max_eff:
                self._arm_vel *= v_max_eff / arm_spd
                self._arm_vel = self._arm_vel
            # Final actuator boundary: no downstream geometric damping or speed
            # clamp may violate the handoff acceleration envelope.
            self._arm_vel[:] = self._apply_handoff_envelope(self._arm_vel, dt)
            self._arm_vel[:] = self._enforce_final_handoff_slew(
                self._arm_vel, dt
            )
            if (
                self._takeover_state == 'PRIMING'
                and (self._phase == self._TRACKING or dist <= thresh_low_counts)
            ):
                self._takeover_state = 'ACTIVE_LOCK'
                self._ic_state = 'hold'
            self.crosshair_velocity[None:None] = self._arm_vel
            self.last_error_dist = dist
            self._prev_control_error[None:None] = error
            self._has_prev_control_error = True
            if self.mode == 'track':
                threshold = thresh_high_counts
            else:
                threshold = thresh_low_counts
            if dist > threshold:
                new_mode = 'flick'
            else:
                new_mode = 'track'
            if new_mode != self.mode:
                _log_cipher.controller_mode(self.mode, new_mode, dist, threshold, 'ct')
                self.mode = new_mode
            else:
                self.mode = new_mode
            self._move_diag_dt = float(dt)
            self._move_diag_cvx = float(self._arm_vel[0])
            self._move_diag_cvy = float(self._arm_vel[1])
            return float(self._arm_vel[0]), float(self._arm_vel[1])

    @staticmethod
    def _clip_vector_norm(vector, max_norm):
        norm = float(np.linalg.norm(vector))
        if norm > max_norm:
            if norm > 1e-09:
                return vector * (max_norm / norm)
        return vector

    def tick_mouse(self, dt_override=None):
        with self._lock as __temp_637:
            now = time.perf_counter()
            if dt_override is None:
                if self._last_mouse_time is None:
                    self._last_mouse_time = now
                    self._accum_emit_tick(0, 0, 0.0, 0.0)
                    return 0, 0
                dt = now - self._last_mouse_time
            else:
                dt = float(dt_override)
            if not self._emit_mouse:
                self._last_mouse_time = now
                self._accum_emit_tick(0, 0, 0.0, 0.0)
                return 0, 0
            self._last_mouse_time = now
            if dt > 0.02:
                _log_cipher.warning(
                    'tick_mouse dt=%.1fms > 20ms (scheduler stall?), clamped', dt *
                    1000.0)
                dt = 0.02
            if dt <= 0.0:
                self._accum_emit_tick(0, 0, 0.0, 0.0)
                return 0, 0
            sqrt_dt = dt ** 0.5
            dW_ou = self._rng.randn(2).astype(np.float64)
            if self._phase == self._BALLISTIC:
                ou_sigma_base = self._ou_sigma_ball
            else:
                head_r_ct = max(self._head_radius_px * self._px_to_ct, 1e-06)
                if self.last_error_dist < head_r_ct * 0.45:
                    ou_sigma_base = self._ou_sigma_track * 0.22
                elif self.last_error_dist < head_r_ct * 0.5:
                    ou_sigma_base = self._ou_sigma_track * 0.4
                elif self.last_error_dist < head_r_ct:
                    ou_sigma_base = self._ou_sigma_track * 0.7
                else:
                    ou_sigma_base = self._ou_sigma_track
            alpha_damp = 0.22 + 0.78 * (1.0 - self._alpha)
            attention_mod = 1.0 / max(self._attention_state, 0.5)
            ou_sigma = ou_sigma_base * alpha_damp * attention_mod
            self._ou_state += (-self._ou_theta * self._ou_state * dt + ou_sigma *
                sqrt_dt * dW_ou)
            self._ou_state = self._ou_state
            dW_drift = self._rng.randn(2).astype(np.float64)
            self._drift_state += (-self._drift_theta * self._drift_state * dt +
                self._drift_sigma * sqrt_dt * dW_drift)
            self._drift_state = self._drift_state
            tremor_vx = self._ou_state[0] * self._ou_vel_scale
            tremor_vy = self._ou_state[1] * self._ou_vel_scale
            noise_scale_1f = (1.0 - self._alpha) * 0.5 + 0.1
            tremor_vx += (self._noise_hand[0] + self._noise_fatigue[0]
                ) * self._ou_vel_scale * noise_scale_1f
            tremor_vy += (self._noise_hand[1] + self._noise_fatigue[1]
                ) * self._ou_vel_scale * noise_scale_1f
            arm_vx_eff = self._arm_vel[0] + tremor_vx
            arm_vy_eff = self._arm_vel[1] + tremor_vy
            alpha_w = float(np.clip(1.0 - np.exp(-dt / self._tau_wrist), 0.01, 0.35))
            self._wrist_vel[0] = (1.0 - alpha_w) * self._wrist_vel[0
                ] + alpha_w * arm_vx_eff
            self._wrist_vel[1] = (1.0 - alpha_w) * self._wrist_vel[1
                ] + alpha_w * arm_vy_eff
            drift_x = self._drift_state[0] * self._drift_pos_scale
            drift_y = self._drift_state[1] * self._drift_pos_scale
            delta_x = self._wrist_vel[0] * dt + drift_x + self._subpixel[0]
            delta_y = self._wrist_vel[1] * dt + drift_y + self._subpixel[1]
            mx = int(np.floor(delta_x))
            my = int(np.floor(delta_y))
            self._subpixel[0] = delta_x - mx
            self._subpixel[1] = delta_y - my
            self._last_delta_float = self._wrist_vel[0
                ] * dt + drift_x, self._wrist_vel[1] * dt + drift_y
            self._accum_emit_tick(mx, my, self._last_delta_float[0], self.
                _last_delta_float[1])
            return mx, my
