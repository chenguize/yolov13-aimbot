# control.py
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import threading
import random

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
# 必须保证仓库根在 sys.path 最前：直接跑 `python test/control.py` 时 Python 会把
# test/ 放在 path[0]，若只「根在 path 里但不在首位」会错 import 标准库同名的 `test`。
if _ROOT_DIR in sys.path:
    sys.path.remove(_ROOT_DIR)
sys.path.insert(0, _ROOT_DIR)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import config
from test.scenarios.base import BaseScenario
from test.scenarios.takeover import TakeoverScenario
from test.scenarios.pure_ai_ball import PureAIBallScenario
# from .base_controller import BaseController
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ====================== 全局工具函数 ======================
def safe_mean(arr, default=0.0):
    return float(np.mean(arr)) if len(arr) > 0 else default


def safe_percentile(arr, q, default=0.0):
    return float(np.percentile(arr, q)) if len(arr) > 0 else default


# ══════════════════════════════════════════════════════════════════════════════
# § 模块D：评分阈值配置化（[TestScoring] section，缺省回退到 v4.1 默认值）
# ══════════════════════════════════════════════════════════════════════════════
def _ts(key: str, default: float) -> float:
    """从 [TestScoring] 读阈值，缺省用 default。"""
    return float(config.getfloat("TestScoring", key, default))


SCORING_THRESHOLDS = {
    # TTK 评分（0.18s→100, 0.40s→0）
    'ttk_full':        _ts('ttk_full_s',       0.18),
    'ttk_zero':        _ts('ttk_zero_s',       0.40),
    # 精度评分（10px→100, 20px→0）
    'prec_full_px':    _ts('prec_full_px',     10.0),
    'prec_zero_px':    _ts('prec_zero_px',     20.0),
    # 新增：过冲评分（overshoot <5px→100, >25px→0）
    'overshoot_full':  _ts('overshoot_full_px', 5.0),
    'overshoot_zero':  _ts('overshoot_zero_px', 25.0),
    # 新增：加速度峰值评分（<500 ct/s²→100, >2000→0）
    'accel_full':      _ts('accel_full_cts2',  500.0),
    'accel_zero':      _ts('accel_zero_cts2',  2000.0),
    # 新增：接管延迟评分（<100ms→100, >500ms→0，v5.0 放宽让 jerk² 改进可见）
    # 10/50ms 旧值对仿真接管（典型 200-400ms）太严，永远 0 分看不出 jerk² 效果
    'takeover_full':   _ts('takeover_full_ms', 100.0),
    'takeover_zero':   _ts('takeover_zero_ms', 500.0),
    # 锁定阈值（error<此值且连续帧<1.5倍 → 锁定）
    'lock_thresh_px':  _ts('lock_thresh_px',   15.0),
    'lock_hold_px':    _ts('lock_hold_px',     22.0),
    'lock_hold_frames':int(_ts('lock_hold_frames', 10)),
    # 分层桶边界
    'dist_near_max':   _ts('dist_near_max_px', 120.0),
    'dist_far_min':    _ts('dist_far_min_px',  220.0),
    'vel_slow_max':    _ts('vel_slow_max_pxs', 200.0),
    'vel_fast_min':    _ts('vel_fast_min_pxs', 800.0),
    # 评分权重（总和=1.0）
    'w_ttk':           _ts('w_ttk',            0.30),
    'w_precision':     _ts('w_precision',      0.15),
    'w_natural':       _ts('w_natural',        0.20),
    'w_bio':           _ts('w_bio',            0.10),
    'w_overshoot':     _ts('w_overshoot',      0.10),
    'w_smoothness':    _ts('w_smoothness',     0.10),
    'w_takeover':      _ts('w_takeover',       0.05),
}


def _linear_score(value: float, full_thresh: float, zero_thresh: float) -> float:
    """线性插值评分：value=full_thresh→100, value=zero_thresh→0。支持递增/递减。"""
    if full_thresh == zero_thresh:
        return 50.0
    if zero_thresh > full_thresh:
        # 递减型（如 TTK：越小越好）
        return float(np.clip(100.0 * (zero_thresh - value) / (zero_thresh - full_thresh), 0.0, 100.0))
    else:
        # 递增型（如精度：越大越好）
        return float(np.clip(100.0 * (value - full_thresh) / (zero_thresh - full_thresh), 0.0, 100.0))


def analyze_tracking_quality(logs: List[Dict]) -> Dict:
    """
    Valorant Biomimetic Scoring v5.0 — 模块D 分层评分 + 新维度指标 + 鲁棒性
    ────────────────────────────────────────────────────────────────────────────
    升级要点：
      1. 分层评分：按初始距离 / 目标速度分桶，防"简单场景高分"过拟合
      2. 新维度：overshoot（过冲）/ accel_peak（加速度峰值）/ takeover_delay（接管延迟）
      3. 鲁棒性：P95 / worst-case / cross-bucket 一致性
      4. 阈值配置化：[TestScoring] section，缺省回退 v4.1
      5. 权重重平衡：TTK 30% + Prec 15% + Natural 20% + Bio 10%
                     + Overshoot 10% + Smooth 10% + Takeover 5%
    """
    if not logs:
        return {}

    ST = SCORING_THRESHOLDS
    kills_data = {}
    for log in logs:
        kid = log['kill_id']
        kills_data.setdefault(kid, {'logs': [], 'mode': log['mode']})['logs'].append(log)

    metrics = {
        'pure_ai': {
            'phase_times': [], 'steady_maes': [], 'headshot_errors': [],
            'final_errors': [],
            'natural_scores': [], 'bio_bonuses': [], 'vel_rms': [],
            # 模块D 新增
            'overshoots': [], 'accel_peaks': [], 'takeover_delays': [],
            'takeover_smoothness': [], 'takeover_precision': [],
            'takeover_composites': [],
            'initial_distances': [], 'target_velocities': [],
            'per_kill_scores': [],
        },
        'human_flick': {
            'phase_times': [], 'steady_maes': [], 'headshot_errors': [],
            'final_errors': [],
            'natural_scores': [], 'bio_bonuses': [], 'vel_rms': [],
            'overshoots': [], 'accel_peaks': [], 'takeover_delays': [],
            'takeover_smoothness': [], 'takeover_precision': [],
            'takeover_composites': [],
            'initial_distances': [], 'target_velocities': [],
            'per_kill_scores': [],
        }
    }

    print("\n" + "=" * 185)
    print(
        f"{'KillID':<6} | {'Mode':<10} | {'Phase(s)':<9} | {'Dist':<6} | {'Vel':<6} | "
        f"{'MAE':<7} | {'Over':<6} | {'AccPk':<7} | {'Natural':<7} | {'Bio':<6} | "
        f"{'TkDelay':<8} | Status")
    print("=" * 185)

    for kid, data in kills_data.items():
        k_logs = data['logs']
        if len(k_logs) < 12:
            continue

        times = np.array([l['t'] for l in k_logs])
        targets = np.array([l['target'] for l in k_logs])
        crosshairs = np.array([l['crosshair'] for l in k_logs])
        errors = np.linalg.norm(targets - crosshairs, axis=1)

        see_idx = next((i for i, l in enumerate(k_logs) if l.get('is_valid', False)), 0)

        # ── 接管场景: 人类松手时间 ──
        release_t = max((l.get('takeover_release_time', 0.0) for l in k_logs), default=0.0)
        if release_t > 0:
            release_idx = next((i for i, l in enumerate(k_logs) if l['t'] >= release_t), see_idx)
            if release_idx > see_idx:
                see_idx = release_idx

        # ── 锁定判定（配置化阈值）──
        lock_thresh = ST['lock_thresh_px']
        lock_hold = ST['lock_hold_px']
        lock_n = ST['lock_hold_frames']
        lock_idx = None
        for i in range(see_idx, len(errors) - lock_n):
            if errors[i] < lock_thresh and np.all(errors[i:i + lock_n] < lock_hold):
                lock_idx = i
                break
        if lock_idx is None:
            lock_idx = see_idx + min(60, len(errors) - see_idx - 1)

        phase_time = times[lock_idx] - times[see_idx]
        dt_med = float(np.median(np.diff(times))) if len(times) > 1 else 0.002
        eval_window = max(8, int(0.12 / max(dt_med, 1e-4)))
        eval_end = min(len(errors), lock_idx + eval_window)
        steady_errors = errors[lock_idx:eval_end]

        if len(steady_errors) > 0:
            low_q, high_q = np.percentile(steady_errors, [15, 85])
            trimmed = steady_errors[(steady_errors >= low_q) & (steady_errors <= high_q)]
            raw_mae = float(np.mean(trimmed)) if len(trimmed) > 0 else float(np.mean(steady_errors))
            headshot_err = float(np.percentile(steady_errors, 70))
        else:
            raw_mae = 0.0
            headshot_err = 0.0

        # ════════════════════════════════════════════════════════════════════════
        # 模块D 新增指标 1：初始距离 + 目标速度（分层桶用）
        # ════════════════════════════════════════════════════════════════════════
        initial_dist = float(errors[see_idx]) if see_idx < len(errors) else 0.0
        target_vel_mean = float(np.mean([l.get('target_vel', 0.0) for l in k_logs[see_idx:lock_idx + 1]]))

        # ════════════════════════════════════════════════════════════════════════
        # 模块D 新增指标 2：过冲（overshoot）
        # 定义：锁定后窗口内 error 最大值 - 锁定点 error，若 >0 说明过冲
        # ════════════════════════════════════════════════════════════════════════
        lock_err = float(errors[lock_idx]) if lock_idx < len(errors) else 0.0
        post_lock_max_err = float(np.max(errors[lock_idx:eval_end])) if eval_end > lock_idx else lock_err
        overshoot = max(0.0, post_lock_max_err - lock_err)

        # ════════════════════════════════════════════════════════════════════════
        # 模块D 新增指标 3：加速度峰值（BALLISTIC→TRACKING 转换段平滑度）
        # 定义：[see_idx, lock_idx+window] 内加速度峰值（ct/s²）
        # ════════════════════════════════════════════════════════════════════════
        accel_peak = 0.0
        if len(k_logs) > 5:
            dt_arr = np.diff(times)
            dt_arr[dt_arr < 1e-6] = 0.002
            vels = np.diff(crosshairs, axis=0) / dt_arr[:, None]
            accel = np.diff(vels, axis=0)
            # 仅取 [see_idx, lock_idx+window] 段
            seg_end = min(len(accel), eval_end)
            if seg_end > see_idx + 1:
                accel_seg = accel[see_idx:seg_end]
                accel_peak = float(np.max(np.linalg.norm(accel_seg, axis=1)))

        # ════════════════════════════════════════════════════════════════════════
        # 模块D 新增指标 4：接管延迟（takeover_delay）
        # 定义：从人类松手到 AI 达到稳定锁定的延迟（仅 takeover 场景）
        # ════════════════════════════════════════════════════════════════════════
        takeover_delay = 0.0
        takeover_accel_peak = 0.0
        takeover_error_50ms = 0.0
        takeover_composite = 0.0
        if release_t > 0:
            takeover_delay = phase_time  # 松手到锁定的总时间
            # Controller slew limits are defined in simulation time. Wall-clock
            # deltas vary with host load and falsely inflate acceleration when
            # the simulator runs faster than real time.
            handoff_times = times
            handoff_t0 = handoff_times[release_idx]
            guard_end = int(np.searchsorted(
                handoff_times, handoff_t0 + 0.100, side='right'
            ))
            guard_start = max(0, release_idx - 2)
            if guard_end - guard_start >= 4:
                handoff_vel = np.array([
                    l.get('ctrl_velocity', (0.0, 0.0))
                    for l in k_logs[release_idx:guard_end]
                ], dtype=np.float64)
                handoff_t = handoff_times[release_idx:guard_end]
                # The diagnostics loop can sample the simulation thread more
                # than once per 2 ms control frame. Keep the last observation
                # of each frame before differentiating velocity.
                _, reverse_indices = np.unique(
                    handoff_t[::-1], return_index=True
                )
                unique_indices = np.sort(len(handoff_t) - 1 - reverse_indices)
                handoff_t = handoff_t[unique_indices]
                handoff_vel = handoff_vel[unique_indices]
                handoff_dt = np.diff(handoff_t)
                handoff_accel = (
                    np.diff(handoff_vel, axis=0) / handoff_dt[:, None]
                    if len(handoff_dt) else np.empty((0, 2), dtype=np.float64)
                )
                # The first sample changes authority source (human velocity to
                # controller velocity). C0 continuity is evaluated on the total
                # crosshair trajectory; actuator slew starts with the next pair.
                if len(handoff_accel) > 1:
                    handoff_accel = handoff_accel[1:]
                if len(handoff_accel):
                    takeover_accel_peak = float(
                        np.max(np.linalg.norm(handoff_accel, axis=1))
                    )
            commit_idx = next((
                i for i in range(release_idx, len(k_logs))
                if k_logs[i].get('takeover_state') in ('PRIMING', 'ACTIVE_LOCK')
            ), release_idx)
            commit_t = handoff_times[commit_idx]
            error_50_idx = int(np.searchsorted(
                handoff_times, commit_t + 0.050, side='left'
            ))
            error_50_idx = min(max(error_50_idx, commit_idx), len(errors) - 1)
            takeover_error_50ms = float(errors[error_50_idx])
            delay_score = _linear_score(
                takeover_delay * 1000.0, ST['takeover_full'], ST['takeover_zero']
            )
            handoff_smooth_score = _linear_score(
                takeover_accel_peak, 1500.0, 5000.0
            )
            handoff_precision_score = _linear_score(
                takeover_error_50ms, 5.0, 30.0
            )
            takeover_composite = (
                0.4 * delay_score
                + 0.3 * handoff_smooth_score
                + 0.3 * handoff_precision_score
            )

        # ==================== Naturalness v3.0 (Valorant Edition) ====================
        vel_rms = 0.0
        natural_score = 75.0
        if len(k_logs) > 5:
            vel_rms = float(np.mean(np.linalg.norm(vels, axis=1)))
            high_freq = float(np.mean(np.abs(accel[::2])))
            hf_penalty = max(0.0, high_freq / 40.0 * 25.0)
            dead_lock_penalty = 15.0 if float(np.std(crosshairs[lock_idx:, 0])) < 0.1 else 0.0
            natural_score = np.clip(100.0 - hf_penalty - dead_lock_penalty, 0.0, 100.0)

        # ==================== Biomimetic Micro-correction v3.0 ====================
        bio_bonus = 0.0
        if lock_idx > see_idx + 5 and len(steady_errors) > 5:
            pre_lock_err = errors[lock_idx - 5: lock_idx]
            if np.max(pre_lock_err) > 2.0 and np.min(pre_lock_err) < 18.0:
                pre_lock_dt = times[lock_idx] - times[lock_idx - 5]
                if pre_lock_dt > 0.008:
                    bio_bonus += 30.0
            if errors[lock_idx] < 1.0 and np.mean(errors[lock_idx:lock_idx + 5]) < 1.0:
                bio_bonus -= 40.0
        bio_bonus = np.clip(bio_bonus, 0.0, 100.0)

        # ==================== 单 kill 综合分（用于 P95/worst-case）====================
        ttk_s = _linear_score(phase_time, ST['ttk_full'], ST['ttk_zero'])
        prec_s = _linear_score(headshot_err, ST['prec_full_px'], ST['prec_zero_px'])
        over_s = _linear_score(overshoot, ST['overshoot_full'], ST['overshoot_zero'])
        smooth_s = _linear_score(accel_peak, ST['accel_full'], ST['accel_zero'])
        per_kill_score = (
            ST['w_ttk'] * ttk_s + ST['w_precision'] * prec_s +
            ST['w_natural'] * natural_score + ST['w_bio'] * bio_bonus +
            ST['w_overshoot'] * over_s + ST['w_smoothness'] * smooth_s
        )
        if release_t > 0:
            tk_s = takeover_composite
            # 接管场景：takeover 权重替换 smoothness 的一部分
            per_kill_score = (
                ST['w_ttk'] * ttk_s + ST['w_precision'] * prec_s +
                ST['w_natural'] * natural_score + ST['w_bio'] * bio_bonus +
                ST['w_overshoot'] * over_s +
                (ST['w_smoothness'] + ST['w_takeover']) * 0.5 * smooth_s +
                (ST['w_smoothness'] + ST['w_takeover']) * 0.5 * tk_s
            )

        # ── 记录 ──
        mode_key = data['mode']
        metrics[mode_key].setdefault('phase_times', []).append(phase_time)
        metrics[mode_key]['steady_maes'].append(raw_mae)
        metrics[mode_key]['headshot_errors'].append(headshot_err)
        metrics[mode_key]['final_errors'].append(float(errors[-1]))
        metrics[mode_key]['natural_scores'].append(natural_score)
        metrics[mode_key]['bio_bonuses'].append(bio_bonus)
        metrics[mode_key]['vel_rms'].append(vel_rms)
        metrics[mode_key]['overshoots'].append(overshoot)
        metrics[mode_key]['accel_peaks'].append(accel_peak)
        metrics[mode_key]['takeover_delays'].append(takeover_delay)
        metrics[mode_key]['takeover_smoothness'].append(takeover_accel_peak)
        metrics[mode_key]['takeover_precision'].append(takeover_error_50ms)
        metrics[mode_key]['takeover_composites'].append(takeover_composite)
        metrics[mode_key]['initial_distances'].append(initial_dist)
        metrics[mode_key]['target_velocities'].append(target_vel_mean)
        metrics[mode_key]['per_kill_scores'].append(per_kill_score)

        status = "✅ 爆头"
        if phase_time > 0.28: status = "⚠️ 反应慢"
        elif phase_time < 0.12: status = "🤖 机器瞬锁"
        if raw_mae > 12.0: status += " ⚠️ 空枪"
        if natural_score < 60: status += " (高频抖动)"
        if bio_bonus > 20: status += " ✨微调"
        if overshoot > 15: status += " ↗过冲"
        if accel_peak > 1500: status += " ⚡加速脉冲"

        print(f"{kid:<6} | {'🧑人机' if mode_key == 'human_flick' else '🤖纯AI':<10} | "
              f"{phase_time:<9.3f} | {initial_dist:<6.0f} | {target_vel_mean:<6.0f} | "
              f"{raw_mae:<7.2f} | {overshoot:<6.1f} | {accel_peak:<7.0f} | "
              f"{natural_score:<7.1f} | {bio_bonus:<6.1f} | "
              f"{takeover_delay*1000:<8.1f} | {status}")

    # ════════════════════════════════════════════════════════════════════════════
    # 模块D：分层评分 — 按距离/速度桶统计，防"简单场景高分"过拟合
    # ════════════════════════════════════════════════════════════════════════════
    all_dists = (metrics['pure_ai']['initial_distances'] +
                 metrics['human_flick']['initial_distances'])
    all_vels = (metrics['pure_ai']['target_velocities'] +
                metrics['human_flick']['target_velocities'])
    all_phases = (metrics['pure_ai']['phase_times'] +
                  metrics['human_flick']['phase_times'])
    all_overshoots = (metrics['pure_ai']['overshoots'] +
                      metrics['human_flick']['overshoots'])
    all_accels = (metrics['pure_ai']['accel_peaks'] +
                  metrics['human_flick']['accel_peaks'])
    all_per_kill = (metrics['pure_ai']['per_kill_scores'] +
                    metrics['human_flick']['per_kill_scores'])

    # 距离桶
    def _dist_bucket(d):
        if d < ST['dist_near_max']: return 'near'
        if d > ST['dist_far_min']: return 'far'
        return 'mid'
    # 速度桶
    def _vel_bucket(v):
        if v < ST['vel_slow_max']: return 'static'
        if v > ST['vel_fast_min']: return 'fast'
        return 'slow'

    bucket_stats = {}
    for i, (d, v, ph, ov, ac, pks) in enumerate(zip(
        all_dists, all_vels, all_phases, all_overshoots, all_accels, all_per_kill
    )):
        bk = f"{_dist_bucket(d)}_{_vel_bucket(v)}"
        if bk not in bucket_stats:
            bucket_stats[bk] = {'count': 0, 'ttk': [], 'overshoot': [], 'accel': [], 'score': []}
        bucket_stats[bk]['count'] += 1
        bucket_stats[bk]['ttk'].append(ph)
        bucket_stats[bk]['overshoot'].append(ov)
        bucket_stats[bk]['accel'].append(ac)
        bucket_stats[bk]['score'].append(pks)

    # ════════════════════════════════════════════════════════════════════════════
    # 模块D：整体评分 — 新权重 + 鲁棒性
    # ════════════════════════════════════════════════════════════════════════════
    per_kill_ttk_scores = [_linear_score(t, ST['ttk_full'], ST['ttk_zero']) for t in all_phases]
    acq_score = safe_mean(per_kill_ttk_scores)

    mean_err = safe_mean(metrics['pure_ai']['headshot_errors'] + metrics['human_flick']['headshot_errors'])
    mean_final_error = safe_mean(
        metrics['pure_ai']['final_errors'] + metrics['human_flick']['final_errors']
    )
    precision_score = _linear_score(mean_err, ST['prec_full_px'], ST['prec_zero_px'])

    natural_avg = safe_mean(metrics['pure_ai']['natural_scores'] + metrics['human_flick']['natural_scores'])
    bio_avg = safe_mean(metrics['pure_ai']['bio_bonuses'] + metrics['human_flick']['bio_bonuses'])

    # 新维度评分
    mean_overshoot = safe_mean(all_overshoots)
    overshoot_score = _linear_score(mean_overshoot, ST['overshoot_full'], ST['overshoot_zero'])

    mean_accel = safe_mean(all_accels)
    smoothness_score = _linear_score(mean_accel, ST['accel_full'], ST['accel_zero'])

    # 接管延迟评分（仅 takeover 场景有值）
    takeover_rows = []
    for mode_metrics in metrics.values():
        takeover_rows.extend(
            (delay, accel, error_50ms)
            for delay, accel, error_50ms in zip(
                mode_metrics['takeover_delays'],
                mode_metrics['takeover_smoothness'],
                mode_metrics['takeover_precision'],
            )
            if delay > 0.0
        )
    has_takeover = bool(takeover_rows)
    takeover_delays = [row[0] for row in takeover_rows]
    takeover_accels = [row[1] for row in takeover_rows]
    takeover_errors_50ms = [row[2] for row in takeover_rows]
    mean_takeover_delay = safe_mean(takeover_delays)
    takeover_composites = (
        metrics['pure_ai']['takeover_composites']
        + metrics['human_flick']['takeover_composites']
    )
    takeover_score = safe_mean(
        [score for score in takeover_composites if score > 0.0]
    ) if has_takeover else 0.0
    mean_takeover_accel = safe_mean(takeover_accels)
    mean_takeover_error_50ms = safe_mean(takeover_errors_50ms)

    # 鲁棒性指标
    overall_per_kill = all_per_kill
    robustness_p95 = safe_percentile(overall_per_kill, 5)   # P5（worst 5%）—— 越高越好
    worst_case = float(np.min(overall_per_kill)) if overall_per_kill else 0.0
    cross_bucket_std = 0.0
    bucket_means = [safe_mean(bs['score']) for bs in bucket_stats.values() if bs['count'] > 0]
    if len(bucket_means) > 1:
        cross_bucket_std = float(np.std(bucket_means))

    # 整体评分（配置化权重）
    overall_score = (
        ST['w_ttk'] * acq_score +
        ST['w_precision'] * precision_score +
        ST['w_natural'] * natural_avg +
        ST['w_bio'] * bio_avg +
        ST['w_overshoot'] * overshoot_score +
        ST['w_smoothness'] * smoothness_score
    )
    if has_takeover:
        overall_score += ST['w_takeover'] * takeover_score
    else:
        # 非接管场景：takeover 权重分给 TTK 和精度
        overall_score += ST['w_takeover'] * 0.5 * acq_score
        overall_score += ST['w_takeover'] * 0.5 * precision_score

    # 跨桶一致性惩罚：桶间 std 过大说明参数在简单/难场景表现差异大
    consistency_penalty = float(np.clip(cross_bucket_std * 0.3, 0.0, 15.0))
    overall_score = float(np.clip(overall_score - consistency_penalty, 0.0, 100.0))

    analysis = {
        # ── 向后兼容键（simulation.py 依赖）──
        'total_kills': len(metrics['pure_ai']['steady_maes']) + len(metrics['human_flick']['steady_maes']),
        'overall_score': float(overall_score),
        'acq_score': float(acq_score),
        'precision_score': float(precision_score),
        'natural_score': float(natural_avg),
        'bio_bonus': float(bio_avg),
        'ttk_base_score': float((acq_score + precision_score) / 2),
        'jitter_penalty': float(max(0.0, (100 - natural_avg) * 1.4)),
        'mae_penalty': float(max(0.0, (mean_err - 8.0) * 3.0)),
        'metrics': metrics,
        # ── 模块D 新增键 ──
        'overshoot_score': float(overshoot_score),
        'smoothness_score': float(smoothness_score),
        'takeover_score': float(takeover_score),
        'mean_overshoot': float(mean_overshoot),
        'mean_accel_peak': float(mean_accel),
        'mean_takeover_delay': float(mean_takeover_delay),
        'takeover_delay_p50': safe_percentile(takeover_delays, 50),
        'takeover_delay_p95': safe_percentile(takeover_delays, 95),
        'takeover_delay_max': max(takeover_delays, default=0.0),
        'mean_takeover_accel_peak': float(mean_takeover_accel),
        'takeover_accel_p95': safe_percentile(takeover_accels, 95),
        'takeover_accel_max': max(takeover_accels, default=0.0),
        'mean_takeover_error_50ms': float(mean_takeover_error_50ms),
        'takeover_error_50ms_p95': safe_percentile(takeover_errors_50ms, 95),
        'takeover_samples': len(takeover_rows),
        'mean_error': float(mean_err),
        'mean_final_error': float(mean_final_error),
        'robustness_p5': float(robustness_p95),     # P5 = worst 5%
        'worst_case': float(worst_case),
        'cross_bucket_std': float(cross_bucket_std),
        'consistency_penalty': float(consistency_penalty),
        'bucket_stats': bucket_stats,
        'per_kill_scores': overall_per_kill,
    }

    # ── 打印分层报告 ──
    print("\n" + "=" * 95)
    print(f"🎯 Valorant Biomimetic v5.0 Final Score: {overall_score:.1f}/100")
    print(f"   TTK: {acq_score:.1f} | Prec: {precision_score:.1f} | Natural: {natural_avg:.1f} | Bio: +{bio_avg:.1f}")
    print(f"   Overshoot: {overshoot_score:.1f} (mean {mean_overshoot:.1f}px) | "
          f"Smooth: {smoothness_score:.1f} (peak {mean_accel:.0f}ct/s²) | "
          f"Takeover: {takeover_score:.1f} ({mean_takeover_delay*1000:.1f}ms)")
    if has_takeover:
        print(
            f"   Handoff 100ms accel: {mean_takeover_accel:.0f}ct/s² | "
            f"50ms error: {mean_takeover_error_50ms:.1f}px | "
            "score weights: delay 40% / smooth 30% / precision 30%"
        )
    print(f"   Robustness: P5={robustness_p95:.1f} | worst={worst_case:.1f} | "
          f"cross-bucket σ={cross_bucket_std:.1f} (penalty -{consistency_penalty:.1f})")
    if bucket_stats:
        print("   分层桶统计:")
        for bk, bs in sorted(bucket_stats.items()):
            if bs['count'] > 0:
                print(f"     {bk:<14}: n={bs['count']:<3} TTK={safe_mean(bs['ttk']):.3f}s "
                      f"over={safe_mean(bs['overshoot']):.1f}px accel={safe_mean(bs['accel']):.0f} "
                      f"score={safe_mean(bs['score']):.1f}")
    print("=" * 95)

    return analysis


def assess_takeover_level(analysis: Dict) -> Dict:
    """Grade shared-control behavior without claiming real-player equivalence."""
    if int(analysis.get('takeover_samples', 0)) == 0:
        return {
            'grade': 'N/A',
            'label': '未运行接管场景',
            'gates': [],
            'scope': '本次数据不包含人机接管样本。',
        }

    elapsed = float(analysis.get('sim_elapsed', float('inf')))
    kills = int(analysis.get('total_kills', 0))
    gates = [
        ('吞吐', kills >= 30 and bool(analysis.get('completed_target')) and elapsed <= 15.0,
         f'{kills}球/{elapsed:.3f}s', '30球<=15s'),
        ('样本量', int(analysis['takeover_samples']) >= 20,
         str(int(analysis['takeover_samples'])), '>=20'),
        ('平均延迟', analysis['mean_takeover_delay'] <= 0.250,
         f"{analysis['mean_takeover_delay'] * 1000:.1f}ms", '<=250ms'),
        ('P95延迟', analysis['takeover_delay_p95'] <= 0.350,
         f"{analysis['takeover_delay_p95'] * 1000:.1f}ms", '<=350ms'),
        ('P95加速度', analysis['takeover_accel_p95'] <= 1500.5,
         f"{analysis['takeover_accel_p95']:.0f}ct/s^2", '<=1500ct/s^2'),
        ('最终误差', analysis['mean_final_error'] <= 8.0,
         f"{analysis['mean_final_error']:.2f}px", '<=8px'),
        ('50ms误差P95', analysis['takeover_error_50ms_p95'] <= 100.0,
         f"{analysis['takeover_error_50ms_p95']:.1f}px", '<=100px'),
        ('最差5%得分', analysis['robustness_p5'] >= 50.0,
         f"{analysis['robustness_p5']:.1f}", '>=50'),
    ]
    if all(passed for _, passed, _, _ in gates):
        grade, label = 'S', '研究目标级'
    else:
        competitive = (
            kills >= 30
            and bool(analysis.get('completed_target'))
            and elapsed <= 15.0
            and int(analysis['takeover_samples']) >= 20
            and analysis['mean_takeover_delay'] <= 0.300
            and analysis['takeover_delay_p95'] <= 0.500
            and analysis['takeover_accel_p95'] <= 2000.0
            and analysis['mean_final_error'] <= 10.0
            and analysis['takeover_error_50ms_p95'] <= 130.0
            and analysis['robustness_p5'] >= 35.0
        )
        usable = (
            bool(analysis.get('completed_target'))
            and elapsed <= 18.0
            and analysis['mean_takeover_delay'] <= 0.400
            and analysis['takeover_accel_p95'] <= 3500.0
            and analysis['mean_final_error'] <= 12.0
        )
        if competitive:
            grade, label = 'A', '高水平合成接管'
        elif usable:
            grade, label = 'B', '可用接管'
        else:
            grade, label = 'C', '未达稳定接管门槛'
    return {
        'grade': grade,
        'label': label,
        'gates': gates,
        'scope': '这是合成场景等级；缺少真人轨迹对照时，不等同于职业选手水平。',
    }


def plot_diagnostics(logs: List[Dict], analysis: Dict, title_prefix: str = ""):
    """绘制带 NAN 断点的图表"""
    plot_times = []
    plot_targets_x, plot_targets_y = [], []
    plot_cross_x,   plot_cross_y   = [], []

    last_kid = logs[0]['kill_id']
    for log in logs:
        if log['kill_id'] != last_kid:
            for arr in [plot_times, plot_targets_x, plot_targets_y, plot_cross_x, plot_cross_y]:
                arr.append(np.nan)
            last_kid = log['kill_id']
        plot_times.append(log['t'])
        plot_targets_x.append(log['target'][0])
        plot_targets_y.append(log['target'][1])
        plot_cross_x.append(log['crosshair'][0])
        plot_cross_y.append(log['crosshair'][1])

    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
    fig.suptitle(
        f"{title_prefix} test result | kills: {analysis['total_kills']} | "
        f"score: {analysis['overall_score']:.1f}  "
        f"(TTK: {analysis['ttk_base_score']:.1f} "
        f"- jitter: {analysis['jitter_penalty']:.1f} "
        f"- MAE: {analysis['mae_penalty']:.1f})",
        fontsize=14, fontweight='bold'
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(plot_targets_x, plot_targets_y, 'r--', alpha=0.5, label='Target')
    ax1.plot(plot_cross_x,   plot_cross_y,   'b-',  alpha=0.8, label='Crosshair')
    ax1.set_title('2D Target Switching Map')
    ax1.axis('equal')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(plot_times, plot_targets_x, 'r--', alpha=0.6)
    ax2.plot(plot_times, plot_cross_x,   'g-')
    ax2.set_title('X-Axis Tracking')

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(plot_times, plot_targets_y, 'r--', alpha=0.6)
    ax3.plot(plot_times, plot_cross_y,   'b-')
    ax3.set_title('Y-Axis Tracking')

    ax4 = fig.add_subplot(gs[1, 1])
    err = np.sqrt(
        (np.array(plot_targets_x) - np.array(plot_cross_x)) ** 2 +
        (np.array(plot_targets_y) - np.array(plot_cross_y)) ** 2
    )
    ax4.plot(plot_times, err, color='orange')
    ax4.axhline(25.0, color='green', linestyle='--', label='deadzone')
    ax4.set_title('Error Distance & Deadzone')

    plt.show()
def print_analysis_report(analysis: Dict, prefix: str = ""):
    m = analysis['metrics']
    takeover_assessment = assess_takeover_level(analysis)
    analysis['takeover_assessment'] = takeover_assessment
    print("\n" + "=" * 75)
    print(f"🎮 {prefix} Valorant Biomimetic v3.0 深度诊断报告")
    print("=" * 75)
    print(f"【综合得分】: {analysis['overall_score']:8.1f} / 100")
    if 'sim_elapsed' in analysis:
        elapsed = float(analysis['sim_elapsed'])
        kills = int(analysis.get('total_kills', 0))
        passed = bool(analysis.get('completed_target', False)) and elapsed <= 15.0
        print(
            f"  任务完成度   : {kills:2d} 球 / {elapsed:6.3f}s  "
            f"{'PASS' if passed else 'FAIL'} (目标: 30球≤15s)"
        )
    if takeover_assessment['grade'] != 'N/A':
        passed_gates = sum(
            1 for _, gate_passed, _, _ in takeover_assessment['gates']
            if gate_passed
        )
        print(
            f"【接管等级】: {takeover_assessment['grade']} - "
            f"{takeover_assessment['label']} "
            f"(S级硬门槛 {passed_gates}/{len(takeover_assessment['gates'])})"
        )
        print(
            "  延迟分布     : "
            f"P50={analysis['takeover_delay_p50'] * 1000:.1f}ms  "
            f"P95={analysis['takeover_delay_p95'] * 1000:.1f}ms  "
            f"max={analysis['takeover_delay_max'] * 1000:.1f}ms"
        )
        print(
            "  接管加速度   : "
            f"mean={analysis['mean_takeover_accel_peak']:.0f}  "
            f"P95={analysis['takeover_accel_p95']:.0f}  "
            f"max={analysis['takeover_accel_max']:.0f} ct/s²"
        )
        print("  S级硬门槛:")
        for name, gate_passed, actual, target in takeover_assessment['gates']:
            print(
                f"    {'PASS' if gate_passed else 'FAIL':4s}  "
                f"{name:<12s} {actual:<16s} 目标 {target}"
            )
        print(f"  证据边界     : {takeover_assessment['scope']}")
    print(f"  TTK (Acq/Rec): {analysis['acq_score']:6.1f}   精度: {analysis['precision_score']:6.1f}")
    print(f"  平滑与急停   : {analysis['natural_score']:6.1f}   微调奖励: +{analysis['bio_bonus']:.1f}")
    print("-" * 75)

    if m['pure_ai'].get('phase_times'):
        print(f"🤖 [纯 AI 模式] (初始 dist 60~256 px)")
        print(f"   - 平均 TTK(see→lock): {safe_mean(m['pure_ai']['phase_times']):.3f}s | "
              f"平均误差: {safe_mean(m['pure_ai']['steady_maes']):.2f}px | "
              f"自然度: {safe_mean(m['pure_ai']['natural_scores']):.1f} | "
              f"微调: +{safe_mean(m['pure_ai']['bio_bonuses']):.1f}")

    if m['human_flick'].get('phase_times'):
        print(f"🧑 [人机协同模式] (人类甩 1000+→256px 内，AI 接管)")
        print(f"   - 平均 TTK(see→lock): {safe_mean(m['human_flick']['phase_times']):.3f}s | "
              f"平均误差: {safe_mean(m['human_flick']['steady_maes']):.2f}px | "
              f"自然度: {safe_mean(m['human_flick']['natural_scores']):.1f} | "
              f"微调: +{safe_mean(m['human_flick']['bio_bonuses']):.1f}")

def run_simulation_with_diagnostics(
    plot: bool = True,
    verbose: bool = True,
    use_fixed: bool = True,
    duration: float = 60.0,
    params: Dict = None,
    scenario: BaseScenario = None,
) -> Tuple[float, float, Dict]:
    try:
        from test.sim_agent import SimAIAgent
    except ModuleNotFoundError:
        from sim_agent import SimAIAgent
    from config import config
    config.reload()

    wm_align = config.getbool("Test", "align_worldmodel_to_zero_sim_delay", False)
    has_params = params is not None
    original_getfloat = config.getfloat
    original_getint = config.getint
    original_getbool = config.getbool

    def combined_getfloat(section, key, fallback=None):
        if has_params:
            # 模块E：支持 section-qualified 参数名（如 "Kalman.R", "IMM.markov_diag"）
            qualified = f"{section}.{key}"
            if qualified in params:
                return float(params[qualified])
            # 向后兼容：bare key（cipher_* 参数）
            if key in params:
                return float(params[key])
        if wm_align and section == "WorldModel":
            if key in ("moonlight_latency_ms", "virtualhere_latency_ms"):
                return 0.0
            if key == "base_hardware_lag":
                return min(float(original_getfloat(section, key, fallback)), 0.003)
        if has_params and key == "base_hardware_lag" and "fixed_lead_time" in params:
            return float(params["fixed_lead_time"])
        return original_getfloat(section, key, fallback)

    def combined_getint(section, key, fallback=None):
        if has_params:
            # 模块E：支持 section-qualified 参数名
            qualified = f"{section}.{key}"
            if qualified in params:
                return int(params[qualified])
            if key in params:
                return int(params[key])
        return original_getint(section, key, fallback)

    def combined_getbool(section, key, fallback=False):
        if wm_align and section == "WorldModel" and key in (
            "adaptive_latency_enable",
            "wan_mode",
        ):
            return False
        return original_getbool(section, key, fallback)

    if wm_align or has_params:
        config.getfloat = combined_getfloat
        config.getbool = combined_getbool
        if has_params:
            config.getint = combined_getint

    try:
        agent = SimAIAgent(scenario=scenario)
        fixed_dt, logs = 0.002, []

        if verbose:
            print(f"▶ 正在载入实战模拟环境...")
            if wm_align:
                print(
                    "  [Test] align_worldmodel_to_zero_sim_delay=ON → "
                    "WorldModel 串流/硬件延迟已压到≈0（仅本轮仿真）"
                )

        max_steps = max(10000, int(duration / fixed_dt) * 20)
        while (not agent.is_done and agent.sim_time < duration
               and agent.sim_steps < max_steps):
            agent.step()
            # ── 模块D：扩展日志采集（alpha/takeover_state/conf/phase/num_targets）──
            _ctrl = agent.world_model.controller
            _dets = agent.ctx.targets
            logs.append({
                't':        agent.sim_time,
                'wall_t':   agent._perf_time,
                'target':   agent.enemy_pos.copy(),
                'crosshair':agent.crosshair_pos.copy(),
                'kill_id':  agent.kill_count,
                'mode':     agent.chase_mode,
                'is_valid': agent.ctx.p_predict is not None,
                'ai_factor':getattr(agent, 'last_ai_factor', 1.0),
                'takeover_release_time': getattr(agent, 'takeover_release_time', 0.0),
                # 新增：α 连续融合状态
                'alpha':    float(getattr(_ctrl, 'alpha', 1.0)),
                'takeover_state': str(getattr(_ctrl, 'takeover_state', 'ACTIVE_LOCK')),
                'handoff_generation': int(
                    getattr(_ctrl, '_handoff_generation', 0)
                ),
                'ctrl_mode': str(getattr(_ctrl, 'mode', 'track')),
                'ctrl_velocity': np.asarray(
                    getattr(_ctrl, 'crosshair_velocity', (0.0, 0.0)),
                    dtype=np.float64,
                ).copy(),
                'px_to_ct': float(getattr(_ctrl, '_px_to_ct', 1.0)),
                # 新增：检测置信度 + 目标数（多目标场景）
                'conf':     float(_dets[0].conf) if _dets else 0.0,
                'num_targets': len(_dets) if _dets else 0,
                # 新增：目标速度（用于分层评分）
                'target_vel': float(np.linalg.norm(agent.enemy_vel)) if hasattr(agent, 'enemy_vel') else 0.0,
            })

        # kill_id denotes the target currently being attempted. After N
        # completed kills, records with kill_id == N belong to the unfinished
        # next target and must not be scored as a completed kill.
        completed_logs = [
            log for log in logs if int(log['kill_id']) < int(agent.kill_count)
        ]
        analysis = analyze_tracking_quality(completed_logs)
        analysis['total_kills'] = int(agent.kill_count)
        analysis['sim_elapsed'] = float(agent.sim_time)
        analysis['completed_target'] = bool(agent.is_done)
        scenario_name = type(scenario).__name__ if scenario else "BallTrackingScenario"
        tag = "Optuna调参" if params else f"{scenario_name}"

        if verbose:
            print_analysis_report(analysis, tag)
        if plot:
            plot_diagnostics(logs, analysis, tag)

        return 0.0, 0.0, analysis
    finally:
        if wm_align or has_params:
            config.getfloat = original_getfloat
            config.getbool = original_getbool
            if has_params:
                config.getint = original_getint




def run_takeover_test(
    plot: bool = True,
    verbose: bool = True,
    duration: float = 45.0,
    max_kills: int = 25,
) -> Dict:
    """
    运行 AI 接管能力专项测试。

    与默认的 BallTrackingScenario 不同:
      - 目标 spawn 距离 200-600px（始终在 AI FOV 内）
      - 人类故意打偏: undershoot(15-40%短) / overshoot(5-20%过) / near_miss(10-35px擦边)
      - 测试人类→AI 中途接管的平滑度与延迟
    """
    scenario = TakeoverScenario(max_kills=max_kills)
    _, _, analysis = run_simulation_with_diagnostics(
        plot=plot,
        verbose=verbose,
        duration=duration,
        scenario=scenario,
    )

    # 打印接管专项报告
    scenario.print_takeover_report()

    return analysis


def run_pure_ai_ball_test(
    plot: bool = True,
    verbose: bool = True,
    duration: float = 60.0,
    max_kills: int = 30,
) -> Dict:
    """
    纯 AI 瞄准专项测试（Neon 身法单球，无 human_flick）。

    在 config.ini 中切换 [WorldModel] predict_ahead / ctrl_lead_enable /
    adaptive_latency_enable 等，对比本场景的得分与 TTK，即可评估「开预测」的收益。
    """
    scenario = PureAIBallScenario(max_kills=max_kills)
    _, _, analysis = run_simulation_with_diagnostics(
        plot=plot,
        verbose=verbose,
        duration=duration,
        scenario=scenario,
    )
    return analysis


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="测试场景运行器")
    parser.add_argument(
        "--scenario", type=str, default="ball",
        choices=["ball", "takeover", "pure_ai"],
        help="场景: ball=混合模式小球, takeover=人机接管, pure_ai=纯AI瞄准(测预测)",
    )
    parser.add_argument(
        "--kills", type=int, default=30,
        help="最大击杀数 (takeover / pure_ai；混合 ball 场景仍用内置 30)",
    )
    parser.add_argument(
        "--duration", type=float, default=60.0,
        help="仿真时长上限 (秒)。takeover 可缩短如 45",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="评分完成后显示诊断图（默认不显示，便于自动化运行）",
    )
    parser.add_argument(
        "--seed", type=int, default=7,
        help="随机种子（默认 7，保证评分可复现）",
    )
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.scenario == "takeover":
        run_takeover_test(
            plot=args.plot, duration=args.duration, max_kills=args.kills
        )
    elif args.scenario == "pure_ai":
        run_pure_ai_ball_test(
            plot=args.plot, duration=args.duration, max_kills=args.kills
        )
    else:
        run_simulation_with_diagnostics(
            plot=args.plot, use_fixed=True, duration=args.duration
        )
