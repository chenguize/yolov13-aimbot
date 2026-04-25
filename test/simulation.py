# simulation.py  ── CIPHER v1.0 CMA-ES 参数搜索（warm-start 版）
# ═══════════════════════════════════════════════════════════════════════════════
# Optuna + CMA-ES 炼丹炉，搜索 PROController (CIPHER) 的全部 cipher_* 参数。
#
# 配套评分系统 (test/control.py) 为 v4.1：
#   · 两种模式（pure_ai / human_flick）统一用 phase_times = see_idx → lock_idx
#   · TTK 满分阈值 0.18s，零分阈值 0.40s
#   · 权重 TTK 40% / Precision 20% / Natural 25% / Bio 15%
#   · pure_ai 的 spawn_radius 已限制为 60~256px（sim_agent），反映 AI 只接管
#     "256px 内瞄准"的实战语境
#
# 配套算法改动（不调参，只改架构）：
#   · PROController.get_expected_lead() —— 把 BALLISTIC 剩余执行时间暴露给 WM
#   · WorldModel 把 ctrl_lead 叠加到 smart_lead，BALLISTIC 瞄准 "动作结束时刻"
#   · TargetState 协方差合理初始化，避免刚 spawn 时 cov_penalty 被虚假压低
#
# 与上一版的根本差异：
#   · warm-start：用 pro_controller.py 的默认值作为 CMA-ES 的 x0；
#   · enqueue baseline：第 0 号 trial 强制用 baseline 跑一次，锁住一个保底分；
#   · bounds 全部在 baseline 周边 ×0.5 / ×2.0 范围，避免远离局部最优；
#   · 10 个验证种子 + 多段场景，降低目标函数噪声；
#   · objective 只在 trial 结束后算 robust（减小 pruner 误判）。
#
# ─── 使用 ────────────────────────────────────────────────────────────────
#   python test/simulation.py                 ← 多进程炼丹
#   python test/simulation.py --single        ← 单进程跑（方便 debug）
#   python test/simulation.py --resume        ← 续跑现有 study（不重建 sampler）
# ═══════════════════════════════════════════════════════════════════════════════
import argparse
import contextlib
import io
import multiprocessing
import os
import random as pyrand
import sys

import numpy as np
import optuna

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from test import control as c


# ══════════════════════════════════════════════════════════════════════════════
# § 1  BASELINE — 手调 + 算法重构后的起点，用作 warm-start
# ══════════════════════════════════════════════════════════════════════════════
# 这些值必须与 pro_controller.py 里 config.getfloat(…, default) 的默认值一致，
# 否则 "trial==baseline" 无法复现 bench 基线分数。
BASELINE = {
    "cipher_max_speed":       10000.0,
    "cipher_ff_gain":             0.95,
    "cipher_ff_acc_sec":          0.022,
    "cipher_thresh_high_px":     50.0,
    "cipher_thresh_low_px":      35.0,
    "cipher_deadzone_scale":      0.035,

    "cipher_k_pursuit":         135.0,
    "cipher_k_flick":           460.0,
    "cipher_b_pursuit":           0.32,
    "cipher_k_correction":       55.0,
    "cipher_b_correction":        0.18,

    "cipher_ff_speed_knee":     200.0,
    "cipher_ff_speed_scale":   1800.0,
    "cipher_ff_scale_min":        0.35,

    "cipher_fitts_a":             0.028,
    "cipher_fitts_b":             0.040,
    "cipher_fitts_sigma":         0.06,
    "cipher_fitts_T_min":         0.055,
    "cipher_fitts_T_max":         0.28,

    "cipher_undershoot_loc":      0.15,
    "cipher_undershoot_sigma":    1.8,

    "cipher_ramp_ms":            42.0,
    "cipher_ramp_min":            0.85,
    "cipher_tau_arm":             0.012,
    "cipher_tau_wrist":           0.008,

    "cipher_ou_sigma_ball":       0.18,
    "cipher_ou_sigma_track":      0.38,
    "cipher_drift_sigma":         0.10,
    "cipher_ou_vel_scale":      120.0,
    "cipher_drift_pos_scale":     0.025,

    "cipher_prog_interval":       0.06,

    # Micro-adjust entry（仅 human_flick 模式激活；pure_ai 被跳过以求最快 TTK）
    # entry_ticks 是 int（getint 读取），override 需确保 int 路径也被拦截
    "cipher_entry_ticks":        12,
    "cipher_entry_dz_px":         2.5,
}


# ══════════════════════════════════════════════════════════════════════════════
# § 2  搜索空间 —— bounds 统一围绕 BASELINE 设置（0.5× … 2×），
#        用 suggest_float(low, high) 指定；对数参数加 log=True
# ══════════════════════════════════════════════════════════════════════════════
def _bounds_for(name: str, baseline: float, *, log: bool = False,
                ratio_lo: float = 0.5, ratio_hi: float = 2.0,
                abs_lo: float | None = None, abs_hi: float | None = None
                ) -> tuple[float, float, bool]:
    lo = baseline * ratio_lo
    hi = baseline * ratio_hi
    if abs_lo is not None:
        lo = max(lo, abs_lo)
    if abs_hi is not None:
        hi = min(hi, abs_hi)
    return lo, hi, log


# 每个参数：(low, high, log-scale?) —— 全部在 BASELINE 周边取
PARAM_BOUNDS: dict[str, tuple[float, float, bool]] = {
    # 速度 / 前馈
    "cipher_max_speed":       _bounds_for("cipher_max_speed",     BASELINE["cipher_max_speed"],     ratio_lo=0.6, ratio_hi=1.6),
    "cipher_ff_gain":         (0.80, 1.00, False),
    "cipher_ff_acc_sec":      (0.000, 0.050, False),

    # 阈值 / 死区
    "cipher_thresh_high_px":  (35.0, 70.0, False),
    "cipher_thresh_low_px":   (20.0, 50.0, False),
    "cipher_deadzone_scale":  (0.015, 0.06, False),

    # 阻抗刚度 / 阻尼
    "cipher_k_pursuit":       (70.0,  220.0, False),
    "cipher_k_flick":         (300.0, 650.0, False),
    "cipher_b_pursuit":       (0.15,   0.60, False),
    "cipher_k_correction":    (30.0,  100.0, False),
    "cipher_b_correction":    (0.08,   0.50, False),

    # 前馈速度门控
    "cipher_ff_speed_knee":   (100.0,  500.0, False),
    "cipher_ff_speed_scale":  (800.0, 3000.0, False),
    "cipher_ff_scale_min":    (0.15,    0.60, False),

    # Fitts' Law
    "cipher_fitts_a":         (0.015, 0.050, False),
    "cipher_fitts_b":         (0.025, 0.065, False),
    "cipher_fitts_sigma":     (0.02,  0.12, False),
    "cipher_fitts_T_min":     (0.035, 0.090, False),
    "cipher_fitts_T_max":     (0.22,  0.35, False),

    # Undershoot (v1.1: zero-mean bias, 搜索范围限制在 ±0.15·head_r 内)
    "cipher_undershoot_loc":    (0.0,  0.15, False),
    "cipher_undershoot_sigma":  (0.8,  3.5,  False),

    # 冷启动
    "cipher_ramp_ms":           (25.0, 75.0, False),
    "cipher_ramp_min":           (0.70, 1.00, False),

    # 神经肌肉滤波（生理上限非常严格，不让 CMA-ES 乱跑）
    "cipher_tau_arm":            (0.008, 0.020, False),
    "cipher_tau_wrist":          (0.005, 0.015, False),

    # OU 噪声 —— 上一轮 CMA-ES 被这里坑惨（σ=0.82），压紧上限
    "cipher_ou_sigma_ball":      (0.08, 0.40, False),
    "cipher_ou_sigma_track":     (0.15, 0.55, False),
    "cipher_drift_sigma":        (0.04, 0.18, False),
    "cipher_ou_vel_scale":       (40.0, 180.0, False),
    "cipher_drift_pos_scale":    (0.005, 0.040, False),

    # Motor program 重规划冷却
    "cipher_prog_interval":      (0.03, 0.12, False),
}


# ══════════════════════════════════════════════════════════════════════════════
# § 3  验证种子 & 评估
# ══════════════════════════════════════════════════════════════════════════════
# 10 seed × 30s，足够覆盖 human_flick / pure_ai 的不同组合，并把 std 压下去。
VALIDATION_SEEDS = (11, 23, 47, 71, 103, 137, 199, 251, 317, 401)


def run_once(params: dict, seed: int, duration: float = 30.0) -> dict:
    np.random.seed(seed)
    pyrand.seed(seed)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _, _, analysis = c.run_simulation_with_diagnostics(
            duration=duration,
            plot=False,
            verbose=False,
            use_fixed=True,
            params=params,
        )
    return analysis


def _safe(a: dict, key: str, default=0.0) -> float:
    return float(a.get(key, default))


def evaluate(params: dict, seeds=VALIDATION_SEEDS) -> tuple[float, float, list[float]]:
    """跑一组 seeds，返回 (mean, std, [per_seed_scores])."""
    scores: list[float] = []
    for seed in seeds:
        a = run_once(params, seed)
        overall = _safe(a, "overall_score")
        total   = int(a.get("total_kills", 0))
        prec    = _safe(a, "precision_score")

        # 软惩罚：击杀过少 / 精度塌陷，避免 CMA-ES 把"0 kill 0 precision"当作平局
        penalty = 0.0
        if total < 8:
            penalty += (8 - total) * 6.0
        if prec < 3.0:
            penalty += (3.0 - prec) * 3.0

        scores.append(overall - penalty)
    return float(np.mean(scores)), float(np.std(scores)), scores


# ══════════════════════════════════════════════════════════════════════════════
# § 4  Optuna Objective
# ══════════════════════════════════════════════════════════════════════════════
def objective(trial: optuna.Trial) -> float:
    params: dict[str, float] = {}
    for name, (lo, hi, log_scale) in PARAM_BOUNDS.items():
        params[name] = trial.suggest_float(name, lo, hi, log=log_scale)

    # 物理 / 逻辑约束剪枝
    if params["cipher_thresh_low_px"] >= params["cipher_thresh_high_px"]:
        raise optuna.exceptions.TrialPruned()
    if params["cipher_fitts_T_min"] >= params["cipher_fitts_T_max"]:
        raise optuna.exceptions.TrialPruned()

    mean_s, std_s, per_seed = evaluate(params)
    # 鲁棒目标：减小 std 权重（0.8 对 10 seed 已经够了）
    robust = mean_s - 0.5 * std_s
    trial.set_user_attr("mean", mean_s)
    trial.set_user_attr("std", std_s)
    trial.set_user_attr("per_seed", per_seed)
    return -robust


# ══════════════════════════════════════════════════════════════════════════════
# § 5  研究对象构造：warm-start + enqueue baseline
# ══════════════════════════════════════════════════════════════════════════════
def build_study(db_url: str, study_name: str, resume: bool) -> optuna.Study:
    # sampler 的 x0 就是 baseline；sigma0 ≈ bounds 宽度的 1/6
    x0 = {k: BASELINE[k] for k in PARAM_BOUNDS.keys()}
    sigma0 = min(
        (hi - lo) / 6.0
        for (lo, hi, _) in PARAM_BOUNDS.values()
    )
    sampler = optuna.samplers.CmaEsSampler(
        x0=x0,
        sigma0=sigma0,
        seed=42,
        n_startup_trials=0,           # 直接从 x0 开始，不要再撒随机
        warn_independent_sampling=False,
    )
    pruner = optuna.pruners.NopPruner()   # 10-seed 评估已经稳定，别剪枝

    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=db_url,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=resume,
        )
    except optuna.exceptions.DuplicatedStudyError:
        print(f"\n❌ study_name '{study_name}' 已存在于 {db_url}\n")
        print("  选项：")
        print("    A) 续跑旧曲线（算法架构未改动时推荐）：")
        print("       python test/simulation.py --resume")
        print("    B) 删除旧 study 重来（算法有大改时推荐）：")
        print(f"       optuna delete-study --study-name {study_name} --storage {db_url}")
        print("    C) 改 simulation.py 里的 study_name 再跑")
        raise SystemExit(1)

    # 第 0 号 trial：强制用 baseline 跑一次（仅当 study 为空时）
    if len(study.trials) == 0:
        study.enqueue_trial({k: BASELINE[k] for k in PARAM_BOUNDS.keys()})

    return study


# ══════════════════════════════════════════════════════════════════════════════
# § 6  Worker / entrypoint
# ══════════════════════════════════════════════════════════════════════════════
def optimize_worker(db_url: str, study_name: str, n_trials: int):
    study = optuna.load_study(study_name=study_name, storage=db_url)
    study.optimize(objective, n_trials=n_trials)


def _print_best_params(study: optuna.Study):
    print("\n" + "=" * 78)
    best = study.best_trial
    m  = best.user_attrs.get("mean",  float("nan"))
    s  = best.user_attrs.get("std",   float("nan"))
    ps = best.user_attrs.get("per_seed", [])
    print(f"🏆 CIPHER CMA-ES 完成：鲁棒分 {-study.best_value:.2f} "
          f"(mean={m:.2f}, std={s:.2f})")
    print(f"   per-seed: {['%.1f' % x for x in ps]}")
    print("=" * 78)

    baseline_diff = []
    max_k = max(len(k) for k in best.params)
    for k in sorted(best.params):
        v  = best.params[k]
        bv = BASELINE[k]
        delta_pct = (v - bv) / max(abs(bv), 1e-9) * 100
        flag = "↑" if delta_pct > 5 else ("↓" if delta_pct < -5 else " ")
        baseline_diff.append((k, v, bv, delta_pct, flag))
        print(f"  {flag} {k:<{max_k}} = {v:<12.6g}  (baseline {bv:g}, Δ{delta_pct:+.0f}%)")

    print("\n— 粘贴进 config.ini 的 [Controller] section —")
    for k, v, _, _, _ in baseline_diff:
        print(f"{k} = {v:.6g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true", help="单进程跑，方便 debug")
    parser.add_argument("--resume", action="store_true", help="续跑现有 study")
    parser.add_argument("--trials", type=int, default=400, help="总 trial 数")
    parser.add_argument("--workers", type=int, default=0, help="worker 数量（0 = 自动）")
    args = parser.parse_args()

    db_url     = "sqlite:///aimbot_cipher_v1.db"
    # v4.1 架构改动（WorldModel↔Controller 时间轴对齐、Kalman cov 合理初值、
    # pure_ai 限制 256px、评分统一 phase_times）后，旧 study 的 trial 参数与
    # 当前代码已不同源，不应 resume。用新 study_name 起新曲线；旧 study 仍然
    # 保留在 db 里方便对比。
    study_name = "cipher_v4_1_archfix"

    print("\n🚀 启动 CMA-ES 炼丹炉 (CIPHER v1.0, warm-start)")
    print(f"   x0 = BASELINE（{len(PARAM_BOUNDS)} 维），10 seeds × 30s")
    print(f"   storage = {db_url}, study = {study_name}\n")

    # 主进程先建 study（包含 enqueue baseline），worker 用 load_study
    study = build_study(db_url, study_name, resume=args.resume)

    if args.single or args.workers == 1:
        study.optimize(objective, n_trials=args.trials)
    else:
        n_workers = args.workers or max(1, multiprocessing.cpu_count() // 2)
        per = args.trials // n_workers
        procs = []
        for _ in range(n_workers):
            p = multiprocessing.Process(
                target=optimize_worker,
                args=(db_url, study_name, per),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    _print_best_params(study)


if __name__ == "__main__":
    main()
