# simulation.py  ── CIPHER v2.0 防过拟合参数搜索（模块E 全面重构）
# ═══════════════════════════════════════════════════════════════════════════════
# 核心升级（解决"自动化调参太受测试任务影响"问题）：
#   1. 场景随机化：每 trial 用不同场景配置（距离/速度/时序随机化）
#   2. 多场景评估：每 trial 跑 2-3 种场景类型，防"单场景过拟合"
#   3. 留出测试集：OPT_SEEDS 优化 / HELD_OUT_SEEDS 最终验证
#   4. 两阶段剪枝：3-seed 粗筛 + 8-seed 精评 + MedianPruner
#   5. 搜索空间扩展：controller(37) + IMM(5) + ReID(5) + α(2) = 49 维
#   6. 物理约束：临界阻尼 ζ = B/(2√(K·τ)) ∈ [0.3, 1.5]
#   7. 鲁棒目标：mean - 0.5·std + P5 惩罚 + 跨桶一致性惩罚
#
# ─── 使用 ────────────────────────────────────────────────────────────────
#   python test/simulation.py                    ← 默认多场景防过拟合炼丹
#   python test/simulation.py --scenario ball    ← 单场景调试
#   python test/simulation.py --single           ← 单进程 debug
#   python test/simulation.py --resume           ← 续跑现有 study
#   python test/simulation.py --validate-only    ← 仅用留出集验证 best params
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
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)

from test import control as c


# ══════════════════════════════════════════════════════════════════════════════
# § 1  BASELINE — 手调 + 算法重构后的起点，用作 warm-start
# ══════════════════════════════════════════════════════════════════════════════
# Controller 参数（37 维，与 pro_controller.py config.getfloat default 一致）
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
    "cipher_ff_predict_dt":       0.030,
    "cipher_ff_smooth_alpha":     0.30,
    "cipher_ff_scale_near_min":   0.45,
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
    "cipher_tau_arm_dyn_gain":     0.6,    # v8.3 τ_arm 距离自适应斜率 (0.4+gain·d_norm)
    "cipher_motor_endpoint_ema_alpha": 0.15,  # v8.2 mini-snap endpoint EMA α
    "cipher_tau_wrist":           0.008,
    "cipher_ou_sigma_ball":       0.18,
    "cipher_ou_sigma_track":      0.38,
    "cipher_drift_sigma":         0.10,
    "cipher_ou_vel_scale":      120.0,
    "cipher_drift_pos_scale":     0.025,
    "cipher_prog_interval":       0.06,
    "cipher_entry_ticks":        12,
    "cipher_entry_dz_px":         2.5,
    # ── 高级 override（默认 0 = 自动计算；>0 则强制覆盖）──
    "cipher_zeta_correction":     0.0,
    "cipher_zeta_pursuit":        0.0,
    "cipher_thresh_hyst_ratio":   0.0,
    "cipher_tau_wrist_ratio":     0.0,
}

# 模块E 新增：IMM-Kalman / ReID / α 调度参数（12 维）
BASELINE_EXTENDED = {
    # IMM-Kalman（[Kalman] + [IMM] section）
    "Kalman.R":                   2.0,
    "Kalman.Q_pos":               5.0,
    "Kalman.Q_vel":              80.0,
    "Kalman.Q_acc":             200.0,
    "IMM.markov_diag":            0.90,
    # ReID / TrackManager（[WorldModel] section）
    "WorldModel.track_reid_weight":      0.15,
    "WorldModel.track_ws_w_dist":        1.0,
    "WorldModel.track_ws_w_conf":        0.5,
    "WorldModel.track_ws_w_threat":      0.3,
    "WorldModel.track_ws_w_velocity":    0.2,
    # α 调度（[General] section）
    "General.min_aim_conf":       0.32,
    "General.aim_lock_confirm_count": 3,
}

# 合并完整 baseline（49 维）
BASELINE_ALL = {**BASELINE, **BASELINE_EXTENDED}


# ══════════════════════════════════════════════════════════════════════════════
# § 2  搜索空间 — 围绕 BASELINE 设定 bounds
# ══════════════════════════════════════════════════════════════════════════════
def _bounds(baseline: float, ratio_lo: float = 0.5, ratio_hi: float = 2.0,
            abs_lo: float = None, abs_hi: float = None) -> tuple:
    lo = baseline * ratio_lo
    hi = baseline * ratio_hi
    if abs_lo is not None: lo = max(lo, abs_lo)
    if abs_hi is not None: hi = min(hi, abs_hi)
    return (lo, hi, False)

PARAM_BOUNDS: dict[str, tuple] = {
    # ── Controller (37 维) ──
    "cipher_max_speed":       _bounds(BASELINE["cipher_max_speed"],     0.6, 1.6),
    "cipher_ff_gain":         (0.80, 1.00, False),
    "cipher_ff_acc_sec":      (0.000, 0.050, False),
    "cipher_thresh_high_px":  (35.0, 70.0, False),
    "cipher_thresh_low_px":   (20.0, 50.0, False),
    "cipher_deadzone_scale":  (0.015, 0.06, False),
    "cipher_k_pursuit":       (70.0,  220.0, False),
    "cipher_k_flick":         (300.0, 650.0, False),
    "cipher_b_pursuit":       (0.15,   0.60, False),
    "cipher_k_correction":    (30.0,  100.0, False),
    "cipher_b_correction":    (0.08,   0.50, False),
    "cipher_ff_speed_knee":   (100.0,  500.0, False),
    "cipher_ff_speed_scale":  (800.0, 3000.0, False),
    "cipher_ff_scale_min":    (0.15,    0.60, False),
    "cipher_ff_predict_dt":   (0.010,   0.060, False),
    "cipher_ff_smooth_alpha": (0.10,    0.60, False),
    "cipher_ff_scale_near_min": (0.20,  0.70, False),
    "cipher_fitts_a":         (0.015, 0.050, False),
    "cipher_fitts_b":         (0.025, 0.065, False),
    "cipher_fitts_sigma":     (0.02,  0.12, False),
    "cipher_fitts_T_min":     (0.035, 0.090, False),
    "cipher_fitts_T_max":     (0.22,  0.35, False),
    "cipher_undershoot_loc":    (0.0,  0.15, False),
    "cipher_undershoot_sigma":  (0.8,  3.5,  False),
    "cipher_ramp_ms":           (25.0, 75.0, False),
    "cipher_ramp_min":           (0.70, 1.00, False),
    "cipher_tau_arm":            (0.008, 0.020, False),
    "cipher_tau_arm_dyn_gain":   (0.3, 1.0, False),     # v8.3 τ_arm 距离自适应斜率
    "cipher_motor_endpoint_ema_alpha": (0.05, 0.30, False),  # v8.2 mini-snap endpoint EMA
    "cipher_tau_wrist":          (0.005, 0.015, False),
    "cipher_ou_sigma_ball":      (0.08, 0.40, False),
    "cipher_ou_sigma_track":     (0.15, 0.55, False),
    "cipher_drift_sigma":        (0.04, 0.18, False),
    "cipher_ou_vel_scale":       (40.0, 180.0, False),
    "cipher_drift_pos_scale":    (0.005, 0.040, False),
    "cipher_prog_interval":      (0.03, 0.12, False),
    "cipher_entry_ticks":        (6, 20, False),    # int
    "cipher_entry_dz_px":         (1.0, 5.0, False),
    # 高级 override（0 = 自动；>0 强制覆盖；bounds 上限保守）
    "cipher_zeta_correction":   (0.0, 1.5, False),
    "cipher_zeta_pursuit":      (0.0, 1.5, False),
    "cipher_thresh_hyst_ratio": (0.0, 0.30, False),
    "cipher_tau_wrist_ratio":   (0.0, 0.50, False),
    # ── IMM-Kalman (5 维) ──
    "Kalman.R":               (0.5, 8.0, False),
    "Kalman.Q_pos":           (1.0, 20.0, False),
    "Kalman.Q_vel":           (20.0, 200.0, False),
    "Kalman.Q_acc":           (50.0, 500.0, False),
    "IMM.markov_diag":        (0.75, 0.97, False),
    # ── ReID / TrackManager (5 维) ──
    "WorldModel.track_reid_weight":   (0.03, 0.35, False),
    "WorldModel.track_ws_w_dist":     (0.3, 2.5, False),
    "WorldModel.track_ws_w_conf":     (0.1, 1.2, False),
    "WorldModel.track_ws_w_threat":   (0.05, 0.8, False),
    "WorldModel.track_ws_w_velocity": (0.05, 0.6, False),
    # ── α 调度 (2 维) ──
    "General.min_aim_conf":           (0.22, 0.48, False),
    "General.aim_lock_confirm_count": (1, 6, False),  # int
}

# int 参数集合（需要 suggest_int 而非 suggest_float）
_INT_PARAMS = {
    "cipher_entry_ticks",
    "General.aim_lock_confirm_count",
}


# ══════════════════════════════════════════════════════════════════════════════
# § 3  场景池 + 随机化（防过拟合核心 1：场景多样化）
# ══════════════════════════════════════════════════════════════════════════════
_SCENARIO_TYPES_ALL = [
    "ball", "takeover", "pure_ai",
    "multi_target", "occlusion", "peek", "directional_change", "coast",
]

# 模块级全局（供多进程 worker 访问）
_SCENARIO_TYPES: tuple = ("ball", "takeover", "pure_ai")  # 默认多场景
_SCENARIO_MAX_KILLS: int = 20
_SCENARIO_DURATION: float = 10.0


def _make_randomized_scenario(scenario_type: str, seed: int):
    """
    根据 seed 创建场景实例，参数在合理范围内随机化。
    关键：不同 seed 产生不同场景配置 → 防止优化器记忆特定场景模式。
    """
    rng = np.random.RandomState(seed * 7 + 13)

    if scenario_type == "ball":
        return None  # BallTrackingScenario 用内置随机化
    if scenario_type == "takeover":
        from test.scenarios.takeover import TakeoverScenario
        return TakeoverScenario(max_kills=_SCENARIO_MAX_KILLS)
    if scenario_type == "pure_ai":
        from test.scenarios.pure_ai_ball import PureAIBallScenario
        return PureAIBallScenario(max_kills=_SCENARIO_MAX_KILLS)
    if scenario_type == "multi_target":
        from test.scenarios.multi_target import MultiTargetScenario
        return MultiTargetScenario(
            max_kills=_SCENARIO_MAX_KILLS,
            num_targets_range=(2, 4),
            dist_range=(rng.uniform(3, 8), rng.uniform(20, 40)),
            speed_range=(rng.uniform(2, 4), rng.uniform(6, 10)),
            seed=seed,
        )
    if scenario_type == "occlusion":
        from test.scenarios.occlusion import OcclusionScenario
        return OcclusionScenario(
            max_kills=_SCENARIO_MAX_KILLS,
            seed=seed,
            dist_range=(rng.uniform(3, 8), rng.uniform(15, 30)),
            speed_range=(rng.uniform(2, 4), rng.uniform(6, 10)),
        )
    if scenario_type == "peek":
        from test.scenarios.peek import PeekScenario
        return PeekScenario(
            max_kills=_SCENARIO_MAX_KILLS,
            seed=seed,
            dist_range=(rng.uniform(5, 10), rng.uniform(15, 25)),
            strafe_speed_range=(rng.uniform(3, 5), rng.uniform(6, 9)),
        )
    if scenario_type == "directional_change":
        from test.scenarios.directional_change import DirectionalChangeScenario
        return DirectionalChangeScenario(
            max_kills=_SCENARIO_MAX_KILLS,
            seed=seed,
            dist_range=(rng.uniform(3, 8), rng.uniform(15, 30)),
            speed_range=(rng.uniform(3, 5), rng.uniform(6, 10)),
        )
    if scenario_type == "coast":
        from test.scenarios.coast import CoastScenario
        return CoastScenario(
            max_kills=_SCENARIO_MAX_KILLS,
            seed=seed,
            dist_range=(rng.uniform(3, 8), rng.uniform(15, 30)),
            speed_range=(rng.uniform(2, 4), rng.uniform(6, 10)),
        )
    return None


# ══════════════════════════════════════════════════════════════════════════════
# § 4  种子池 + 留出测试集（防过拟合核心 2：数据隔离）
# ══════════════════════════════════════════════════════════════════════════════
# 优化种子池（30 个，足够覆盖场景多样性）
OPT_SEEDS = tuple(range(100, 130))
# 留出种子池（10 个，仅用于最终验证，优化过程不可见）
HELD_OUT_SEEDS = tuple(range(200, 210))

# 每 trial 评估的场景-种子组合数
_N_COARSE_SEEDS = 3       # 粗筛阶段种子数
_N_FINE_SEEDS = 6         # 精评阶段种子数
_N_SCENARIOS_PER_SEED = 2  # 每个种子评估的场景类型数


def _pick_scenario_types(seed: int, n: int) -> list:
    """根据 seed 确定性地选取 n 个场景类型（覆盖多样性）。"""
    rng = np.random.RandomState(seed * 3 + 7)
    pool = list(_SCENARIO_TYPES)
    if n >= len(pool):
        return pool
    return list(rng.choice(pool, size=n, replace=False))


# ══════════════════════════════════════════════════════════════════════════════
# § 5  评估函数（两阶段：粗筛 + 精评）
# ══════════════════════════════════════════════════════════════════════════════
def run_once(params: dict, seed: int, scenario_type: str = None,
             duration: float = None) -> dict:
    """跑单次仿真，返回 analysis dict。"""
    np.random.seed(seed)
    pyrand.seed(seed)

    if scenario_type is None:
        scenario_type = _SCENARIO_TYPES[0]

    scenario = _make_randomized_scenario(scenario_type, seed)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _, _, analysis = c.run_simulation_with_diagnostics(
            duration=(duration if duration is not None else _SCENARIO_DURATION),
            plot=False,
            verbose=False,
            use_fixed=True,
            params=params,
            scenario=scenario,
        )
    return analysis


def _safe(a: dict, key: str, default=0.0) -> float:
    return float(a.get(key, default))


def evaluate_multi(params: dict, seeds: tuple, n_scenarios: int = 2) -> dict:
    """
    多场景多种子评估（防过拟合核心）。
    每个 seed 跑 n_scenarios 种场景类型，收集所有分数。
    """
    all_scores = []
    all_overall = []
    all_p5 = []
    all_worst = []
    all_bucket_stds = []
    per_seed_detail = []

    for seed in seeds:
        scenario_types = _pick_scenario_types(seed, n_scenarios)
        seed_scores = []
        for st in scenario_types:
            a = run_once(params, seed, st)
            overall = _safe(a, "overall_score")
            p5 = _safe(a, "robustness_p5")
            worst = _safe(a, "worst_case")
            bucket_std = _safe(a, "cross_bucket_std")

            # 软惩罚：击杀过少 / 精度塌陷
            total = int(a.get("total_kills", 0))
            prec = _safe(a, "precision_score")
            penalty = 0.0
            if total < 5:
                penalty += (5 - total) * 6.0
            if prec < 3.0:
                penalty += (3.0 - prec) * 3.0

            score = overall - penalty
            all_scores.append(score)
            all_overall.append(overall)
            all_p5.append(p5)
            all_worst.append(worst)
            if bucket_std > 0:
                all_bucket_stds.append(bucket_std)
            seed_scores.append({"scenario": st, "score": score, "overall": overall,
                                "p5": p5, "worst": worst, "kills": total})
        per_seed_detail.append({"seed": seed, "scores": seed_scores})

    return {
        "mean": float(np.mean(all_scores)),
        "std": float(np.std(all_scores)),
        "p5": float(np.percentile(all_scores, 5)) if len(all_scores) > 1 else float(np.mean(all_scores)),
        "worst": float(np.min(all_scores)) if all_scores else 0.0,
        "cross_bucket_std": float(np.mean(all_bucket_stds)) if all_bucket_stds else 0.0,
        "n_runs": len(all_scores),
        "per_seed": per_seed_detail,
    }


def evaluate_coarse(params: dict) -> dict:
    """阶段 1：粗筛（3 seed × 1 scenario = 3 runs）。"""
    seeds = OPT_SEEDS[:_N_COARSE_SEEDS]
    return evaluate_multi(params, seeds, n_scenarios=1)


def evaluate_fine(params: dict, n_seeds: int = _N_FINE_SEEDS) -> dict:
    """阶段 2：精评（n_seeds × 2 scenarios）。"""
    seeds = OPT_SEEDS[:n_seeds]
    return evaluate_multi(params, seeds, n_scenarios=_N_SCENARIOS_PER_SEED)


# ══════════════════════════════════════════════════════════════════════════════
# § 6  物理约束剪枝（防过拟合核心 3：物理一致性）
# ══════════════════════════════════════════════════════════════════════════════
def check_physical_constraints(params: dict) -> bool:
    """检查物理约束，返回 True 表示合法。"""
    # 1. 阈值有序
    if params["cipher_thresh_low_px"] >= params["cipher_thresh_high_px"]:
        return False
    if params["cipher_fitts_T_min"] >= params["cipher_fitts_T_max"]:
        return False
    # 2. 神经肌肉时间常数：tau_arm > tau_wrist
    if params["cipher_tau_arm"] <= params["cipher_tau_wrist"]:
        return False
    # 3. 临界阻尼一致性：ζ = B/(2√(K·τ)) ∈ [0.1, 2.0]
    #    游戏瞄准允许高度欠阻尼（ζ~0.1-0.3）以换取快速响应，
    #    下限 0.1 排除极端震荡，上限 2.0 排除过度迟钝。
    K = params["cipher_k_pursuit"]
    B = params["cipher_b_pursuit"]
    tau = params["cipher_tau_arm"]
    zeta = B / (2.0 * np.sqrt(max(K * tau, 1e-9)))
    if not (0.1 <= zeta <= 2.0):
        return False
    # 4. flick 刚度应 > pursuit 刚度（远距需要更强弹簧）
    if params["cipher_k_flick"] < params["cipher_k_pursuit"]:
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# § 7  Optuna Objective（两阶段 + MedianPruner）
# ══════════════════════════════════════════════════════════════════════════════
def objective(trial: optuna.Trial) -> float:
    # ── 采样参数 ──
    params: dict = {}
    for name, (lo, hi, log_scale) in PARAM_BOUNDS.items():
        if name in _INT_PARAMS:
            params[name] = int(trial.suggest_int(name, int(lo), int(hi)))
        else:
            params[name] = trial.suggest_float(name, lo, hi, log=log_scale)

    # ── 物理约束剪枝 ──
    if not check_physical_constraints(params):
        raise optuna.exceptions.TrialPruned()

    # ── 阶段 1：粗筛（3 seed × 1 scenario）──
    coarse = evaluate_coarse(params)
    coarse_score = coarse["mean"] - 0.5 * coarse["std"]

    # 报告中间结果给 pruner
    trial.report(coarse_score, step=0)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    # ── 阶段 2：精评（6 seed × 2 scenarios = 12 runs）──
    fine = evaluate_fine(params)

    # ── 鲁棒目标：mean - 0.5·std + P5 惩罚 + 跨桶一致性惩罚 ──
    robust = fine["mean"] - 0.5 * fine["std"]
    p5_penalty = max(0.0, 40.0 - fine["p5"]) * 0.3      # P5 低于 40 则惩罚
    consistency_penalty = min(fine["cross_bucket_std"] * 0.3, 15.0)  # 桶间差异大则惩罚
    final_score = robust - p5_penalty - consistency_penalty

    trial.set_user_attr("mean", fine["mean"])
    trial.set_user_attr("std", fine["std"])
    trial.set_user_attr("p5", fine["p5"])
    trial.set_user_attr("worst", fine["worst"])
    trial.set_user_attr("cross_bucket_std", fine["cross_bucket_std"])
    trial.set_user_attr("n_runs", fine["n_runs"])
    trial.set_user_attr("coarse_score", coarse_score)
    trial.set_user_attr("per_seed", fine["per_seed"])

    return -final_score


# ══════════════════════════════════════════════════════════════════════════════
# § 8  Study 构造 + 留出验证
# ══════════════════════════════════════════════════════════════════════════════
def build_study(db_url: str, study_name: str, resume: bool) -> optuna.Study:
    x0 = {k: BASELINE_ALL[k] for k in PARAM_BOUNDS.keys()}
    sigma0 = min((hi - lo) / 6.0 for (lo, hi, _) in PARAM_BOUNDS.values())

    sampler = optuna.samplers.CmaEsSampler(
        x0=x0,
        sigma0=sigma0,
        seed=42,
        n_startup_trials=0,
        warn_independent_sampling=False,
    )
    # MedianPruner：粗筛阶段后剪掉底 50%
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=1,
        interval_steps=1,
        n_min_trials=5,
    )

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
        print(f"\n❌ study '{study_name}' 已存在于 {db_url}")
        print("  选项：")
        print("    A) 续跑：python test/simulation.py --resume")
        print(f"    B) 删除：optuna delete-study --study-name {study_name} --storage {db_url}")
        raise SystemExit(1)

    if len(study.trials) == 0:
        study.enqueue_trial({k: BASELINE_ALL[k] for k in PARAM_BOUNDS.keys()})

    return study


def validate_on_held_out(params: dict) -> dict:
    """
    在留出种子集上验证 best params（防过拟合核心 4：数据隔离）。
    对比优化分数 vs 留出分数，差距大 = 过拟合。
    """
    print("\n" + "=" * 78)
    print("🔒 留出测试集验证（HELD_OUT_SEEDS）")
    print("=" * 78)

    # 优化集分数
    opt_result = evaluate_fine(params, n_seeds=6)
    print(f"  优化集  (OPT_SEEDS)    : mean={opt_result['mean']:.1f}  std={opt_result['std']:.1f}  "
          f"P5={opt_result['p5']:.1f}  worst={opt_result['worst']:.1f}")

    # 留出集分数
    held_out_result = evaluate_multi(params, HELD_OUT_SEEDS, n_scenarios=_N_SCENARIOS_PER_SEED)
    print(f"  留出集  (HELD_OUT)     : mean={held_out_result['mean']:.1f}  std={held_out_result['std']:.1f}  "
          f"P5={held_out_result['p5']:.1f}  worst={held_out_result['worst']:.1f}")

    gap = opt_result["mean"] - held_out_result["mean"]
    gap_pct = gap / max(opt_result["mean"], 1.0) * 100
    print(f"  泛化差距: {gap:+.1f} ({gap_pct:+.1f}%)")
    if gap_pct > 15:
        print("  ⚠️  过拟合警告：留出集分数比优化集低 >15%")
    elif gap_pct > 8:
        print("  ⚠️  轻度过拟合：留出集分数比优化集低 8-15%")
    else:
        print("  ✅ 泛化良好：留出集与优化集分数接近")
    print("=" * 78)

    return {
        "opt_mean": opt_result["mean"],
        "held_out_mean": held_out_result["mean"],
        "gap": gap,
        "gap_pct": gap_pct,
        "overfit": gap_pct > 15,
    }


# ══════════════════════════════════════════════════════════════════════════════
# § 9  Worker / 报告 / Main
# ══════════════════════════════════════════════════════════════════════════════
def optimize_worker(db_url: str, study_name: str, n_trials: int):
    study = optuna.load_study(study_name=study_name, storage=db_url)
    study.optimize(objective, n_trials=n_trials)


def _print_best_params(study: optuna.Study):
    print("\n" + "=" * 78)
    # 检查是否有完成的 trial（被物理约束全部剪掉时无 best）
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
        print(f"⚠️  无已完成 trial：{len(study.trials)} 个 trial 中 {n_pruned} 个被剪枝。")
        print("   多数 trial 被物理约束排除 —— 请放宽 PARAM_BOUNDS 或增加 trials 数。")
        print("=" * 78)
        return
    best = study.best_trial
    m  = best.user_attrs.get("mean",  float("nan"))
    s  = best.user_attrs.get("std",   float("nan"))
    p5 = best.user_attrs.get("p5",    float("nan"))
    w  = best.user_attrs.get("worst", float("nan"))
    print(f"🏆 CIPHER v2.0 防过拟合搜索完成")
    print(f"   鲁棒分: {-study.best_value:.2f}")
    print(f"   mean={m:.2f}  std={s:.2f}  P5={p5:.2f}  worst={w:.2f}")
    print("=" * 78)

    # 参数对比
    print("\n— 参数对比 (vs BASELINE) —")
    for k in sorted(best.params):
        v  = best.params[k]
        bv = BASELINE_ALL.get(k, 0.0)
        delta_pct = (v - bv) / max(abs(bv), 1e-9) * 100
        flag = "↑" if delta_pct > 5 else ("↓" if delta_pct < -5 else " ")
        print(f"  {flag} {k:<40} = {v:<12.6g}  (baseline {bv:g}, Δ{delta_pct:+.0f}%)")

    # 留出集验证
    validate_on_held_out(best.params)

    # ── 调优值输出：指引用户写入位置 ─────────────────────────────────────────
    # config.ini 已重构为「功能开关 + 硬约束」，不再存放 cipher_*/Kalman.R 等调优参数。
    # 调优值应写入 test/simulation.py 的 BASELINE / BASELINE_EXTENDED，下次启动即生效。
    print("\n— 下一步：把调优值写入 test/simulation.py 的 BASELINE / BASELINE_EXTENDED —")
    print("   (config.ini 不再存放调优参数，避免污染功能开关层)")
    controller_params = {k: v for k, v in best.params.items() if k.startswith("cipher_")}
    ext_params       = {k: v for k, v in best.params.items() if not k.startswith("cipher_")}
    if controller_params:
        print("\n  ── controller 37 维 (写入 BASELINE) ──")
        for k, v in sorted(controller_params.items()):
            bv = BASELINE.get(k, 0.0)
            print(f"    \"{k}\": {v:.6g},  # was {bv:g}")
    if ext_params:
        print("\n  ── 扩展 12 维 (写入 BASELINE_EXTENDED) ──")
        for k, v in sorted(ext_params.items()):
            bv = BASELINE_EXTENDED.get(k, 0.0)
            print(f"    \"{k}\": {v:.6g},  # was {bv:g}")


# ══════════════════════════════════════════════════════════════════════════════
#  实时评分工具（算法迭代加速器）
#
#  --quick  ── 1 seed × 5 场景 × 0.5s sim，~3 秒出 7 维指标
#  --save   ── 把当前 7 维指标 + 整体分存为 JSON
#  --compare── 与基线 JSON diff 指标涨跌
#  --diagnose── 详细分析 + 自动给算法修改建议
# ══════════════════════════════════════════════════════════════════════════════
import json
import time
from pathlib import Path

_QUICK_SEED = 200  # 快速评分专用固定种子（脱离 OPT_SEEDS 避免被调优污染）_QUICK_SEED = 200
# 5 场景覆盖 5 类典型使用：
#   pure_ai        - 纯算法层（无接管）
#   multi_target   - 多目标切换 / ReID / 威胁打分
#   directional_change - 急停变向 / IMM 模型切换
#   occlusion      - 遮挡恢复
#   coast          - 平稳追踪 / 拟人化
#   takeover       - 人机接管（jerk² / 紧急通道）→ v5.0 加入，测接管类优化
_QUICK_SCENARIOS = ("pure_ai", "multi_target", "directional_change",
                    "occlusion", "coast", "takeover")
_QUICK_DURATION = 0.5  # 秒/场景

_METRIC_LABELS = [
    ("total",     "总分",   "overall_score"),
    ("ttk",       "TTK",   "acq_score"),
    ("precision", "Prec",  "precision_score"),
    ("natural",   "Nat",   "natural_score"),
    ("bio",       "Bio",   "bio_bonus"),
    ("over",      "Over",  "overshoot_score"),
    ("smooth",    "Smth",  "smoothness_score"),
    ("takeover",  "Take",  "takeover_score"),
]


def evaluate_quick(params: dict, duration: float = _QUICK_DURATION) -> dict:
    """
    快速评分：固定 1 seed × 5 场景 × duration 秒，~3 秒出分。
    适用于算法迭代的实时反馈。返回结构化的 7 维指标 + 整体分。
    """
    per_scenario = {}
    for st in _QUICK_SCENARIOS:
        try:
            a = run_once(params, _QUICK_SEED, st, duration=duration)
        except Exception as e:
            per_scenario[st] = {"error": str(e), "overall_score": 0.0}
            continue
        per_scenario[st] = {
            "overall":       float(_safe(a, "overall_score")),
            "ttk":           float(_safe(a, "acq_score")),
            "precision":     float(_safe(a, "precision_score")),
            "natural":       float(_safe(a, "natural_score")),
            "bio":           float(_safe(a, "bio_bonus")),
            "overshoot":     float(_safe(a, "overshoot_score")),
            "smoothness":    float(_safe(a, "smoothness_score")),
            "takeover":      float(_safe(a, "takeover_score")),
            "p5":            float(_safe(a, "robustness_p5")),
            "worst":         float(_safe(a, "worst_case")),
            "mean_over_px":  float(_safe(a, "mean_overshoot")),
            "mean_accel":    float(_safe(a, "mean_accel_peak")),
            "kills":         int(a.get("total_kills", 0)),
        }
    # 聚合（场景间 mean）
    def _mean(key: str) -> float:
        vals = [d[key] for d in per_scenario.values() if key in d]
        return float(np.mean(vals)) if vals else 0.0
    aggregate = {
        "total":      _mean("overall"),
        "ttk":        _mean("ttk"),
        "precision":  _mean("precision"),
        "natural":    _mean("natural"),
        "bio":        _mean("bio"),
        "overshoot":  _mean("overshoot"),
        "smoothness": _mean("smoothness"),
        "takeover":   _mean("takeover"),
    }
    return {"aggregate": aggregate, "per_scenario": per_scenario}


def _print_quick_table(res: dict, baseline: dict | None = None) -> None:
    """
    打印 7 维指标表（带 baseline 对比）。
    baseline: dict 含 total/ttk/precision/... 共 7 个键
    """
    agg = res["aggregate"]
    headers = ["指标", "本次"]
    if baseline:
        headers += ["基线", "Δ", "状态"]
    print("─" * 64)
    print(f"  {'指标':<10} | " + " | ".join(f"{h:>10}" for h in headers[1:]))
    print("─" * 64)
    keys = ["total", "ttk", "precision", "natural", "bio", "overshoot", "smoothness", "takeover"]
    for k in keys:
        v = agg[k]
        if baseline and k in baseline:
            base = baseline[k]
            delta = v - base
            mark = "✅" if delta > 0.5 else ("❌" if delta < -0.5 else "  ")
            print(f"  {k:<10} | {v:>10.2f} | {base:>10.2f} | {delta:+7.2f} | {mark}")
        else:
            print(f"  {k:<10} | {v:>10.2f}")
    print("─" * 64)
    # 场景细分（简洁）
    print(f"  场景明细：")
    for st, d in res["per_scenario"].items():
        if "error" in d:
            print(f"    {st:<22} ERROR: {d['error']}")
        else:
            print(f"    {st:<22} total={d['overall']:.1f} ttk={d['ttk']:.1f} "
                  f"over={d['overshoot']:.1f}({d['mean_over_px']:.1f}px) "
                  f"accel={d['mean_accel']:.0f} kills={d['kills']}")


def _diagnose_suggestions(agg: dict, baseline: dict | None) -> list[str]:
    """
    基于 7 维指标异常给出算法修改建议。
    返回建议列表（每条一句话）。
    """
    s = []
    if baseline is None:
        return s
    over_d  = agg["overshoot"]   - baseline["overshoot"]
    smt_d   = agg["smoothness"]  - baseline["smoothness"]
    nat_d   = agg["natural"]     - baseline["natural"]
    ttk_d   = agg["ttk"]         - baseline["ttk"]
    pre_d   = agg["precision"]   - baseline["precision"]
    take_d  = agg["takeover"]    - baseline["takeover"]
    total_d = agg["total"]       - baseline["total"]

    # ── 总分下降时的优先级诊断 ──
    if total_d < -1.0:
        # 找最差指标
        deltas = {"overshoot": over_d, "smoothness": smt_d, "natural": nat_d,
                  "ttk": ttk_d, "precision": pre_d, "takeover": take_d}
        worst = min(deltas, key=deltas.get)
        if worst == "overshoot" and over_d < -3:
            s.append(f"💡 Over -{-over_d:.1f}：目标过冲加重。检查 cipher_ff_scale_near_min (近端前馈) 是否过高，"
                     f"或 cipher_b_pursuit (阻尼) 是否过低。")
        elif worst == "smoothness" and smt_d < -3:
            s.append(f"💡 Smth -{-smt_d:.1f}：加速度峰值失控。检查 cipher_max_speed (限速) 是否过低，"
                     f"或 cipher_ff_acc_sec (加速度前馈) 是否过大。")
        elif worst == "ttk" and ttk_d < -3:
            s.append(f"💡 TTK -{-ttk_d:.1f}：击杀变慢。检查 cipher_k_flick (flick 增益) 是否被改小，"
                     f"或 cipher_ff_gain (前馈强度) 是否降低。")
        elif worst == "natural" and nat_d < -3:
            s.append(f"💡 Nat -{-nat_d:.1f}：拟人度下降。检查 cipher_ou_sigma_* (噪声) 是否过大，"
                     f"或 attention 调制范围是否被关掉。")
        elif worst == "precision" and pre_d < -3:
            s.append(f"💡 Prec -{-pre_d:.1f}：稳态精度下降。检查 cipher_ou_sigma_track (追踪噪声) 是否过大，"
                     f"或 cipher_deadzone_scale (死区) 是否被改小。")
        elif worst == "takeover" and take_d < -3:
            s.append(f"💡 Take -{-take_d:.1f}：接管延迟增加。检查 [Aim].min_aim_conf (锁目标门槛) 是否过高，"
                     f"或 aim_lock_confirm_count (首锁确认) 是否被加大。")
    elif total_d > 1.0:
        s.append(f"✅ 总分 +{total_d:.1f}：本次改动有效。")
    return s


def _save_snapshot(path: str, res: dict, params: dict) -> None:
    """保存当前评分快照（aggregate + 用到的关键参数）。"""
    snap = {
        "aggregate": res["aggregate"],
        "per_scenario": res["per_scenario"],
        "params_subset": {k: v for k, v in params.items() if k.startswith("cipher_")},
        "timestamp": time.time(),
    }
    Path(path).write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"💾 已保存评分快照 → {path}")


def _load_snapshot(path: str) -> dict:
    """加载评分快照。仅返回 aggregate（用于对比表）。"""
    snap = json.loads(Path(path).read_text(encoding="utf-8"))
    print(f"📂 已加载基线快照 ← {path} (timestamp={snap.get('timestamp','?')})")
    return snap["aggregate"]


def main():
    parser = argparse.ArgumentParser(
        description="CIPHER v2.0 — 防过拟合调优 + 实时评分"
    )
    # ── 实时评分（算法迭代主用）──
    parser.add_argument("--quick", action="store_true",
                        help="快速评分：1 seed × 5 场景 × 0.5s，~3 秒出 7 维指标")
    parser.add_argument("--save", type=str, metavar="PATH", default=None,
                        help="保存评分快照到 JSON（与 --quick 联用）")
    parser.add_argument("--compare", type=str, metavar="PATH", default=None,
                        help="与基线 JSON 对比（与 --quick 联用，自动触发诊断）")
    parser.add_argument("--diagnose", action="store_true",
                        help="详细诊断 + 修改建议（与 --quick 联用）")
    parser.add_argument("--baseline", type=str, metavar="PATH", default="baseline.json",
                        help="--diagnose 自动加载的基线路径（默认 baseline.json）")
    # ── 大规模调优（最后一步才用）──
    parser.add_argument("--single", action="store_true", help="单进程跑 Optuna，方便 debug")
    parser.add_argument("--resume", action="store_true", help="续跑现有 study")
    parser.add_argument("--scenario", type=str, default="multi",
                        help="场景: multi=多场景(默认) | ball/takeover/pure_ai=单场景调试")
    parser.add_argument("--trials", type=int, default=400, help="总 trial 数")
    parser.add_argument("--workers", type=int, default=0, help="worker 数量（0 = 自动）")
    parser.add_argument("--validate-only", action="store_true",
                        help="仅用留出集验证现有 best params（不跑优化）")
    args = parser.parse_args()

    # ══════════════════════════════════════════════════════════════════════
    # § 0  快速评分分支（算法迭代主用，秒级出分 + 7 维指标 + 修改建议）
    # ══════════════════════════════════════════════════════════════════════
    if args.quick or args.save or args.compare or args.diagnose:
        _run_quick_scorer(args)
        return

    # ══════════════════════════════════════════════════════════════════════
    # § 1  大规模 Optuna 调优（最后一步才用）
    # ══════════════════════════════════════════════════════════════════════
    global _SCENARIO_TYPES
    if args.scenario == "multi":
        _SCENARIO_TYPES = tuple(_SCENARIO_TYPES_ALL)
    else:
        _SCENARIO_TYPES = (args.scenario,)

    db_url = "sqlite:///aimbot_cipher_v2.db"
    study_name = "cipher_v2_antioverfit"

    # ── 仅验证模式 ──
    if args.validate_only:
        try:
            study = optuna.load_study(study_name=study_name, storage=db_url)
        except KeyError:
            print(f"❌ study '{study_name}' 不存在，无法验证")
            return
        if not study.best_trial:
            print("❌ study 无已完成 trial")
            return
        print(f"\n🔒 验证 study '{study_name}' 的 best params")
        validate_on_held_out(study.best_trial.params)
        return

    scenario_label = "多场景防过拟合" if args.scenario == "multi" else f"单场景({args.scenario})"

    print(f"\n🚀 CIPHER v2.0 防过拟合搜索")
    print(f"   模式: {scenario_label}")
    # 动态分组统计（v5.0：实际维度由 PARAM_BOUNDS 决定，分类由 key 前缀）
    n_total = len(PARAM_BOUNDS)
    n_ctrl  = sum(1 for k in PARAM_BOUNDS if k.startswith("cipher_"))
    n_kf    = sum(1 for k in PARAM_BOUNDS if k.startswith("Kalman."))   # Kalman filter Q/R
    n_imm   = sum(1 for k in PARAM_BOUNDS if k.startswith("IMM."))       # IMM model 转移
    n_reid  = sum(1 for k in PARAM_BOUNDS if k.startswith("WorldModel."))
    n_alpha = sum(1 for k in PARAM_BOUNDS if k.startswith("General."))
    n_other = n_total - n_ctrl - n_kf - n_imm - n_reid - n_alpha
    n_imm_total = n_kf + n_imm
    print(f"   搜索空间: {n_total} 维 (controller {n_ctrl} + IMM {n_imm_total} "
          f"[KF {n_kf}+IMM {n_imm}] + ReID {n_reid} + α {n_alpha}"
          + (f" + 其它 {n_other}" if n_other else "") + ")")
    print(f"   优化种子: {len(OPT_SEEDS)} 个 | 留出种子: {len(HELD_OUT_SEEDS)} 个")
    print(f"   两阶段: 粗筛 {_N_COARSE_SEEDS}×1 + 精评 {_N_FINE_SEEDS}×{_N_SCENARIOS_PER_SEED} = "
          f"{_N_COARSE_SEEDS + _N_FINE_SEEDS * _N_SCENARIOS_PER_SEED} runs/trial")
    print(f"   storage = {db_url}, study = {study_name}\n")

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


def _run_quick_scorer(args) -> None:
    """
    快速评分入口（算法迭代主用）。
    用法：
      python test/simulation.py --quick                       # 仅出分
      python test/simulation.py --quick --save baseline.json  # 出分 + 保存
      python test/simulation.py --compare baseline.json       # 出分 + 对比
      python test/simulation.py --diagnose                    # 出分 + 诊断建议
    """
    # 1. 构造当前 params（从 BASELINE_ALL 读，确保 49 维完整）
    params = dict(BASELINE_ALL)

    # 2. 跑 5 场景 × 0.5s
    t0 = time.time()
    res = evaluate_quick(params)
    dt = time.time() - t0

    # 3. 解析基线（--compare / --diagnose / baseline.json 存在时）
    baseline_agg = None
    baseline_path = None
    if args.compare:
        baseline_agg = _load_snapshot(args.compare)
        baseline_path = args.compare
    elif args.diagnose:
        if Path(args.baseline).exists():
            baseline_agg = _load_snapshot(args.baseline)
            baseline_path = args.baseline
        else:
            print(f"⚠️  --diagnose 需要基线文件，未找到 {args.baseline}，跳过对比")
    elif Path(args.baseline).exists() and not args.save:
        # 默认行为：当前目录下若有 baseline.json，自动加载并对比
        baseline_agg = _load_snapshot(args.baseline)
        baseline_path = args.baseline

    # 4. 打印
    n_scenarios = len(_QUICK_SCENARIOS)
    print(f"\n📊 CIPHER 实时评分 (1 seed × {n_scenarios} 场景 × {_QUICK_DURATION}s, 耗时 {dt:.1f}s)")
    if baseline_path:
        print(f"   对比基线: {baseline_path}")
    _print_quick_table(res, baseline=baseline_agg)

    # 5. 保存（如有）
    if args.save:
        _save_snapshot(args.save, res, params)

    # 6. 诊断（--diagnose 或 --compare 触发）
    if args.diagnose or args.compare:
        suggestions = _diagnose_suggestions(res["aggregate"], baseline_agg)
        if suggestions:
            print(f"\n🔧 诊断建议：")
            for line in suggestions:
                print(f"  {line}")
        elif baseline_agg is not None:
            print(f"\n✅ 各项指标稳定，无明显退化")


if __name__ == "__main__":
    main()
