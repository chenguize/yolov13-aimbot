# simulation.py  ── v5.1 瓦罗兰特专属修复版
import numpy as np
import optuna
import multiprocessing
from optuna.samplers import CmaEsSampler
from optuna.pruners import MedianPruner
from control import run_simulation_with_diagnostics


def run_once(duration, params, seed):
    np.random.seed(seed)
    result = run_simulation_with_diagnostics(
        duration=duration,
        plot=False,
        verbose=False,
        use_fixed=True,
        params=params,
    )
    return result[2]  # analysis dict


def _extract_metrics(analysis: dict) -> tuple[float, float, int, float]:
    overall_score = analysis.get("overall_score", 0.0)
    precision_score = analysis.get("precision_score", 0.0)
    total_kills = analysis.get("total_kills", 0)

    # 提取原始稳态误差，专门用于在 0 分地带提供“引路梯度”
    m = analysis.get("metrics", {})
    maes = m.get('pure_ai', {}).get('steady_maes', []) + m.get('human_flick', {}).get('steady_maes', [])
    raw_mae = float(np.mean(maes)) if maes else 999.0

    return overall_score, precision_score, total_kills, raw_mae


def objective(trial):
    # ── 终极搜索空间 v5.4 (修复物理摩擦死锁) ──────────────────
    params = {
        # 1. 物理极限 (释放爆发力，允许瞬间拉枪)
        "max_accel": trial.suggest_float("max_accel", 50000.0, 150000.0),
        "max_speed": trial.suggest_float("max_speed", 5000.0, 12000.0),

        # 2. 拟人手腕阻尼 (寻找 500Hz 下最丝滑的物理惯性)
        "wrist_alpha": trial.suggest_float("wrist_alpha", 0.15, 0.65),

        # 3. 大脑刚性配置 (注意 r_track 的真实量级 1e-5 到 1e-3)
        "r_track": trial.suggest_float("r_track", 1e-5, 2e-3, log=True),
        "q_pos_track": trial.suggest_float("q_pos_track", 15.0, 80.0),
        "q_vel_track": trial.suggest_float("q_vel_track", 0.1, 1.5),

        "r_flick": trial.suggest_float("r_flick", 1e-7, 1e-4, log=True),
        "q_pos_flick": trial.suggest_float("q_pos_flick", 1.0, 15.0),
        "q_vel_flick": trial.suggest_float("q_vel_flick", 1e-5, 1e-3, log=True),

        # 4. 卡尔曼与环境参数 (保持不变，或微调)
        "deadzone_scale": trial.suggest_float("deadzone_scale", 0.05, 0.15),
        "velocity_friction": trial.suggest_float("velocity_friction", 0.980, 0.998),
        "ki_gain": trial.suggest_float("ki_gain", 0.005, 0.050),
        "ff_acc_gain": trial.suggest_float("ff_acc_gain", 0.10, 0.50),

        # 5. 生理硬直时间 (搜索最佳的反应死区 80ms~140ms)
        "coldstart_ramp_ms": trial.suggest_float("coldstart_ramp_ms", 80.0, 140.0),
    }

    # ── 物理约束软剪枝 ─────────────────────────
    deadzone_px = 60.0 * params["deadzone_scale"]
    v_at_deadzone = (2.0 * params["max_accel"] * deadzone_px) ** 0.5
    if v_at_deadzone > params["max_speed"] * 1.8:
        raise optuna.exceptions.TrialPruned()

    validation_seeds = [42, 1337, 2025]
    validation_durations = [15.0, 25.0, 35.0]

    trial_scores = []

    for step, (seed, duration) in enumerate(zip(validation_seeds, validation_durations)):
        analysis = run_once(duration=duration, params=params, seed=seed)
        overall_score, precision_score, total_kills, raw_mae = _extract_metrics(analysis)

        # 惩罚项同步放宽
        score_penalty = 0.0
        if total_kills < 5:
            score_penalty += (5 - total_kills) * 20.0
        if precision_score < 1.0:
            score_penalty += raw_mae * 1.5  # 降低惩罚斜率

        final_score = overall_score - score_penalty
        trial_scores.append(final_score)

        trial.report(final_score, step)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    mean_score = float(np.mean(trial_scores))
    score_std = float(np.std(trial_scores))

    robust_score = mean_score - score_std * 0.8
    return -robust_score

def optimize_worker(db_url, study_name, n_trials):
    study = optuna.load_study(study_name=study_name, storage=db_url)
    study.optimize(objective, n_trials=n_trials)

if __name__ == "__main__":
    print("\n🚀 启动 CMA-ES 炼丹炉 (V5.5 真实Hitbox 收敛版)\n")

    # 🔥 重命名数据库，开启最后一次探索
    db_url = "sqlite:///aimbot_valorant_v5_5.db"
    study_name = "pro_tuning_valorant_v5_5"

    pruner = optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=1)

    study = optuna.create_study(
        study_name=study_name,
        storage=db_url,
        direction="minimize",
        sampler=optuna.samplers.CmaEsSampler(seed=42, warn_independent_sampling=True),
        pruner=pruner,
        load_if_exists=True
    )

    n_workers = 10
    total_trials = 300
    trials_per_worker = total_trials // n_workers

    processes = []
    for i in range(n_workers):
        p = multiprocessing.Process(
            target=optimize_worker,
            args=(db_url, study_name, trials_per_worker)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n" + "=" * 50)
    print(f"🏆 炼丹完成！最高鲁棒得分: {-study.best_value:.2f}")
    print("=" * 50)

    print("【最优参数】:")
    best_p = study.best_params
    for k, v in sorted(best_p.items()):
        if any(x in k for x in ['q_vel', 'r_', 'ki_gain']) or k == 'R':
            print(f"  {k:22s} = {v:.6e}")
        else:
            print(f"  {k:22s} = {v:.4f}")

    print("\n生成参数重要性图...")
    try:
        import optuna.visualization as vis
        fig = vis.plot_param_importances(study)
        fig.show()
    except Exception as e:
        print(f"可视化失败: {e}")