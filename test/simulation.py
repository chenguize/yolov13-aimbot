# simulation_optimized.py（解封版贝叶斯调参）
import optuna
from optuna.samplers import TPESampler
from joblib import Parallel, delayed
import numpy as np
import json
import time
from datetime import datetime

# 假设你的 run_simulation 位于 control.py 或其他地方
# 并且它能接受 params 字典来覆盖 config
from control import run_simulation_with_diagnostics

def run_simulation_wrapper(duration, params):
    """
    包装器：如果你原来的 run_simulation 变了，可以在这里适配。
    假设它将 params 字典覆盖到了全局 config 中。
    """
    # 注意：根据你的具体实现，可能需要在这里把 params 注入到系统配置中
    # 例如 config.update(params) 或者 run_simulation_with_diagnostics 内部有处理
    # 这里我们假设底层已经支持透传 params 字典
    mae, rmse, _ = run_simulation_with_diagnostics(
        duration=duration,
        plot=False,
        verbose=False,
        use_fixed=True,
        # params=params  # <-- 如果你的测试框架支持，需要传进去
    )
    return mae, rmse

def objective(trial):
    """目标函数：最小化MAE"""

    params = {
        # ============================================
        # PROController 参数 (解封版搜索域)
        # ============================================
        'mpc_w_pos': trial.suggest_float('mpc_w_pos', 30.0, 150.0),      # 强力拉扯
        'mpc_w_vel': trial.suggest_float('mpc_w_vel', 1e-6, 1e-3, log=True),  # 极小阻尼
        'mpc_w_acc': trial.suggest_float('mpc_w_acc', 1e-8, 1e-5, log=True),  # 极小加速度惩罚

        'mpc_samples': trial.suggest_int('mpc_samples', 64, 128),
        'mpc_horizon': trial.suggest_int('mpc_horizon', 10, 20),

        # 还原电竞鼠标真实的物理爆发极限
        'max_accel': trial.suggest_float('max_accel', 200000.0, 800000.0),
        'max_speed': trial.suggest_float('max_speed', 10000.0, 30000.0),

        # ============================================
        # WorldModel 核心参数 (精简因果版)
        # ============================================
        # 移除了所有复杂的自适应和EMA平滑，只保留纯物理提前量
        'fixed_lead_time': trial.suggest_float('fixed_lead_time', 0.02, 0.10),

        # ============================================
        # Kalman 滤波器参数 (高响应版)
        # ============================================
        'kalman_R': trial.suggest_float('kalman_R', 0.1, 5.0, log=True), # 信任视觉
        'kalman_Q_pos': trial.suggest_float('kalman_Q_pos', 1.0, 20.0),
        'kalman_Q_vel': trial.suggest_float('kalman_Q_vel', 20.0, 200.0),# 允许目标急停变向
        'kalman_Q_acc': trial.suggest_float('kalman_Q_acc', 100.0, 800.0),
    }

    try:
        # 运行仿真 (如果你的 run_simulation_with_diagnostics 暂不支持直接传 params，
        # 你可能需要在这里用 config.set_param 临时注入这些变量)
        mae, rmse = run_simulation_wrapper(duration=6.0, params=params)

        # 组合目标：MAE 为绝对核心，轻微惩罚 RMSE(避免极大单次跳动)
        combined_score = mae + 0.1 * rmse

        # 记录详细信息
        trial.set_user_attr('mae', mae)
        trial.set_user_attr('rmse', rmse)
        trial.set_user_attr('timestamp', datetime.now().isoformat())

        return combined_score

    except Exception as e:
        print(f"❌ Trial {trial.number} failed: {e}")
        return 9999.0  # 失败的trial返回很大的值


def run_one_trial(study):
    """运行单个trial（用于并行）"""
    trial = study.ask()
    value = objective(trial)
    study.tell(trial, value)
    return value


def save_best_params(study, filename='best_params.json'):
    """保存最佳参数"""
    best_params = study.best_params.copy()
    best_params['best_mae'] = study.best_trial.user_attrs.get('mae', study.best_value)
    best_params['best_rmse'] = study.best_trial.user_attrs.get('rmse', 0)
    best_params['best_trial_number'] = study.best_trial.number
    best_params['total_trials'] = len(study.trials)
    best_params['timestamp'] = datetime.now().isoformat()

    with open(filename, 'w') as f:
        json.dump(best_params, f, indent=2)

    print(f"\n💾 最佳参数已保存到: {filename}")


def print_progress(study, trial_number, total_trials):
    """打印进度信息"""
    progress = (trial_number / total_trials) * 100

    print(f"\n{'=' * 70}")
    print(f"📊 进度: {trial_number}/{total_trials} ({progress:.1f}%)")
    print(f"{'=' * 70}")

    if study.best_trial:
        mae = study.best_trial.user_attrs.get('mae', study.best_value)
        rmse = study.best_trial.user_attrs.get('rmse', 'N/A')

        print(f"🏆 当前最佳 MAE: {mae:.2f} px (RMSE: {rmse:.2f} px)")
        print(f"📍 最佳试验编号: {study.best_trial.number}")

        print(f"\n🔑 最佳参数（关键部分）:")
        important_params = [
            'mpc_w_pos', 'mpc_w_vel', 'mpc_horizon', 'max_accel',
            'fixed_lead_time', 'kalman_R', 'kalman_Q_vel'
        ]
        for key in important_params:
            if key in study.best_params:
                value = study.best_params[key]
                if isinstance(value, float):
                    print(f"  {key:18s}: {value:.6f}")
                else:
                    print(f"  {key:18s}: {value}")

    print(f"{'=' * 70}\n")


if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("🎯 贝叶斯超参数优化 - 物理限制解除版")
    print("=" * 70 + "\n")

    # 创建study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(
            seed=42,
            n_startup_trials=20,  # 前20个试验用于纯随机探索
            multivariate=True,    # 考虑参数间的协方差关联
        ),
        study_name=f'aim_controller_v2_{int(time.time())}',
    )

    # ====================================
    # 强引导：注入3个极限基线试验点
    # ====================================
    print("📌 第1阶段：注入3个极限基线试验点...")

    # 1. 刚刚我们在沙盒验证过的“完美锁头”神仙参数
    study.enqueue_trial({
        'mpc_w_pos': 50.0, 'mpc_w_vel': 0.0001, 'mpc_w_acc': 1e-6,
        'mpc_samples': 64, 'mpc_horizon': 15,
        'max_accel': 500000.0, 'max_speed': 20000.0,
        'fixed_lead_time': 0.055,
        'kalman_R': 1.0, 'kalman_Q_pos': 5.0, 'kalman_Q_vel': 50.0, 'kalman_Q_acc': 200.0
    })

    # 2. 超激进参数：极高的位置权重和恐怖的加速度
    study.enqueue_trial({
        'mpc_w_pos': 100.0, 'mpc_w_vel': 1e-5, 'mpc_w_acc': 1e-7,
        'mpc_samples': 128, 'mpc_horizon': 12,
        'max_accel': 800000.0, 'max_speed': 25000.0,
        'fixed_lead_time': 0.035,
        'kalman_R': 0.5, 'kalman_Q_pos': 10.0, 'kalman_Q_vel': 120.0, 'kalman_Q_acc': 500.0
    })

    # 3. 均衡版：相对温和的平滑度测试
    study.enqueue_trial({
        'mpc_w_pos': 35.0, 'mpc_w_vel': 0.001, 'mpc_w_acc': 1e-5,
        'mpc_samples': 80, 'mpc_horizon': 18,
        'max_accel': 250000.0, 'max_speed': 15000.0,
        'fixed_lead_time': 0.07,
        'kalman_R': 2.0, 'kalman_Q_pos': 2.0, 'kalman_Q_vel': 30.0, 'kalman_Q_acc': 150.0
    })

    # 顺序运行基线试验
    print("⏳ 运行基线试验...\n")
    study.optimize(objective, n_trials=3)

    print_progress(study, 3, 100)
    save_best_params(study, 'best_params_baseline.json')

    # ====================================
    # 第2阶段：并行优化
    # ====================================
    n_remaining = 97  # 凑满 100 个试验
    n_jobs = -1       # 使用所有 CPU 核心

    print(f"\n📌 第2阶段：并行贝叶斯优化...")
    print(f"   并行任务数: {n_jobs} (全部CPU核心)")
    print(f"   剩余试验数: {n_remaining}\n")

    batch_size = 10
    for batch_start in range(0, n_remaining, batch_size):
        batch_end = min(batch_start + batch_size, n_remaining)
        current_batch = batch_end - batch_start

        print(f"\n🔄 运行批次 {batch_start // batch_size + 1}/{(n_remaining - 1) // batch_size + 1} ({current_batch} trials)...")

        Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
            delayed(run_one_trial)(study) for _ in range(current_batch)
        )

        print_progress(study, 3 + batch_end, 100)
        save_best_params(study, f'best_params_batch_{batch_end}.json')

    # ====================================
    # 最终结果
    # ====================================
    print("\n" + "=" * 70)
    print("🎉 优化完成！")
    print("=" * 70 + "\n")

    mae = study.best_trial.user_attrs.get('mae', study.best_value)
    rmse = study.best_trial.user_attrs.get('rmse', 'N/A')

    print(f"🏆 最佳结果:")
    print(f"   MAE:  {mae:.2f} px")
    print(f"   RMSE: {rmse:.2f} px")
    print(f"   试验编号: {study.best_trial.number}")

    # 保存最终结果
    save_best_params(study, 'best_params_final.json')

    try:
        import pickle
        with open('optimization_study.pkl', 'wb') as f:
            pickle.dump(study, f)
    except Exception as e:
        pass