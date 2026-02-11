# analyze_optimization.py（优化结果分析工具）
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import optuna


def load_study(study_path='optimization_study.pkl'):
    """加载保存的study"""
    try:
        with open(study_path, 'rb') as f:
            study = pickle.load(f)
        return study
    except FileNotFoundError:
        print(f"❌ 找不到文件: {study_path}")
        return None


def plot_optimization_history(study):
    """绘制优化历史"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('贝叶斯优化过程分析', fontsize=16, fontweight='bold')

    # 1. MAE随试验次数的变化
    ax = axes[0, 0]
    trial_numbers = [t.number for t in study.trials]
    maes = [t.user_attrs.get('mae', t.value) for t in study.trials]

    ax.plot(trial_numbers, maes, 'o-', alpha=0.6, markersize=4, label='每次试验')

    # 绘制最佳值曲线
    best_so_far = []
    current_best = float('inf')
    for mae in maes:
        current_best = min(current_best, mae)
        best_so_far.append(current_best)

    ax.plot(trial_numbers, best_so_far, 'r-', linewidth=2, label='历史最佳')
    ax.set_xlabel('试验编号')
    ax.set_ylabel('MAE [px]')
    ax.set_title('优化收敛曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. MAE分布直方图
    ax = axes[0, 1]
    ax.hist(maes, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(min(maes), color='red', linestyle='--', linewidth=2,
               label=f'最佳: {min(maes):.2f}px')
    ax.axvline(np.median(maes), color='green', linestyle='--', linewidth=2,
               label=f'中位数: {np.median(maes):.2f}px')
    ax.set_xlabel('MAE [px]')
    ax.set_ylabel('频数')
    ax.set_title('MAE分布')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 3. 参数重要性（如果有足够的试验）
    ax = axes[1, 0]
    try:
        importances = optuna.importance.get_param_importances(study)
        # 取前10个最重要的参数
        top_params = dict(sorted(importances.items(),
                                key=lambda x: x[1],
                                reverse=True)[:10])

        params = list(top_params.keys())
        values = list(top_params.values())

        ax.barh(params, values, color='coral')
        ax.set_xlabel('重要性')
        ax.set_title('参数重要性排名（Top 10）')
        ax.grid(True, alpha=0.3, axis='x')

    except Exception as e:
        ax.text(0.5, 0.5, f'参数重要性分析失败\n{str(e)}',
                ha='center', va='center', fontsize=10)
        ax.set_title('参数重要性')

    # 4. 关键参数vs MAE散点图
    ax = axes[1, 1]

    # 选择最重要的参数（w_pos）
    key_param = 'w_pos'
    if key_param in study.best_params:
        param_values = [t.params.get(key_param, None) for t in study.trials]
        param_values = [v for v in param_values if v is not None]
        param_maes = [t.user_attrs.get('mae', t.value)
                     for t in study.trials
                     if key_param in t.params]

        ax.scatter(param_values, param_maes, alpha=0.6, s=50)
        ax.set_xlabel(f'{key_param}')
        ax.set_ylabel('MAE [px]')
        ax.set_title(f'{key_param} vs MAE')
        ax.grid(True, alpha=0.3)

        # 标注最佳点
        best_trial = study.best_trial
        if key_param in best_trial.params:
            ax.scatter([best_trial.params[key_param]],
                      [best_trial.user_attrs.get('mae', best_trial.value)],
                      color='red', s=200, marker='*',
                      edgecolors='black', linewidths=2,
                      label='最佳配置', zorder=5)
            ax.legend()

    plt.tight_layout()
    plt.savefig('optimization_analysis.png', dpi=150, bbox_inches='tight')
    print("\n📊 分析图已保存: optimization_analysis.png")
    plt.show()


def compare_configurations(study, top_n=5):
    """比较最佳的N个配置"""

    # 按MAE排序
    sorted_trials = sorted(study.trials,
                          key=lambda t: t.user_attrs.get('mae', t.value))

    print("\n" + "="*70)
    print(f"🏆 Top {top_n} 配置对比")
    print("="*70 + "\n")

    for i, trial in enumerate(sorted_trials[:top_n], 1):
        mae = trial.user_attrs.get('mae', trial.value)
        rmse = trial.user_attrs.get('rmse', 'N/A')

        print(f"第 {i} 名 - Trial #{trial.number}")
        print(f"   MAE:  {mae:.2f} px")
        print(f"   RMSE: {rmse:.2f} px" if rmse != 'N/A' else f"   RMSE: N/A")

        # 显示关键参数
        print(f"   关键参数:")
        important = ['w_pos', 'w_vel', 'w_acc', 'horizon', 'samples',
                    'perception_lag', 'alpha_flick', 'alpha_track']
        for key in important:
            if key in trial.params:
                value = trial.params[key]
                if isinstance(value, float):
                    print(f"      {key:18s}: {value:.4f}")
                else:
                    print(f"      {key:18s}: {value}")
        print()


def export_for_code(study, output_file='best_params_code.py'):
    """导出为可直接使用的Python代码"""

    best = study.best_params
    mae = study.best_trial.user_attrs.get('mae', study.best_value)

    code = f'''# 自动生成的最佳参数配置
# MAE: {mae:.2f} px
# 生成时间: {Path(__file__).stat().st_mtime if Path(__file__).exists() else "N/A"}

BEST_PARAMS = {{
    # PROController 参数
    'w_pos': {best.get('w_pos', 25.0):.6f},
    'w_vel': {best.get('w_vel', 0.05):.6f},
    'w_acc': {best.get('w_acc', 0.02):.6f},
    'samples': {best.get('samples', 150)},
    'horizon': {best.get('horizon', 15)},
    'max_accel': {best.get('max_accel', 45000.0):.1f},
    'max_speed': {best.get('max_speed', 6000.0):.1f},
    
    # WorldModel 参数
    'perception_lag': {best.get('perception_lag', 0.025):.6f},
    'lead_gain': {best.get('lead_gain', 2.5):.6f},
    'lead_max': {best.get('lead_max', 0.22):.6f},
    'lag_lr': {best.get('lag_lr', 0.018):.6f},
    'lag_alpha': {best.get('lag_alpha', 0.15):.6f},
    'lag_deadzone': {best.get('lag_deadzone', 1.5):.6f},
    'lag_min': {best.get('lag_min', 0.025):.6f},
    'lag_max': {best.get('lag_max', 0.10):.6f},
    'alpha_flick': {best.get('alpha_flick', 0.70):.6f},
    'alpha_track': {best.get('alpha_track', 0.60):.6f},
    'head_width_m': {best.get('head_width_m', 0.22):.6f},
    
    # Kalman 参数
    'kalman_R': {best.get('kalman_R', 5.0):.6f},
    'kalman_Q_pos': {best.get('kalman_Q_pos', 0.02):.6f},
    'kalman_Q_vel': {best.get('kalman_Q_vel', 10.0):.6f},
    'kalman_Q_acc': {best.get('kalman_Q_acc', 150.0):.6f},
    'confidence_threshold': {best.get('confidence_threshold', 0.5):.6f},
}}

# 使用示例:
# from best_params_code import BEST_PARAMS
# mae, rmse = run_simulation(duration=8.0, params=BEST_PARAMS)
'''

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(code)

    print(f"\n💾 可执行代码已导出到: {output_file}")


def print_summary(study):
    """打印优化摘要"""

    maes = [t.user_attrs.get('mae', t.value) for t in study.trials]

    print("\n" + "="*70)
    print("📊 优化结果摘要")
    print("="*70 + "\n")

    print(f"总试验次数:      {len(study.trials)}")
    print(f"完成的试验:      {len([t for t in study.trials if t.state.name == 'COMPLETE'])}")
    print(f"失败的试验:      {len([t for t in study.trials if t.state.name == 'FAIL'])}")

    print(f"\n最佳 MAE:        {min(maes):.2f} px")
    print(f"中位数 MAE:      {np.median(maes):.2f} px")
    print(f"平均 MAE:        {np.mean(maes):.2f} px")
    print(f"标准差:          {np.std(maes):.2f} px")

    # 改进幅度（相比基线）
    if len(maes) >= 4:
        baseline_mae = maes[0]  # 第一个试验作为基线
        best_mae = min(maes)
        improvement = ((baseline_mae - best_mae) / baseline_mae) * 100
        print(f"\n相比基线改进:    {improvement:.1f}%")
        print(f"   (基线: {baseline_mae:.2f} px → 最佳: {best_mae:.2f} px)")

    print("\n" + "="*70)


if __name__ == "__main__":

    print("\n" + "="*70)
    print("📈 优化结果分析工具")
    print("="*70 + "\n")

    # 加载study
    study_path = 'optimization_study.pkl'
    print(f"📂 加载Study: {study_path}")

    study = load_study(study_path)

    if study is None:
        print("\n⚠️  找不到Study文件，请先运行优化:")
        print("   python simulation_optimized.py\n")
        exit(1)

    # 打印摘要
    print_summary(study)

    # 对比Top 5配置
    compare_configurations(study, top_n=5)

    # 绘制分析图
    print("\n📊 生成可视化分析...")
    plot_optimization_history(study)

    # 导出代码
    export_for_code(study)

    print("\n✨ 分析完成！")
    print("   - 查看 optimization_analysis.png 了解优化过程")
    print("   - 查看 best_params_code.py 获取可用参数")
    print()