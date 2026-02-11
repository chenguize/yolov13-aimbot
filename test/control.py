# control_enhanced.py
# 增强版仿真控制和诊断工具
# 核心功能：
# 1. 详细的性能分析
# 2. 可视化诊断
# 3. 原版vs修复版对比测试

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


def analyze_tracking_quality(logs: List[Dict]) -> Dict:
    """
    分析跟踪质量

    Returns:
        包含多种性能指标的字典
    """
    if not logs:
        return {}

    # 提取时间序列数据
    times = np.array([log['t'] for log in logs])
    targets = np.array([log['target'] for log in logs])
    crosshairs = np.array([log['crosshair'] for log in logs])

    # ========== 精度指标 ==========
    errors = targets - crosshairs
    distances = np.linalg.norm(errors, axis=1)

    mae = np.mean(distances)
    rmse = np.sqrt(np.mean(distances ** 2))
    max_error = np.max(distances)
    p95_error = np.percentile(distances, 95)
    p99_error = np.percentile(distances, 99)

    # ========== 速度指标 ==========
    dt = np.diff(times)
    velocities = np.diff(crosshairs, axis=0) / dt[:, np.newaxis]
    speeds = np.linalg.norm(velocities, axis=1)

    mean_speed = np.mean(speeds)
    max_speed = np.max(speeds)
    speed_std = np.std(speeds)

    # ========== 加速度指标 ==========
    if len(velocities) > 1:
        accelerations = np.diff(velocities, axis=0) / dt[1:, np.newaxis]
        accels = np.linalg.norm(accelerations, axis=1)

        mean_accel = np.mean(accels)
        max_accel = np.max(accels)
        accel_std = np.std(accels)
    else:
        mean_accel = max_accel = accel_std = 0.0

    # ========== 平滑度指标（高频抖动）==========
    jitter_x = np.std(np.diff(crosshairs[:, 0]))
    jitter_y = np.std(np.diff(crosshairs[:, 1]))

    # ========== 响应延迟估计 ==========
    # 计算误差与目标速度的相位关系
    if len(logs) > 10:
        # 简化：使用误差峰值与目标峰值的时间差
        target_peaks_x = np.where(np.diff(np.sign(np.diff(targets[:, 0]))) < 0)[0] + 1
        error_peaks_x = np.where(np.diff(np.sign(np.diff(errors[:, 0]))) < 0)[0] + 1

        if len(target_peaks_x) > 0 and len(error_peaks_x) > 0:
            # 平均相位滞后
            avg_lag_samples = np.mean([
                np.abs(target_peaks_x[0] - ep)
                for ep in error_peaks_x[:min(3, len(error_peaks_x))]
            ])
            avg_lag_time = avg_lag_samples * np.mean(dt) if len(dt) > 0 else 0.0
        else:
            avg_lag_time = 0.0
    else:
        avg_lag_time = 0.0

    # ========== 模式统计 ==========
    modes = [log.get('mode', 'unknown') for log in logs]
    mode_counts = {}
    for mode in modes:
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    # ========== 综合评分 ==========
    # 评分公式：精度(60%) + 平滑度(20%) + 响应(20%)
    accuracy_score = max(0, 100 - mae * 2)  # MAE<5px得满分
    smoothness_score = max(0, 100 - jitter_x * 10)  # 低抖动得高分
    responsiveness_score = max(0, 100 - avg_lag_time * 1000)  # 低延迟得高分

    overall_score = (
            accuracy_score * 0.6 +
            smoothness_score * 0.2 +
            responsiveness_score * 0.2
    )

    return {
        # 精度指标
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'p95_error': p95_error,
        'p99_error': p99_error,

        # 速度指标
        'mean_speed': mean_speed,
        'max_speed': max_speed,
        'speed_std': speed_std,

        # 加速度指标
        'mean_accel': mean_accel,
        'max_accel': max_accel,
        'accel_std': accel_std,

        # 平滑度指标
        'jitter_x': jitter_x,
        'jitter_y': jitter_y,

        # 延迟估计
        'avg_lag_time': avg_lag_time,

        # 模式统计
        'modes': modes,
        'mode_distribution': mode_counts,

        # 综合评分
        'overall_score': overall_score,
        'accuracy_score': accuracy_score,
        'smoothness_score': smoothness_score,
        'responsiveness_score': responsiveness_score
    }


def plot_diagnostics(logs: List[Dict], analysis: Dict, title_prefix: str = ""):
    """
    绘制详细诊断图表
    """
    times = np.array([log['t'] for log in logs])
    targets = np.array([log['target'] for log in logs])
    crosshairs = np.array([log['crosshair'] for log in logs])
    errors = targets - crosshairs
    distances = np.linalg.norm(errors, axis=1)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    main_title = f"{title_prefix}诊断报告 | 综合评分: {analysis['overall_score']:.1f}/100"
    fig.suptitle(main_title, fontsize=16, fontweight='bold')

    # ========== 1. X轴跟踪 ==========
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(times, targets[:, 0], 'r--', linewidth=2, label='Target X', alpha=0.7)
    ax.plot(times, crosshairs[:, 0], 'g-', linewidth=1.5, label='Crosshair X')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('X Position [px]')
    ax.set_title('X-Axis Tracking')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ========== 2. Y轴跟踪 ==========
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(times, targets[:, 1], 'r--', linewidth=2, label='Target Y', alpha=0.7)
    ax.plot(times, crosshairs[:, 1], 'b-', linewidth=1.5, label='Crosshair Y')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Y Position [px]')
    ax.set_title('Y-Axis Tracking')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ========== 3. 误差时间序列 ==========
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(times, distances, color='orange', linewidth=1.5, label='Distance Error')
    ax.axhline(analysis['mae'], color='green', linestyle='--',
               linewidth=2, label=f'MAE: {analysis["mae"]:.2f}px')
    ax.axhline(analysis['p95_error'], color='red', linestyle='--',
               linewidth=1, label=f'P95: {analysis["p95_error"]:.2f}px')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Error [px]')
    ax.set_title('Tracking Error Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ========== 4. 误差分布直方图 ==========
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(distances, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(analysis['mae'], color='green', linestyle='--',
               linewidth=2, label=f'MAE: {analysis["mae"]:.2f}px')
    ax.axvline(analysis['p95_error'], color='red', linestyle='--',
               linewidth=2, label=f'P95: {analysis["p95_error"]:.2f}px')
    ax.set_xlabel('Error [px]')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ========== 5. 速度曲线 ==========
    ax = fig.add_subplot(gs[1, 1])
    if len(times) > 1:
        dt = np.diff(times)
        velocities = np.diff(crosshairs, axis=0) / dt[:, np.newaxis]
        speeds = np.linalg.norm(velocities, axis=1)
        ax.plot(times[1:], speeds, color='purple', linewidth=1.5)
        ax.axhline(analysis['mean_speed'], color='blue', linestyle='--',
                   linewidth=1, label=f'Mean: {analysis["mean_speed"]:.1f}px/s')
        ax.axhline(analysis['max_speed'], color='red', linestyle=':',
                   linewidth=1, label=f'Max: {analysis["max_speed"]:.1f}px/s')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Speed [px/s]')
        ax.set_title('Crosshair Speed')
        ax.legend()
    ax.grid(True, alpha=0.3)

    # ========== 6. 加速度曲线 ==========
    ax = fig.add_subplot(gs[1, 2])
    if len(times) > 2:
        dt = np.diff(times)
        velocities = np.diff(crosshairs, axis=0) / dt[:, np.newaxis]
        accelerations = np.diff(velocities, axis=0) / dt[1:, np.newaxis]
        accels = np.linalg.norm(accelerations, axis=1)
        ax.plot(times[2:], accels, color='brown', linewidth=1.5)
        ax.axhline(analysis['mean_accel'], color='blue', linestyle='--',
                   linewidth=1, label=f'Mean: {analysis["mean_accel"]:.1f}px/s²')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Acceleration [px/s²]')
        ax.set_title('Crosshair Acceleration')
        ax.legend()
    ax.grid(True, alpha=0.3)

    # ========== 7. 2D轨迹图 ==========
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(targets[:, 0], targets[:, 1], 'r--', linewidth=2,
            label='Target', alpha=0.6)
    ax.plot(crosshairs[:, 0], crosshairs[:, 1], 'g-', linewidth=1.5,
            label='Crosshair', alpha=0.8)
    ax.set_xlabel('X Position [px]')
    ax.set_ylabel('Y Position [px]')
    ax.set_title('2D Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # ========== 8. X误差分量 ==========
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(times, errors[:, 0], color='coral', linewidth=1.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('X Error [px]')
    ax.set_title('X-Axis Error')
    ax.grid(True, alpha=0.3)

    # ========== 9. Y误差分量 ==========
    ax = fig.add_subplot(gs[2, 2])
    ax.plot(times, errors[:, 1], color='teal', linewidth=1.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Y Error [px]')
    ax.set_title('Y-Axis Error')
    ax.grid(True, alpha=0.3)

    plt.show()


def print_analysis_report(analysis: Dict, prefix: str = ""):
    """打印详细分析报告"""

    print("\n" + "=" * 70)
    print(f"📊 {prefix}跟踪质量分析报告")
    print("=" * 70)

    print("\n【精度指标】")
    print(f"  MAE (平均绝对误差):     {analysis['mae']:8.2f} px")
    print(f"  RMSE (均方根误差):      {analysis['rmse']:8.2f} px")
    print(f"  最大误差:               {analysis['max_error']:8.2f} px")
    print(f"  P95 误差:              {analysis['p95_error']:8.2f} px")
    print(f"  P99 误差:              {analysis['p99_error']:8.2f} px")

    print("\n【运动特性】")
    print(f"  平均速度:               {analysis['mean_speed']:8.1f} px/s")
    print(f"  最大速度:               {analysis['max_speed']:8.1f} px/s")
    print(f"  速度标准差:             {analysis['speed_std']:8.1f} px/s")
    print(f"  平均加速度:             {analysis['mean_accel']:8.1f} px/s²")
    print(f"  最大加速度:             {analysis['max_accel']:8.1f} px/s²")
    print(f"  加速度标准差:           {analysis['accel_std']:8.1f} px/s²")

    print("\n【平滑度指标】")
    print(f"  X轴抖动 (std):         {analysis['jitter_x']:8.2f} px")
    print(f"  Y轴抖动 (std):         {analysis['jitter_y']:8.2f} px")

    print("\n【响应延迟】")
    print(f"  估计延迟:               {analysis['avg_lag_time'] * 1000:8.1f} ms")

    print("\n【模式分布】")
    total_frames = len(analysis['modes'])
    for mode, count in analysis['mode_distribution'].items():
        percentage = (count / total_frames) * 100
        print(f"  {mode:12s}: {count:6d} 次 ({percentage:5.1f}%)")

    print("\n【综合评分】")
    print(f"  精度评分:               {analysis['accuracy_score']:8.1f} / 100")
    print(f"  平滑度评分:             {analysis['smoothness_score']:8.1f} / 100")
    print(f"  响应性评分:             {analysis['responsiveness_score']:8.1f} / 100")
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  总分:                   {analysis['overall_score']:8.1f} / 100")

    # 评级
    score = analysis['overall_score']
    if score >= 90:
        grade = "🌟 卓越 (Excellent)"
    elif score >= 75:
        grade = "✅ 优秀 (Good)"
    elif score >= 60:
        grade = "⚠️  合格 (Fair)"
    else:
        grade = "❌ 需要改进 (Poor)"

    print(f"  等级:                   {grade}")
    print("=" * 70 + "\n")


def run_simulation_with_diagnostics(
        duration: float = 8.0,
        plot: bool = True,
        verbose: bool = True,
        use_fixed: bool = True
) -> Tuple[float, float, Dict]:
    """
    运行带完整诊断的仿真

    Args:
        duration: 仿真时长（秒）
        plot: 是否绘图
        verbose: 是否打印详细报告
        use_fixed: 是否使用修复版控制器
    """
    # 选择版本
    if use_fixed:
        try:
            from sim_agent_improved import SimAIAgent
            print("✅ 使用修复版控制器")
        except ImportError:
            print("⚠️  修复版未找到，使用原版")
            from sim_agent import SimAIAgent
    else:
        from sim_agent import SimAIAgent
        print("📌 使用原版控制器")

    agent = SimAIAgent()

    # 固定时间步长仿真
    fixed_dt = 0.002  # 2ms
    sim_time = 0.0
    next_step_time = time.perf_counter()

    logs = []

    print(f"▶ 运行仿真 (dt={fixed_dt * 1000:.1f}ms, duration={duration}s)")

    while sim_time < duration:
        now = time.perf_counter()

        # 精确时间控制
        if now < next_step_time:
            time.sleep(max(0, next_step_time - now))

        # 执行仿真步骤
        agent.step()

        # 记录日志
        logs.append({
            't': sim_time,
            'target': agent.enemy_pos.copy(),
            'crosshair': agent.crosshair_pos.copy(),
            'velocity': agent.ctx.v_real if agent.ctx.v_real else (0, 0),
            'mode': agent.world_model.controller.mode
        })

        sim_time += fixed_dt
        next_step_time += fixed_dt

    # 分析结果
    analysis = analyze_tracking_quality(logs)

    if verbose:
        prefix = "修复版 " if use_fixed else "原版 "
        print_analysis_report(analysis, prefix)

    if plot:
        title_prefix = "修复版 " if use_fixed else "原版 "
        plot_diagnostics(logs, analysis, title_prefix)

    return analysis['mae'], analysis['rmse'], analysis


def compare_versions(duration: float = 8.0):
    """
    对比原版和修复版性能
    """
    print("\n" + "=" * 70)
    print("🔬 开始版本对比测试")
    print("=" * 70)

    # 测试原版
    print("\n📍 测试原版控制器...")
    mae_orig, rmse_orig, analysis_orig = run_simulation_with_diagnostics(
        duration=duration,
        plot=False,
        verbose=False,
        use_fixed=False
    )
    print_analysis_report(analysis_orig, "原版")

    # 测试修复版
    print("\n📍 测试修复版控制器...")
    mae_fixed, rmse_fixed, analysis_fixed = run_simulation_with_diagnostics(
        duration=duration,
        plot=False,
        verbose=False,
        use_fixed=True
    )
    print_analysis_report(analysis_fixed, "修复版")

    # 对比报告
    print("\n" + "=" * 70)
    print("📊 版本对比总结")
    print("=" * 70)

    improvement_mae = (mae_orig - mae_fixed) / mae_orig * 100
    improvement_score = (
            analysis_fixed['overall_score'] - analysis_orig['overall_score']
    )

    print(f"\n{'指标':<20} {'原版':>12} {'修复版':>12} {'改进':>12}")
    print("-" * 70)
    print(f"{'MAE (px)':<20} {mae_orig:>12.2f} {mae_fixed:>12.2f} {improvement_mae:>11.1f}%")
    print(f"{'RMSE (px)':<20} {rmse_orig:>12.2f} {rmse_fixed:>12.2f}")
    print(f"{'综合评分':<20} {analysis_orig['overall_score']:>12.1f} "
          f"{analysis_fixed['overall_score']:>12.1f} {improvement_score:>+11.1f}")
    print("-" * 70)

    if improvement_mae > 0:
        print(f"\n✅ 修复版性能提升 {improvement_mae:.1f}%")
    else:
        print(f"\n⚠️  修复版性能下降 {abs(improvement_mae):.1f}%")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    # 单独测试修复版
    print("🚀 测试修复版控制器")
    mae, rmse, analysis = run_simulation_with_diagnostics(
        duration=8.0,
        plot=True,
        verbose=True,
        use_fixed=True
    )

    # 可选：运行对比测试
    # compare_versions(duration=8.0)