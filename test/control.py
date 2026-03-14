# control.py
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ====================== 全局工具函数 ======================
def safe_mean(arr, default=0.0):
    return float(np.mean(arr)) if len(arr) > 0 else default


def analyze_tracking_quality(logs: List[Dict]) -> Dict:
    """
    Valorant-Specific Biomimetic Scoring v3.0
    专为《无畏契约》定制：极短TTK + 严苛首发精度 + 微调拟人特征(Micro-correction) + 零高频抖动
    """
    if not logs:
        return {}

    kills_data = {}
    for log in logs:
        kid = log['kill_id']
        kills_data.setdefault(kid, {'logs': [], 'mode': log['mode']})['logs'].append(log)

    metrics = {
        'pure_ai': {'acq_times': [], 'steady_maes': [], 'natural_scores': [], 'bio_bonuses': [], 'vel_rms': []},
        'human_flick': {'recovery_times': [], 'steady_maes': [], 'natural_scores': [], 'bio_bonuses': [], 'vel_rms': []}
    }

    print("\n" + "=" * 165)
    print(
        f"{'KillID':<6} | {'Mode':<10} | {'Acq/Rec(s)':<10} | {'SteadyMAE':<9} | {'Natural':<7} | {'BioBonus':<8} | {'VelRMS':<7} | Status")
    print("=" * 165)

    for kid, data in kills_data.items():
        k_logs = data['logs']
        if len(k_logs) < 12:
            continue

        times = np.array([l['t'] for l in k_logs])
        targets = np.array([l['target'] for l in k_logs])
        crosshairs = np.array([l['crosshair'] for l in k_logs])
        errors = np.linalg.norm(targets - crosshairs, axis=1)
        ai_factors = np.array([l.get('ai_factor', 1.0) for l in k_logs])

        see_idx = next((i for i, l in enumerate(k_logs) if l.get('is_valid', False)), 0)

        # 【锁定判定重构】：Valorant的头部很小，锁定阈值从 42px 缩紧到 15px
        lock_idx = None
        for i in range(see_idx, len(errors) - 10):
            if errors[i] < 15.0 and np.all(errors[i:i + 10] < 22.0):
                lock_idx = i
                break
        if lock_idx is None:
            lock_idx = see_idx + min(60, len(errors) - see_idx - 1)

        phase_time = times[lock_idx] - times[see_idx]
        steady_errors = errors[lock_idx:]

        raw_mae = float(np.mean(steady_errors)) if len(steady_errors) > 0 else 0.0

        # ==================== Naturalness v3.0 (Valorant Edition) ====================
        vel_rms = 0.0
        natural_score = 75.0
        if len(k_logs) > 5:
            dt_arr = np.diff(times)
            dt_arr[dt_arr < 1e-6] = 0.002
            vels = np.diff(crosshairs, axis=0) / dt_arr[:, None]
            vel_rms = float(np.mean(np.linalg.norm(vels, axis=1)))

            # 取消对高速度的惩罚（允许拉枪），但极度惩罚高频加速度（机械抖动）
            accel = np.diff(vels, axis=0)
            high_freq = float(np.mean(np.abs(accel[::2])))
            hf_penalty = max(0.0, high_freq / 40.0 * 25.0)  # 对高频颤抖更加敏感

            # 如果锁定时完全不动（像机器死锁），给予惩罚；需要有极其微弱的呼吸游离
            dead_lock_penalty = 15.0 if float(np.std(crosshairs[lock_idx:, 0])) < 0.1 else 0.0

            natural_score = np.clip(100.0 - hf_penalty - dead_lock_penalty, 0.0, 100.0)

        # ==================== Biomimetic Micro-correction v3.0 ====================
        # 奖励瓦罗兰特特征的微调：高速甩枪 -> 降速停顿且存在2~8px误差 -> 二次修正入魂
        bio_bonus = 0.0
        if lock_idx > see_idx + 5 and len(steady_errors) > 5:
            pre_lock_err = errors[lock_idx - 5: lock_idx]
            # 检查锁定前夕是否出现了轻微的 Under-flick 或 Over-flick（误差在 2 到 10 之间）
            if np.max(pre_lock_err) > 2.0 and np.min(pre_lock_err) < 18.0:
                # 检查这段时间内速度是否有明显下降（人类大脑在确认目标位置的短暂降速）
                pre_lock_dt = times[lock_idx] - times[lock_idx - 5]
                if pre_lock_dt > 0.008:  # 至少有短暂的停顿期
                    bio_bonus += 30.0

            # 如果 AI 是以一条极其完美的直线零误差砸在目标上（误差瞬间从几十掉到 <1.0），说明极其机械
            if errors[lock_idx] < 1.0 and np.mean(errors[lock_idx:lock_idx + 5]) < 1.0:
                bio_bonus -= 40.0  # 扣除死锁分

        bio_bonus = np.clip(bio_bonus, 0.0, 100.0)

        # 记录
        mode_key = data['mode']
        if mode_key == 'human_flick':
            takeover_idx = see_idx
            for i in range(see_idx, len(ai_factors)):
                if ai_factors[i] > 0.45 and (i == see_idx or ai_factors[i - 1] <= 0.45):
                    takeover_idx = i
                    break
            recovery_time = times[lock_idx] - times[takeover_idx] if lock_idx > takeover_idx else phase_time
            metrics['human_flick']['recovery_times'].append(recovery_time)
        else:
            metrics['pure_ai']['acq_times'].append(phase_time)

        metrics[mode_key]['steady_maes'].append(raw_mae)
        metrics[mode_key]['natural_scores'].append(natural_score)
        metrics[mode_key]['bio_bonuses'].append(bio_bonus)
        metrics[mode_key]['vel_rms'].append(vel_rms)

        # 【瓦罗兰特严苛评级标准】
        status = "✅ 爆头"
        if phase_time > 0.28:
            status = "⚠️ 反应慢"
        elif phase_time < 0.12:
            status = "🤖 机器瞬锁"

        if raw_mae > 12.0: status += " ⚠️ 空枪"

        if natural_score < 60: status += " (高频抖动)"
        if bio_bonus > 20: status += " ✨真人类微调"

        print(f"{kid:<6} | {'🧑 人机' if mode_key == 'human_flick' else '🤖 纯AI':<10} | "
              f"{phase_time:<10.3f} | {raw_mae:<9.2f} | {natural_score:<7.1f} | "
              f"+{bio_bonus:<6.1f} | {vel_rms:<7.0f} | {status}")

        # ====================== 最终评分 (V5.5 真实 Hitbox 版) ======================
    acq_all = metrics['pure_ai']['acq_times'] + metrics['human_flick'].get('recovery_times', [])
    mean_ttk = safe_mean(acq_all)
    acq_score = np.clip(100.0 - max(0.0, mean_ttk - 0.24) * 250.0, 0.0, 100.0)

    # 精度评分：允许 5px 以内的绝对完美，5px~15px 缓慢扣分（对应打在头边缘）
    mean_err = safe_mean(metrics['pure_ai']['steady_maes'] + metrics['human_flick']['steady_maes'])
    precision_score = np.clip(100.0 - max(0.0, mean_err - 5.0) * 8.0, 0.0, 100.0)

    natural_avg = safe_mean(metrics['pure_ai']['natural_scores'] + metrics['human_flick']['natural_scores'])
    bio_avg = safe_mean(metrics['pure_ai']['bio_bonuses'] + metrics['human_flick']['bio_bonuses'])

    overall_score = 0.35 * acq_score + 0.35 * precision_score + 0.15 * natural_avg + 0.15 * bio_avg

    analysis = {
        'total_kills': len(metrics['pure_ai']['steady_maes']) + len(metrics['human_flick']['steady_maes']),
        'overall_score': float(overall_score),
        'acq_score': float(acq_score),
        'precision_score': float(precision_score),
        'natural_score': float(natural_avg),
        'bio_bonus': float(bio_avg),
        'ttk_base_score': float((acq_score + precision_score) / 2),
        'jitter_penalty': float(max(0.0, (100 - natural_avg) * 1.4)),
        # MAE 惩罚放宽：8像素以内完全免罚
        'mae_penalty': float(max(0.0, (mean_err - 8.0) * 3.0)),
        'metrics': metrics,
    }

    print("\n" + "=" * 102)
    print(f"🎯 Valorant Biomimetic v3.0 Final Score: {overall_score:.1f}/100")
    print(f"   TTK (Acq/Rec) : {acq_score:.1f} | Headshot Precision : {precision_score:.1f}")
    print(f"   Smooth & Stop : {natural_avg:.1f} | Micro-adjust Bonus : +{bio_avg:.1f}")
    print("=" * 102)

    return analysis
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
    print("\n" + "=" * 75)
    print(f"🎮 {prefix} Valorant Biomimetic v3.0 深度诊断报告")
    print("=" * 75)
    print(f"【综合得分】: {analysis['overall_score']:8.1f} / 100")
    print(f"  TTK (Acq/Rec): {analysis['acq_score']:6.1f}   精度: {analysis['precision_score']:6.1f}")
    print(f"  平滑与急停   : {analysis['natural_score']:6.1f}   微调奖励: +{analysis['bio_bonus']:.1f}")
    print("-" * 75)

    if m['pure_ai']['acq_times']:
        print(f"🤖 [纯 AI 模式]")
        print(f"   - 平均 TTK: {safe_mean(m['pure_ai']['acq_times']):.3f}s | "
              f"平均误差: {safe_mean(m['pure_ai']['steady_maes']):.2f}px | "
              f"自然度: {safe_mean(m['pure_ai']['natural_scores']):.1f} | "
              f"微调: +{safe_mean(m['pure_ai']['bio_bonuses']):.1f}")

    if m['human_flick'].get('recovery_times'):
        print(f"🧑 [人机协同模式]")
        print(f"   - 平均恢复时间: {safe_mean(m['human_flick']['recovery_times']):.3f}s | "
              f"平均误差: {safe_mean(m['human_flick']['steady_maes']):.2f}px | "
              f"自然度: {safe_mean(m['human_flick']['natural_scores']):.1f} | "
              f"微调: +{safe_mean(m['human_flick']['bio_bonuses']):.1f}")

def run_simulation_with_diagnostics(
    plot: bool = True,
    verbose: bool = True,
    use_fixed: bool = True,
    duration: float = 60.0,
    params: Dict = None,
) -> Tuple[float, float, Dict]:
    from sim_agent import SimAIAgent
    from config import config
    config.reload()

    if params is not None:
        original_getfloat = config.getfloat
        def override_getfloat(section, key, fallback=None):
            if key in params:
                return float(params[key])
            if key == "base_hardware_lag" and "fixed_lead_time" in params:
                return float(params["fixed_lead_time"])
            return original_getfloat(section, key, fallback)
        config.getfloat = override_getfloat

    agent = SimAIAgent()
    sim_time, fixed_dt, logs = 0.0, 0.002, []

    if verbose:
        print(f"▶ 正在载入实战模拟环境...")

    while not agent.is_done and sim_time < duration:
        agent.step()
        logs.append({
            't':        sim_time,
            'target':   agent.enemy_pos.copy(),
            'crosshair':agent.crosshair_pos.copy(),
            'kill_id':  agent.kill_count,
            'mode':     agent.chase_mode,
            'is_valid': agent.ctx.p_predict is not None,
            'ai_factor':getattr(agent, 'last_ai_factor', 1.0),
        })
        sim_time += fixed_dt

    analysis = analyze_tracking_quality(logs)
    if verbose:
        print_analysis_report(analysis, "Optuna 调参版" if params else "诊断版")
    if plot:
        plot_diagnostics(logs, analysis, "诊断版")

    if params is not None:
        config.getfloat = original_getfloat

    return 0.0, 0.0, analysis


if __name__ == "__main__":
    run_simulation_with_diagnostics(use_fixed=True)