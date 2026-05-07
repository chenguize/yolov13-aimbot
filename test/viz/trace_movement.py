# _trace_movement.py
# 目的：让 AI 能"看见" 256px 以内的 crosshair 运动轨迹是否"一段一段"
# 输出：
#   1) 每次 motor program 启动事件 (time, dist, T, peak_v)
#   2) 每个 kill 的速度曲线（找峰/谷，识别 sub-movements）
#   3) BALLISTIC / TRACKING phase 切换时间线
#   4) 保存一个 PNG 图，用户可以肉眼确认
import argparse
import contextlib
import io
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)


def run_traced(seed: int = 317, duration: float = 8.0, scenario_type: str = "ball"):
    np.random.seed(seed)
    import random as pyrand
    pyrand.seed(seed)

    from test.sim_agent import SimAIAgent

    scenario = None
    if scenario_type == "takeover":
        from test.scenarios.takeover import TakeoverScenario
        scenario = TakeoverScenario(max_kills=30)
    elif scenario_type == "pure_ai":
        from test.scenarios.pure_ai_ball import PureAIBallScenario
        scenario = PureAIBallScenario(max_kills=30)

    agent = SimAIAgent(scenario=scenario)
    ctrl  = agent.world_model.controller

    events = []  # [(tag, time, dist_px, extra), ...]

    orig_smp = ctrl._start_motor_program
    def traced_smp(dist, error, head_r_counts):
        result = orig_smp(dist, error, head_r_counts)
        events.append((
            "SMP", agent.sim_time,
            dist / ctrl._px_to_ct,
            {"T": ctrl._prog_T, "peak_v_px": ctrl._prog_peak_v / ctrl._px_to_ct,
             "D_px": ctrl._prog_D / ctrl._px_to_ct,
             "undershoot_px": ctrl._prog_undershoot / ctrl._px_to_ct},
        ))
        return result
    ctrl._start_motor_program = traced_smp

    logs = []
    last_phase = ctrl._phase
    buf = io.StringIO()

    with contextlib.redirect_stdout(buf):
        while agent.sim_time < duration:
            agent.step()
            logs.append({
                "t":         agent.sim_time,
                "target":    agent.enemy_pos.copy(),
                "crosshair": agent.crosshair_pos.copy(),
                "phase":     ctrl._phase,
                "arm_vel":   float(np.linalg.norm(ctrl._arm_vel)),
                "wrist_vel": float(np.linalg.norm(ctrl._wrist_vel)),
                "err_px":    ctrl.last_error_dist / ctrl._px_to_ct,
                "ai_factor": getattr(agent, "last_ai_factor", 0.0),
                "chase_mode": agent.chase_mode,
            })
            if ctrl._phase != last_phase:
                events.append((
                    "PHASE_" + ("BALLISTIC" if ctrl._phase == 0 else "TRACKING"),
                    agent.sim_time,
                    ctrl.last_error_dist / ctrl._px_to_ct,
                    None,
                ))
                last_phase = ctrl._phase

    return logs, events


def analyze(logs, events, title: str):
    times      = np.array([l["t"] for l in logs])
    crosshairs = np.array([l["crosshair"] for l in logs])
    targets    = np.array([l["target"] for l in logs])
    err_pxs    = np.array([l["err_px"] for l in logs])
    phases     = np.array([l["phase"] for l in logs])
    chase_modes = [l["chase_mode"] for l in logs]

    dt = float(np.median(np.diff(times)))
    vels   = np.diff(crosshairs, axis=0) / dt
    speeds = np.linalg.norm(vels, axis=1)  # px/s
    vel_t  = times[:-1]

    smp_events = [e for e in events if e[0] == "SMP"]
    n_ticks    = len(logs)

    print(f"\n===== {title} =====")
    print(f"总时长 {times[-1]:.2f}s / {n_ticks} ticks")
    print(f"Motor program 启动次数: {len(smp_events)} ({len(smp_events)/max(times[-1],1e-3):.2f} /s)")
    print(f"Crosshair 速度: mean={speeds.mean():.0f} max={speeds.max():.0f} px/s")

    # 256 区间内 (error <= 256px) 的速度统计
    err_head = err_pxs[:-1]
    mask256  = err_head <= 256.0
    mask_band_high = (err_head > 50.0) & (err_head <= 256.0)  # BALLISTIC 区
    mask_band_mid  = (err_head > 15.0) & (err_head <= 50.0)   # 接近 TRACKING
    mask_band_low  = err_head <= 15.0                          # lock

    def band_stat(mask, name):
        if mask.sum() < 5:
            print(f"  [{name:<18}] 样本不足")
            return
        s = speeds[mask]
        print(f"  [{name:<18}] n={mask.sum():>5}  "
              f"speed mean={s.mean():>6.0f}  median={np.median(s):>6.0f}  "
              f"p10={np.percentile(s,10):>5.0f}  p90={np.percentile(s,90):>5.0f}  "
              f"低速占比(<200 px/s)={float(np.mean(s<200))*100:.1f}%")

    print("\n  按 error 距离分区的 crosshair 速度统计:")
    band_stat(mask_band_high, "50 < err ≤ 256 px")
    band_stat(mask_band_mid,  "15 < err ≤ 50  px")
    band_stat(mask_band_low,  "     err ≤ 15  px")

    # 每次 motor program 后 200ms 内的速度曲线 ── 检测"一段一段"
    print("\n  每次 Motor program 启动后 200ms 的速度轨迹 (采样 10 点):")
    for i, (_, t_smp, dist, extra) in enumerate(smp_events[:10]):
        idx0 = int(np.searchsorted(times, t_smp))
        idx1 = int(np.searchsorted(times, t_smp + 0.20))
        if idx1 - idx0 < 5:
            continue
        # 采 10 个点均匀分布
        sample_idx = np.linspace(idx0, idx1-2, 10).astype(int)
        sample_idx = np.clip(sample_idx, 0, len(speeds)-1)
        s_trace = speeds[sample_idx]
        # 检测 sub-movements：峰 → 谷 → 峰
        sub = 0
        last_val = 0
        rising = False
        for v in s_trace:
            if v > last_val + 100 and not rising:
                rising = True
                sub += 1
            elif v < last_val - 100 and rising:
                rising = False
            last_val = v
        print(f"    #{i+1:>2}  t={t_smp:.3f}  启动 dist={dist:>5.1f}px  "
              f"T={extra['T']*1000:.0f}ms  peak_v={extra['peak_v_px']:.0f}px/s  "
              f"under={extra['undershoot_px']:.1f}px")
        print(f"        速度曲线(px/s): " + " ".join(f"{int(v):>5}" for v in s_trace)
              + f"   子动作数≈{sub}")

    # 绘图
    fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axs[0].plot(times, err_pxs, "k-", lw=0.8)
    axs[0].axhline(15,  color="g", ls=":", lw=0.8, label="15px (lock)")
    axs[0].axhline(50,  color="orange", ls=":", lw=0.8, label="50px (thresh_high)")
    axs[0].axhline(256, color="r", ls=":", lw=0.8, label="256px (AI FOV)")
    axs[0].set_ylabel("error (px)")
    axs[0].set_yscale("symlog", linthresh=10)
    axs[0].legend(loc="upper right", fontsize=8)
    axs[0].grid(alpha=0.3)

    axs[1].plot(vel_t, speeds, "b-", lw=0.6)
    axs[1].set_ylabel("crosshair speed (px/s)")
    axs[1].grid(alpha=0.3)
    # overlay SMP markers
    for tag, tt, dd, ex in smp_events:
        axs[1].axvline(tt, color="red", alpha=0.25, lw=0.7)
    axs[1].set_ylim(0, max(200, np.percentile(speeds, 99) * 1.1))

    axs[2].plot(times, phases, "g-", lw=0.8, drawstyle="steps-post")
    axs[2].set_ylabel("phase (0=BALL,1=TRK)")
    axs[2].set_xlabel("time (s)")
    axs[2].grid(alpha=0.3)
    # chase_mode 背景色
    for i in range(len(times)-1):
        if chase_modes[i] == "human_flick":
            axs[2].axvspan(times[i], times[i+1], color="yellow", alpha=0.05)

    plt.suptitle(title)
    plt.tight_layout()
    out_path = os.path.join(_ROOT, f"trace_{title.replace(' ','_')}.png")
    plt.savefig(out_path, dpi=110)
    plt.close()
    print(f"  [saved] trajectory figure -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Motor program 深度追踪诊断")
    parser.add_argument("--scenario", type=str, default="ball",
                        choices=["ball", "takeover", "pure_ai"], help="场景类型")
    parser.add_argument("--seed", type=int, default=317, help="随机种子")
    parser.add_argument("--duration", type=float, default=8.0, help="仿真时长 (秒)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"[Trace] {args.scenario} scenario — crosshair movement diagnostics")
    print("=" * 60)
    logs, events = run_traced(seed=args.seed, duration=args.duration, scenario_type=args.scenario)
    analyze(logs, events, args.scenario)


if __name__ == "__main__":
    main()
