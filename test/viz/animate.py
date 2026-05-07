# animate_trajectory.py
import argparse
import os
import sys
import matplotlib as mpl

# 后端须在 import pyplot 之前选定：批量导出用 Agg（更快、无 Tk）；需要弹窗用 --interactive
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--interactive", action="store_true", help=argparse.SUPPRESS)
_pre_args, _remaining = _pre.parse_known_args()
mpl.use("TkAgg" if _pre_args.interactive else "Agg")

import time
import subprocess
import shutil
import platform
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Dict, Optional

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

_SCRIPT_DIR = Path(__file__).resolve().parent
# 默认同目录输出，双击运行或 cwd 任意时都能找到文件
DEFAULT_VIDEO_PATH = _SCRIPT_DIR / "tracking_result.mp4"


def _open_default_video(path: str) -> None:
    """用系统默认播放器打开视频（导出后即时观看，无需手点文件）。"""
    p = Path(path).resolve()
    if not p.is_file():
        return
    try:
        if platform.system() == "Windows":
            os.startfile(str(p))  # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(["xdg-open", str(p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"▶ 已用默认程序打开 {p}")
    except Exception as e:
        print(f"⚠ 无法自动打开视频（请手动打开）: {e}")
def _decimate_path(xy: np.ndarray, max_points: int) -> np.ndarray:
    """路径点过多时均匀抽稀，避免每帧重画整条线变成近似 O(n²) 拖死 CPU。"""
    n = xy.shape[0]
    if n <= max_points or max_points < 2:
        return xy
    idx = np.linspace(0, n - 1, max_points, dtype=np.int64)
    return xy[idx]


def export_opencv_ffmpeg(
    logs: List[Dict],
    save_path: str,
    *,
    step: int,
    max_path_points: int,
    dpi: float,
    nvenc: bool,
    fixed_dt: float = 0.002,
) -> float:
    """用 OpenCV 画帧，经 pipe 喂给 ffmpeg；比 Matplotlib 光栅化快一个数量级以上。

    NVENC 只占导出时间的一小部分（每帧几毫秒级），大头仍是 CPU 画线；任务管理器里请看
    「视频编码 / Video Encode」队列而非「3D」。
    """
    if cv2 is None:
        raise RuntimeError("导出 engine=opencv 需要安装 opencv-python")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("未找到 ffmpeg，请安装并加入 PATH")

    targets = np.array([log['target'] for log in logs])
    crosshairs = np.array([log['crosshair'] for log in logs])

    pad = 100.0
    all_x = np.concatenate([targets[:, 0], crosshairs[:, 0]])
    all_y = np.concatenate([targets[:, 1], crosshairs[:, 1]])
    global_min_x, global_max_x = np.min(all_x) - pad, np.max(all_x) + pad
    global_min_y, global_max_y = np.min(all_y) - pad, np.max(all_y) + pad
    x_range = global_max_x - global_min_x
    y_range = global_max_y - global_min_y
    max_range = max(x_range, y_range) / 2.0
    mid_x = (global_max_x + global_min_x) / 2.0
    mid_y = (global_max_y + global_min_y) / 2.0

    W = max(320, int(round(12 * dpi)))
    H = max(240, int(round(8 * dpi)))
    world_span = 2.0 * max_range
    scale = min(W, H) / world_span
    cx_f, cy_f = W / 2.0, H / 2.0

    def to_pix(xy: np.ndarray) -> np.ndarray:
        x, y = xy[:, 0], xy[:, 1]
        px = np.rint(cx_f + (x - mid_x) * scale).astype(np.int32)
        py = np.rint(cy_f - (y - mid_y) * scale).astype(np.int32)
        return np.stack([px, py], axis=1)

    fps = max(1, int(round(1.0 / (fixed_dt * max(1, step)))))
    hit_r_px = max(1, int(round(15.0 * scale)))

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}",
        "-r", str(fps),
        "-i", "-",
    ]
    if nvenc:
        cmd.extend([
            "-c:v", "h264_nvenc",
            "-pix_fmt", "yuv420p",
            "-preset", "p4",
            "-b:v", "5M",
        ])
    else:
        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-crf", "23",
        ])
    cmd.append(save_path)

    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdin is None:
        raise RuntimeError("无法打开 ffmpeg stdin")

    try:
        for frame_idx in range(0, len(logs), step):
            img = np.full((H, W, 3), 255, dtype=np.uint8)

            t_hist = targets[0 : frame_idx + 1]
            c_hist = crosshairs[0 : frame_idx + 1]
            t_draw = _decimate_path(t_hist, max_path_points)
            c_draw = _decimate_path(c_hist, max_path_points)

            if len(t_draw) > 1:
                tp = to_pix(t_draw).reshape(-1, 1, 2)
                cv2.polylines(img, [tp], False, (0, 0, 255), 1, cv2.LINE_AA)
            if len(c_draw) > 1:
                cp = to_pix(c_draw).reshape(-1, 1, 2)
                cv2.polylines(img, [cp], False, (255, 0, 0), 2, cv2.LINE_AA)

            ctx, cty = float(targets[frame_idx, 0]), float(targets[frame_idx, 1])
            chx, chy = float(crosshairs[frame_idx, 0]), float(crosshairs[frame_idx, 1])
            tcx, tcy = to_pix(np.array([[ctx, cty]], dtype=np.float64))[0]
            qx, qy = to_pix(np.array([[chx, chy]], dtype=np.float64))[0]

            cv2.circle(img, (int(tcx), int(tcy)), hit_r_px, (0, 0, 200), 1, cv2.LINE_AA)
            cv2.circle(img, (int(tcx), int(tcy)), 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (int(qx), int(qy)), 4, (255, 0, 0), -1, cv2.LINE_AA)

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            proc.stdin.write(rgb.tobytes())
    finally:
        proc.stdin.close()

    err = b""
    if proc.stderr:
        err = proc.stderr.read()
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg 退出码 {code}: {err.decode('utf-8', errors='replace').strip()}")

    return time.perf_counter() - t0


def run_minimal_simulation(duration: float = 5.0, scenario_type: str = "ball") -> List[Dict]:
    """静默运行仿真，只收集轨迹数据"""
    import os, sys
    # 仓库根必须排在 sys.path 最前；否则在 test/ 下执行时 cwd 会干扰 import test.*
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT in sys.path:
        sys.path.remove(_ROOT)
    sys.path.insert(0, _ROOT)

    # 【修复1】记录真实的系统底层时钟，防止被模拟器永久劫持
    original_perf_counter = time.perf_counter

    from test.sim_agent import SimAIAgent

    # 场景选择
    scenario = None
    if scenario_type == "takeover":
        from test.scenarios.takeover import TakeoverScenario
        scenario = TakeoverScenario(max_kills=30)
    elif scenario_type == "pure_ai":
        from test.scenarios.pure_ai_ball import PureAIBallScenario
        scenario = PureAIBallScenario(max_kills=30)

    agent = SimAIAgent(scenario=scenario)
    fixed_dt = 0.002
    sim_time = 0.0
    logs = []

    print(f"⚙️ 正在收集轨迹数据 (场景: {scenario_type}, 时长: {duration}秒)...")
    while sim_time < duration:
        agent.step()
        logs.append({
            'target': agent.enemy_pos.copy(),
            'crosshair': agent.crosshair_pos.copy()
        })
        sim_time += fixed_dt

    # 【修复3】仿真结束，必须把时钟归还给系统！否则动画库会陷入时间静止
    time.perf_counter = original_perf_counter

    print("✅ 数据收集完毕，系统时钟已恢复，准备播放动画！")
    return logs


def animate_mouse_trajectory(
    logs: List[Dict],
    save_path: Optional[str] = None,
    *,
    step: int = 20,
    max_path_points: int = 4000,
    dpi: float = 72.0,
    nvenc: bool = True,
    interactive: bool = False,
    engine: str = "opencv",
    auto_open: bool = True,
):
    """绘制动态鼠标追踪轨迹（全局固定视角 + 完整轨迹留存）。

    导出视频时，轨迹会按 ``max_path_points`` 抽稀，避免 Matplotlib 每帧重画整条折线导致
    CPU 单核打满、总耗时近似 O(n²)。Matplotlib 2D 绘制基本不走 GPU；``--nvenc`` 只加速
    最后一步视频编码；``engine=opencv`` 可大幅缩短「画帧」时间。
    """
    if not logs:
        return

    if save_path and engine == "opencv":
        n_frames = len(range(0, len(logs), max(1, step)))
        fps_rt = max(1, int(round(1.0 / (0.002 * max(1, step)))))
        enc = "NVENC" if nvenc else "libx264"
        print(
            f"\n🎬 OpenCV + ffmpeg({enc})：{n_frames} 帧，FPS={fps_rt} → {save_path}"
        )
        exported_ok = False
        try:
            dt = export_opencv_ffmpeg(
                logs,
                save_path,
                step=max(1, step),
                max_path_points=max_path_points,
                dpi=dpi,
                nvenc=nvenc,
            )
            print(f"✅ 导出成功（{dt:.1f}s）")
            exported_ok = True
        except RuntimeError as e:
            if nvenc:
                print(f"⚠️ NVENC 失败，改用 CPU 编码: {e}")
                try:
                    dt = export_opencv_ffmpeg(
                        logs,
                        save_path,
                        step=max(1, step),
                        max_path_points=max_path_points,
                        dpi=dpi,
                        nvenc=False,
                    )
                    print(f"✅ 导出成功（libx264，{dt:.1f}s）")
                    exported_ok = True
                except RuntimeError as e2:
                    print(f"❌ 导出失败: {e2}")
            else:
                print(f"❌ 导出失败: {e}")
        if exported_ok and auto_open:
            _open_default_video(save_path)
        elif interactive:
            print("⚠️ 实时 Matplotlib 窗口请用: --engine matplotlib --interactive")
        return

    targets = np.array([log['target'] for log in logs])
    crosshairs = np.array([log['crosshair'] for log in logs])

    fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)
    ax.set_title("Biomimetic Aim Tracking (Global Fixed View)", fontsize=14, fontweight='bold')
    ax.set_xlabel("X Position [px]")
    ax.set_ylabel("Y Position [px]")
    ax.grid(True, linestyle='--', alpha=0.5)

    # 【新增：计算全局最大边界，把摄像机死死锁在全景】
    all_x = np.concatenate([targets[:, 0], crosshairs[:, 0]])
    all_y = np.concatenate([targets[:, 1], crosshairs[:, 1]])
    global_min_x, global_max_x = np.min(all_x) - 100, np.max(all_x) + 100
    global_min_y, global_max_y = np.min(all_y) - 100, np.max(all_y) + 100

    # 保持长宽比一致，防止画面拉伸变形
    x_range = global_max_x - global_min_x
    y_range = global_max_y - global_min_y
    max_range = max(x_range, y_range) / 2.0
    mid_x = (global_max_x + global_min_x) / 2.0
    mid_y = (global_max_y + global_min_y) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)

    # 绘制元素初始化
    target_path, = ax.plot([], [], 'r--', lw=1.5, alpha=0.3, label='Target Path (Basketball)')
    crosshair_path, = ax.plot([], [], 'b-', lw=1.5, alpha=0.6, label='Crosshair Trajectory')
    target_dot, = ax.plot([], [], 'ro', markersize=6, label='Enemy Head')
    crosshair_dot, = ax.plot([], [], 'bo', markersize=6, label='Crosshair')

    # 爆头判定线
    hitbox = plt.Circle((0, 0), 15.0, color='red', fill=False, linestyle=':', lw=1.5, alpha=0.8)
    ax.add_patch(hitbox)

    ax.legend(loc='upper right')

    frames_to_render = range(0, len(logs), step)

    def init():
        target_path.set_data([], [])
        crosshair_path.set_data([], [])
        target_dot.set_data([], [])
        crosshair_dot.set_data([], [])
        hitbox.center = (0, 0)
        return target_path, crosshair_path, target_dot, crosshair_dot, hitbox

    def update(frame_idx):
        # 【取消拖尾截断，从第0帧开始画，显示完整的“毛线团”轨迹】
        tail_start = 0

        t_hist = targets[tail_start:frame_idx + 1]
        c_hist = crosshairs[tail_start:frame_idx + 1]
        t_draw = _decimate_path(t_hist, max_path_points)
        c_draw = _decimate_path(c_hist, max_path_points)

        target_path.set_data(t_draw[:, 0], t_draw[:, 1])
        crosshair_path.set_data(c_draw[:, 0], c_draw[:, 1])

        curr_t_x, curr_t_y = targets[frame_idx, 0], targets[frame_idx, 1]
        target_dot.set_data([curr_t_x], [curr_t_y])
        crosshair_dot.set_data([crosshairs[frame_idx, 0]], [crosshairs[frame_idx, 1]])
        hitbox.center = (curr_t_x, curr_t_y)

        # 注意：这里彻底删除了原来动态设置 set_xlim 和 set_ylim 的逻辑

        return target_path, crosshair_path, target_dot, crosshair_dot, hitbox

    ani = animation.FuncAnimation(
        fig, update, frames=frames_to_render,
        init_func=init, interval=20, blit=False, repeat=False
    )

    if save_path:
        fps_realtime = max(1, int(round(1.0 / (0.002 * step))))
        n_frames = len(range(0, len(logs), step))
        enc = "NVENC(GPU 编码)" if nvenc else "libx264(CPU 编码)"
        print(
            f"\n🎬 Matplotlib：约 {n_frames} 帧，FPS={fps_realtime}，{enc} → {save_path}"
        )
        save_ok = False
        extra_args = (
            ["-vcodec", "h264_nvenc", "-pix_fmt", "yuv420p", "-preset", "p4", "-b:v", "5M"]
            if nvenc
            else []
        )
        try:
            writer = animation.FFMpegWriter(fps=fps_realtime, bitrate=3000, extra_args=extra_args)
            ani.save(save_path, writer=writer, dpi=dpi)
            save_ok = True
            print(f"✅ 已保存")
        except Exception as e:
            if nvenc:
                print(f"⚠️ NVENC 失败，改用 CPU 编码重试: {e}")
                try:
                    w2 = animation.FFMpegWriter(fps=fps_realtime, bitrate=3000)
                    ani.save(save_path, writer=w2, dpi=dpi)
                    save_ok = True
                    print(f"✅ 已保存（CPU 编码）")
                except Exception as e2:
                    print(f"❌ 导出错误: {e2}")
            else:
                print(f"❌ 导出错误: {e}")
        if save_ok and auto_open and not interactive:
            _open_default_video(save_path)

    if interactive:
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="仿真轨迹 → 视频（零参数即：pure_ai / 20s / OpenCV 快导 / 尝试 NVENC / 结束自动打开）",
    )
    parser.add_argument("--scenario", type=str, default="takeover",
                        choices=["ball", "takeover", "pure_ai"], help="场景（默认 pure_ai）")
    parser.add_argument("--duration", type=float, default=20.0, help="仿真时长（秒），默认 20")
    parser.add_argument(
        "--save",
        type=str,
        default=str(DEFAULT_VIDEO_PATH),
        help=f"输出 mp4 路径（默认 {DEFAULT_VIDEO_PATH.name} 在脚本同目录）",
    )
    parser.add_argument("--stride", type=int, default=20, help="抽帧步长，默认 20")
    parser.add_argument("--max-path-points", type=int, default=4000, dest="max_path_points")
    parser.add_argument("--dpi", type=float, default=72.0, help="导出分辨率系数，默认 72")
    parser.add_argument(
        "--no-nvenc",
        action="store_true",
        help="禁用 NVENC，仅用 CPU libx264",
    )
    parser.add_argument("--no-open", action="store_true", help="导出完成后不自动打开视频")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Matplotlib 弹窗实时播放（慢；与默认快导二选一体验）",
    )
    parser.add_argument(
        "--engine",
        choices=["opencv", "matplotlib"],
        default="opencv",
        help="opencv（默认，快）或 matplotlib（图例/网格全）",
    )
    args = parser.parse_args()

    trajectory_logs = run_minimal_simulation(duration=args.duration, scenario_type=args.scenario)

    save = args.save if args.save.strip() else None
    engine = args.engine
    if engine == "opencv" and save is None:
        print("⚠️ 未指定保存路径，改用 matplotlib")
        engine = "matplotlib"

    animate_mouse_trajectory(
        trajectory_logs,
        save_path=save,
        step=max(1, args.stride),
        max_path_points=max(256, args.max_path_points),
        dpi=max(50.0, args.dpi),
        nvenc=not args.no_nvenc,
        interactive=args.interactive,
        engine=engine,
        auto_open=not args.no_open,
    )