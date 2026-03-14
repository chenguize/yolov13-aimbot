# animate_trajectory.py
import matplotlib

matplotlib.use('TkAgg')

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Dict


def run_minimal_simulation(duration: float = 5.0) -> List[Dict]:
    """静默运行仿真，只收集轨迹数据"""

    # 【修复1】记录真实的系统底层时钟，防止被模拟器永久劫持
    original_perf_counter = time.perf_counter

    # 【修复2】强制从 sim_agent 导入，确保使用你最新修改的 ring_buffer 逻辑
    from sim_agent import SimAIAgent

    agent = SimAIAgent()
    fixed_dt = 0.002
    sim_time = 0.0
    logs = []

    print(f"⚙️ 正在收集轨迹数据 (时长: {duration}秒)...")
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


def animate_mouse_trajectory(logs: List[Dict], save_path: str = None):
    """绘制动态鼠标追踪轨迹（全局固定视角 + 完整轨迹留存）"""
    if not logs:
        return

    targets = np.array([log['target'] for log in logs])
    crosshairs = np.array([log['crosshair'] for log in logs])

    fig, ax = plt.subplots(figsize=(12, 8))
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

    step = 10
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

        target_path.set_data(t_hist[:, 0], t_hist[:, 1])
        crosshair_path.set_data(c_hist[:, 0], c_hist[:, 1])

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
        fps_realtime = int(1.0 / (0.002 * step))
        print(f"\n🎬 正在渲染全景视图视频 (FPS: {fps_realtime}) 并保存至 {save_path} ...")
        try:
            writer = animation.FFMpegWriter(fps=fps_realtime, bitrate=3000)
            ani.save(save_path, writer=writer)
            print(f"✅ 全景视频保存成功！")
        except Exception as e:
            print(f"❌ 导出时发生未知错误: {e}")

    plt.show()

if __name__ == "__main__":
    # 运行仿真
    trajectory_logs = run_minimal_simulation(duration=20.0)

    # 传入 save_path 参数触发视频导出
    animate_mouse_trajectory(trajectory_logs, save_path="tracking_result.mp4")
    #animate_mouse_trajectory(trajectory_logs)