# main.py - YOLOv13 Aimbot + Triggerbot 主控制模块
#
# 核心职责：
# 1. 系统初始化（创建各模块实例）
# 2. 多线程管理（捕获、推理、输出线程）
# 3. 主控制循环（目标检测、瞄准计算、触发控制）
# 4. 优雅关闭（信号处理、资源清理）
#
# 系统架构：完全异步 + 状态驱动 + 世界模型 + 独立高频控制环
#
# 重要优化（2025.12.24改动记录）：
#   - 射击游戏中鼠标位置恒等于屏幕中心（十字准星）
#   - 删除实时 get_current_mouse_pos()，统一使用固定 screen_center
#   - 避免采集→推理延迟导致的坐标误差（15~35ms 内鼠标可能移动 5~20px）
#   - 瞄准 & Triggerbot 均以屏幕中心为参考点，更准确、更稳定
#   - 删除所有预测逻辑（按需求：只要映射，不需要预测）
#   - 增强 Triggerbot 随机延迟（更像人）

import threading
import time
import signal
import sys
import random
from typing import Tuple
from config import config
from perception.capture import CaptureThread
from perception.bus import FrameBus
from inference import InferenceThread
from world_model import WorldModel
from controllers.controller_factory import get_controller
from aim_strategies.factory import create_aim_strategy
from output_ghub import GHUBOutput, gHub  # 使用全局 gHub 实例（DLL 原生调用）
from controllers.humanize.reaction_delay import ReactionDelay
from controllers.humanize.noise import Noise
from controllers.humanize.fatigue import Fatigue
from controllers.humanize.overshoot import Overshoot
from controllers.humanize.curve import Curve
import keyboard  # 新增：全局键盘监听


# 全局控制变量
running = True
shutdown_event = threading.Event()
paused = False  # 新增：暂停状态


def signal_handler(sig, frame):
    global running
    print("\n[Main] 收到退出信号，正在优雅关闭...")
    shutdown_event.set()
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def toggle_pause():
    """P 键切换暂停/恢复"""
    global paused
    paused = not paused
    if paused:
        print("[Main] 项目已暂停 (按 P 恢复)")
    else:
        print("[Main] 项目已恢复")


def main():
    global running, paused

    print("[Main] YOLOv13 Aimbot + Triggerbot 启动 (2025.12.26版) ... Ctrl+C 退出")
    print("[Main] 快捷键：P = 暂停/恢复")

    # 计算屏幕中心（射击游戏中鼠标位置恒等于此点，使用整数）
    screen_center = (
        int(config.getint("General", "screen_width", 1920) / 2),
        int(config.getint("General", "screen_height", 1080) / 2)
    )

    # 初始化核心模块（保持不变）
    frame_bus = FrameBus()
    world_model = WorldModel(frame_bus)
    controller = get_controller()
    strategy = create_aim_strategy()
    output = GHUBOutput(world_model)

    # humanize 模块实例
    h_reaction = ReactionDelay()
    h_noise = Noise()
    h_fatigue = Fatigue()
    h_overshoot = Overshoot()
    h_curve = Curve()

    prev_dx, prev_dy = 0.0, 0.0

    # 启动线程
    capture = CaptureThread(frame_bus, shutdown_event)
    inference = InferenceThread(frame_bus, world_model, shutdown_event)
    output.start()
    capture.start()
    inference.start()

    # 注册 P 键监听（全局热键）
    keyboard.add_hotkey('p', toggle_pause)

    last_time = time.perf_counter()

    try:
        while running and not shutdown_event.wait(timeout=0.003):
            if paused:
                time.sleep(0.1)  # 暂停时降低 CPU 占用
                continue

            now = time.perf_counter()
            dt = max(now - last_time, 1e-6)
            last_time = now

            target = world_model.get_best_target()

            # Aimbot 处理（保持原逻辑）
            if target and config.getbool("General", "enable_aimbot", False):
                dx, dy = strategy.calculate_mouse_move(
                    target["screen_x"],
                    target["screen_y"],
                    screen_center[0],
                    screen_center[1],
                    dt
                )

                dx, dy = controller.compute(target, screen_center, dt)

                h_reaction.apply_delay()
                dx, dy = h_noise.apply(dx, dy)
                dx, dy = h_fatigue.apply(dx, dy)
                dx, dy = h_overshoot.apply(dx, dy, prev_dx, prev_dy)

                t = min(1.0, (abs(dx) + abs(dy)) / 50.0)
                dx *= h_curve.apply(t)
                dy *= h_curve.apply(t)

                prev_dx, prev_dy = dx, dy

                output.send_move(dx, dy)

            # Triggerbot 处理
            if config.getbool("General", "enable_triggerbot", False) and controller.should_trigger(target, screen_center):
                min_delay = config.getfloat("Triggerbot", "trigger_delay_min_ms", 15) / 1000
                max_delay = config.getfloat("Triggerbot", "trigger_delay_max_ms", 80) / 1000
                delay = random.uniform(min_delay, max_delay)
                time.sleep(delay)
                gHub.mouse_down(1)
                hold_time = config.getfloat("Triggerbot", "trigger_hold_time_ms", 40) / 1000
                time.sleep(hold_time)
                gHub.mouse_up(1)

    except Exception as e:
        print(f"[Main] 主循环异常: {e}")

    finally:
        print("[Main] 正在关闭所有线程...")
        shutdown_event.set()
        time.sleep(0.5)
        capture.join(timeout=5.0)
        inference.join(timeout=5.0)
        output.join(timeout=5.0)
        keyboard.unhook_all()  # 清理热键
        print("[Main] 系统已安全退出")


if __name__ == "__main__":
    main()