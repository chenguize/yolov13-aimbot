# main.py - YOLOv13 Aimbot + Triggerbot 主控制模块（同步调用 + 限幅）
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
from output_ghub import gHub
import keyboard

running = True
shutdown_event = threading.Event()
paused = False

# ------------------ 信号处理 ------------------
def signal_handler(sig, frame):
    global running
    print("\n[Main] 收到退出信号，正在优雅关闭...")
    shutdown_event.set()
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ------------------ Triggerbot 封装 ------------------
def triggerbot_fire(target, controller, screen_center):
    """
    封装 Triggerbot 执行逻辑：
    - 容错盒判定
    - 三枪冷却 + 长时间未开枪重置（由 controller.should_trigger 处理）
    - 随机延迟模拟人类反应
    - 按下 → 持续 → 抬起
    """
    if not target:
        return

    if not controller.should_trigger(target, screen_center):
        return

    print(f"[Trigger] 目标已进入容错盒 (cls: {target['cls']})，准备开火")

    # 1. 随机延迟
    min_delay = config.getfloat("Triggerbot", "trigger_delay_min_ms", 15) / 1000
    max_delay = config.getfloat("Triggerbot", "trigger_delay_max_ms", 50) / 1000
    delay = random.uniform(min_delay, max_delay)
    if delay > 0:
        time.sleep(delay)

    # 2. 执行点击
    gHub.mouse_down(1)
    hold_time = config.getfloat("Triggerbot", "trigger_hold_time_ms", 30) / 1000
    time.sleep(hold_time)
    gHub.mouse_up(1)
    print(f"[Trigger] 开火完成 | 延迟: {delay*1000:.1f}ms | 保持: {hold_time*1000:.1f}ms")

# ------------------ 暂停开关 ------------------
def toggle_pause():
    global paused
    paused = not paused
    print(f"[Main] 项目已 {'暂停' if paused else '恢复'} (按 P 切换)")

# ------------------ 主循环 ------------------
def main():
    global running, paused

    print("[Main] YOLOv13 Aimbot + Triggerbot 启动 ... Ctrl+C 退出")
    print("[Main] 快捷键：P = 暂停/恢复")

    screen_center = (
        int(config.getint("General", "screen_width", 1920) / 2),
        int(config.getint("General", "screen_height", 1080) / 2)
    )
    print(f"[Main] 屏幕中心点: {screen_center}")

    frame_bus = FrameBus()
    world_model = WorldModel(frame_bus)
    controller = get_controller()
    strategy = create_aim_strategy()

    print(f"[Main] 使用控制器: {type(controller).__name__}")
    print(f"[Main] 使用瞄准策略: {type(strategy).__name__}")
    if hasattr(strategy, 'game_sens'):
        print(f"[Main] 游戏灵敏度 (game_sens): {strategy.game_sens}")

    capture = CaptureThread(frame_bus, shutdown_event)
    inference = InferenceThread(frame_bus, world_model, shutdown_event)
    capture.start()
    inference.start()

    keyboard.add_hotkey('p', toggle_pause)

    last_time = time.perf_counter()

    try:
        while running and not shutdown_event.wait(timeout=0.003):
            if paused:
                time.sleep(0.1)
                continue

            now = time.perf_counter()
            dt = max(now - last_time, 0.001)
            last_time = now

            target = world_model.get_best_target()

            # 输出当前帧检测结果
            if hasattr(world_model, 'current_detections') and world_model.current_detections:
                print("\n[Inference Raw] 当前帧检测结果（局部坐标）：")
                for det in world_model.current_detections:
                    print(f"  {det}")

            # 输出选中目标信息
            if target:
                print("\n[Target Selected] 最佳目标（屏幕绝对坐标）：")
                print(f"  screen_x = {target['screen_x']:.1f}")
                print(f"  screen_y = {target['screen_y']:.1f}")
                print(f"  conf     = {target['conf']:.3f}")
                print(f"  cls      = {target['cls']}")
                print(f"  width    = {target.get('width', 'N/A'):.1f}")
                print(f"  height   = {target.get('height', 'N/A'):.1f}")

                delta_x = target['screen_x'] - screen_center[0]
                delta_y = target['screen_y'] - screen_center[1]
                print("[Delta] 像素偏差（目标相对于中心）：")
                print(f"  dx = {delta_x:.1f} px")
                print(f"  dy = {delta_y:.1f} px")

            # ------------------ 自瞄逻辑 ------------------
            if target and config.getbool("General", "enable_aimbot", True):
                print("[Aimbot] 已开启自瞄 → 开始计算")
                DEADZONE = 0.5  # 死区
                MIN_MOVE = 1    # 最小移动阈值

                delta_x = target['screen_x'] - screen_center[0]
                delta_y = target['screen_y'] - screen_center[1]

                if abs(delta_x) < DEADZONE and abs(delta_y) < DEADZONE:
                    print(f"[Aimbot] 已在死区内 ({abs(delta_x):.1f}, {abs(delta_y):.1f})")
                    # 死区内也触发 Triggerbot
                    if config.getbool("General", "enable_triggerbot", False):
                        triggerbot_fire(target, controller, screen_center)
                    continue

                try:
                    print("[Strategy] 调用 calculate_mouse_move...")
                    intent_dx, intent_dy = strategy.calculate_mouse_move(
                        target["screen_x"],
                        target["screen_y"],
                        screen_center[0],
                        screen_center[1],
                        dt
                    )
                    print("[Strategy] 输出意图（counts）：")
                    print(f"  intent_dx = {intent_dx:.4f}")
                    print(f"  intent_dy = {intent_dy:.4f}")
                    print(f"  dt        = {dt * 1000:.2f} ms")

                    dx, dy = controller.compute(intent_dx, intent_dy, dt)
                    print("[Controller] PID 最终输出（counts）：")
                    print(f"  dx = {dx:.4f}")
                    print(f"  dy = {dy:.4f}")

                    if abs(dx) < MIN_MOVE and abs(dy) < MIN_MOVE:
                        print(f"[Aimbot] 移动量太小 ({abs(dx):.1f}, {abs(dy):.1f})")
                        # 小移动量下仍触发 Triggerbot
                        if config.getbool("General", "enable_triggerbot", False):
                            triggerbot_fire(target, controller, screen_center)
                        continue

                    print(f"[Final] 发送移动: dx={dx:.0f}, dy={dy:.0f}")
                    gHub.mouse_xy(int(dx), int(dy))

                except Exception as e:
                    print(f"[Main] 瞄准计算异常: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()

            # ------------------ Triggerbot 统一调用 ------------------
            if config.getbool("General", "enable_triggerbot", False):
                triggerbot_fire(target, controller, screen_center)

    except Exception as e:
        print(f"[Main] 主循环异常: {e}")

    finally:
        print("[Main] 正在关闭所有线程...")
        shutdown_event.set()
        time.sleep(0.5)
        capture.join(timeout=5.0)
        inference.join(timeout=5.0)
        keyboard.unhook_all()
        print("[Main] 系统已安全退出")


if __name__ == "__main__":
    main()
