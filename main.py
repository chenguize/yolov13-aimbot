# main.py
import threading
import time
import signal
import sys
from typing import Tuple
import win32api
from config import config
from perception.capture import CaptureThread
from perception.bus import FrameBus
from inference import InferenceThread
from world_model import WorldModel
from controllers.controller_factory import get_controller
from aim_strategies.factory import create_aim_strategy
from output_ghub import GHUBOutput, gHub  # 使用你提供的全局 gHub
from controllers.humanize.reaction_delay import ReactionDelay
from controllers.humanize.noise import Noise
from controllers.humanize.fatigue import Fatigue
from controllers.humanize.overshoot import Overshoot
from controllers.humanize.curve import Curve


# 全局控制
running = True
shutdown_event = threading.Event()


def signal_handler(sig, frame):
    global running
    print("\n[Main] 收到退出信号，正在优雅关闭...")
    shutdown_event.set()
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_current_mouse_pos() -> Tuple[int, int]:
    return win32api.GetCursorPos()


def main():
    global running

    print("[Main] YOLOv13 Aimbot + Triggerbot 启动 (2025.12.26版) ... Ctrl+C 退出")

    # 初始化
    frame_bus = FrameBus()
    world_model = WorldModel(frame_bus)
    controller = get_controller()
    strategy = create_aim_strategy()
    output = GHUBOutput(world_model)

    # humanize 模块（即使全关闭也零开销）
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

    last_time = time.perf_counter()

    try:
        while running and not shutdown_event.wait(timeout=0.003):
            now = time.perf_counter()
            dt = max(now - last_time, 1e-6)
            last_time = now

            current_pos = get_current_mouse_pos()
            target = world_model.get_best_target()

            # Aimbot
            if target and config.getbool("General", "enable_aimbot", False):
                # 预测
                pred_x, pred_y = strategy.calculate_prediction(
                    target["screen_x"], target["screen_y"], dt
                )
                target["screen_x"] = pred_x
                target["screen_y"] = pred_y

                # 计算移动
                dx, dy = controller.compute(target, current_pos, dt)

                # Humanize 完整管道
                h_reaction.apply_delay()
                dx, dy = h_noise.apply(dx, dy)
                dx, dy = h_fatigue.apply(dx, dy)
                dx, dy = h_overshoot.apply(dx, dy, prev_dx, prev_dy)

                # 曲线
                t = min(1.0, (abs(dx) + abs(dy)) / 50.0)
                dx *= h_curve.apply(t)
                dy *= h_curve.apply(t)

                prev_dx, prev_dy = dx, dy

                output.send_move(dx, dy)

            # Triggerbot
            if config.getbool("General", "enable_triggerbot", False) and controller.should_trigger(target, current_pos):
                delay = config.getfloat("Triggerbot", "trigger_delay_min_ms", 15) / 1000
                time.sleep(delay)
                gHub.mouse_down(1)
                time.sleep(config.getfloat("Triggerbot", "trigger_hold_time_ms", 40) / 1000)
                gHub.mouse_up(1)

    except Exception as e:
        print(f"[Main] 主循环异常: {e}")

    finally:
        print("[Main] 正在关闭...")
        shutdown_event.set()
        time.sleep(0.5)
        capture.join(timeout=2)
        inference.join(timeout=2)
        output.join(timeout=2)
        print("[Main] 已安全退出")


if __name__ == "__main__":
    main()