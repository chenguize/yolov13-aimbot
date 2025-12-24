# main.py
import threading
import time
import sys
import signal
from queue import Queue
from typing import Optional

from perception.capture import CaptureThread
from perception.bus import FrameBus
from inference import InferenceThread
from world_model import WorldModel
from controllers.controller_factory import create_controller
from output_ghub import GHUBOutputQueue

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


def main():
    print("[Main] YOLOv13 Aimbot 最简完整结构版启动... (Ctrl+C 退出)")

    # 初始化核心组件
    frame_bus = FrameBus(max_history=8)
    world_model = WorldModel()
    controller = create_controller()           # 目前只返回最简比例控制器
    output_queue = Queue(maxsize=64)           # 输出指令队列
    ghub_output = GHUBOutputQueue(output_queue)

    # 启动各线程
    threads = [
        CaptureThread(frame_bus, shutdown_event),
        InferenceThread(frame_bus, world_model, shutdown_event),
        ghub_output  # 输出线程也是独立线程
    ]

    for t in threads:
        t.start()

    # 主控制循环（决策环）
    try:
        while not shutdown_event.wait(timeout=0.003):
            target = world_model.get_best_target()
            if target is None:
                continue

            # 简单计算移动量
            dx, dy = controller.compute(target)

            # 放入输出队列（带简单背压）
            try:
                output_queue.put_nowait((dx, dy))
            except Queue.Full:
                pass  # 队列满了就丢，防止内存爆炸

    except Exception as e:
        print(f"[Main] 主循环异常: {e}")

    finally:
        print("[Main] 正在关闭系统...")
        shutdown_event.set()

        # 等待所有线程
        for t in threads:
            t.join(timeout=2.0)

        print("[Main] 系统已安全退出")


if __name__ == "__main__":
    main()