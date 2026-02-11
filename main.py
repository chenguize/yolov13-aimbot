import time
import signal
import keyboard
import logging
from ctypes import windll  # [关键] 引入 Windows底层库

# 引入核心 Agent
from agent import AIAgent

# 日志设置
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("Main")

# ==============================================================================
# [关键修复] 全局强制 Windows 定时器精度为 1ms
# 这一步必须在所有线程启动前执行。
# 它可以让 time.sleep(0.001) 真正只睡 1ms，而不是默认的 15.6ms。
# ==============================================================================
try:
    windll.winmm.timeBeginPeriod(1)
except Exception as e:
    logger.warning(f"Failed to set high resolution timer: {e}")


def main():
    # 1. 实例化 Agent
    agent = AIAgent()

    # 2. 定义信号处理 (Ctrl+C)
    def signal_handler(sig, frame):
        print("\n[System] Interrupt received, stopping...")
        agent.stop()

    # 3. 绑定热键
    # 注意：在 Windows 上 signal.SIGINT 可能无法完美捕获所有中断，
    # 配合 try-finally 块是更稳妥的做法。
    signal.signal(signal.SIGINT, signal_handler)

    keyboard.add_hotkey('p', agent.toggle_pause)
    keyboard.add_hotkey('alt+f1', agent.toggle_aimbot)

    # 4. 启动所有子线程 (Capture, Inference, MouseWorker)
    agent.start()
    print("[System] Online. Press 'P' to pause, 'Alt+F1' to toggle Aimbot.")

    # 5. 进入主循环
    try:
        while not agent.shutdown_event.is_set():
            # 执行一帧逻辑 (WorldModel Step -> Decision -> Trigger)
            # 内部包含 wait_for_frame，所以这里会自动阻塞直到新一帧到来
            agent.tick()

            # [优化] 极短休眠
            # 即使 agent.tick 内部有阻塞，这里加一个极短的 sleep (0.1ms)
            # 有助于在极高 FPS (500+) 下让出少量 CPU 时间片给操作系统，
            # 避免主线程抢占导致鼠标线程微卡顿。
            time.sleep(0.0001)

    except KeyboardInterrupt:
        print("\n[System] Keyboard Interrupt (Ctrl+C)")
    finally:
        print("[System] Shutting down...")
        agent.stop()
        keyboard.unhook_all()

        # [清理] 恢复系统定时器精度
        try:
            windll.winmm.timeEndPeriod(1)
        except Exception:
            pass
        print("[System] Goodbye.")


if __name__ == "__main__":
    main()