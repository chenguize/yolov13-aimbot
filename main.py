import time
import signal
import sys
import logging
import keyboard
from ctypes import windll  # [关键] 引入 Windows底层库

from utils.logging_bootstrap import setup_root_logging

# 在 import Agent（进而 import output/推理栈）之前配置 root logger，否则子模块 log 与格式不一致
setup_root_logging()
logger = logging.getLogger("Main")

# 核心 Agent（依赖较多，放日志初始化之后）
from agent import AIAgent

# ==============================================================================
# [关键修复] 全局强制 Windows 定时器精度为 1ms
# 这一步必须在所有线程启动前执行。
# 它可以让 time.sleep(0.001) 真正只睡 1ms，而不是默认的 15.6ms。
# ==============================================================================
try:
    windll.winmm.timeBeginPeriod(1)
except Exception as e:
    logger.warning(f"Failed to set high resolution timer: {e}")


def _log_startup_banner() -> None:
    """首屏排查：解释器、工作目录、配置文件路径（与预期 venv 是否一致）。"""
    from pathlib import Path
    from config import config as cfg

    root = Path(__file__).resolve().parent
    ini = root / "config.ini"
    logger.info("Startup | config.ini -> %s", ini)
    if not cfg.getbool("Debug", "startup_diag", True):
        return
    logger.info("Startup | python_exe=%s", sys.executable)
    logger.info("Startup | cwd=%s", Path.cwd())
    logger.info("Startup | project_root=%s", root)


def main():
    _log_startup_banner()

    # 1. 实例化 Agent
    agent = AIAgent()

    # 2. 定义信号处理 (Ctrl+C)
    def signal_handler(sig, frame):
        logger.warning("Interrupt received, stopping...")
        agent.stop()

    # 3. 绑定热键
    # 注意：在 Windows 上 signal.SIGINT 可能无法完美捕获所有中断，
    # 配合 try-finally 块是更稳妥的做法。
    signal.signal(signal.SIGINT, signal_handler)

    keyboard.add_hotkey('p', agent.toggle_pause)
    keyboard.add_hotkey('alt+f1', agent.toggle_aimbot)

    # 4. 启动所有子线程 (Capture, Inference, MouseWorker)
    agent.start()
    logger.info("Online | hotkeys: P=pause, Alt+F1=aimbot toggle | main loop blocking on inference frame_ready")

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
        logger.warning("KeyboardInterrupt (Ctrl+C)")
    finally:
        logger.info("Shutting down...")
        agent.stop()
        keyboard.unhook_all()

        # [清理] 恢复系统定时器精度
        try:
            windll.winmm.timeEndPeriod(1)
        except Exception:
            pass
        logger.info("Goodbye.")


if __name__ == "__main__":
    main()