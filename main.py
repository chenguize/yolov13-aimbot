import time
import signal
import keyboard
import logging

# 引入新的大脑
from agent import AIAgent

# 日志设置
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def main():
    # 1. 实例化 Agent (它会自动初始化所有子模块)
    agent = AIAgent()

    # 2. 定义系统级控制函数
    def signal_handler(sig, frame):
        print("\n[System] Interrupt received, stopping...")
        agent.stop()

    # 3. 绑定热键与信号
    signal.signal(signal.SIGINT, signal_handler)
    keyboard.add_hotkey('p', agent.toggle_pause)
    keyboard.add_hotkey('alt+f1', agent.toggle_aimbot)

    # 4. 启动系统
    agent.start()
    print("[System] Online. Press 'P' to pause, 'Alt+F1' to toggle Aimbot.")

    # 5. 进入主循环
    # 所有的复杂性都被 agent.tick() 屏蔽了
    try:
        while not agent.shutdown_event.is_set():
            agent.tick()
    except KeyboardInterrupt:
        pass
    finally:
        agent.stop()
        keyboard.unhook_all()


if __name__ == "__main__":
    main()