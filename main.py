# main.py - YOLOv13 Aimbot 主控制模块
# 
# 核心职责：
# 1. 系统初始化（创建各模块实例）
# 2. 多线程管理（捕获、推理、输出线程）
# 3. 主控制循环（目标检测、瞄准计算、触发控制）
# 4. 优雅关闭（信号处理、资源清理）
# 
# 系统架构：完全异步 + 状态驱动 + 世界模型 + 独立高频控制环
# 
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


# 全局控制变量 - 用于控制整个系统的运行状态
running = True
shutdown_event = threading.Event()


def signal_handler(sig, frame):
    """信号处理器 - 处理 Ctrl+C 退出信号"""
    global running
    print("\n[Main] 收到退出信号，正在优雅关闭...")
    shutdown_event.set()
    running = False


# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_current_mouse_pos() -> Tuple[int, int]:
    """获取当前鼠标屏幕绝对坐标"""
    return win32api.GetCursorPos()


def main():
    """主函数 - 系统入口点"""
    global running

    print("[Main] YOLOv13 Aimbot + Triggerbot 启动 (2025.12.26版) ... Ctrl+C 退出")

    # 初始化核心模块
    frame_bus = FrameBus()  # 帧状态总线 - 实现状态驱动架构
    world_model = WorldModel(frame_bus)  # 世界模型 - 状态融合、预测、裁判
    controller = get_controller()  # 控制器 - 决策、控制、拟人化
    strategy = create_aim_strategy()  # 瞄准策略 - 游戏专用预测
    output = GHUBOutput(world_model)  # 输出模块 - 执行层（命令 → 鼠标移动）

    # humanize 模块（即使全关闭也零开销）
    h_reaction = ReactionDelay()  # 反应延迟 - 模拟人类反应时间
    h_noise = Noise()  # 随机噪声 - 模拟手抖
    h_fatigue = Fatigue()  # 疲劳累积 - 长时间瞄准精度下降
    h_overshoot = Overshoot()  # 过冲回正 - 快速移动后的过冲效应
    h_curve = Curve()  # 曲线形状 - 鼠标移动轨迹曲线

    prev_dx, prev_dy = 0.0, 0.0  # 上一帧移动量，用于过冲计算

    # 启动各工作线程
    capture = CaptureThread(frame_bus, shutdown_event)  # 捕获线程 - BetterCam 抓取画面
    inference = InferenceThread(frame_bus, world_model, shutdown_event)  # 推理线程 - YOLOv13 异步推理
    output.start()  # 启动输出线程
    capture.start()  # 启动捕获线程
    inference.start()  # 启动推理线程

    last_time = time.perf_counter()  # 上一帧时间戳

    try:
        # 主控制循环 - 高频执行瞄准和触发逻辑
        while running and not shutdown_event.wait(timeout=0.003):
            now = time.perf_counter()
            dt = max(now - last_time, 1e-6)  # 时间差，防止除零错误
            last_time = now

            current_pos = get_current_mouse_pos()  # 获取当前鼠标位置
            target = world_model.get_best_target()  # 从世界模型获取最佳目标

            # Aimbot 逻辑 - 自动瞄准功能
            if target and config.getbool("General", "enable_aimbot", False):
                # 预测计算 - 根据目标运动预测未来位置
                pred_x, pred_y = strategy.calculate_prediction(
                    target["screen_x"], target["screen_y"], dt
                )
                target["screen_x"] = pred_x  # 更新目标位置为预测位置
                target["screen_y"] = pred_y

                # 计算瞄准移动量 - 使用控制器计算 dx, dy
                dx, dy = controller.compute(target, current_pos, dt)

                # Humanize 完整管道 - 应用拟人化处理
                h_reaction.apply_delay()  # 应用反应延迟
                dx, dy = h_noise.apply(dx, dy)  # 添加随机噪声
                dx, dy = h_fatigue.apply(dx, dy)  # 应用疲劳效应
                dx, dy = h_overshoot.apply(dx, dy, prev_dx, prev_dy)  # 应用过冲回正

                # 曲线整形 - 应用移动轨迹曲线
                t = min(1.0, (abs(dx) + abs(dy)) / 50.0)  # 计算曲线参数
                dx *= h_curve.apply(t)  # 应用 x 轴曲线
                dy *= h_curve.apply(t)  # 应用 y 轴曲线

                prev_dx, prev_dy = dx, dy  # 更新上一帧移动量

                output.send_move(dx, dy)  # 发送移动命令

            # Triggerbot 逻辑 - 自动触发功能
            if config.getbool("General", "enable_triggerbot", False) and controller.should_trigger(target, current_pos):
                # 添加随机触发延迟
                delay = config.getfloat("Triggerbot", "trigger_delay_min_ms", 15) / 1000
                time.sleep(delay)
                gHub.mouse_down(1)  # 按下鼠标左键
                time.sleep(config.getfloat("Triggerbot", "trigger_hold_time_ms", 40) / 1000)  # 按住时间
                gHub.mouse_up(1)  # 释放鼠标左键

    except Exception as e:
        print(f"[Main] 主循环异常: {e}")

    finally:
        print("[Main] 正在关闭...")
        shutdown_event.set()  # 设置关闭事件
        time.sleep(0.5)  # 等待线程响应
        capture.join(timeout=2)  # 等待捕获线程结束
        inference.join(timeout=2)  # 等待推理线程结束
        output.join(timeout=2)  # 等待输出线程结束
        print("[Main] 已安全退出")


if __name__ == "__main__":
    main()